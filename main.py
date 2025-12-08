#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["rich", "psutil"]
# ///


import argparse
import asyncio
import logging
import os
import re
import subprocess
import sys
import time
from asyncio import Event, Queue
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import Path, PurePath

import psutil
from rich.console import Console
from rich.progress import BarColumn, Progress, ProgressColumn, Task, TaskID, TaskProgressColumn, Text, TextColumn

from _version import __version__

LOG_LEVELS = {
    "FATAL": logging.FATAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}

logger = logging.getLogger(__name__)

# logs will be stored here
APP_DIR = Path.home() / ".local" / "share" / "ninja-so-fancy"


# in seconds
PROCESS_TREE_CHECK_INTERVAL = float(os.environ.get("NINJASOFANCY_PROCESS_TREE_CHECK_INTERVAL", "0.1"))

# certain paths will be shortened to this length for display purposes
MAX_PATH_LENGTH = int(os.environ.get("NINJASOFANCY_MAX_PATH_LENGTH", "40"))

# certain lines will be shortened to this length to reduce clutter
MAX_LINE_LENGTH = int(os.environ.get("NINJASOFANCY_MAX_LINE_LENGTH", "320"))

# see LOG_LEVELS for acceptable values
LOG_LEVEL = os.getenv("NINJASOFANCY_LOG_LEVEL", "INFO").upper()


class Severity(IntEnum):
    NOTE = 1
    WARNING = 2
    ERROR = 3
    FATAL_ERROR = 4


class TaskKind(StrEnum):
    COMPILING = "building"
    LINKING_EXE = "linking executable"
    LINKING_SHARED_LIB = "linking shared library"
    LINKING_STATIC_LIB = "linking static library"


@dataclass
class ProcInfo:
    pid: int
    name: str
    cmd: str
    status: str
    create_time: float


@dataclass
class ProgressLine:
    out_path: str
    count_current: int
    count_total: int
    kind: TaskKind


@dataclass
class CompilerLine:
    source_path: str
    line: int
    column: int
    severity: Severity
    message: list[str]


@dataclass
class FailureLine:
    code: int
    out_path: str


@dataclass
class NinjaStoppedLine:
    reason: str
    error: bool


@dataclass
class RawLine:
    output: str


type OutputLine = ProgressLine | CompilerLine | FailureLine | NinjaStoppedLine | RawLine


@dataclass
class ParsingCompilerMessage:
    line: CompilerLine


@dataclass
class ParsingFailureMessage:
    out_path: str
    lines: list[str]


type ParserState = ParsingCompilerMessage | ParsingFailureMessage | None


@dataclass
class FinishedNinjaTask:
    time: float
    out_path: str
    count_current: int | None
    count_total: int | None
    kind: TaskKind


@dataclass
class NewChildProcess:
    info: ProcInfo


@dataclass
class FinishedChildProcess:
    pid: int
    time: float


@dataclass
class FinishedObject:
    out_path: str
    time: float


@dataclass
class Finish:
    pass


@dataclass
class NinjaStopped:
    reason: str
    error: bool


@dataclass
class NinjaExited:
    exit_code: int


@dataclass(eq=True, frozen=True)
class CompilerDiagnostic:
    source_path: str
    line: int
    column: int
    severity: Severity
    message: str


@dataclass
class ErrorMessage:
    out_path: str
    lines: list[str]


type Message = (
    CompilerDiagnostic | ErrorMessage | FinishedNinjaTask | FinishedObject | NinjaStopped | NinjaExited | NewChildProcess | FinishedChildProcess | Finish
)


@dataclass
class NinjaTask:
    out_path: str
    start_time: float
    end_time: float | None
    kind: TaskKind
    proc: ProcInfo | None


@dataclass
class AppState:
    root_dir: str
    tasks: dict[str, NinjaTask]
    child_procs: dict[int, ProcInfo]  # key=pid
    diagnostics: list[CompilerDiagnostic]
    error_messages: list[ErrorMessage]
    watched_dirs: set[str]
    count_current: int | None
    count_total: int | None
    stopped: bool
    stopped_reason: str | int | None
    stopped_error: bool


Q: Queue[Message] = Queue(100)

S = AppState(".", {}, {}, [], [], set(), None, None, False, None, False)

ev_state_changed = Event()

console = Console(soft_wrap=True)


def severity_from_string(s: str) -> Severity:
    match s.strip().lower():
        case "note":
            return Severity.NOTE
        case "warning":
            return Severity.WARNING
        case "error":
            return Severity.ERROR
        case "fatal error":
            return Severity.FATAL_ERROR
    raise ValueError(f'Cannot parse "{s}" to Severity')


def shorten_path(path: str) -> str:
    path = os.path.normpath(path)
    parts = path.split(os.sep)
    path_short = path
    m = 0
    while len(path_short) > MAX_PATH_LENGTH:
        m += 1
        if m >= len(parts):
            return f"...{path_short[-(MAX_PATH_LENGTH - 3) :]}"
        parts_shortened = [part[0] if part and i < m else part for i, part in enumerate(parts)]
        path_short = os.sep.join(parts_shortened)
    return path_short


def shorten_string(s: str, maxlen: int) -> str:
    if len(s) > maxlen:
        return f"{s[: maxlen - 3]}..."
    return s


def get_process_tree(parent_pid: int) -> list[psutil.Process]:
    try:
        parent = psutil.Process(parent_pid)
        children = parent.children(recursive=True)
        return children
    except psutil.NoSuchProcess:
        return []


def extract_process_info(proc: psutil.Process) -> ProcInfo | None:
    try:
        return ProcInfo(proc.pid, proc.name(), " ".join(proc.cmdline()), proc.status(), proc.create_time())
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        return None


def outfile_from_cmd(cmd: str) -> str | None:
    pattern = r"-o\s+(\S+)"
    match = re.search(pattern, cmd)
    if match:
        return match.group(1)


def task_kind_from_outfile(outfile: str) -> TaskKind:
    if outfile.endswith(".o"):
        return TaskKind.COMPILING
    elif outfile.endswith(".so") or outfile.endswith(".dylib"):
        return TaskKind.LINKING_SHARED_LIB
    elif outfile.endswith(".a"):
        return TaskKind.LINKING_STATIC_LIB
    else:
        return TaskKind.LINKING_EXE


def task_kind_from_cmd(cmd: str) -> TaskKind | None:
    out_path = outfile_from_cmd(cmd)
    if out_path is not None:
        if out_path.endswith(".o"):
            return TaskKind.COMPILING
        elif out_path.endswith(".so") or out_path.endswith(".dylib"):
            return TaskKind.LINKING_SHARED_LIB
        elif out_path.endswith(".a"):
            return TaskKind.LINKING_STATIC_LIB
        else:
            return TaskKind.LINKING_EXE


async def monitor_process_tree(parent_pid: int) -> None:
    seen_pids: dict[int, ProcInfo | None] = {parent_pid: None}

    while True:
        try:
            parent = psutil.Process(parent_pid)
            if not parent.is_running():
                break
        except psutil.NoSuchProcess:
            break

        children = get_process_tree(parent_pid)

        for child in children:
            if child.pid not in seen_pids:
                info = extract_process_info(child)
                if info is not None:
                    logger.debug(f"new child process detected: {info}")
                    seen_pids[child.pid] = info
                    await Q.put(NewChildProcess(info))
                else:
                    seen_pids[child.pid] = None

        for pid, info in seen_pids.items():
            if info is not None:
                try:
                    child = psutil.Process(pid)
                    if not child.is_running():
                        logger.debug(f"child process finished: {pid}")
                        await Q.put(FinishedChildProcess(pid, time.time()))
                except psutil.NoSuchProcess:
                    logger.debug(f"child process finished: {pid}")
                    await Q.put(FinishedChildProcess(pid, time.time()))

        await asyncio.sleep(PROCESS_TREE_CHECK_INTERVAL)


async def handle_messages() -> None:
    while True:
        msg = await Q.get()
        match msg:
            case NewChildProcess(info):
                out_path = outfile_from_cmd(info.cmd)
                if out_path is not None:
                    if not os.path.isabs(out_path):
                        out_path = os.path.join(S.root_dir, out_path)
                    kind = task_kind_from_outfile(out_path)
                    S.tasks[out_path] = NinjaTask(out_path, info.create_time, None, kind, info)
                    S.child_procs[info.pid] = info
                    ev_state_changed.set()
            case FinishedChildProcess(pid, end_time):
                if pid in S.child_procs:
                    proc = S.child_procs[pid]
                    out_path = outfile_from_cmd(proc.cmd)
                    if out_path is not None:
                        if not os.path.isabs(out_path):
                            out_path = os.path.join(S.root_dir, out_path)
                        task = S.tasks[out_path]
                        if task.proc is not None and task.proc.pid == pid:
                            S.tasks[out_path].end_time = end_time
                            ev_state_changed.set()
            case FinishedNinjaTask(time_, out_path, count_current, count_total, kind):
                if not os.path.isabs(out_path):
                    out_path = os.path.join(S.root_dir, out_path)
                S.tasks[out_path] = NinjaTask(out_path, time_, time_, kind, None)
                if count_current is not None:
                    S.count_current = count_current
                if count_total is not None:
                    S.count_total = count_total
                ev_state_changed.set()
            case CompilerDiagnostic() as cd:
                S.diagnostics.append(cd)
                ev_state_changed.set()
            case ErrorMessage() as msg:
                S.error_messages.append(msg)
                ev_state_changed.set()
            case FinishedObject(out_path, time):
                if not os.path.isabs(out_path):
                    out_path = os.path.join(S.root_dir, out_path)
                if out_path in S.tasks:
                    task = S.tasks[out_path]
                    if task.end_time is None:
                        task.end_time = max(task.start_time, time)
                        ev_state_changed.set()
            case NinjaStopped(reason, is_error):
                S.stopped = True
                S.stopped_reason = reason
                S.stopped_error = is_error
                ev_state_changed.set()
            case NinjaExited(exit_code):
                S.stopped = True
                S.stopped_reason = exit_code
                S.stopped_error = exit_code != 0
                ev_state_changed.set()
            case Finish():
                await asyncio.sleep(0.1)
                ev_state_changed.set()
                break


def parse_line(line: str) -> OutputLine:
    if "ninja: no work to do" in line:
        return NinjaStoppedLine("no work to do", False)

    pattern = r"^ninja:\s+error:\s+(.+)$"
    match = re.search(pattern, line)
    if match:
        error_msg = match.group(1).strip()
        return NinjaStoppedLine(error_msg, True)

    pattern = r"^ninja:\s+build stopped:\s+(.+)\.$"
    match = re.search(pattern, line)
    if match:
        reason = match.group(1).strip()
        return NinjaStoppedLine(reason, True)

    # extract progress and file path from Ninja build output
    pattern = r"\[(\d+)/(\d+)\]\s+(.+)$"
    match = re.search(pattern, line)
    if match:
        current = match.group(1)
        total = match.group(2)
        cmd = match.group(3).strip()
        outfile = outfile_from_cmd(cmd)
        kind = task_kind_from_cmd(cmd)

        if outfile is not None and kind is not None:
            return ProgressLine(outfile.strip(), int(current), int(total), kind)
        else:
            logger.debug(f"ignoring progess line: {line}")

    # extract components from compiler error/warning messages
    pattern = r"^(.+?):(\d+):(\d+):\s+(fatal error|error|warning|note):\s+(.+)$"
    match = re.search(pattern, line)
    if match:
        filepath = match.group(1).strip()
        line_num = match.group(2)
        col_num = match.group(3)
        severity = match.group(4)
        message = match.group(5).strip()
        return CompilerLine(filepath, int(line_num), int(col_num), severity_from_string(severity), [message])

    # extract error code and filepath from "FAILED" messages
    pattern = r"^FAILED:\s+\[code=(\d+)\]\s+([\w/\-\.]+)(.*)$"
    match = re.search(pattern, line)
    if match:
        error_code = match.group(1)
        filepath = match.group(2).strip()
        return FailureLine(int(error_code), filepath)

    return RawLine(line)


async def process_line(state: ParserState, line: OutputLine) -> ParserState:
    match line:
        case CompilerLine() as cl:
            match state:
                case ParsingCompilerMessage(m):
                    await Q.put(CompilerDiagnostic(m.source_path, m.line, m.column, m.severity, "\n".join(m.message)))
                case ParsingFailureMessage(out_path, lines):
                    await Q.put(ErrorMessage(out_path, lines))
            return ParsingCompilerMessage(cl)
        case RawLine() as rl:
            match state:
                case ParsingCompilerMessage(m):
                    m.message.append(rl.output)
                case ParsingFailureMessage(m, lines):
                    lines.append(rl.output)
            return state
        case FailureLine(_, out_path):
            match state:
                case ParsingCompilerMessage(m):
                    await Q.put(CompilerDiagnostic(m.source_path, m.line, m.column, m.severity, "\n".join(m.message)))
                case ParsingFailureMessage(out_path, lines):
                    await Q.put(ErrorMessage(out_path, lines))
            return ParsingFailureMessage(out_path, [])
        case ProgressLine(out_path, count_current, count_total, kind):
            match state:
                case ParsingCompilerMessage(m):
                    await Q.put(CompilerDiagnostic(m.source_path, m.line, m.column, m.severity, "\n".join(m.message)))
                case ParsingFailureMessage(out_path, lines):
                    await Q.put(ErrorMessage(out_path, lines))
            if not os.path.isabs(out_path):
                out_path = os.path.join(S.root_dir, out_path)
            await Q.put(FinishedNinjaTask(time.time(), out_path, count_current, count_total, kind))
            return None
        case NinjaStoppedLine(reason, is_error):
            match state:
                case ParsingCompilerMessage(m):
                    await Q.put(CompilerDiagnostic(m.source_path, m.line, m.column, m.severity, "\n".join(m.message)))
                case ParsingFailureMessage(out_path, lines):
                    await Q.put(ErrorMessage(out_path, lines))
            await Q.put(NinjaStopped(reason, is_error))
            return None


async def process_stream(stream):
    state = None

    try:
        while True:
            line = await stream.readline()

            # Empty line means EOF
            if not line:
                break

            text: str = line.decode("utf-8").rstrip("\n\r")

            state = await process_line(state, parse_line(text))

    except Exception:
        logger.exception("error reading stream")


async def run_subprocess(command: list[str]) -> None:
    try:
        process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        await asyncio.gather(process_stream(process.stdout), monitor_process_tree(process.pid))
        exit_code = await process.wait()
        await Q.put(NinjaExited(exit_code))

    except Exception:
        logger.exception("error running subprocess")


def keep_task_after_finish(task: NinjaTask) -> bool:
    return task.kind != TaskKind.COMPILING and PurePath(task.out_path).is_relative_to(S.root_dir)


class SecondsElapsedColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            return Text("-", style="progress.elapsed")
        return Text(f"{elapsed:6.1f}s", style="progress.elapsed")


async def render_loop() -> None:
    columns = [
        TextColumn("[progress.description]{task.description}"),
        TextColumn("{task.fields[proc_name]}", justify="left", style="cyan"),
        BarColumn(bar_width=10),
        TaskProgressColumn(),
        # TimeElapsedColumn(),
        SecondsElapsedColumn(),
    ]
    task_ids: dict[str, TaskID] = {}
    with Progress(*columns, console=console) as progress:
        diags_seen: set[CompilerDiagnostic] = set()
        errors_seen: set[str] = set()

        max_severity_seen = Severity.NOTE

        id0 = progress.add_task("🥷", total=None, proc_name="")

        keep_going = True

        while True:
            await ev_state_changed.wait()

            for diag in S.diagnostics:
                if diag not in diags_seen:
                    diags_seen.add(diag)
                    max_severity_seen = max(max_severity_seen, diag.severity)
                    match diag.severity:
                        case Severity.FATAL_ERROR:
                            console.print(f"🛑 [red]{diag.source_path} {diag.line}:{diag.column}")
                            console.print(f"[bold red]fatal error[/bold red]: {diag.message}")
                            console.line()
                        case Severity.ERROR:
                            console.print(f"‼️ [red]{diag.source_path} {diag.line}:{diag.column}")
                            console.print(f"[bold red]error[/bold red]: {diag.message}")
                            console.line()
                        case Severity.WARNING:
                            console.print(f"⚠️ [yellow]{diag.source_path} {diag.line}:{diag.column}")
                            console.print(f"[bold yellow]warning[/bold yellow]: {diag.message}")
                            console.line()
                        case Severity.NOTE:
                            console.print(f"💡 [magenta]{diag.source_path} {diag.line}:{diag.column}")
                            console.print(f"[bold magenta]note[/bold magenta]: {diag.message}")
                            console.line()

            for em in S.error_messages:
                m = "\n".join([shorten_string(line, MAX_LINE_LENGTH) for line in em.lines])
                if m not in errors_seen:
                    errors_seen.add(m)
                    # after the first compile error, we skip printing non-compiler errors for readability.
                    if max_severity_seen < Severity.ERROR:
                        console.print(f"‼️ [red]{em.out_path}")
                        console.print(m)
                        console.line()

            progress.update(id0, total=S.count_total, completed=S.count_current)

            for out_path, task in S.tasks.items():
                if not PurePath(out_path).is_relative_to(S.root_dir):  # don't render progress on temporary files
                    continue

                extras: dict = {"proc_name": f"({shorten_string(task.proc.name, 10)})" if task.proc is not None else ""}

                if out_path not in task_ids:
                    match task.kind:
                        case TaskKind.COMPILING:
                            task_ids[out_path] = progress.add_task(f"🛠️ {shorten_path(os.path.relpath(out_path, S.root_dir))}", total=None, **extras)
                        case TaskKind.LINKING_EXE:
                            task_ids[out_path] = progress.add_task(f"⚡️ {shorten_path(os.path.relpath(out_path, S.root_dir))}", total=None, **extras)
                        case TaskKind.LINKING_SHARED_LIB:
                            task_ids[out_path] = progress.add_task(f"📚 {shorten_path(os.path.relpath(out_path, S.root_dir))}", total=None, **extras)
                        case TaskKind.LINKING_STATIC_LIB:
                            task_ids[out_path] = progress.add_task(f"📚 {shorten_path(os.path.relpath(out_path, S.root_dir))}", total=None, **extras)

                tid = task_ids[out_path]

                if tid in progress.task_ids:
                    if task.end_time is not None:
                        progress.update(tid, total=100, completed=100, proc_name="", refresh=True)
                        if not keep_task_after_finish(task):
                            progress.remove_task(tid)
                    else:
                        progress.update(tid, **extras)

            if not keep_going:
                progress.remove_task(id0)
                progress.refresh()
                break

            if S.stopped:
                if not S.stopped_error:
                    for out_path, task in S.tasks.items():
                        tid = task_ids.get(out_path)
                        if tid is not None and tid in progress.task_ids:
                            progress.update(tid, total=100, completed=100, proc_name="", refresh=True)
                            if not keep_task_after_finish(task):
                                progress.remove_task(tid)
                    if S.count_total is not None and S.count_total > 0:
                        progress.update(id0, completed=S.count_total, refresh=True)
                    else:
                        progress.update(id0, total=100, completed=100, refresh=True)
                keep_going = False
                await Q.put(Finish())

            ev_state_changed.clear()

    if len(task_ids) > 0:
        console.line()


async def main_async(ninja_args) -> None:
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(run_subprocess(["ninja", "-v", *ninja_args]))
            tg.create_task(handle_messages())
            tg.create_task(render_loop())
    except* Exception:
        logger.exception("exception in task group")


def shortcircuit(ninja_args: list[str]) -> bool:
    for arg in ninja_args:
        if arg.strip() == "--version":  # print ninja version and exit
            return True
        if arg.startswith("cmTC_"):  # CMake try-compile targets
            return True
        if "/CMakeFiles/CMakeTmp" in arg:  # CMake try-compile stuff
            return True
        if "/CMakeFiles/CMakeScratch" in arg:  # CMake try-compile stuff
            return True
    return False


def main():
    APP_DIR.mkdir(parents=True, exist_ok=True)

    log_level = LOG_LEVELS.get(LOG_LEVEL, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(APP_DIR / "ninja-so-fancy.log")],
    )

    ninja_args = sys.argv[1:]

    logger.info(f"started with args: {ninja_args}")

    if shortcircuit(ninja_args):
        logger.info("short circuiting...")
        proc = subprocess.run(["ninja", *ninja_args])
        return proc.returncode

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-C",
        help="change to DIR before doing anything else",
        metavar="DIR",
    )

    parser.add_argument(
        "-f",
        help="specify the input build file [default=build.ninja]",
        metavar="FILE",
    )

    parser.add_argument(
        "--nsf-version",
        action="store_true",
        help="print ninja-so-fancy version and exit",
    )

    args, _ = parser.parse_known_args()

    if args.nsf_version:
        print(__version__)
        return 0

    try:
        ninja_proc = subprocess.run(["ninja", "--version"], capture_output=True)
        ninja_version = ninja_proc.stdout.decode("utf-8").strip()
    except FileNotFoundError:
        print("error: ninja executable not found", file=sys.stderr)
        return 1
    except Exception as ex:
        print("failed to invoke `ninja --version`; be sure a recent version of ninja is on your PATH!", file=sys.stderr)
        print(f"error message: {ex}", file=sys.stderr)
        return 1

    ninja_version_match = re.search(r"(\d+)\.(\d+)\.(\w+)", ninja_version)
    if ninja_version_match:
        major = int(ninja_version_match.group(1))
        minor = int(ninja_version_match.group(2))
        # patch = ninja_version_match.group(3)
        if (major < 1) or (major < 2 and minor < 10):
            print(f"warning: old version of ninja detected: {ninja_version}", file=sys.stderr)
    else:
        print("output of `ninja --version` doesn't have expected format.", file=sys.stderr)
        return 1

    S.root_dir = os.getcwd()

    if args.C is not None:
        S.root_dir = os.path.abspath(args.C)

    if args.f is not None:
        S.root_dir = os.path.dirname(os.path.abspath(args.f))

    t_start = time.time()

    asyncio.run(main_async(ninja_args))

    t_elapsed = time.time() - t_start

    if S.stopped_reason:
        if S.stopped_error:
            console.print(f"🥷 [red]failed ({S.stopped_reason})")
        else:
            console.print(f"🥷 [green]finished ({S.stopped_reason})")
    else:
        if S.stopped_error:
            console.print("🥷 [red]failed")
        else:
            console.print(f"🥷 [green]finished in {t_elapsed:.1f}s")


if __name__ == "__main__":
    main()
