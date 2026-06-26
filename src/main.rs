use std::collections::{HashMap, HashSet};
use std::env;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use console::{style, Style};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use regex::Regex;
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System, UpdateKind};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio::sync::Mutex;
use tokio::sync::Notify;

const VERSION: &str = env!("CARGO_PKG_VERSION");

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

fn env_f64(name: &str, default: f64) -> f64 {
    env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Severity {
    Note,
    Warning,
    Error,
    FatalError,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TaskKind {
    Compiling,
    LinkingExe,
    LinkingSharedLib,
    LinkingStaticLib,
}

#[derive(Debug, Clone)]
struct ProcInfo {
    pid: u32,
    name: String,
    cmd: String,
    create_time: f64,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct NinjaTask {
    out_path: String,
    start_time: f64,
    end_time: Option<f64>,
    kind: TaskKind,
    proc_name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CompilerDiagnostic {
    source_path: String,
    line: u32,
    column: u32,
    severity: Severity,
    message: String,
}

#[derive(Debug, Clone)]
struct ErrorMessage {
    out_path: String,
    lines: Vec<String>,
}

// ---------------------------------------------------------------------------
// Messages (sent between tasks)
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum Message {
    NewChildProcess(ProcInfo),
    FinishedChildProcess { pid: u32, time: f64 },
    FinishedNinjaTask {
        time: f64,
        out_path: String,
        count_current: Option<u64>,
        count_total: Option<u64>,
        kind: TaskKind,
    },
    CompilerDiag(CompilerDiagnostic),
    Error(ErrorMessage),
    NinjaStopped { reason: String, is_error: bool },
    NinjaExited { exit_code: i32 },
    Finish,
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

struct AppState {
    root_dir: PathBuf,
    tasks: HashMap<String, NinjaTask>,
    child_procs: HashMap<u32, ProcInfo>,
    diagnostics: Vec<CompilerDiagnostic>,
    error_messages: Vec<ErrorMessage>,
    count_current: Option<u64>,
    count_total: Option<u64>,
    stopped: bool,
    stopped_reason: Option<String>,
    stopped_error: bool,
}

impl AppState {
    fn new(root_dir: PathBuf) -> Self {
        Self {
            root_dir,
            tasks: HashMap::new(),
            child_procs: HashMap::new(),
            diagnostics: Vec::new(),
            error_messages: Vec::new(),
            count_current: None,
            count_total: None,
            stopped: false,
            stopped_reason: None,
            stopped_error: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

fn outfile_from_cmd(cmd: &str) -> Option<&str> {
    let re = Regex::new(r"-o\s+(\S+)").unwrap();
    re.captures(cmd).map(|c| c.get(1).unwrap().as_str())
}

fn task_kind_from_outfile(outfile: &str) -> TaskKind {
    if outfile.ends_with(".o") {
        TaskKind::Compiling
    } else if outfile.ends_with(".so") || outfile.ends_with(".dylib") {
        TaskKind::LinkingSharedLib
    } else if outfile.ends_with(".a") {
        TaskKind::LinkingStaticLib
    } else {
        TaskKind::LinkingExe
    }
}

enum OutputLine {
    Progress {
        out_path: String,
        count_current: u64,
        count_total: u64,
        kind: TaskKind,
    },
    Compiler {
        source_path: String,
        line: u32,
        column: u32,
        severity: Severity,
        message: String,
    },
    Failure {
        out_path: String,
    },
    NinjaStopped {
        reason: String,
        is_error: bool,
    },
    Raw {
        text: String,
    },
}

fn severity_from_str(s: &str) -> Option<Severity> {
    match s.trim().to_lowercase().as_str() {
        "note" => Some(Severity::Note),
        "warning" => Some(Severity::Warning),
        "error" => Some(Severity::Error),
        "fatal error" => Some(Severity::FatalError),
        _ => None,
    }
}

struct LineParser {
    re_no_work: Regex,
    re_ninja_error: Regex,
    re_build_stopped: Regex,
    re_progress: Regex,
    re_compiler: Regex,
    re_failed: Regex,
    re_outfile: Regex,
}

impl LineParser {
    fn new() -> Self {
        Self {
            re_no_work: Regex::new(r"ninja: no work to do").unwrap(),
            re_ninja_error: Regex::new(r"^ninja:\s+error:\s+(.+)$").unwrap(),
            re_build_stopped: Regex::new(r"^ninja:\s+build stopped:\s+(.+)\.$").unwrap(),
            re_progress: Regex::new(r"\[(\d+)/(\d+)\]\s+(.+)$").unwrap(),
            re_compiler: Regex::new(
                r"^(.+?):(\d+):(\d+):\s+(fatal error|error|warning|note):\s+(.+)$",
            )
            .unwrap(),
            re_failed: Regex::new(r"^FAILED:\s+\[code=(\d+)\]\s+([\w/\-\.]+)").unwrap(),
            re_outfile: Regex::new(r"-o\s+(\S+)").unwrap(),
        }
    }

    fn parse(&self, line: &str) -> OutputLine {
        if self.re_no_work.is_match(line) {
            return OutputLine::NinjaStopped {
                reason: "no work to do".into(),
                is_error: false,
            };
        }

        if let Some(caps) = self.re_ninja_error.captures(line) {
            return OutputLine::NinjaStopped {
                reason: caps[1].trim().to_string(),
                is_error: true,
            };
        }

        if let Some(caps) = self.re_build_stopped.captures(line) {
            return OutputLine::NinjaStopped {
                reason: caps[1].trim().to_string(),
                is_error: true,
            };
        }

        if let Some(caps) = self.re_progress.captures(line) {
            let current: u64 = caps[1].parse().unwrap();
            let total: u64 = caps[2].parse().unwrap();
            let cmd = caps[3].trim();

            if let Some(outfile_caps) = self.re_outfile.captures(cmd) {
                let outfile = outfile_caps[1].to_string();
                let kind = task_kind_from_outfile(&outfile);
                return OutputLine::Progress {
                    out_path: outfile,
                    count_current: current,
                    count_total: total,
                    kind,
                };
            }
        }

        if let Some(caps) = self.re_compiler.captures(line) {
            let filepath = caps[1].trim().to_string();
            let line_num: u32 = caps[2].parse().unwrap();
            let col_num: u32 = caps[3].parse().unwrap();
            let severity_str = &caps[4];
            let message = caps[5].trim().to_string();

            if let Some(severity) = severity_from_str(severity_str) {
                return OutputLine::Compiler {
                    source_path: filepath,
                    line: line_num,
                    column: col_num,
                    severity,
                    message,
                };
            }
        }

        if let Some(caps) = self.re_failed.captures(line) {
            let filepath = caps[2].trim().to_string();
            return OutputLine::Failure { out_path: filepath };
        }

        OutputLine::Raw {
            text: line.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Parser state machine (multi-line messages)
// ---------------------------------------------------------------------------

enum ParserState {
    Idle,
    ParsingCompiler {
        source_path: String,
        line: u32,
        column: u32,
        severity: Severity,
        messages: Vec<String>,
    },
    ParsingFailure {
        out_path: String,
        lines: Vec<String>,
    },
}

struct StreamParser {
    state: ParserState,
    tx: mpsc::Sender<Message>,
    root_dir: PathBuf,
}

impl StreamParser {
    fn new(tx: mpsc::Sender<Message>, root_dir: PathBuf) -> Self {
        Self {
            state: ParserState::Idle,
            tx,
            root_dir,
        }
    }

    async fn flush_state(&mut self) {
        match std::mem::replace(&mut self.state, ParserState::Idle) {
            ParserState::Idle => {}
            ParserState::ParsingCompiler {
                source_path,
                line,
                column,
                severity,
                messages,
            } => {
                let _ = self
                    .tx
                    .send(Message::CompilerDiag(CompilerDiagnostic {
                        source_path,
                        line,
                        column,
                        severity,
                        message: messages.join("\n"),
                    }))
                    .await;
            }
            ParserState::ParsingFailure { out_path, lines } => {
                let _ = self
                    .tx
                    .send(Message::Error(ErrorMessage { out_path, lines }))
                    .await;
            }
        }
    }

    fn resolve_path(&self, path: &str) -> String {
        let p = Path::new(path);
        if p.is_absolute() {
            path.to_string()
        } else {
            self.root_dir.join(path).to_string_lossy().to_string()
        }
    }

    async fn process_line(&mut self, output: OutputLine) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        match output {
            OutputLine::Compiler {
                source_path,
                line,
                column,
                severity,
                message,
            } => {
                self.flush_state().await;
                self.state = ParserState::ParsingCompiler {
                    source_path,
                    line,
                    column,
                    severity,
                    messages: vec![message],
                };
            }
            OutputLine::Raw { text } => match &mut self.state {
                ParserState::ParsingCompiler { messages, .. } => {
                    messages.push(text);
                }
                ParserState::ParsingFailure { lines, .. } => {
                    lines.push(text);
                }
                ParserState::Idle => {}
            },
            OutputLine::Failure { out_path } => {
                self.flush_state().await;
                self.state = ParserState::ParsingFailure {
                    out_path,
                    lines: Vec::new(),
                };
            }
            OutputLine::Progress {
                out_path,
                count_current,
                count_total,
                kind,
            } => {
                self.flush_state().await;
                let resolved = self.resolve_path(&out_path);
                let _ = self
                    .tx
                    .send(Message::FinishedNinjaTask {
                        time: now,
                        out_path: resolved,
                        count_current: Some(count_current),
                        count_total: Some(count_total),
                        kind,
                    })
                    .await;
            }
            OutputLine::NinjaStopped { reason, is_error } => {
                self.flush_state().await;
                let _ = self
                    .tx
                    .send(Message::NinjaStopped { reason, is_error })
                    .await;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Path shortening
// ---------------------------------------------------------------------------

fn shorten_path(path: &str, max_len: usize) -> String {
    let normalized = Path::new(path)
        .components()
        .collect::<PathBuf>()
        .to_string_lossy()
        .to_string();
    let parts: Vec<&str> = normalized.split('/').collect();

    if normalized.len() <= max_len {
        return normalized;
    }

    let mut m = 0;
    let mut short = normalized.clone();
    while short.len() > max_len {
        m += 1;
        if m >= parts.len() {
            let suffix =
                &normalized[normalized.len().saturating_sub(max_len.saturating_sub(3))..];
            return format!("...{}", suffix);
        }
        let shortened_parts: Vec<String> = parts
            .iter()
            .enumerate()
            .map(|(i, part)| {
                if i < m && !part.is_empty() {
                    part.chars().next().unwrap().to_string()
                } else {
                    part.to_string()
                }
            })
            .collect();
        short = shortened_parts.join("/");
    }
    short
}

fn shorten_string(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    } else {
        s.to_string()
    }
}

// ---------------------------------------------------------------------------
// Process tree monitoring
// ---------------------------------------------------------------------------

async fn monitor_process_tree(parent_pid: u32, tx: mpsc::Sender<Message>, interval: Duration) {
    let mut sys = System::new();
    let mut seen_pids: HashMap<u32, Option<ProcInfo>> = HashMap::new();
    seen_pids.insert(parent_pid, None);

    let refresh_kind = ProcessRefreshKind::nothing()
        .with_cmd(UpdateKind::Always)
        .with_exe(UpdateKind::Always);

    loop {
        sys.refresh_processes_specifics(ProcessesToUpdate::All, true, refresh_kind);

        let parent_sysinfo_pid = Pid::from_u32(parent_pid);
        if sys.process(parent_sysinfo_pid).is_none() {
            break;
        }

        let children: Vec<(u32, String, String, f64)> = sys
            .processes()
            .iter()
            .filter(|(_, proc_)| {
                proc_.parent().is_some_and(|ppid| {
                    ppid == parent_sysinfo_pid
                        || sys
                            .process(ppid)
                            .and_then(|p| p.parent())
                            .is_some_and(|gppid| gppid == parent_sysinfo_pid)
                })
            })
            .map(|(pid, proc_)| {
                let cmd_str = proc_
                    .cmd()
                    .iter()
                    .map(|s| s.to_string_lossy().to_string())
                    .collect::<Vec<_>>()
                    .join(" ");
                (
                    pid.as_u32(),
                    proc_.name().to_string_lossy().to_string(),
                    cmd_str,
                    proc_.start_time() as f64,
                )
            })
            .collect();

        for (pid, name, cmd, create_time) in &children {
            if !seen_pids.contains_key(pid) {
                let info = ProcInfo {
                    pid: *pid,
                    name: name.clone(),
                    cmd: cmd.clone(),
                    create_time: *create_time,
                };
                log::debug!("new child process detected: {:?}", info);
                seen_pids.insert(*pid, Some(info.clone()));
                let _ = tx.send(Message::NewChildProcess(info)).await;
            }
        }

        let active_pids: HashSet<u32> = children.iter().map(|(pid, ..)| *pid).collect();
        let finished: Vec<u32> = seen_pids
            .iter()
            .filter(|(pid, info)| {
                info.is_some() && !active_pids.contains(pid) && **pid != parent_pid
            })
            .map(|(pid, _)| *pid)
            .collect();

        for pid in finished {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64();
            log::debug!("child process finished: {}", pid);
            let _ = tx
                .send(Message::FinishedChildProcess { pid, time: now })
                .await;
            seen_pids.remove(&pid);
        }

        tokio::time::sleep(interval).await;
    }
}

// ---------------------------------------------------------------------------
// Message handler (updates state)
// ---------------------------------------------------------------------------

async fn handle_messages(
    state: Arc<Mutex<AppState>>,
    mut rx: mpsc::Receiver<Message>,
    notify: Arc<Notify>,
) {
    while let Some(msg) = rx.recv().await {
        let mut s = state.lock().await;
        match msg {
            Message::NewChildProcess(info) => {
                if let Some(out_path) = outfile_from_cmd(&info.cmd) {
                    let resolved = if Path::new(out_path).is_absolute() {
                        out_path.to_string()
                    } else {
                        s.root_dir.join(out_path).to_string_lossy().to_string()
                    };
                    let kind = task_kind_from_outfile(out_path);
                    s.tasks.insert(
                        resolved.clone(),
                        NinjaTask {
                            out_path: resolved,
                            start_time: info.create_time,
                            end_time: None,
                            kind,
                            proc_name: Some(info.name.clone()),
                        },
                    );
                    s.child_procs.insert(info.pid, info);
                }
            }
            Message::FinishedChildProcess { pid, time } => {
                if let Some(proc_) = s.child_procs.get(&pid) {
                    if let Some(out_path) = outfile_from_cmd(&proc_.cmd) {
                        let resolved = if Path::new(out_path).is_absolute() {
                            out_path.to_string()
                        } else {
                            s.root_dir.join(out_path).to_string_lossy().to_string()
                        };
                        if let Some(task) = s.tasks.get_mut(&resolved) {
                            if task.end_time.is_none() {
                                task.end_time = Some(time);
                            }
                        }
                    }
                }
            }
            Message::FinishedNinjaTask {
                time,
                out_path,
                count_current,
                count_total,
                kind,
            } => {
                let task = s.tasks.entry(out_path.clone()).or_insert(NinjaTask {
                    out_path,
                    start_time: time,
                    end_time: None,
                    kind,
                    proc_name: None,
                });
                task.end_time = Some(time);
                if let Some(c) = count_current {
                    s.count_current = Some(c);
                }
                if let Some(t) = count_total {
                    s.count_total = Some(t);
                }
            }
            Message::CompilerDiag(diag) => {
                s.diagnostics.push(diag);
            }
            Message::Error(err) => {
                s.error_messages.push(err);
            }
            Message::NinjaStopped { reason, is_error } => {
                s.stopped = true;
                s.stopped_reason = Some(reason);
                s.stopped_error = is_error;
            }
            Message::NinjaExited { exit_code } => {
                s.stopped = true;
                s.stopped_reason = Some(exit_code.to_string());
                s.stopped_error = exit_code != 0;
            }
            Message::Finish => {
                notify.notify_one();
                break;
            }
        }
        notify.notify_one();
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

fn kind_emoji(kind: TaskKind) -> &'static str {
    match kind {
        TaskKind::Compiling => "\u{1f6e0}\u{fe0f} ",
        TaskKind::LinkingExe => "\u{26a1}\u{fe0f}",
        TaskKind::LinkingSharedLib => "\u{1f4da}",
        TaskKind::LinkingStaticLib => "\u{1f4da}",
    }
}

fn keep_task_after_finish(task: &NinjaTask, root_dir: &Path) -> bool {
    task.kind != TaskKind::Compiling && Path::new(&task.out_path).starts_with(root_dir)
}

async fn render_loop(
    state: Arc<Mutex<AppState>>,
    notify: Arc<Notify>,
    max_path_length: usize,
    max_line_length: usize,
) {
    let mp = MultiProgress::new();

    let overall_style =
        ProgressStyle::with_template("\u{1f977} [{bar:20.cyan/blue}] {pos}/{len}")
            .unwrap()
            .progress_chars("=> ");

    let task_style =
        ProgressStyle::with_template("  {spinner:.cyan} {prefix} {wide_msg}")
            .unwrap()
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏", "✓"]);

    let overall_pb = mp.add(ProgressBar::new(0));
    overall_pb.set_style(overall_style.clone());
    overall_pb.enable_steady_tick(Duration::from_millis(100));

    let mut task_bars: HashMap<String, ProgressBar> = HashMap::new();
    let mut task_start_times: HashMap<String, Instant> = HashMap::new();
    let mut task_frozen_elapsed: HashMap<String, f64> = HashMap::new();
    let mut diags_seen: HashSet<CompilerDiagnostic> = HashSet::new();
    let mut errors_seen: HashSet<String> = HashSet::new();
    let mut max_severity = Severity::Note;
    let mut finished = false;
    let mut tick_interval = tokio::time::interval(Duration::from_millis(100));

    loop {
        tokio::select! {
            _ = notify.notified() => {}
            _ = tick_interval.tick() => {}
        }

        let s = state.lock().await;
        let term_width = console::Term::stdout().size().1 as usize;

        // Print diagnostics
        for diag in &s.diagnostics {
            if diags_seen.contains(diag) {
                continue;
            }
            diags_seen.insert(diag.clone());
            if diag.severity > max_severity {
                max_severity = diag.severity;
            }

            let (icon, sev_style, label) = match diag.severity {
                Severity::FatalError => ("\u{1f6d1}", Style::new().red().bold(), "fatal error"),
                Severity::Error => ("\u{203c}\u{fe0f} ", Style::new().red().bold(), "error"),
                Severity::Warning => {
                    ("\u{26a0}\u{fe0f} ", Style::new().yellow().bold(), "warning")
                }
                Severity::Note => ("\u{1f4a1}", Style::new().magenta().bold(), "note"),
            };

            let loc_style = match diag.severity {
                Severity::FatalError | Severity::Error => Style::new().red(),
                Severity::Warning => Style::new().yellow(),
                Severity::Note => Style::new().magenta(),
            };

            mp.println(format!(
                "{} {}",
                icon,
                loc_style.apply_to(format!(
                    "{} {}:{}",
                    diag.source_path, diag.line, diag.column
                ))
            ))
            .ok();
            mp.println(format!(
                "{}: {}",
                sev_style.apply_to(label),
                diag.message
            ))
            .ok();
            mp.println("").ok();
        }

        // Print error messages
        for em in &s.error_messages {
            let m: String = em
                .lines
                .iter()
                .map(|l| shorten_string(l, max_line_length))
                .collect::<Vec<_>>()
                .join("\n");
            if errors_seen.contains(&m) {
                continue;
            }
            errors_seen.insert(m.clone());
            if max_severity < Severity::Error {
                mp.println(format!(
                    "\u{203c}\u{fe0f}  {}",
                    style(&em.out_path).red()
                ))
                .ok();
                mp.println(&m).ok();
                mp.println("").ok();
            }
        }

        // Update overall progress
        if let Some(total) = s.count_total {
            overall_pb.set_length(total);
        }
        if let Some(current) = s.count_current {
            overall_pb.set_position(current);
        }

        // Update task bars
        let root_dir = s.root_dir.clone();
        for (out_path, task) in &s.tasks {
            if !Path::new(out_path).starts_with(&root_dir) {
                continue;
            }

            let rel_path = Path::new(out_path)
                .strip_prefix(&root_dir)
                .unwrap_or(Path::new(out_path))
                .to_string_lossy()
                .to_string();
            let short = shorten_path(&rel_path, max_path_length);
            let proc_name_styled = task
                .proc_name
                .as_ref()
                .map(|n| format!("{}", style(format!("({})", shorten_string(n, 10))).magenta()))
                .unwrap_or_default();
            let prefix = format!("{} {} {}", kind_emoji(task.kind), short, proc_name_styled);

            if !task_bars.contains_key(out_path) {
                let pb = mp.insert_after(&overall_pb, ProgressBar::new_spinner());
                pb.set_style(task_style.clone());
                pb.set_prefix(prefix.clone());
                pb.enable_steady_tick(Duration::from_millis(100));
                task_start_times.insert(out_path.clone(), Instant::now());
                task_bars.insert(out_path.clone(), pb);
            }

            if let Some(pb) = task_bars.get(out_path) {
                pb.set_prefix(prefix.clone());

                if task.end_time.is_some() {
                    if !task_frozen_elapsed.contains_key(out_path) {
                        let elapsed = task_start_times
                            .get(out_path)
                            .map(|t| t.elapsed().as_secs_f64())
                            .unwrap_or(0.0);
                        task_frozen_elapsed.insert(out_path.clone(), elapsed);
                    }
                    let elapsed = task_frozen_elapsed[out_path];
                    let elapsed_str = format!("{:.1}s", elapsed);
                    let prefix_width = console::measure_text_width(&prefix);
                    let left_width = 2 + prefix_width + 1 + 1 + 1;
                    let fill_width = term_width.saturating_sub(left_width + elapsed_str.len());
                    let msg = format!("{:>width$}", elapsed_str, width = fill_width + elapsed_str.len());
                    pb.set_message(msg);

                    if keep_task_after_finish(task, &root_dir) {
                        pb.finish();
                    } else {
                        pb.finish_and_clear();
                        mp.remove(pb);
                    }
                } else {
                    let elapsed = task_start_times
                        .get(out_path)
                        .map(|t| t.elapsed().as_secs_f64())
                        .unwrap_or(0.0);
                    let elapsed_str = format!("{:.1}s", elapsed);
                    let prefix_width = console::measure_text_width(&prefix);
                    let left_width = 2 + prefix_width + 1 + 1 + 1;
                    let fill_width = term_width.saturating_sub(left_width + elapsed_str.len());
                    let msg = format!("{:>width$}", elapsed_str, width = fill_width + elapsed_str.len());
                    pb.set_message(msg);
                }
            }
        }

        if s.stopped && !finished {
            finished = true;
            if !s.stopped_error {
                for (out_path, task) in &s.tasks {
                    if let Some(pb) = task_bars.get(out_path) {
                        if keep_task_after_finish(task, &root_dir) {
                            pb.finish();
                        } else {
                            pb.finish_and_clear();
                            mp.remove(pb);
                        }
                    }
                }
                if let Some(total) = s.count_total {
                    overall_pb.set_position(total);
                }
            }
            overall_pb.finish_and_clear();
        }

        if finished {
            break;
        }
    }

    if !task_bars.is_empty() {
        eprintln!();
    }
}

// ---------------------------------------------------------------------------
// Short-circuit detection
// ---------------------------------------------------------------------------

fn should_shortcircuit(args: &[String]) -> bool {
    args.iter().any(|arg| {
        let a = arg.trim();
        a == "--version"
            || a.starts_with("cmTC_")
            || a.contains("/CMakeFiles/CMakeTmp")
            || a.contains("/CMakeFiles/CMakeScratch")
    })
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let process_check_interval =
        Duration::from_secs_f64(env_f64("NINJASOFANCY_PROCESS_TREE_CHECK_INTERVAL", 0.1));
    let max_path_length = env_usize("NINJASOFANCY_MAX_PATH_LENGTH", 40);
    let max_line_length = env_usize("NINJASOFANCY_MAX_LINE_LENGTH", 320);

    let app_dir = app_data_dir();
    std::fs::create_dir_all(&app_dir).ok();

    let log_level = env::var("NINJASOFANCY_LOG_LEVEL").unwrap_or_else(|_| "info".into());

    // Set up file-based logging
    let log_path = app_dir.join("ninja-so-fancy.log");
    if let Ok(file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
    {
        env_logger::Builder::new()
            .filter_level(log::LevelFilter::Info)
            .parse_filters(&format!("ninja_so_fancy={}", log_level))
            .target(env_logger::Target::Pipe(Box::new(file)))
            .init();
    }

    let ninja_args: Vec<String> = env::args().skip(1).collect();

    log::info!("started with args: {:?}", ninja_args);

    // Handle --nsf-version
    if ninja_args.iter().any(|a| a == "--nsf-version") {
        println!("{}", VERSION);
        return;
    }

    // Short-circuit for CMake probes and --version
    if should_shortcircuit(&ninja_args) {
        log::info!("short circuiting...");
        let status = std::process::Command::new("ninja")
            .args(&ninja_args)
            .status();
        match status {
            Ok(s) => std::process::exit(s.code().unwrap_or(1)),
            Err(_) => std::process::exit(1),
        }
    }

    // Check ninja version
    let ninja_version = match std::process::Command::new("ninja")
        .arg("--version")
        .output()
    {
        Ok(output) => String::from_utf8_lossy(&output.stdout).trim().to_string(),
        Err(_) => {
            eprintln!("error: ninja executable not found");
            std::process::exit(1);
        }
    };

    let version_re = Regex::new(r"(\d+)\.(\d+)\.\w+").unwrap();
    if let Some(caps) = version_re.captures(&ninja_version) {
        let major: u32 = caps[1].parse().unwrap_or(0);
        let minor: u32 = caps[2].parse().unwrap_or(0);
        if major < 1 || (major < 2 && minor < 10) {
            eprintln!("warning: old version of ninja detected: {}", ninja_version);
        }
    } else {
        eprintln!("output of `ninja --version` doesn't have expected format.");
        std::process::exit(1);
    }

    // Determine root directory
    let root_dir = determine_root_dir(&ninja_args);

    let state = Arc::new(Mutex::new(AppState::new(root_dir.clone())));

    let (tx, rx) = mpsc::channel::<Message>(100);
    let notify = Arc::new(Notify::new());

    let start_time = Instant::now();

    // Spawn ninja subprocess
    let tx_stream = tx.clone();
    let tx_monitor = tx.clone();
    let root_dir_clone = root_dir.clone();

    let ninja_handle = tokio::spawn(async move {
        let mut child = match Command::new("ninja")
            .arg("-v")
            .args(&ninja_args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(e) => {
                eprintln!("failed to start ninja: {}", e);
                let _ = tx_stream.send(Message::NinjaExited { exit_code: 1 }).await;
                let _ = tx_stream.send(Message::Finish).await;
                return;
            }
        };

        let pid = child.id().unwrap_or(0);

        // Monitor process tree
        let monitor_handle = tokio::spawn(monitor_process_tree(
            pid,
            tx_monitor,
            process_check_interval,
        ));

        // Process stdout
        let stdout = child.stdout.take().unwrap();
        let reader = BufReader::new(stdout);
        let mut lines = reader.lines();

        let parser = LineParser::new();
        let mut stream_parser = StreamParser::new(tx_stream.clone(), root_dir_clone);

        while let Ok(Some(line)) = lines.next_line().await {
            let output = parser.parse(&line);
            stream_parser.process_line(output).await;
        }
        stream_parser.flush_state().await;

        let status = child.wait().await;
        let exit_code = status.map(|s| s.code().unwrap_or(1)).unwrap_or(1);

        monitor_handle.abort();

        let _ = tx_stream
            .send(Message::NinjaExited { exit_code })
            .await;
        let _ = tx_stream.send(Message::Finish).await;
    });

    // Spawn message handler
    let state_clone = state.clone();
    let notify_clone = notify.clone();
    let msg_handle = tokio::spawn(handle_messages(state_clone, rx, notify_clone));

    // Spawn render loop
    let state_clone = state.clone();
    let notify_clone = notify.clone();
    let render_handle = tokio::spawn(render_loop(
        state_clone,
        notify_clone,
        max_path_length,
        max_line_length,
    ));

    let _ = ninja_handle.await;
    let _ = msg_handle.await;
    let _ = render_handle.await;

    let elapsed = start_time.elapsed().as_secs_f64();

    let s = state.lock().await;
    if let Some(ref reason) = s.stopped_reason {
        if let Ok(code) = reason.parse::<i32>() {
            if code != 0 {
                eprintln!("\u{1f977} {}", style("failed").red());
                std::process::exit(code);
            } else {
                eprintln!(
                    "\u{1f977} {}",
                    style(format!("finished in {:.1}s", elapsed)).green()
                );
            }
        } else if s.stopped_error {
            eprintln!(
                "\u{1f977} {}",
                style(format!("failed ({})", reason)).red()
            );
            std::process::exit(1);
        } else {
            eprintln!(
                "\u{1f977} {}",
                style(format!("finished ({})", reason)).green()
            );
        }
    } else if s.stopped_error {
        eprintln!("\u{1f977} {}", style("failed").red());
        std::process::exit(1);
    } else {
        eprintln!(
            "\u{1f977} {}",
            style(format!("finished in {:.1}s", elapsed)).green()
        );
    }
}

fn determine_root_dir(args: &[String]) -> PathBuf {
    let mut i = 0;
    while i < args.len() {
        if args[i] == "-C" {
            if i + 1 < args.len() {
                return PathBuf::from(&args[i + 1])
                    .canonicalize()
                    .unwrap_or_else(|_| PathBuf::from(&args[i + 1]));
            }
        } else if args[i].starts_with("-C") {
            let dir = &args[i][2..];
            return PathBuf::from(dir)
                .canonicalize()
                .unwrap_or_else(|_| PathBuf::from(dir));
        } else if args[i] == "-f" {
            if i + 1 < args.len() {
                let p = PathBuf::from(&args[i + 1]);
                if let Some(parent) = p.parent() {
                    return parent
                        .canonicalize()
                        .unwrap_or_else(|_| parent.to_path_buf());
                }
            }
        }
        i += 1;
    }
    env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn app_data_dir() -> PathBuf {
    if let Some(data_dir) = env::var_os("XDG_DATA_HOME") {
        PathBuf::from(data_dir).join("ninja-so-fancy")
    } else if let Some(home) = env::var_os("HOME") {
        PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("ninja-so-fancy")
    } else {
        PathBuf::from("/tmp/ninja-so-fancy")
    }
}
