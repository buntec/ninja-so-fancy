# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```sh
cargo build --release     # build optimized binary
cargo install --path .    # install locally
cargo fmt                 # format code
cargo set-version --bump patch  # bump version (requires cargo-edit)
```

A `justfile` provides shortcuts: `just build`, `just install`, `just format`, `just bump`.

## What This Is

`ninja-so-fancy` is a drop-in replacement wrapper for the `ninja` build tool that provides human-friendly output (progress bars, colored diagnostics, elapsed time per task). It's used with CMake via `-DCMAKE_MAKE_PROGRAM=ninja-so-fancy`.

## Architecture

All code lives in `src/main.rs` (~1150 lines). The runtime is async (tokio) with three concurrent tasks communicating via `mpsc` channels:

1. **Ninja subprocess + process tree monitor** — spawns `ninja -v`, reads stdout line-by-line, and polls `/proc` (via `sysinfo`) to detect child compiler/linker processes and their lifetimes.
2. **Message handler** — receives parsed events and updates shared `AppState` (behind `Arc<Mutex>`).
3. **Render loop** — reads `AppState` on every notify/tick and drives `indicatif` progress bars (overall + per-task spinners with elapsed time).

The output parsing pipeline: raw line → `LineParser` (regex-based single-line classification) → `StreamParser` (state machine for multi-line compiler diagnostics and failure blocks) → `Message` enum sent to handler.

Short-circuit path: CMake probe builds and `--version` bypass all fancy rendering and exec ninja directly.

## Key Dependencies

- `tokio` — async runtime
- `indicatif` + `console` — terminal progress bars and styling
- `regex` — ninja output parsing
- `sysinfo` — process tree monitoring
- `clap` — CLI arg parsing (derive mode, though args are currently passed through to ninja)

## Configuration

Behavior is controlled via environment variables (prefix `NINJASOFANCY_`). Logs go to `~/.local/share/ninja-so-fancy/ninja-so-fancy.log`.
