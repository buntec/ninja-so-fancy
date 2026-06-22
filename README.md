# ninja-so-fancy

`ninja-so-fancy` is a thin wrapper around the `ninja` build tool.
It enhances and enriches ninja's output to make it more useful and pleasant to human users.
If you spend a lot of time staring at build outputs, you may find this useful.

The name is inspired by the lovely [diff-so-fancy](https://github.com/so-fancy/diff-so-fancy).

<img alt="Demo" src="demo.gif" width="600" />

Tested on macOS and Linux using CMake/Clang/GCC.

## Install

You need:

- [ninja](https://ninja-build.org/)
- [Rust toolchain](https://rustup.rs/)

```sh
# install from source
cargo install --git https://github.com/buntec/ninja-so-fancy

# or build locally
git clone https://github.com/buntec/ninja-so-fancy
cd ninja-so-fancy
cargo install --path .

# use
ninja-so-fancy --version      # shows ninja version
ninja-so-fancy --nsf-version  # shows ninja-so-fancy version
```

## Use with CMake

```sh
cmake -G Ninja -DCMAKE_MAKE_PROGRAM=ninja-so-fancy ...
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `NINJASOFANCY_PROCESS_TREE_CHECK_INTERVAL` | `0.1` | Process tree polling interval in seconds |
| `NINJASOFANCY_MAX_PATH_LENGTH` | `40` | Max display length for paths |
| `NINJASOFANCY_MAX_LINE_LENGTH` | `320` | Max display length for error lines |
| `NINJASOFANCY_LOG_LEVEL` | `info` | Log level (debug, info, warn, error) |

Logs are written to `~/.local/share/ninja-so-fancy/ninja-so-fancy.log`.
