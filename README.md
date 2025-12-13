# 🥷 ninja-so-fancy 💫

`ninja-so-fancy` is a thin wrapper around the `ninja` build tool.
It enhances and enriches ninja's output to make it more useful and pleasant to human users.
If you spend a lot of time staring at build outputs, you may find this useful.

The name is inspired by the lovely [diff-so-fancy](https://github.com/so-fancy/diff-so-fancy).

<img alt="Demo" src="demo.gif" width="600" />

Tested on macOS and Linux using CMake/Clang/GCC.

## Install

You need:

- [ninja](https://ninja-build.org/)
- [uv](https://github.com/astral-sh/uv)

```sh
# install
uv tool install git+http://github.com/buntec/ninja-so-fancy

# update
uv tool upgrade ninja-so-fancy

# use
ninja-so-fancy --version # shows ninja version
ninja-so-fancy --nsf-version # shows ninja-so-fancy version
```

## Use with CMake

```sh
cmake -G Ninja -DCMAKE_MAKE_PROGRAM=ninja-so-fancy ...
```
