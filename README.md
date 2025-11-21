# ninja-so-fancy

`ninja-so-fancy` is a drop-in replacement for and wrapper around the `ninja` build tool.
It enhances and enriches ninja's output to make it more useful and pleasant to human users.
If you spend a lot of time staring at build outputs, you may find this useful.

Disclaimer: this is work-in-progress - expect bugs!

## Install

You need:

- [ninja](https://ninja-build.org/)
- [uv](https://github.com/astral-sh/uv)

```sh
# to install
uv tool install git+http://github.com/buntec/ninja-so-fancy

# to update
uv tool upgrade ninja-so-fancy
```

## Use with CMake

```sh
cmake -G Ninja -B build -DCMAKE_MAKE_PROGRAM=ninja-so-fancy
```
