build:
    cargo build --release

install:
    cargo install --path .

format:
    cargo fmt

bump part="patch":
    cargo set-version --bump {{part}}
