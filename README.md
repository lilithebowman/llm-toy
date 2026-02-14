# llm-toy

Minimal Rust CLI that runs downloaded LLM modules on a laptop NPU. This is a starter scaffold with a placeholder NPU backend.

## What’s included

- CLI with `run` and `info` subcommands
- Model loading checks
- NPU backend trait and a placeholder implementation

## Build

```bash
cargo build
```

## Run (placeholder backend)

```bash
cargo run -- run --backend ryzen-ai --prompt "Hello" --max-tokens 64
```

If `--model` is omitted, the app downloads a default Qwen model into the cache directory and uses it automatically.

```bash
cargo run -- info --backend ryzen-ai
```

## Next steps

- Replace the Ryzen AI placeholder backend in `src/lib.rs` with the real AMD NPU SDK integration (Windows-native).
- Add model format adapters (GGUF, ONNX, or vendor-specific formats).
- Implement tokenizer and streaming output.
