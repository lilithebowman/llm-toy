# llm-toy

Minimal Rust CLI that runs downloaded LLM modules on a laptop NPU. This is a starter scaffold with a placeholder NPU backend.

## What’s included

- CLI with `run` and `info` subcommands
- Model loading checks
- NPU backend trait and a placeholder implementation
- CPU backend using ONNX Runtime (feature `cpu`)

## Build

```bash
cargo build
```

## Build (CPU backend)

```bash
cargo build --features cpu
```

## Build on Windows (Ryzen AI backend)

1. Install Rust for Windows (MSVC toolchain).

```powershell
winget install Rustlang.Rustup
rustup default stable-x86_64-pc-windows-msvc
```

2. Build with the `ryzen-ai` feature:

```powershell
cargo build --features ryzen-ai
```

3. Set the AMD ONNX Runtime DLL path (PowerShell):

```powershell
$env:RYZEN_AI_ORT_DLL = "C:\Path\To\amd\onnxruntime.dll"
```

4. Run (PowerShell):

```powershell
cargo run --features ryzen-ai -- run --backend ryzen-ai --model path\to\model.onnx --input-ids "1,2,3" --prompt "Hello" --max-tokens 64
```

If you omit `--model`, you can provide a URL instead and it will auto-download into the cache directory:

```powershell
$env:RYZEN_AI_MODEL_URL = "https://huggingface.co/onnx-community/Qwen2.5-1.5B/resolve/main/onnx/model_int8.onnx?download=true"
cargo run --features ryzen-ai -- run --backend ryzen-ai --input-ids "1,2,3" --prompt "Hello" --max-tokens 64
```

## Run (placeholder backend)

```bash
cargo run -- run --backend placeholder --prompt "Hello" --max-tokens 64
```

If `--model` is omitted, the app downloads a default Qwen model into the cache directory and uses it automatically.

```bash
cargo run -- info --backend placeholder
```

## Run (CPU backend)

The CPU backend expects an ONNX model plus token IDs (until tokenizer support lands).

```bash
cargo run --features cpu -- run --backend cpu --model path/to/model.onnx --input-ids "1,2,3" --prompt "Hello" --max-tokens 64
```

On Windows, you may need a compatible ONNX Runtime DLL (>= 1.23). If you have multiple versions installed, point to the correct one:

```powershell
$env:CPU_ORT_DLL = "C:\Path\To\onnxruntime.dll"
```

If you omit `--model`, you can provide a URL instead and it will auto-download into the cache directory:

```bash
$env:CPU_MODEL_URL = "https://huggingface.co/onnx-community/Qwen2.5-1.5B/resolve/main/onnx/model_int8.onnx?download=true"
cargo run --features cpu -- run --backend cpu --input-ids "1,2,3" --prompt "Hello" --max-tokens 64
```

## Next steps

- Windows-native Ryzen AI backend uses ONNX Runtime (AMD build) when built with feature `ryzen-ai`.
- Set `RYZEN_AI_ORT_DLL` to the AMD-provided `onnxruntime.dll` path so the backend loads the Ryzen AI execution provider.
- Provide ONNX models and token IDs via `--model` and `--input-ids` until tokenizer support is implemented.
- Add model format adapters (GGUF, ONNX, or vendor-specific formats).
- Implement tokenizer and streaming output.
