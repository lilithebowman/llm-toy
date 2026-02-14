use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use llm_toy::{load_model, InferenceRequest, ModelConfig};
use std::fs;
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "llm-toy", version, about = "Run downloaded LLM modules on a laptop NPU")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Run {
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long, default_value = "placeholder")]
        backend: String,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
    },
    Info {
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long, default_value = "placeholder")]
        backend: String,
    },
}

const DEFAULT_QWEN_URL: &str =
    "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf";
const DEFAULT_QWEN_FILENAME: &str = "qwen2.5-1.5b-instruct-q4_k_m.gguf";

fn default_cache_dir() -> Result<PathBuf> {
    let base = dirs::cache_dir().context("Failed to determine cache directory")?;
    Ok(base.join("llm-toy"))
}

fn ensure_qwen_model() -> Result<PathBuf> {
    let cache_dir = default_cache_dir()?;
    fs::create_dir_all(&cache_dir)?;
    let model_path = cache_dir.join(DEFAULT_QWEN_FILENAME);

    if model_path.exists() {
        return Ok(model_path);
    }

    println!("Downloading default Qwen model to {}", model_path.display());
    let response = ureq::get(DEFAULT_QWEN_URL)
        .call()
        .context("Failed to download Qwen model")?;

    let mut reader = response.into_reader();
    let mut file = fs::File::create(&model_path)?;
    std::io::copy(&mut reader, &mut file)?;
    file.flush()?;

    Ok(model_path)
}

fn resolve_model_path(model: Option<PathBuf>) -> Result<PathBuf> {
    match model {
        Some(path) => Ok(path),
        None => ensure_qwen_model(),
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            model,
            backend,
            prompt,
            max_tokens,
        } => {
            let model = resolve_model_path(model)?;
            let config = ModelConfig {
                name: model
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("unknown")
                    .to_string(),
                path: model.to_string_lossy().to_string(),
                npu_backend: backend,
            };
            let backend = load_model(&config)?;
            let response = backend.run(&InferenceRequest { prompt, max_tokens })?;
            println!("{}", response.text);
        }
        Commands::Info { model, backend } => {
            let model = resolve_model_path(model)?;
            let config = ModelConfig {
                name: model
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("unknown")
                    .to_string(),
                path: model.to_string_lossy().to_string(),
                npu_backend: backend,
            };
            let backend = load_model(&config)?;
            let metadata = fs::metadata(model)?;
            println!("Model: {}", config.name);
            println!("Backend: {}", backend.name());
            println!("Size: {} bytes", metadata.len());
        }
    }

    Ok(())
}
