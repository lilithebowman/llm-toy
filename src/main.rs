use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use llm_toy::{load_model, InferenceRequest, ModelConfig};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

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
        #[arg(long)]
        model_url: Option<String>,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long)]
        tokenizer_url: Option<String>,
        #[arg(long, default_value = "placeholder")]
        backend: String,
        #[arg(long)]
        prompt: String,
        #[arg(long)]
        input_ids: Option<String>,
        #[arg(long)]
        input_name: Option<String>,
        #[arg(long)]
        output_name: Option<String>,
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
        #[arg(long)]
        eos_token_id: Option<i64>,
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,
        #[arg(long, default_value_t = 40)]
        top_k: usize,
        #[arg(long, default_value_t = 0.9)]
        top_p: f32,
        #[arg(long, default_value_t = 1.1)]
        repetition_penalty: f32,
        #[arg(long)]
        seed: Option<u64>,
    },
    Info {
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long)]
        model_url: Option<String>,
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
    let response = download_agent()?
        .get(DEFAULT_QWEN_URL)
        .call()
        .context("Failed to download Qwen model")?;

    let mut reader = response.into_reader();
    let mut file = fs::File::create(&model_path)?;
    std::io::copy(&mut reader, &mut file)?;
    file.flush()?;

    Ok(model_path)
}

fn model_filename_from_url(model_url: &str) -> String {
    if let Ok(url) = url::Url::parse(model_url) {
        if let Some(name) = url.path_segments().and_then(|segments| segments.last()) {
            if !name.is_empty() {
                return name.to_string();
            }
        }
    }
    "model.onnx".to_string()
}

fn ensure_model_from_url(model_url: &str) -> Result<PathBuf> {
    let cache_dir = default_cache_dir()?;
    fs::create_dir_all(&cache_dir)?;

    let filename = model_filename_from_url(model_url);
    let model_path = cache_dir.join(filename);
    if model_path.exists() {
        return Ok(model_path);
    }

    println!("Downloading model to {}", model_path.display());
    let response = download_agent()?
        .get(model_url)
        .call()
        .context("Failed to download model")?;

    let mut reader = response.into_reader();
    let mut file = fs::File::create(&model_path)?;
    std::io::copy(&mut reader, &mut file)?;
    file.flush()?;

    Ok(model_path)
}

fn ensure_tokenizer_from_url(tokenizer_url: &str) -> Result<PathBuf> {
    let cache_dir = default_cache_dir()?;
    fs::create_dir_all(&cache_dir)?;

    let filename = model_filename_from_url(tokenizer_url);
    let tokenizer_path = cache_dir.join(filename);
    if tokenizer_path.exists() {
        return Ok(tokenizer_path);
    }

    println!("Downloading tokenizer to {}", tokenizer_path.display());
    let response = download_agent()?
        .get(tokenizer_url)
        .call()
        .context("Failed to download tokenizer")?;

    let mut reader = response.into_reader();
    let mut file = fs::File::create(&tokenizer_path)?;
    std::io::copy(&mut reader, &mut file)?;
    file.flush()?;

    Ok(tokenizer_path)
}

fn download_agent() -> Result<ureq::Agent> {
    let connector = native_tls::TlsConnector::new().context("Failed to init native TLS")?;
    Ok(ureq::AgentBuilder::new()
        .tls_connector(Arc::new(connector))
        .build())
}

fn resolve_model_path(model: Option<PathBuf>, model_url: Option<String>, backend: &str) -> Result<PathBuf> {
    match model {
        Some(path) => Ok(path),
        None => {
            if backend == "ryzen-ai" {
                let model_url = model_url
                    .or_else(|| std::env::var("RYZEN_AI_MODEL_URL").ok())
                    .context("ryzen-ai backend requires --model or --model-url (or RYZEN_AI_MODEL_URL)")?;
                return ensure_model_from_url(&model_url);
            }
            if backend == "cpu" {
                let model_url = model_url
                    .or_else(|| std::env::var("CPU_MODEL_URL").ok())
                    .context("cpu backend requires --model or --model-url (or CPU_MODEL_URL)")?;
                return ensure_model_from_url(&model_url);
            }
            ensure_qwen_model()
        }
    }
}

fn resolve_tokenizer_path(
    tokenizer: Option<PathBuf>,
    tokenizer_url: Option<String>,
    backend: &str,
    needs_tokenizer: bool,
) -> Result<Option<PathBuf>> {
    if let Some(path) = tokenizer {
        return Ok(Some(path));
    }

    let tokenizer_url = tokenizer_url.or_else(|| {
        if backend == "cpu" {
            std::env::var("CPU_TOKENIZER_URL").ok()
        } else {
            None
        }
    });

    if let Some(url) = tokenizer_url {
        return Ok(Some(ensure_tokenizer_from_url(&url)?));
    }

    if needs_tokenizer {
        bail!("cpu backend requires --tokenizer or --tokenizer-url (or CPU_TOKENIZER_URL) when --input-ids is omitted");
    }

    Ok(None)
}

fn parse_input_ids(value: Option<String>) -> Result<Option<Vec<i64>>> {
    let Some(raw) = value else {
        return Ok(None);
    };
    if raw.trim().is_empty() {
        return Ok(None);
    }

    let mut ids = Vec::new();
    for part in raw.split(',') {
        let token = part.trim();
        if token.is_empty() {
            continue;
        }
        let id: i64 = token.parse().context("Invalid token id in --input-ids")?;
        ids.push(id);
    }
    Ok(Some(ids))
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            model,
            model_url,
            tokenizer,
            tokenizer_url,
            backend,
            prompt,
            input_ids,
            input_name,
            output_name,
            max_tokens,
            eos_token_id,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            seed,
        } => {
            let model = resolve_model_path(model, model_url, &backend)?;
            let parsed_input_ids = parse_input_ids(input_ids)?;
            let tokenizer_path = resolve_tokenizer_path(
                tokenizer,
                tokenizer_url,
                &backend,
                parsed_input_ids.is_none(),
            )?;
            let config = ModelConfig {
                name: model
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("unknown")
                    .to_string(),
                path: model.to_string_lossy().to_string(),
                npu_backend: backend,
            };
            let mut backend = load_model(&config)?;
            let response = backend.run(&InferenceRequest {
                prompt,
                max_tokens,
                input_ids: parsed_input_ids,
                input_name,
                output_name,
                tokenizer_path: tokenizer_path.map(|path| path.to_string_lossy().to_string()),
                eos_token_id,
                temperature,
                top_k: Some(top_k),
                top_p: Some(top_p),
                repetition_penalty,
                seed,
            })?;
            println!("{}", response.text);
        }
        Commands::Info { model, model_url, backend } => {
            let model = resolve_model_path(model, model_url, &backend)?;
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
