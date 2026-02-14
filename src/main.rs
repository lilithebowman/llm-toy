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
        #[arg(long, default_value_t = false)]
        memory: bool,
        #[arg(long)]
        memory_file: Option<PathBuf>,
        #[arg(long, default_value_t = false)]
        memory_clear: bool,
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

fn default_memory_path() -> Result<PathBuf> {
    let cache_dir = default_cache_dir()?;
    fs::create_dir_all(&cache_dir)?;
    Ok(cache_dir.join("memory.json"))
}

#[derive(serde::Serialize, serde::Deserialize, Default, Clone)]
struct MemoryEntry {
    prompt: String,
    response: String,
}

#[derive(serde::Serialize, serde::Deserialize, Default)]
struct MemoryState {
    last_prompt: Option<String>,
    last_response: Option<String>,
    #[serde(default)]
    conversation_history: Vec<MemoryEntry>,
}

fn load_memory(path: &PathBuf) -> Result<MemoryState> {
    if !path.exists() {
        return Ok(MemoryState::default());
    }
    let data = fs::read_to_string(path)?;
    let state = serde_json::from_str(&data).unwrap_or_default();
    Ok(state)
}

fn save_memory(path: &PathBuf, state: &MemoryState) -> Result<()> {
    let data = serde_json::to_string_pretty(state)?;
    fs::write(path, data)?;
    Ok(())
}

fn apply_memory(prompt: &str, memory: &MemoryState) -> String {
    const MAX_MEMORY_CHARS: usize = 2000;
    const MAX_MEMORY_LINES: usize = 20;
    const MAX_HISTORY: usize = 3;

    fn clamp_text(text: &str) -> String {
        let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
        let mut lines: Vec<&str> = normalized.lines().collect();
        if lines.len() > MAX_MEMORY_LINES {
            lines.truncate(MAX_MEMORY_LINES);
            lines.push("[...]");
        }
        let mut s = lines.join("\n");
        if s.len() > MAX_MEMORY_CHARS {
            s.truncate(MAX_MEMORY_CHARS);
            s.push_str("\n[...]");
        }
        s
    }

    let mut combined = String::new();
    if !memory.conversation_history.is_empty() || memory.last_prompt.is_some() || memory.last_response.is_some() {
        combined.push_str("### Previous\n");
        if !memory.conversation_history.is_empty() {
            let start = memory.conversation_history.len().saturating_sub(MAX_HISTORY);
            for entry in &memory.conversation_history[start..] {
                combined.push_str("User:\n");
                combined.push_str(&clamp_text(&entry.prompt));
                combined.push_str("\n\nAssistant:\n");
                combined.push_str(&clamp_text(&entry.response));
                combined.push_str("\n\n");
            }
        }
        if let Some(prev) = memory.last_prompt.as_ref() {
            combined.push_str("User:\n");
            combined.push_str(&clamp_text(prev));
            combined.push_str("\n\n");
        }
        if let Some(resp) = memory.last_response.as_ref() {
            combined.push_str("Assistant:\n");
            combined.push_str(&clamp_text(resp));
            combined.push_str("\n\n");
        }
    }
    combined.push_str("### Current\nUser:\n");
    combined.push_str(prompt);
    combined.push_str("\n\nAssistant:");
    combined
}

fn clean_answer(original_prompt: &str, answer: &str) -> String {
    let mut out = answer.trim().to_string();
    if let Some(idx) = out.rfind("Assistant:") {
        out = out[idx + "Assistant:".len()..].trim_start().to_string();
    }
    if out.starts_with(original_prompt) {
        out = out[original_prompt.len()..].trim_start().to_string();
    }
    out
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
            memory,
            memory_file,
            memory_clear,
        } => {
            let model = resolve_model_path(model, model_url, &backend)?;
            let parsed_input_ids = parse_input_ids(input_ids)?;
            let tokenizer_path = resolve_tokenizer_path(
                tokenizer,
                tokenizer_url,
                &backend,
                parsed_input_ids.is_none(),
            )?;
            let original_prompt = prompt.clone();
            let memory_path = if memory || memory_clear {
                Some(memory_file.unwrap_or(default_memory_path()?))
            } else {
                None
            };
            if memory_clear {
                if let Some(path) = memory_path.as_ref() {
                    let _ = fs::remove_file(path);
                }
            }
            let mut memory_state = if memory {
                if let Some(path) = memory_path.as_ref() {
                    load_memory(path)?
                } else {
                    MemoryState::default()
                }
            } else {
                MemoryState::default()
            };
            let prompt = if memory {
                apply_memory(&original_prompt, &memory_state)
            } else {
                original_prompt.clone()
            };
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
            let answer = clean_answer(&original_prompt, &response.text);
            println!("Q: {}", original_prompt);
            println!("A:\n{}", answer);
            if memory {
                memory_state.last_prompt = Some(original_prompt.clone());
                memory_state.last_response = Some(answer.clone());
                memory_state.conversation_history.push(MemoryEntry {
                    prompt: original_prompt,
                    response: answer.clone(),
                });
                if let Some(path) = memory_path.as_ref() {
                    save_memory(path, &memory_state)?;
                }
            }
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
