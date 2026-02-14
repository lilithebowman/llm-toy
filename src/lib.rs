use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub path: String,
    pub npu_backend: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub prompt: String,
    pub max_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub text: String,
}

pub trait NpuBackend {
    fn name(&self) -> &str;
    fn is_available(&self) -> bool;
    fn load_model(&self, model_path: &Path) -> Result<()>;
    fn run(&self, request: &InferenceRequest) -> Result<InferenceResponse>;
}

pub struct PlaceholderNpuBackend {
    backend_name: String,
}

impl PlaceholderNpuBackend {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            backend_name: name.into(),
        }
    }
}

impl NpuBackend for PlaceholderNpuBackend {
    fn name(&self) -> &str {
        &self.backend_name
    }

    fn is_available(&self) -> bool {
        true
    }

    fn load_model(&self, model_path: &Path) -> Result<()> {
        if !model_path.exists() {
            bail!("Model file not found: {}", model_path.display());
        }
        Ok(())
    }

    fn run(&self, request: &InferenceRequest) -> Result<InferenceResponse> {
        let text = format!(
            "[placeholder:{}] {}",
            self.backend_name, request.prompt
        );
        Ok(InferenceResponse { text })
    }
}

pub struct AmdXdnaBackend {
    backend_name: String,
}

impl AmdXdnaBackend {
    pub fn new() -> Self {
        Self {
            backend_name: "amd-xdna".to_string(),
        }
    }
}

impl NpuBackend for AmdXdnaBackend {
    fn name(&self) -> &str {
        &self.backend_name
    }

    fn is_available(&self) -> bool {
        true
    }

    fn load_model(&self, model_path: &Path) -> Result<()> {
        if !model_path.exists() {
            bail!("Model file not found: {}", model_path.display());
        }
        Ok(())
    }

    fn run(&self, request: &InferenceRequest) -> Result<InferenceResponse> {
        let text = format!("[amd-xdna:placeholder] {}", request.prompt);
        Ok(InferenceResponse { text })
    }
}

#[cfg(windows)]
pub struct RyzenAiBackend {
    backend_name: String,
}

#[cfg(windows)]
impl RyzenAiBackend {
    pub fn new() -> Self {
        Self {
            backend_name: "ryzen-ai".to_string(),
        }
    }
}

#[cfg(windows)]
impl NpuBackend for RyzenAiBackend {
    fn name(&self) -> &str {
        &self.backend_name
    }

    fn is_available(&self) -> bool {
        true
    }

    fn load_model(&self, model_path: &Path) -> Result<()> {
        if !model_path.exists() {
            bail!("Model file not found: {}", model_path.display());
        }
        Ok(())
    }

    fn run(&self, request: &InferenceRequest) -> Result<InferenceResponse> {
        let text = format!("[ryzen-ai:placeholder] {}", request.prompt);
        Ok(InferenceResponse { text })
    }
}

pub fn load_backend(name: &str) -> Result<Box<dyn NpuBackend>> {
    match name {
        #[cfg(windows)]
        "ryzen-ai" => Ok(Box::new(RyzenAiBackend::new())),
        "amd-xdna" => Ok(Box::new(AmdXdnaBackend::new())),
        _ => Ok(Box::new(PlaceholderNpuBackend::new(name))),
    }
}

pub fn load_model(config: &ModelConfig) -> Result<Box<dyn NpuBackend>> {
    let backend = load_backend(&config.npu_backend)?;
    let model_path = Path::new(&config.path);
    if !backend.is_available() {
        bail!("NPU backend '{}' is not available", backend.name());
    }
    backend.load_model(model_path)?;
    Ok(backend)
}
