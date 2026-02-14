use anyhow::{bail, Context, Result};
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
    pub input_ids: Option<Vec<i64>>,
    pub input_name: Option<String>,
    pub output_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub text: String,
}

pub trait NpuBackend {
    fn name(&self) -> &str;
    fn is_available(&self) -> bool;
    fn load_model(&mut self, model_path: &Path) -> Result<()>;
    fn run(&mut self, request: &InferenceRequest) -> Result<InferenceResponse>;
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

    fn load_model(&mut self, model_path: &Path) -> Result<()> {
        if !model_path.exists() {
            bail!("Model file not found: {}", model_path.display());
        }
        Ok(())
    }

    fn run(&mut self, request: &InferenceRequest) -> Result<InferenceResponse> {
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

    fn load_model(&mut self, model_path: &Path) -> Result<()> {
        if !model_path.exists() {
            bail!("Model file not found: {}", model_path.display());
        }
        Ok(())
    }

    fn run(&mut self, request: &InferenceRequest) -> Result<InferenceResponse> {
        let text = format!("[amd-xdna:placeholder] {}", request.prompt);
        Ok(InferenceResponse { text })
    }
}

#[cfg(feature = "cpu")]
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    tensor::{Shape, TensorElementType},
    value::{DynTensor, DynValue, Tensor, ValueType},
};

#[cfg(feature = "cpu")]
pub struct CpuBackend {
    backend_name: String,
    session: Option<Session>,
}

#[cfg(feature = "cpu")]
impl CpuBackend {
    pub fn new() -> Self {
        Self {
            backend_name: "cpu".to_string(),
            session: None,
        }
    }

    fn init_environment() -> Result<()> {
        if let Ok(path) = std::env::var("CPU_ORT_DLL") {
            ort::init_from(path)?.with_name("llm-toy").commit();
        } else {
            ort::init().with_name("llm-toy").commit();
        }
        Ok(())
    }

    fn resolve_dynamic_shape(name: &str, shape: &Shape, seq_len: usize) -> Shape {
        let mut resolved = Vec::with_capacity(shape.len());
        let mut used_batch = false;
        let is_past = name.contains("past_key_values") || name.contains("past");

        for dim in shape.iter().copied() {
            if dim >= 0 {
                resolved.push(dim);
                continue;
            }

            if is_past {
                if !used_batch {
                    resolved.push(1);
                    used_batch = true;
                } else {
                    resolved.push(0);
                }
            } else if !used_batch {
                resolved.push(1);
                used_batch = true;
            } else {
                resolved.push(seq_len as i64);
            }
        }

        Shape::from(resolved)
    }

    fn token_shape(shape: &Shape, seq_len: usize) -> Shape {
        match shape.len() {
            1 => Shape::from([seq_len as i64]),
            _ => Shape::from([1_i64, seq_len as i64]),
        }
    }

    fn build_int_tensor(
        ty: TensorElementType,
        shape: Shape,
        data: Vec<i64>
    ) -> Result<DynValue> {
        match ty {
            TensorElementType::Int64 => Ok(Tensor::from_array((shape, data))?.into_dyn()),
            TensorElementType::Int32 => {
                let data: Vec<i32> = data.into_iter().map(|v| v as i32).collect();
                Ok(Tensor::from_array((shape, data))?.into_dyn())
            }
            TensorElementType::Int8 => {
                let data: Vec<i8> = data.into_iter().map(|v| v as i8).collect();
                Ok(Tensor::from_array((shape, data))?.into_dyn())
            }
            TensorElementType::Uint8 => {
                let data: Vec<u8> = data.into_iter().map(|v| v as u8).collect();
                Ok(Tensor::from_array((shape, data))?.into_dyn())
            }
            TensorElementType::Uint16 => {
                let data: Vec<u16> = data.into_iter().map(|v| v as u16).collect();
                Ok(Tensor::from_array((shape, data))?.into_dyn())
            }
            TensorElementType::Uint32 => {
                let data: Vec<u32> = data.into_iter().map(|v| v as u32).collect();
                Ok(Tensor::from_array((shape, data))?.into_dyn())
            }
            TensorElementType::Uint64 => {
                let data: Vec<u64> = data.into_iter().map(|v| v as u64).collect();
                Ok(Tensor::from_array((shape, data))?.into_dyn())
            }
            _ => bail!("Unsupported integer tensor type: {ty}"),
        }
    }

    fn tensor_meta(value_type: &ValueType) -> Option<(TensorElementType, Shape)> {
        match value_type {
            ValueType::Tensor { ty, shape, .. } => Some((*ty, shape.clone())),
            ValueType::Optional(inner) => match inner.as_ref() {
                ValueType::Tensor { ty, shape, .. } => Some((*ty, shape.clone())),
                _ => None,
            },
            _ => None,
        }
    }
}

#[cfg(feature = "cpu")]
impl NpuBackend for CpuBackend {
    fn name(&self) -> &str {
        &self.backend_name
    }

    fn is_available(&self) -> bool {
        true
    }

    fn load_model(&mut self, model_path: &Path) -> Result<()> {
        if !model_path.exists() {
            bail!("Model file not found: {}", model_path.display());
        }

        Self::init_environment()?;
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level1)?
            .commit_from_file(model_path)?;

        self.session = Some(session);
        Ok(())
    }

    fn run(&mut self, request: &InferenceRequest) -> Result<InferenceResponse> {
        let session = self
            .session
            .as_mut()
            .context("Model is not loaded")?;

        let input_ids = request
            .input_ids
            .as_ref()
            .context("cpu backend requires --input-ids")?;

        let input_name = request.input_name.as_deref().unwrap_or("input_ids");
        let output_name = request.output_name.as_deref().unwrap_or("logits");

        let seq_len = input_ids.len();
        let mut inputs: Vec<(String, DynValue)> = Vec::new();

        for outlet in session.inputs() {
            let name = outlet.name();
            let Some((ty, shape)) = Self::tensor_meta(outlet.dtype()) else {
                continue;
            };

            if name == input_name {
                let token_shape = Self::token_shape(&shape, seq_len);
                let tensor = Self::build_int_tensor(ty, token_shape, input_ids.clone())?;
                inputs.push((name.to_string(), tensor));
                continue;
            }

            if name.contains("attention_mask") {
                let token_shape = Self::token_shape(&shape, seq_len);
                let data = vec![1_i64; seq_len];
                let tensor = Self::build_int_tensor(ty, token_shape, data)?;
                inputs.push((name.to_string(), tensor));
                continue;
            }

            if name.contains("position_ids") {
                let token_shape = Self::token_shape(&shape, seq_len);
                let data = (0..seq_len as i64).collect::<Vec<_>>();
                let tensor = Self::build_int_tensor(ty, token_shape, data)?;
                inputs.push((name.to_string(), tensor));
                continue;
            }

            if name.contains("token_type_ids") {
                let token_shape = Self::token_shape(&shape, seq_len);
                let data = vec![0_i64; seq_len];
                let tensor = Self::build_int_tensor(ty, token_shape, data)?;
                inputs.push((name.to_string(), tensor));
                continue;
            }

            let resolved = Self::resolve_dynamic_shape(name, &shape, seq_len);
            let tensor = DynTensor::new(session.allocator(), ty, resolved)?;
            inputs.push((name.to_string(), tensor.into_dyn()));
        }

        let outputs = session.run(inputs)?;
        let output = outputs[output_name].try_extract_array::<f32>()?;
        let shape = output
            .shape()
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join("x");
        let first = output.iter().next().copied().unwrap_or(0.0);

        Ok(InferenceResponse {
            text: format!("[cpu] output shape={} first={}", shape, first),
        })
    }
}

#[cfg(not(feature = "cpu"))]
pub struct CpuBackend {
    backend_name: String,
}

#[cfg(not(feature = "cpu"))]
impl CpuBackend {
    pub fn new() -> Self {
        Self {
            backend_name: "cpu".to_string(),
        }
    }
}

#[cfg(not(feature = "cpu"))]
impl NpuBackend for CpuBackend {
    fn name(&self) -> &str {
        &self.backend_name
    }

    fn is_available(&self) -> bool {
        false
    }

    fn load_model(&mut self, model_path: &Path) -> Result<()> {
        if !model_path.exists() {
            bail!("Model file not found: {}", model_path.display());
        }
        Ok(())
    }

    fn run(&mut self, _request: &InferenceRequest) -> Result<InferenceResponse> {
        bail!("cpu backend requires the 'cpu' feature")
    }
}

#[cfg(all(windows, feature = "ryzen-ai"))]
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor
};

#[cfg(all(windows, feature = "ryzen-ai"))]
pub struct RyzenAiBackend {
    backend_name: String,
    session: Option<Session>
}

#[cfg(all(windows, feature = "ryzen-ai"))]
impl RyzenAiBackend {
    pub fn new() -> Self {
        Self {
            backend_name: "ryzen-ai".to_string(),
            session: None
        }
    }

    fn init_environment() -> Result<()> {
        if let Ok(path) = std::env::var("RYZEN_AI_ORT_DLL") {
            ort::init_from(path)?.with_name("llm-toy").commit();
        } else {
            ort::init().with_name("llm-toy").commit();
        }
        Ok(())
    }
}

#[cfg(all(windows, feature = "ryzen-ai"))]
impl NpuBackend for RyzenAiBackend {
    fn name(&self) -> &str {
        &self.backend_name
    }

    fn is_available(&self) -> bool {
        true
    }

    fn load_model(&mut self, model_path: &Path) -> Result<()> {
        if !model_path.exists() {
            bail!("Model file not found: {}", model_path.display());
        }

        Self::init_environment()?;
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level1)?
            .commit_from_file(model_path)?;

        self.session = Some(session);
        Ok(())
    }

    fn run(&mut self, request: &InferenceRequest) -> Result<InferenceResponse> {
        let session = self
            .session
            .as_mut()
            .context("Model is not loaded")?;

        let input_ids = request
            .input_ids
            .as_ref()
            .context("ryzen-ai backend requires --input-ids")?;

        let input_name = request.input_name.as_deref().unwrap_or("input_ids");
        let output_name = request.output_name.as_deref().unwrap_or("logits");

        let input_tensor = Tensor::from_array((
            [1usize, input_ids.len()],
            input_ids.clone(),
        ))
        .context("Failed to build input tensor")?;

        let outputs = session.run(ort::inputs![input_name => input_tensor])?;
        let output = outputs[output_name].try_extract_array::<f32>()?;
        let shape = output
            .shape()
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join("x");
        let first = output.iter().next().copied().unwrap_or(0.0);

        Ok(InferenceResponse {
            text: format!("[ryzen-ai] output shape={} first={}", shape, first)
        })
    }
}

#[cfg(all(windows, not(feature = "ryzen-ai")))]
pub struct RyzenAiBackend {
    backend_name: String,
}

#[cfg(all(windows, not(feature = "ryzen-ai")))]
impl RyzenAiBackend {
    pub fn new() -> Self {
        Self {
            backend_name: "ryzen-ai".to_string(),
        }
    }
}

#[cfg(all(windows, not(feature = "ryzen-ai")))]
impl NpuBackend for RyzenAiBackend {
    fn name(&self) -> &str {
        &self.backend_name
    }

    fn is_available(&self) -> bool {
        true
    }

    fn load_model(&mut self, model_path: &Path) -> Result<()> {
        if !model_path.exists() {
            bail!("Model file not found: {}", model_path.display());
        }
        Ok(())
    }

    fn run(&mut self, request: &InferenceRequest) -> Result<InferenceResponse> {
        let text = format!("[ryzen-ai:placeholder] {}", request.prompt);
        Ok(InferenceResponse { text })
    }
}

pub fn load_backend(name: &str) -> Result<Box<dyn NpuBackend>> {
    match name {
        "cpu" => Ok(Box::new(CpuBackend::new())),
        #[cfg(windows)]
        "ryzen-ai" => Ok(Box::new(RyzenAiBackend::new())),
        "amd-xdna" => Ok(Box::new(AmdXdnaBackend::new())),
        _ => Ok(Box::new(PlaceholderNpuBackend::new(name))),
    }
}

pub fn load_model(config: &ModelConfig) -> Result<Box<dyn NpuBackend>> {
    let mut backend = load_backend(&config.npu_backend)?;
    let model_path = Path::new(&config.path);
    if !backend.is_available() {
        bail!("NPU backend '{}' is not available", backend.name());
    }
    backend.load_model(model_path)?;
    Ok(backend)
}
