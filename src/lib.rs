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
    pub tokenizer_path: Option<String>,
    pub eos_token_id: Option<i64>,
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
use tokenizers::Tokenizer;
#[cfg(feature = "cpu")]
use ndarray::Axis;

#[cfg(feature = "cpu")]
pub struct CpuBackend {
    backend_name: String,
    session: Option<Session>,
    tokenizer: Option<tokenizers::Tokenizer>,
    tokenizer_path: Option<String>,
}

#[cfg(feature = "cpu")]
impl CpuBackend {
    pub fn new() -> Self {
        Self {
            backend_name: "cpu".to_string(),
            session: None,
            tokenizer: None,
            tokenizer_path: None,
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

    fn ensure_tokenizer(&mut self, path: &str) -> Result<&Tokenizer> {
        if self.tokenizer_path.as_deref() != Some(path) {
            let tokenizer = Tokenizer::from_file(path)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from {path}: {e}"))?;
            self.tokenizer = Some(tokenizer);
            self.tokenizer_path = Some(path.to_string());
        }

        self.tokenizer
            .as_ref()
            .context("Tokenizer is not loaded")
    }

    fn build_inputs(
        session: &Session,
        input_ids: &[i64],
        input_name: &str,
    ) -> Result<Vec<(String, DynValue)>> {
        let seq_len = input_ids.len();
        let mut inputs: Vec<(String, DynValue)> = Vec::new();

        for outlet in session.inputs() {
            let name = outlet.name();
            let Some((ty, shape)) = Self::tensor_meta(outlet.dtype()) else {
                continue;
            };

            if name == input_name {
                let token_shape = Self::token_shape(&shape, seq_len);
                let tensor = Self::build_int_tensor(ty, token_shape, input_ids.to_vec())?;
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

        Ok(inputs)
    }

    fn pick_next_token(output: ndarray::ArrayViewD<'_, f32>) -> Result<i64> {
        if output.ndim() == 3 {
            let batch = output.index_axis(Axis(0), 0);
            let seq = batch.len_of(Axis(0));
            let logits = batch.index_axis(Axis(0), seq.saturating_sub(1));
            let mut best_id = 0i64;
            let mut best = f32::MIN;
            for (idx, val) in logits.iter().enumerate() {
                if *val > best {
                    best = *val;
                    best_id = idx as i64;
                }
            }
            return Ok(best_id);
        }

        if output.ndim() == 2 {
            let seq = output.len_of(Axis(0));
            let logits = output.index_axis(Axis(0), seq.saturating_sub(1));
            let mut best_id = 0i64;
            let mut best = f32::MIN;
            for (idx, val) in logits.iter().enumerate() {
                if *val > best {
                    best = *val;
                    best_id = idx as i64;
                }
            }
            return Ok(best_id);
        }

        bail!("Unsupported logits rank {}", output.ndim());
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
        let input_name = request.input_name.as_deref().unwrap_or("input_ids");
        let output_name = request.output_name.as_deref().unwrap_or("logits");

        let tokenizer = if let Some(path) = request.tokenizer_path.as_deref() {
            self.ensure_tokenizer(path)?;
            self.tokenizer.clone()
        } else {
            None
        };

        let session = self
            .session
            .as_mut()
            .context("Model is not loaded")?;

        let mut all_ids = if let Some(ids) = request.input_ids.as_ref() {
            ids.clone()
        } else {
            let tokenizer = tokenizer
                .as_ref()
                .context("cpu backend requires a tokenizer when --input-ids is omitted")?;
            let encoding = tokenizer
                .encode(request.prompt.as_str(), true)
                .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {e}"))?;
            encoding.get_ids().iter().map(|id| *id as i64).collect()
        };

        let mut last_shape_first: Option<(String, f32)> = None;
        if request.max_tokens == 0 {
            if let Some(tokenizer) = tokenizer.as_ref() {
                let text = tokenizer
                    .decode(&all_ids.iter().map(|v| *v as u32).collect::<Vec<u32>>(), true)
                    .map_err(|e| anyhow::anyhow!("Failed to decode tokens: {e}"))?;
                return Ok(InferenceResponse { text });
            }
            return Ok(InferenceResponse { text: request.prompt.clone() });
        }

        for _ in 0..request.max_tokens {
            let inputs = Self::build_inputs(session, &all_ids, input_name)?;
            let outputs = session.run(inputs)?;
            let output = outputs[output_name].try_extract_array::<f32>()?;
            let next_id = Self::pick_next_token(output.view())?;
            let shape = output
                .shape()
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join("x");
            let first = output.iter().next().copied().unwrap_or(0.0);
            last_shape_first = Some((shape, first));
            all_ids.push(next_id);

            if let Some(eos) = request.eos_token_id {
                if next_id == eos {
                    break;
                }
            }
        }

        if let Some(tokenizer) = tokenizer.as_ref() {
            let ids: Vec<u32> = all_ids.iter().map(|v| *v as u32).collect();
            let text = tokenizer
                .decode(&ids, true)
                .map_err(|e| anyhow::anyhow!("Failed to decode tokens: {e}"))?;
            return Ok(InferenceResponse { text });
        }

        if let Some((shape, first)) = last_shape_first {
            return Ok(InferenceResponse {
                text: format!("[cpu] output shape={} first={}", shape, first),
            });
        }

        Ok(InferenceResponse {
            text: "[cpu] no output".to_string(),
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
