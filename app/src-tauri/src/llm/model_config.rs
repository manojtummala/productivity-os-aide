// app/src-tauri/src/llm/model_config.rs

pub struct ModelConfig {
    pub model_path: String,
}

impl ModelConfig {
    pub fn default() -> Self {
        Self {
            model_path: "../models/phi-3-mini.gguf".to_string(),
        }
    }
}
