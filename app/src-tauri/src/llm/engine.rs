// app/src-tauri/src/llm/engine.rs

pub struct LlmEngine;

impl LlmEngine {
    pub fn new() -> Self {
        Self {}
    }

    pub fn generate(&self, prompt: &str) -> String {
        // TODO: replace with real MLX call
        format!("(stubbed LLM response) You said: {}", prompt)
    }
}