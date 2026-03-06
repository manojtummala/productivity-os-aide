// app/src-tauri/src/commands.rs

use crate::llm::engine::LlmEngine;
use std::sync::Mutex;
use tauri::State;

pub struct AppState {
    pub llm: Mutex<LlmEngine>,
}

#[tauri::command]
pub fn invoke_generate(prompt: String, state: State<AppState>) -> Result<String, String> {
    let llm = state.llm.lock().map_err(|_| "LLM lock poisoned".to_string())?;
    Ok(llm.generate(&prompt))
}