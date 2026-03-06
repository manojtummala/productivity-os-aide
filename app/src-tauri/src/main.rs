// app/src-tauri/src/main.rs

mod commands;
mod llm;

use commands::{invoke_generate, AppState};
use llm::engine::LlmEngine;

fn main() {
    tauri::Builder::default()
        .manage(AppState {
            llm: std::sync::Mutex::new(LlmEngine::new()),
        })
        .invoke_handler(tauri::generate_handler![invoke_generate])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
