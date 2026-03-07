// app/src-tauri/src/main.rs

mod commands;
mod llm;

use commands::AppState;
use llm::engine::LlmEngine;

fn main() {
    tauri::Builder::default()
        .manage(AppState {
            llm: std::sync::Arc::new(LlmEngine::new()),
        })
        .invoke_handler(tauri::generate_handler![
            commands::invoke_rag_generate_stream,
            commands::cancel_generation
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
