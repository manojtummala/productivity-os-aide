// app/src-tauri/src/commands.rs

use crate::llm::engine::LlmEngine;
use serde_json::Value;
use std::process::Command;
use std::sync::Arc;
use tauri::{Emitter, State, Window};

pub struct AppState {
    pub llm: Arc<LlmEngine>,
}

fn python() -> String {
    if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
        format!("{}/bin/python", venv)
    } else {
        "python3".to_string()
    }
}

fn should_skip_rag(query: &str) -> bool {
    let q = query.trim().to_lowercase();
    if q.len() <= 12 {
        return true;
    }
    matches!(
        q.as_str(),
        "hi" | "hello" | "hey" | "thanks" | "thank you" | "help" | "what can you do"
    )
}

#[tauri::command]
pub fn cancel_generation(window: Window, state: State<AppState>) -> Result<(), String> {
    state.llm.cancel()?;
    let _ = window.emit("llm-stopped", true);
    Ok(())
}

#[tauri::command]
pub fn invoke_rag_generate_stream(
    query: String,
    window: Window,
    state: State<AppState>,
) -> Result<(), String> {
    let window = window.clone();
    let llm = state.llm.clone();

    tauri::async_runtime::spawn(async move {
        let use_rag = !should_skip_rag(&query);

        if use_rag {
            let _ = window.emit("rag-status", "retrieving");
        } else {
            let _ = window.emit("rag-status", "generating");
        }

        let prompt = if use_rag {
            let manifest_dir = env!("CARGO_MANIFEST_DIR");
            let rag_script = format!("{}/../embeddings/retrieve_and_prompt.py", manifest_dir);

            let output = match Command::new(python()).arg(&rag_script).arg(&query).output() {
                Ok(o) => o,
                Err(e) => {
                    let _ = window.emit("rag-error", format!("Failed to run RAG script: {e}"));
                    return;
                }
            };

            if !output.status.success() {
                let _ = window.emit(
                    "rag-error",
                    String::from_utf8_lossy(&output.stderr).to_string(),
                );
                return;
            }

            let parsed: Value = match serde_json::from_slice(&output.stdout) {
                Ok(v) => v,
                Err(e) => {
                    let _ = window.emit("rag-error", format!("Bad JSON from RAG script: {e}"));
                    return;
                }
            };

            match parsed["prompt"].as_str() {
                Some(p) => p.to_string(),
                None => {
                    let _ = window.emit("rag-error", "Missing prompt in RAG output".to_string());
                    return;
                }
            }
        } else {
            query.clone()
        };

        let _ = window.emit("rag-status", "generating");

        let res = llm.generate_stream(&prompt, |chunk| {
            let _ = window.emit("llm-token", chunk);
        });

        match res {
            Ok(_) => {
                let _ = window.emit("rag-status", "done");
                let _ = window.emit("llm-done", true);
            }
            Err(e) => {
                // If cancelled, you’ll typically see "Generation stopped" or a non-zero exit.
                let _ = window.emit("rag-error", e);
            }
        }
    });

    Ok(())
}
