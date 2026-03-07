// app/src-tauri/src/llm/engine.rs

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct LlmEngine {
    active_child: Arc<Mutex<Option<Child>>>,
}

impl LlmEngine {
    pub fn new() -> Self {
        Self {
            active_child: Arc::new(Mutex::new(None)),
        }
    }

    fn python() -> String {
        if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
            format!("{}/bin/python", venv)
        } else {
            "python3".to_string()
        }
    }

    pub fn cancel(&self) -> Result<(), String> {
        let mut guard = self.active_child.lock().map_err(|_| "LLM lock poisoned")?;
        if let Some(child) = guard.as_mut() {
            child.kill().map_err(|e| format!("Failed to kill process: {e}"))?;
        }
        *guard = None;
        Ok(())
    }

    pub fn generate_stream<F>(&self, prompt: &str, mut on_chunk: F) -> Result<(), String>
    where
        F: FnMut(String),
    {
        let manifest_dir = env!("CARGO_MANIFEST_DIR"); // app/src-tauri
        let script = format!("{}/../llm/stream_generate.py", manifest_dir);

        let mut child = Command::new(Self::python())
            .arg("-u")
            .arg(&script)
            .arg(prompt)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn python: {e}"))?;

        {
            let mut guard = self.active_child.lock().map_err(|_| "LLM lock poisoned")?;
            *guard = Some(child);
        }

        // Take stdout from the stored child
        let stdout = {
            let mut guard = self.active_child.lock().map_err(|_| "LLM lock poisoned")?;
            guard
                .as_mut()
                .and_then(|c| c.stdout.take())
                .ok_or("No stdout")?
        };

        let reader = BufReader::new(stdout);

        for line in reader.lines() {
            let chunk = line.map_err(|e| e.to_string())?;
            if !chunk.is_empty() {
                on_chunk(chunk);
            }
        }

        // Wait + clear active child
        let status = {
            let mut guard = self.active_child.lock().map_err(|_| "LLM lock poisoned")?;
            if let Some(mut c) = guard.take() {
                c.wait().map_err(|e| e.to_string())?
            } else {
                // If it was cancelled mid-flight, treat as stopped
                return Err("Generation stopped".to_string());
            }
        };

        if !status.success() {
            return Err(format!("Python exited with status {status}"));
        }

        Ok(())
    }
}
