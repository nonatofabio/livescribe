#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use tauri::State;

struct PythonProcess(Mutex<Option<Child>>);

#[tauri::command]
fn start_backend(state: State<PythonProcess>) -> Result<String, String> {
    let mut process = state.0.lock().map_err(|e| e.to_string())?;
    
    if process.is_some() {
        return Ok("Backend already running".to_string());
    }
    
    let child = Command::new("python")
        .args(["backend/transcription_server.py"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start backend: {}", e))?;
    
    *process = Some(child);
    Ok("Backend started".to_string())
}

#[tauri::command]
fn stop_backend(state: State<PythonProcess>) -> Result<String, String> {
    let mut process = state.0.lock().map_err(|e| e.to_string())?;
    
    if let Some(mut child) = process.take() {
        child.kill().map_err(|e| format!("Failed to stop backend: {}", e))?;
        Ok("Backend stopped".to_string())
    } else {
        Ok("Backend not running".to_string())
    }
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(PythonProcess(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![start_backend, stop_backend])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

