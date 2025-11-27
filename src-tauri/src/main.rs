//! LiveScribe - Tauri Backend
//!
//! This module handles the Tauri application lifecycle and manages
//! the Python transcription server as a child process.
//!
//! Architecture:
//! - Tauri (Rust) serves as the desktop app shell
//! - Python backend runs as a subprocess, providing WebSocket API
//! - Frontend communicates with Python via WebSocket for real-time transcription

// Hides console window on Windows in release builds
// In debug mode, console stays visible for debugging output
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use tauri::State;

/// Wrapper struct for managing the Python backend process.
///
/// Uses Mutex to allow safe concurrent access from multiple Tauri commands.
/// Option<Child> represents:
/// - None: Backend not running
/// - Some(Child): Backend process is active
struct PythonProcess(Mutex<Option<Child>>);

/// Starts the Python transcription server as a child process.
///
/// This command is invoked from the frontend to initialize the backend.
/// The server runs on port 8765 and provides:
/// - REST API for device/model listing
/// - WebSocket endpoint for real-time transcription streaming
///
/// # Returns
/// - `Ok(String)` - Success message
/// - `Err(String)` - Error description if spawn fails
///
/// # Thread Safety
/// Uses Mutex lock to prevent race conditions when checking/modifying process state.
#[tauri::command]
fn start_backend(state: State<PythonProcess>) -> Result<String, String> {
    // Acquire lock on the process state
    let mut process = state.0.lock().map_err(|e| e.to_string())?;

    // Prevent multiple instances - backend is singleton
    if process.is_some() {
        return Ok("Backend already running".to_string());
    }

    // Spawn Python process with piped stdout/stderr for potential logging
    // Note: Uses relative path, assumes CWD is project root
    let child = Command::new("python")
        .args(["backend/transcription_server.py"])
        .stdout(Stdio::piped()) // Capture stdout (could be used for logging)
        .stderr(Stdio::piped()) // Capture stderr (could be used for error reporting)
        .spawn()
        .map_err(|e| format!("Failed to start backend: {}", e))?;

    // Store the child process handle for later cleanup
    *process = Some(child);
    Ok("Backend started".to_string())
}

/// Stops the Python transcription server.
///
/// Sends SIGKILL to the child process and cleans up the handle.
/// Called when:
/// - User closes the application
/// - User explicitly stops the backend
/// - Application needs to restart the backend
///
/// # Returns
/// - `Ok(String)` - Status message (stopped or was not running)
/// - `Err(String)` - Error if kill signal fails
#[tauri::command]
fn stop_backend(state: State<PythonProcess>) -> Result<String, String> {
    let mut process = state.0.lock().map_err(|e| e.to_string())?;

    // Take ownership of the child process (if exists) and kill it
    // .take() replaces the Option with None and returns the previous value
    if let Some(mut child) = process.take() {
        child
            .kill()
            .map_err(|e| format!("Failed to stop backend: {}", e))?;
        Ok("Backend stopped".to_string())
    } else {
        Ok("Backend not running".to_string())
    }
}

/// Application entry point.
///
/// Initializes Tauri with:
/// 1. Shell plugin - for potential shell command execution
/// 2. Managed state - PythonProcess singleton for backend lifecycle
/// 3. Command handlers - expose start/stop functions to frontend
fn main() {
    tauri::Builder::default()
        // Enable shell plugin for spawning processes
        .plugin(tauri_plugin_shell::init())
        // Register shared state - accessible from all commands via State<T>
        .manage(PythonProcess(Mutex::new(None)))
        // Register IPC commands callable from JavaScript via invoke()
        .invoke_handler(tauri::generate_handler![start_backend, stop_backend])
        // Build and run the application
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
