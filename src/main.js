/**
 * LiveScribe - Real-time Audio Transcription
 * Frontend JavaScript for Tauri app
 */

const API_BASE = 'http://127.0.0.1:8765';
const WS_URL = 'ws://127.0.0.1:8765/ws/transcribe';

// State
let ws = null;
let isRecording = false;
let transcriptEntries = [];
let totalWords = 0;
let totalChunks = 0;

// DOM Elements
const elements = {
  statusIndicator: document.getElementById('status-indicator'),
  statusText: document.getElementById('status-text'),
  modelSelect: document.getElementById('model-select'),
  deviceSelect: document.getElementById('device-select'),
  chunkSlider: document.getElementById('chunk-slider'),
  chunkValue: document.getElementById('chunk-value'),
  languageSelect: document.getElementById('language-select'),
  startBtn: document.getElementById('start-btn'),
  stopBtn: document.getElementById('stop-btn'),
  clearBtn: document.getElementById('clear-btn'),
  copyBtn: document.getElementById('copy-btn'),
  saveBtn: document.getElementById('save-btn'),
  chunksCount: document.getElementById('chunks-count'),
  wordsCount: document.getElementById('words-count'),
  recordingIndicator: document.getElementById('recording-indicator'),
  transcriptContent: document.getElementById('transcript-content'),
  toastContainer: document.getElementById('toast-container'),
};

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
  setupEventListeners();
  await loadDevices();
  updateStatus('ready', 'Ready');
});

function setupEventListeners() {
  elements.chunkSlider.addEventListener('input', (e) => {
    elements.chunkValue.textContent = e.target.value;
  });

  elements.startBtn.addEventListener('click', startRecording);
  elements.stopBtn.addEventListener('click', stopRecording);
  elements.clearBtn.addEventListener('click', clearTranscript);
  elements.copyBtn.addEventListener('click', copyToClipboard);
  elements.saveBtn.addEventListener('click', saveToFile);
}

// API Functions
async function loadDevices() {
  try {
    const response = await fetch(`${API_BASE}/devices`);
    const data = await response.json();
    
    elements.deviceSelect.innerHTML = '';
    data.devices.forEach(device => {
      const option = document.createElement('option');
      option.value = device.index;
      option.textContent = device.name;
      if (device.index === data.default) {
        option.selected = true;
      }
      elements.deviceSelect.appendChild(option);
    });
    
    updateStatus('connected', 'Connected');
  } catch (error) {
    console.error('Failed to load devices:', error);
    updateStatus('disconnected', 'Backend offline');
    showToast('Cannot connect to backend. Make sure it is running.', 'error');
  }
}

// Recording Functions
async function startRecording() {
  if (isRecording) return;
  
  try {
    // Connect WebSocket
    ws = new WebSocket(WS_URL);
    
    ws.onopen = () => {
      // Send configuration
      const config = {
        device: parseInt(elements.deviceSelect.value) || null,
        chunk_duration: parseInt(elements.chunkSlider.value),
        language: elements.languageSelect.value,
      };
      ws.send(JSON.stringify(config));
      
      isRecording = true;
      updateUI();
      updateStatus('recording', 'Recording');
      clearEmptyState();
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleWebSocketMessage(data);
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      showToast('Connection error', 'error');
    };
    
    ws.onclose = () => {
      if (isRecording) {
        isRecording = false;
        updateUI();
        updateStatus('connected', 'Connected');
      }
    };
    
  } catch (error) {
    console.error('Failed to start recording:', error);
    showToast('Failed to start recording', 'error');
  }
}

async function stopRecording() {
  if (!isRecording) return;
  
  try {
    await fetch(`${API_BASE}/stop`, { method: 'POST' });
    
    if (ws) {
      ws.close();
      ws = null;
    }
    
    isRecording = false;
    updateUI();
    updateStatus('connected', 'Connected');
    showToast('Recording stopped', 'success');
    
  } catch (error) {
    console.error('Failed to stop recording:', error);
  }
}

function handleWebSocketMessage(data) {
  switch (data.type) {
    case 'status':
      addStatusMessage(data.message);
      break;
      
    case 'started':
      addStatusMessage(`Using device: ${data.device}`);
      break;
      
    case 'recording':
      // Visual feedback - recording chunk
      break;
      
    case 'processing':
      // Visual feedback - processing
      break;
      
    case 'transcription':
      addTranscriptEntry(data.timestamp, data.text);
      totalChunks++;
      totalWords += data.text.split(/\s+/).filter(w => w.length > 0).length;
      updateStats();
      break;
      
    case 'silence':
      // Optionally show silence indicator
      break;
      
    case 'error':
      showToast(data.message, 'error');
      break;
  }
}

// UI Functions
function updateStatus(state, text) {
  elements.statusIndicator.className = 'status-dot';
  
  if (state === 'connected' || state === 'ready') {
    elements.statusIndicator.classList.add('connected');
  } else if (state === 'recording') {
    elements.statusIndicator.classList.add('recording');
  }
  
  elements.statusText.textContent = text;
}

function updateUI() {
  elements.startBtn.disabled = isRecording;
  elements.stopBtn.disabled = !isRecording;
  elements.modelSelect.disabled = isRecording;
  elements.deviceSelect.disabled = isRecording;
  elements.chunkSlider.disabled = isRecording;
  elements.languageSelect.disabled = isRecording;
  
  if (isRecording) {
    elements.recordingIndicator.classList.remove('hidden');
  } else {
    elements.recordingIndicator.classList.add('hidden');
  }
}

function updateStats() {
  elements.chunksCount.textContent = totalChunks;
  elements.wordsCount.textContent = totalWords;
}

function clearEmptyState() {
  const emptyState = elements.transcriptContent.querySelector('.empty-state');
  if (emptyState) {
    emptyState.remove();
  }
}

function addStatusMessage(message) {
  clearEmptyState();
  
  const div = document.createElement('div');
  div.className = 'status-message';
  div.textContent = message;
  elements.transcriptContent.appendChild(div);
  scrollToBottom();
}

function addTranscriptEntry(timestamp, text) {
  clearEmptyState();
  
  const entry = { timestamp, text };
  transcriptEntries.push(entry);
  
  const div = document.createElement('div');
  div.className = 'transcript-entry';
  div.innerHTML = `
    <div class="timestamp">[${timestamp}]</div>
    <div class="text">${escapeHtml(text)}</div>
  `;
  elements.transcriptContent.appendChild(div);
  scrollToBottom();
}

function scrollToBottom() {
  elements.transcriptContent.scrollTop = elements.transcriptContent.scrollHeight;
}

function clearTranscript() {
  transcriptEntries = [];
  totalWords = 0;
  totalChunks = 0;
  updateStats();
  
  elements.transcriptContent.innerHTML = `
    <div class="empty-state">
      <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.4">
        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
        <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
        <line x1="12" x2="12" y1="19" y2="22"/>
      </svg>
      <p>Start recording to see transcription</p>
      <span>Configure settings and click "Start Recording"</span>
    </div>
  `;
  
  showToast('Transcript cleared', 'success');
}

async function copyToClipboard() {
  if (transcriptEntries.length === 0) {
    showToast('Nothing to copy', 'error');
    return;
  }
  
  const text = transcriptEntries
    .map(e => `[${e.timestamp}] ${e.text}`)
    .join('\n');
  
  try {
    await navigator.clipboard.writeText(text);
    showToast('Copied to clipboard', 'success');
  } catch (error) {
    showToast('Failed to copy', 'error');
  }
}

function saveToFile() {
  if (transcriptEntries.length === 0) {
    showToast('Nothing to save', 'error');
    return;
  }
  
  const header = `${'='.repeat(60)}\nTranscription - ${new Date().toLocaleString()}\n${'='.repeat(60)}\n\n`;
  const content = transcriptEntries
    .map(e => `[${e.timestamp}] ${e.text}`)
    .join('\n');
  const footer = `\n\n${'='.repeat(60)}\nTotal: ${totalChunks} chunks, ${totalWords} words\n${'='.repeat(60)}`;
  
  const blob = new Blob([header + content + footer], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  
  const a = document.createElement('a');
  a.href = url;
  a.download = `transcription_${Date.now()}.txt`;
  a.click();
  
  URL.revokeObjectURL(url);
  showToast('File saved', 'success');
}

function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  
  elements.toastContainer.appendChild(toast);
  
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(20px)';
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

