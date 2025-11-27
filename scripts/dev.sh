#!/bin/bash
# Development script - starts both backend and frontend

echo "🚀 Starting LiveScribe Development Environment"
echo "=============================================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo -e "${BLUE}Activating Python virtual environment...${NC}"
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo -e "${BLUE}Activating Python virtual environment...${NC}"
    source venv/bin/activate
else
    echo -e "${BLUE}No virtual environment found. Using system Python.${NC}"
    echo -e "${BLUE}Tip: Create one with 'python -m venv .venv'${NC}"
fi

# Check if Python backend dependencies are installed
echo -e "${BLUE}Checking Python dependencies...${NC}"
pip install -q -r backend/requirements.txt

# Start Python backend
echo -e "${GREEN}Starting Python backend on port 8765...${NC}"
python backend/transcription_server.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 2

# Check if backend started
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Failed to start backend"
    exit 1
fi

echo -e "${GREEN}Backend started (PID: $BACKEND_PID)${NC}"

# Start frontend dev server
echo -e "${GREEN}Starting Vite dev server...${NC}"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "=============================================="
echo -e "${GREEN}✅ Development servers running${NC}"
echo "   Backend:  http://127.0.0.1:8765"
echo "   Frontend: http://localhost:1420"
echo ""
echo "Press Ctrl+C to stop all servers"
echo "=============================================="

# Trap Ctrl+C and kill both processes
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup INT

# Wait for either process to exit
wait

