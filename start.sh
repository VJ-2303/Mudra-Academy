#!/bin/bash

###############################################################################
# MUDRA ACADEMY - UNIFIED STARTUP SCRIPT
###############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
API_PORT=5000
FRONTEND_PORT=8000
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_DIR="$PROJECT_ROOT/ml"
PID_FILE_API="$PROJECT_ROOT/.api_server.pid"
PID_FILE_FRONTEND="$PROJECT_ROOT/.frontend_server.pid"

print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_info() { echo -e "${BLUE}ℹ${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }

check_dependencies() {
    print_info "Checking dependencies..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    print_success "Python 3 found: $(python3 --version)"
    
    # Check if virtual environment exists in ml folder
    if [ ! -d "$ML_DIR/.venv" ]; then
        print_warning "Virtual environment not found. Creating..."
        python3 -m venv "$ML_DIR/.venv"
        print_success "Virtual environment created"
    fi
    
    # Check if requirements are installed using the venv's python
    print_info "Checking Python packages..."
    if ! "$ML_DIR/.venv/bin/python" -c "import flask" 2>/dev/null; then
        print_warning "Installing required packages..."
        "$ML_DIR/.venv/bin/pip" install -q -r "$ML_DIR/requirements.txt"
        print_success "Packages installed"
    else
        print_success "All required packages installed"
    fi
    
    if [ ! -f "$ML_DIR/mudra_rf_model.pkl" ]; then
        print_error "ML model file not found: $ML_DIR/mudra_rf_model.pkl"
        exit 1
    fi
    print_success "ML model found"
}

stop_servers() {
    print_info "Stopping servers..."
    
    if [ -f "$PID_FILE_API" ]; then
        API_PID=$(cat "$PID_FILE_API")
        if ps -p "$API_PID" > /dev/null 2>&1; then
            kill "$API_PID" 2>/dev/null || true
            print_success "API server stopped (PID: $API_PID)"
        fi
        rm -f "$PID_FILE_API"
    fi
    
    if [ -f "$PID_FILE_FRONTEND" ]; then
        FRONTEND_PID=$(cat "$PID_FILE_FRONTEND")
        if ps -p "$FRONTEND_PID" > /dev/null 2>&1; then
            kill "$FRONTEND_PID" 2>/dev/null || true
            print_success "Frontend server stopped (PID: $FRONTEND_PID)"
        fi
        rm -f "$PID_FILE_FRONTEND"
    fi
    
    lsof -ti:$API_PORT | xargs kill -9 2>/dev/null || true
    lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null || true
    
    print_success "All servers stopped"
}

start_api_server() {
    print_info "Starting Flask API server..."
    
    cd "$ML_DIR"
    
    # Use the venv's python directly (no log file - output discarded)
    nohup "$ML_DIR/.venv/bin/python" api_server.py > /dev/null 2>&1 &
    API_PID=$!
    echo $API_PID > "$PID_FILE_API"
    
    sleep 2
    
    if ps -p $API_PID > /dev/null 2>&1; then
        print_success "API server started (PID: $API_PID)"
        print_success "API running at: http://localhost:$API_PORT"
    else
        print_error "Failed to start API server"
        exit 1
    fi
    
    cd "$PROJECT_ROOT"
}

start_frontend_server() {
    print_info "Starting frontend server..."
    
    cd "$PROJECT_ROOT"
    
    # No log file - output discarded
    nohup python3 -m http.server $FRONTEND_PORT > /dev/null 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > "$PID_FILE_FRONTEND"
    
    sleep 1
    
    if ps -p $FRONTEND_PID > /dev/null 2>&1; then
        print_success "Frontend server started (PID: $FRONTEND_PID)"
        print_success "Frontend running at: http://localhost:$FRONTEND_PORT"
    else
        print_error "Failed to start frontend server"
        exit 1
    fi
}

main() {
    
    if [ "$1" = "--stop" ] || [ "$1" = "stop" ]; then
        stop_servers
        echo ""
        print_info "Use './start.sh' to start the servers again"
        echo ""
        exit 0
    fi
    
    if [ -f "$PID_FILE_API" ] || [ -f "$PID_FILE_FRONTEND" ]; then
        print_warning "Existing servers found. Stopping them first..."
        stop_servers
        echo ""
    fi
    
    check_dependencies
    echo ""
    
    start_api_server
    start_frontend_server
    
    echo ""
    echo -e "${GREEN}                    SERVERS STARTED! ${NC}"
    echo ""
    echo -e "${CYAN}📡 API Server:${NC}      http://localhost:$API_PORT"
    echo -e "${CYAN}🌐 Frontend:${NC}        http://localhost:$FRONTEND_PORT"
    echo ""
    echo -e "${YELLOW}Quick Links:${NC}"
    echo -e "   • Home:              http://localhost:$FRONTEND_PORT"
    echo -e "   • Detection Mode:    http://localhost:$FRONTEND_PORT/pages/detection-mode.html"
    echo -e "   • Live Detection:    http://localhost:$FRONTEND_PORT/pages/live-detection.html"
    echo ""
    echo -e "${RED}To stop servers:${NC}  ./start.sh --stop"
    echo ""
}

main "$@"
