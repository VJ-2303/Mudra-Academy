@echo off
setlocal enabledelayedexpansion

:: ============================================
:: Mudra Academy - Windows Startup Script
:: ============================================

title Mudra Academy Server

:: Colors for output (using echo)
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "CYAN=[96m"
set "RESET=[0m"

:: Check for help or stop commands
if "%1"=="--help" goto :help
if "%1"=="-h" goto :help
if "%1"=="--stop" goto :stop

echo.
echo %CYAN%============================================%RESET%
echo %CYAN%       MUDRA ACADEMY STARTUP SCRIPT        %RESET%
echo %CYAN%============================================%RESET%
echo.

:: Check Python installation
echo %YELLOW%[1/5] Checking Python installation...%RESET%
python --version >nul 2>&1
if errorlevel 1 (
    echo %RED%ERROR: Python is not installed or not in PATH%RESET%
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo %GREEN%Python %PYTHON_VERSION% found%RESET%

:: Navigate to ML directory
cd /d "%~dp0ml"

:: Check/Create virtual environment
echo.
echo %YELLOW%[2/5] Setting up virtual environment...%RESET%
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo %RED%ERROR: Failed to create virtual environment%RESET%
        pause
        exit /b 1
    )
    echo %GREEN%Virtual environment created%RESET%
) else (
    echo %GREEN%Virtual environment exists%RESET%
)

:: Activate virtual environment
call .venv\Scripts\activate.bat

:: Install requirements
echo.
echo %YELLOW%[3/5] Installing dependencies...%RESET%
if exist "requirements.txt" (
    pip install -r requirements.txt --quiet
    if errorlevel 1 (
        echo %RED%ERROR: Failed to install dependencies%RESET%
        pause
        exit /b 1
    )
    echo %GREEN%Dependencies installed%RESET%
) else (
    echo %RED%WARNING: requirements.txt not found%RESET%
)

:: Check for ML model
echo.
echo %YELLOW%[4/5] Checking ML model...%RESET%
if exist "mudra_model.pkl" (
    echo %GREEN%ML model found%RESET%
) else (
    echo %YELLOW%WARNING: mudra_model.pkl not found%RESET%
    echo Detection will use rule-based classification only
)

:: Start servers
echo.
echo %YELLOW%[5/5] Starting servers...%RESET%
echo.

:: Kill any existing processes on ports 5000 and 8000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5000.*LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000.*LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)

:: Start Flask API server in background
echo Starting Flask API server on port 5000...
start /B python api_server.py > api_server.log 2>&1

:: Wait for API server to start
timeout /t 3 /nobreak >nul

:: Check if API server started
netstat -an | findstr ":5000.*LISTENING" >nul
if errorlevel 1 (
    echo %RED%ERROR: Flask API server failed to start%RESET%
    echo Check ml\api_server.log for details
    pause
    exit /b 1
)
echo %GREEN%Flask API server running on http://localhost:5000%RESET%

:: Start frontend server
cd /d "%~dp0"
echo Starting frontend server on port 8000...
start /B python -m http.server 8000 > frontend_server.log 2>&1

:: Wait for frontend server to start
timeout /t 2 /nobreak >nul

:: Check if frontend server started
netstat -an | findstr ":8000.*LISTENING" >nul
if errorlevel 1 (
    echo %RED%ERROR: Frontend server failed to start%RESET%
    echo Check frontend_server.log for details
    pause
    exit /b 1
)
echo %GREEN%Frontend server running on http://localhost:8000%RESET%

echo.
echo %CYAN%============================================%RESET%
echo %GREEN%All servers started successfully!%RESET%
echo %CYAN%============================================%RESET%
echo.
echo %CYAN%Website:%RESET%      http://localhost:8000
echo %CYAN%API Server:%RESET%   http://localhost:5000
echo.
echo %YELLOW%Press any key to open the website...%RESET%
pause >nul

:: Open browser
start http://localhost:8000

echo.
echo %YELLOW%Servers are running in background.%RESET%
echo %YELLOW%Run 'start.bat --stop' to stop all servers.%RESET%
echo.
pause
goto :eof

:: ============================================
:: Stop servers
:: ============================================
:stop
echo.
echo %YELLOW%Stopping servers...%RESET%

:: Kill processes on port 5000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5000.*LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
    echo %GREEN%Stopped API server (PID: %%a)%RESET%
)

:: Kill processes on port 8000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000.*LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
    echo %GREEN%Stopped frontend server (PID: %%a)%RESET%
)

echo.
echo %GREEN%All servers stopped%RESET%
pause
goto :eof

:: ============================================
:: Help
:: ============================================
:help
echo.
echo %CYAN%Mudra Academy - Startup Script%RESET%
echo.
echo Usage: start.bat [option]
echo.
echo Options:
echo   (none)      Start all servers and open browser
echo   --stop      Stop all running servers
echo   --help, -h  Show this help message
echo.
echo Servers:
echo   API Server:      http://localhost:5000
echo   Frontend Server: http://localhost:8000
echo.
pause
goto :eof
