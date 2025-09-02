@echo off
REM ====================================================================
REM Casino Customer Segmentation - One-Click Launcher
REM University of Bath - Academic Thesis Project
REM Student: Muhammed Yavuzhan CANLI
REM Ethics Approval: 10351-12382
REM ====================================================================

Title: Casino Customer Segmentation - Academic Environment


REM Navigate to project directory
cd /d "%~dp0"

REM Check if Docker is installed and running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Docker not running. Attempting to start Docker Desktop...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    echo [INFO] Waiting for Docker Desktop to start...
    timeout /t 30 /nobreak >nul
    
    REM Check again
    docker info >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Docker Desktop failed to start.
        echo Please manually start Docker Desktop and try again.
        pause
        exit /b 1
    )
)

echo [INFO] Docker is running. Starting casino environment...
echo.

REM Stop any existing containers
docker-compose -f docker-compose.academic.yml down >nul 2>&1

REM Start the academic environment
docker-compose -f docker-compose.academic.yml up -d

if %errorlevel% neq 0 (
    echo [ERROR] Failed to start casino environment!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo CASINO ENVIRONMENT READY!
echo ============================================================
echo.
echo 	PostgreSQL Database: localhost:5432
echo    Database: casino_research
echo    Username: researcher
echo    Password: academic_password_2024
echo.
echo 	Casino Demo Application: Running
echo.
echo    Academic Analysis Tools: Available
echo.
echo ============================================================
echo MANAGEMENT COMMANDS:
echo ============================================================
echo View logs:  docker-compose -f docker-compose.academic.yml logs -f
echo Stop:       docker-compose -f docker-compose.academic.yml down
echo Restart:    docker-compose -f docker-compose.academic.yml restart
echo ============================================================
echo.
echo Environment ready for thesis evaluation and demonstration!
echo.

REM Keep window open
pause
