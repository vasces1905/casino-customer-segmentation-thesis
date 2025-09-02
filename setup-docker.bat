@echo off
REM ====================================================================
REM Casino Customer Segmentation - Automated Docker Setup
REM University of Bath - Academic Thesis Project
REM Student: Muhammed Yavuzhan CANLI
REM Ethics Approval: 10351-12382
REM ====================================================================

echo.
echo ============================================================
echo Casino Customer Segmentation - Docker Setup
echo University of Bath Academic Project
echo ============================================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not installed or not in PATH
    echo.
    echo Please install Docker Desktop from:
    echo https://www.docker.com/products/docker-desktop/
    echo.
    echo After installation, restart this script.
    pause
    exit /b 1
)

REM Check if Docker Desktop is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Starting Docker Desktop...
    echo Please wait for Docker Desktop to start, then press any key.
    pause
)

REM Navigate to project directory
cd /d "%~dp0"

echo [INFO] Current directory: %CD%
echo.

REM Check if docker-compose files exist
if not exist "docker-compose.academic.yml" (
    echo [ERROR] docker-compose.academic.yml not found!
    echo Make sure you are running this script from the project root directory.
    pause
    exit /b 1
)

echo [INFO] Building and starting containers...
echo This may take a few minutes on first run...
echo.

REM Stop any existing containers
docker-compose -f docker-compose.academic.yml down >nul 2>&1

REM Build and start the academic environment
docker-compose -f docker-compose.academic.yml up --build -d

if %errorlevel% neq 0 (
    echo [ERROR] Failed to start containers!
    echo Check the error messages above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo SUCCESS! Casino Customer Segmentation Environment Ready
echo ============================================================
echo.
echo PostgreSQL Database: Running on port 5432
echo   - Database: casino_research
echo   - Username: researcher
echo   - Password: academic_password_2024
echo.
echo Casino Demo Application: Running
echo.
echo To view logs: docker-compose -f docker-compose.academic.yml logs -f
echo To stop: docker-compose -f docker-compose.academic.yml down
echo.
echo Academic environment is ready for thesis evaluation!
echo ============================================================

pause
