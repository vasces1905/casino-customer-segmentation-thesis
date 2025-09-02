# ====================================================================
# Casino Customer Segmentation - Docker Installation Script
# University of Bath - Academic Thesis Project
# Student: Muhammed Yavuzhan CANLI
# Ethics Approval: 10351-12382
# ====================================================================

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Casino Customer Segmentation - Docker Installation" -ForegroundColor White
Write-Host "University of Bath Academic Project" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "[WARNING] This script should be run as Administrator for Docker installation." -ForegroundColor Yellow
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    Write-Host ""
}

# Function to check if Docker is installed
function Test-DockerInstalled {
    try {
        $dockerVersion = docker --version 2>$null
        if ($dockerVersion) {
            Write-Host "[INFO] Docker is already installed: $dockerVersion" -ForegroundColor Green
            return $true
        }
    }
    catch {
        return $false
    }
    return $false
}

# Function to download and install Docker Desktop
function Install-DockerDesktop {
    Write-Host "[INFO] Downloading Docker Desktop..." -ForegroundColor Yellow
    
    $dockerUrl = "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe"
    $installerPath = "$env:TEMP\DockerDesktopInstaller.exe"
    
    try {
        # Download Docker Desktop installer
        Invoke-WebRequest -Uri $dockerUrl -OutFile $installerPath -UseBasicParsing
        
        Write-Host "[INFO] Installing Docker Desktop..." -ForegroundColor Yellow
        Write-Host "This may take several minutes..." -ForegroundColor Yellow
        
        # Install Docker Desktop silently
        Start-Process -FilePath $installerPath -ArgumentList "install", "--quiet" -Wait
        
        Write-Host "[SUCCESS] Docker Desktop installation completed!" -ForegroundColor Green
        Write-Host ""
        Write-Host "[IMPORTANT] Please restart your computer to complete Docker installation." -ForegroundColor Red
        Write-Host "After restart, run setup-docker.bat to start the casino environment." -ForegroundColor Yellow
        
        # Clean up installer
        Remove-Item $installerPath -Force -ErrorAction SilentlyContinue
        
    }
    catch {
        Write-Host "[ERROR] Failed to download or install Docker Desktop: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Please manually download from: https://www.docker.com/products/docker-desktop/" -ForegroundColor Yellow
        return $false
    }
    
    return $true
}

# Main execution
if (Test-DockerInstalled) {
    Write-Host "[INFO] Docker is ready! Starting casino environment setup..." -ForegroundColor Green
    Write-Host ""
    
    # Change to script directory
    Set-Location -Path $PSScriptRoot
    
    # Run the setup script
    if (Test-Path "setup-docker.bat") {
        Write-Host "[INFO] Launching casino environment..." -ForegroundColor Green
        Start-Process -FilePath "setup-docker.bat" -Wait
    } else {
        Write-Host "[ERROR] setup-docker.bat not found in current directory!" -ForegroundColor Red
    }
} else {
    Write-Host "[INFO] Docker not found. Installing Docker Desktop..." -ForegroundColor Yellow
    
    if (Install-DockerDesktop) {
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host "NEXT STEPS:" -ForegroundColor White
        Write-Host "1. Restart your computer" -ForegroundColor Yellow
        Write-Host "2. Double-click 'setup-docker.bat' to start casino environment" -ForegroundColor Yellow
        Write-Host "============================================================" -ForegroundColor Cyan
    }
}

Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
