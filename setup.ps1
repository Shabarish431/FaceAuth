# FaceAuth setup helper for Windows PowerShell.
# Run from this project folder: .\setup.ps1

$ErrorActionPreference = "Stop"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python Launcher (py) was not found. Install Python 3.10+ from https://www.python.org/downloads/ and select 'Add Python to PATH', then rerun this script."
}

try {
    py -3 --version | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Python launcher did not find a Python 3 interpreter."
    }
} catch {
    throw "The Python Launcher is present, but no Python 3 interpreter is installed. Install Python 3.10+ from https://www.python.org/downloads/, then rerun this script."
}

if (-not (Test-Path ".venv")) {
    py -3 -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "`nSetup complete. Next run:" -ForegroundColor Green
Write-Host ".\.venv\Scripts\python.exe app.py register --name Alice --samples 20"
