# Downloads the trained model into the repo's models/ directory.
# Run once before starting the inference server.
# From the repo root:
#   PowerShell -ExecutionPolicy Bypass -File Demo_Project\Server\download_model.ps1

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$dest      = "$scriptDir\rf_vent4.pkl"

$modelUrl = "https://huggingface.co/schauppi/UAS_MKL_I40_SS2026/resolve/main/rf_vent4.pkl"


Write-Host "Downloading rf_vent4.pkl ..."
Invoke-WebRequest -Uri $modelUrl -OutFile $dest -UseBasicParsing
Write-Host "Saved -> $dest"
Write-Host "Done."
