$ErrorActionPreference = "Stop"

$venvPython = ".\.venv\Scripts\python.exe"

Write-Host "Checking existing PyTorch CUDA installation..."

$checkCode = @'
try:
    import torch
    import torchvision

    correct_versions = (
        torch.__version__.split("+")[0] == "2.13.0"
        and torchvision.__version__.split("+")[0] == "0.28.0"
    )

    if correct_versions and torch.cuda.is_available():
        print("CUDA-enabled PyTorch is already installed.")
        raise SystemExit(0)

    raise SystemExit(1)
except ImportError:
    raise SystemExit(1)
'@

$oldErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"

& $venvPython -c $checkCode 2>$null
$cudaCheckExitCode = $LASTEXITCODE

$ErrorActionPreference = $oldErrorActionPreference

if ($cudaCheckExitCode -eq 0) {
    Write-Host "Correct CUDA-enabled PyTorch is already installed."
    Write-Host "Skipping all installation steps."
    exit 0
}

Write-Host "CUDA-enabled PyTorch not detected. Continuing setup..."

# Upgrade pip
Write-Host "Upgrading pip..."
& $venvPython -m pip install --upgrade pip

# Install project dependencies from pyproject.toml
Write-Host "Installing project dependencies..."
& $venvPython -m pip install -e .

# Install CUDA-enabled PyTorch separately
Write-Host "Installing CUDA-enabled PyTorch..."
& $venvPython -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126

# Verify installation
Write-Host "Verifying installation..."
& $venvPython -c "import torch, torchvision, numpy; print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('NumPy:', numpy.__version__); print('CUDA available:', torch.cuda.is_available())"

if ($LASTEXITCODE -ne 0) {
    throw "Installation verification failed."
}

Write-Host "Setup complete!"
Write-Host "Activate with: .\.venv\Scripts\Activate.ps1"