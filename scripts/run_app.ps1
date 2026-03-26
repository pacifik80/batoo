param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$StreamlitArgs
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

Push-Location $repoRoot
try {
    uv sync --python 3.11 --inexact --no-install-package torch --no-install-package llama-cpp-python
    if ($LASTEXITCODE -ne 0) {
        throw "uv sync failed."
    }

    if (-not (Test-Path $venvPython)) {
        throw "Expected project interpreter at $venvPython after uv sync."
    }

    & $venvPython -m taboo_arena.cli.main ensure-runtime
    if ($LASTEXITCODE -ne 0) {
        throw "Runtime bootstrap failed."
    }

    & $venvPython -m streamlit run src/taboo_arena/app/main.py @StreamlitArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Streamlit exited with code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
