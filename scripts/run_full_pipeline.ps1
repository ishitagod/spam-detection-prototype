# Runs the full embeddings -> FAISS -> train -> compare/promote ->
# diagnostics pipeline end to end, on both sources, from scratch.
#
# Does NOT touch venv creation or package installs - that's one-time
# environment setup (see docs on the dill/feast pin, CUDA torch install),
# not something to blindly redo on every pipeline run.
#
# Resolves venv\Scripts\python.exe explicitly rather than relying on bare
# `python` on PATH - this machine has shown `python` resolving to a
# different account's Anaconda install ahead of the activated venv, so
# every call below goes through $python directly instead.
#
# Usage (from project root, or anywhere - path is resolved relative to
# this script's own location):
#   .\scripts\run_full_pipeline.ps1
#   .\scripts\run_full_pipeline.ps1 -MinImprovement 0.01

param(
    [double]$MinImprovement = 0.0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $ProjectRoot "venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "No venv found at $python - create + install it first." -ForegroundColor Red
    exit 1
}

$StartTime = Get-Date
$LogPath = Join-Path $ProjectRoot "scripts\run_full_pipeline.log"
Write-Host "Logging to $LogPath"
"=== Run started $StartTime ===" | Out-File -FilePath $LogPath -Encoding utf8

function Step {
    # NOTE: param must NOT be named $Args - that collides with
    # PowerShell's reserved automatic $args variable and silently
    # breaks splatting (python.exe launches with zero arguments).
    param([string]$Name, [string[]]$PyArgs)
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    "`n=== $Name ($(Get-Date)) ===" | Out-File -FilePath $LogPath -Append -Encoding utf8
    # No 2>&1 here on purpose: under Windows PowerShell 5.1, redirecting
    # a native exe's stderr wraps every stderr LINE (even benign
    # warnings - numpy/torch/mlflow all print some) as a
    # NativeCommandError and sets $? = $false, which would abort this
    # script on a harmless warning. stdout is teed to the log; stderr
    # still prints straight to console, just isn't captured in the file.
    & $python @PyArgs | Tee-Object -FilePath $LogPath -Append
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Name (exit $LASTEXITCODE) - see $LogPath" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Push-Location $ProjectRoot
try {
    Step "1/8 Text embeddings - full, both sources (SMPP + SS7)" `
        -PyArgs @("-m", "features.text_embeddings", "--processed_dir", "data/processed", "--device", "cuda")

    Step "2/8 FAISS near-dup - SMPP" `
        -PyArgs @("-m", "features.faiss_index", "--source_dir", "data/processed/SMPP")

    Step "3/8 FAISS near-dup - SS7" `
        -PyArgs @("-m", "features.faiss_index", "--source_dir", "data/processed/SS7",
          "--messages_path", "data/processed/SS7/messages_with_behavioral.csv",
          "--out_path", "data/processed/SS7/faiss_output.parquet")

    Step "4/8 Isolation Forest (anomaly_score)" `
        -PyArgs @("-m", "models.anomaly.train")

    Step "5/8 LightGBM with embeddings (rule_pattern_score)" `
        -PyArgs @("-m", "models.rule_pattern.train", "--with_embeddings")

    Step "6/8 Compare/promote - anomaly_score" `
        -PyArgs @("-m", "models.compare_versions",
          "--experiment_name", "anomaly_score",
          "--registered_name", "anomaly_score_model",
          "--metric_key", "overall_pr_auc",
          "--min_improvement", "$MinImprovement")

    Step "7/8 Compare/promote - rule_pattern_score_with_embeddings" `
        -PyArgs @("-m", "models.compare_versions",
          "--experiment_name", "rule_pattern_score_with_embeddings",
          "--registered_name", "rule_pattern_score_model",
          "--metric_key", "test_overall_pr_auc",
          "--min_improvement", "$MinImprovement")

    Step "8/8 Embedding-dominance diagnostic" `
        -PyArgs @("-m", "scripts.check_embedding_dominance")
}
finally {
    Pop-Location
}

$Elapsed = (Get-Date) - $StartTime
Write-Host "`nAll steps completed in $($Elapsed.ToString('hh\:mm\:ss'))." -ForegroundColor Green
"`n=== Completed in $($Elapsed.ToString('hh\:mm\:ss')) ===" | Out-File -FilePath $LogPath -Append -Encoding utf8
