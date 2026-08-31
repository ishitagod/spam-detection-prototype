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
#   .\scripts\run_full_pipeline.ps1 -StartAt 2   # skip step 1 (embeddings) -
#                                                 # e.g. already computed and
#                                                 # you're just re-running FAISS on
#   .\scripts\run_full_pipeline.ps1 -FaissGpu    # use GPU faiss for steps 2/3
#                                                 # (see requirements-gpu.txt)
#   .\scripts\run_full_pipeline.ps1 -SplitBySource -StartAt 4
#                                                 # train + promote SMPP and SS7 as
#                                                 # separate models (steps 4-7 run
#                                                 # twice) instead of one combined model

param(
    [double]$MinImprovement = 0.0,
    [int]$StartAt = 1,  # first step NUMBER to actually run - earlier steps are
                         # printed as SKIPPED, not executed. Use when a step's
                         # output already exists on disk from a prior run.
    [switch]$FaissGpu,  # pass --gpu to the FAISS steps (2/8, 3/8). Off by
                         # default - requires requirements-gpu.txt installed
                         # (see that file; UNTESTED as of writing). Safe to
                         # try even if not installed - features/faiss_index.py
                         # detects a missing GPU build and falls back to CPU
                         # with a printed message rather than failing.
    [switch]$SplitBySource  # train + compare/promote SMPP and SS7 as
                         # SEPARATE models (steps 4-7 run twice, once per
                         # source) instead of one combined model. Each
                         # source gets its own MLflow experiment/registered
                         # model name (suffixed by source - see
                         # models/anomaly/train.py and
                         # models/rule_pattern/train.py). Off by default -
                         # keep the combined model unless
                         # scripts/check_source_split_justified.py showed a
                         # real, persistent per-source gap.
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $ProjectRoot "venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "No venv found at $python - create + install it first." -ForegroundColor Red
    exit 1
}

# Without these two lines, the log ends up full of embedded null bytes:
# python.exe's actual stdout encoding and PowerShell's assumption of what
# encoding a piped native process is using ($OutputEncoding) can silently
# disagree on Windows (commonly UTF-8 vs UTF-16) - Tee-Object then
# re-encodes each captured byte under the wrong assumption, which shows up
# as a stray \x00 interleaved between characters. Pinning both sides to
# UTF-8 explicitly, rather than relying on whatever the console/codepage
# defaults to, is the fix - not a cosmetic Out-File -Encoding change,
# which only affects how PowerShell's OWN strings get written, not how it
# interprets python.exe's bytes on the way in.
$env:PYTHONIOENCODING = "utf-8"
$OutputEncoding = [System.Text.Encoding]::UTF8
# Piping python's stdout through Tee-Object (below) makes Python switch
# from line-buffered to block-buffered (~8KB) - prints in faiss_index.py's
# per-chunk progress loop genuinely execute but sit unflushed for minutes
# on a long step, making a live run look hung when it isn't. Unbuffered
# stdout restores live progress through the pipe.
$env:PYTHONUNBUFFERED = "1"

$StartTime = Get-Date
$LogPath = Join-Path $ProjectRoot "scripts\run_full_pipeline.log"
Write-Host "Logging to $LogPath"
"=== Run started $StartTime ===" | Out-File -FilePath $LogPath -Encoding utf8

function Step {
    # NOTE: param must NOT be named $Args - that collides with
    # PowerShell's reserved automatic $args variable and silently
    # breaks splatting (python.exe launches with zero arguments).
    param([int]$Number, [string]$Name, [string[]]$PyArgs)
    if ($Number -lt $StartAt) {
        Write-Host "`n=== $Name === SKIPPED (-StartAt $StartAt)" -ForegroundColor DarkGray
        return
    }
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

$FaissGpuArgs = if ($FaissGpu) { @("--gpu") } else { @() }

Push-Location $ProjectRoot
try {
    Step -Number 1 -Name "1/8 Text embeddings - full, both sources (SMPP + SS7)" `
        -PyArgs @("-m", "features.text_embeddings", "--processed_dir", "data/processed", "--device", "cuda")

    Step -Number 2 -Name "2/8 FAISS near-dup - SMPP" `
        -PyArgs (@("-m", "features.faiss_index", "--source_dir", "data/processed/SMPP") + $FaissGpuArgs)

    Step -Number 3 -Name "3/8 FAISS near-dup - SS7" `
        -PyArgs (@("-m", "features.faiss_index", "--source_dir", "data/processed/SS7") + $FaissGpuArgs)

    if ($SplitBySource) {
        # Two independent models per layer, one per source - each its own
        # MLflow experiment/registered name (suffixed by source, see
        # models/anomaly/train.py and models/rule_pattern/train.py). Same
        # step NUMBERs as the combined path (so -StartAt still lines up),
        # just run twice.
        foreach ($src in @("SMPP", "SS7")) {
            Step -Number 4 -Name "4/8 Isolation Forest (anomaly_score) - $src" `
                -PyArgs @("-m", "models.anomaly.train", "--sources", $src)

            Step -Number 5 -Name "5/8 LightGBM with embeddings (rule_pattern_score) - $src" `
                -PyArgs @("-m", "models.rule_pattern.train", "--with_embeddings", "--sources", $src)

            Step -Number 6 -Name "6/8 Compare/promote - isolation_forest_$src" `
                -PyArgs @("-m", "models.compare_versions",
                  "--experiment_name", "isolation_forest_$src",
                  "--registered_name", "anomaly_score_model_$src",
                  "--metric_key", "overall_pr_auc",
                  "--min_improvement", "$MinImprovement")

            Step -Number 7 -Name "7/8 Compare/promote - rule_pattern_score_experimental_$src" `
                -PyArgs @("-m", "models.compare_versions",
                  "--experiment_name", "rule_pattern_score_experimental_$src",
                  "--registered_name", "rule_pattern_score_model_$src",
                  "--metric_key", "test_overall_pr_auc",
                  "--min_improvement", "$MinImprovement")
        }
    }
    else {
        Step -Number 4 -Name "4/8 Isolation Forest (anomaly_score)" `
            -PyArgs @("-m", "models.anomaly.train")

        Step -Number 5 -Name "5/8 LightGBM with embeddings (rule_pattern_score)" `
            -PyArgs @("-m", "models.rule_pattern.train", "--with_embeddings")

        Step -Number 6 -Name "6/8 Compare/promote - isolation_forest" `
            -PyArgs @("-m", "models.compare_versions",
              # models/anomaly/train.py's MLFLOW_EXPERIMENT_NAME is "isolation_forest",
              # not "anomaly_score" - same mismatch as step 7, same fix.
              "--experiment_name", "isolation_forest",
              "--registered_name", "anomaly_score_model",
              "--metric_key", "overall_pr_auc",
              "--min_improvement", "$MinImprovement")

        Step -Number 7 -Name "7/8 Compare/promote - rule_pattern_score" `
            -PyArgs @("-m", "models.compare_versions",
              # Step 5 passes --with_embeddings, which models/rule_pattern/train.py
              # routes to MLFLOW_EXPERIMENTAL_EXPERIMENT_NAME
              # ("rule_pattern_score_experimental"), NOT a
              # "rule_pattern_score_with_embeddings" experiment (that name never
              # gets created) - this used to point at the wrong name and would
              # hard-fail every run with "No MLflow experiment named ...".
              "--experiment_name", "rule_pattern_score_experimental",
              "--registered_name", "rule_pattern_score_model",
              "--metric_key", "test_overall_pr_auc",
              "--min_improvement", "$MinImprovement")
    }

    Step -Number 8 -Name "8/8 Embedding-dominance diagnostic" `
        -PyArgs @("-m", "scripts.check_embedding_dominance")
}
finally {
    Pop-Location
}

$Elapsed = (Get-Date) - $StartTime
Write-Host "`nAll steps completed in $($Elapsed.ToString('hh\:mm\:ss'))." -ForegroundColor Green
"`n=== Completed in $($Elapsed.ToString('hh\:mm\:ss')) ===" | Out-File -FilePath $LogPath -Append -Encoding utf8
