$ErrorActionPreference = "Stop"

$root = (Get-Location).Path
Set-Location $root

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $root ("train_log\pushmerge_e300_launch_{0}.out.log" -f $stamp)
$stderr = Join-Path $root ("train_log\pushmerge_e300_launch_{0}.err.log" -f $stamp)
$status = Join-Path $root ("train_log\pushmerge_e300_status_{0}.json" -f $stamp)
$ping = Join-Path $root ("train_log\pushmerge_e300_ping_{0}.txt" -f $stamp)
$python = "D:\Anaconda\envs\gym_airl\python.exe"

@{
    "PPO_RL_TAG" = "PPO_RL_Goal_PushMerge_E300"
    "PPO_RL_EPOCHS" = "300"
    "PPO_RL_SEED" = "44"
    "PPO_RL_PPO_EPOCHS" = "6"
    "PPO_RL_PPO_MINI_BATCH_SIZE" = "256"
    "PPO_RL_ENT_COEF" = "0.005"
    "PPO_RL_LR" = "8e-5"
    "PPO_RL_GAMMA" = "0.99"
    "PPO_RL_GAE_LAMBDA" = "0.95"
    "PPO_RL_CLIP_RANGE" = "0.2"
    "PPO_RL_VF_COEF" = "0.5"
    "PPO_RL_MAX_GRAD_NORM" = "0.5"
    "PPO_RL_REWARD_CLIP_MIN" = "-10.0"
    "PPO_RL_REWARD_CLIP_MAX" = "3.0"
    "PPO_RL_USE_GOAL" = "1"
    "PPO_RL_USE_ATTENTION" = "0"
    "PPO_RL_W_EFF" = "0.20"
    "PPO_RL_W_SAFETY" = "0.45"
    "PPO_RL_W_THW" = "0.0"
    "PPO_RL_W_COMFORT" = "0.05"
    "PPO_RL_W_GOAL" = "1.40"
    "PPO_RL_COLLISION_PENALTY" = "-2.0"
    "PPO_RL_SUCCESS_BONUS" = "0.5"
    "PPO_RL_MERGE_BONUS" = "0.3"
    "PPO_RL_TIMEOUT_PENALTY" = "-2.0"
    "PPO_RL_GOAL_PROGRESS_SCALE" = "40.0"
    "PPO_RL_THW_SAFE_SECONDS" = "2.0"
    "PPO_RL_SAVE_FREQ_EPOCHS" = "1"
    "PPO_RL_QUICK_EVAL_EPISODES" = "8"
    "PPO_RL_FULL_EVAL_EPISODES" = "100"
    "PPO_RL_FULL_EVAL_FREQ_EPOCHS" = "1"
    "PPO_RL_EPOCH0_EVAL_EPISODES" = "100"
    "PPO_RL_BEST_SELECT_START_EPOCH" = "270"
    "PYTHONUNBUFFERED" = "1"
}.GetEnumerator() | ForEach-Object {
    [System.Environment]::SetEnvironmentVariable($_.Key, $_.Value, "Process")
}

Set-Content -Path $ping -Value "launching`ntrain_ppo_rl_baseline_entry.py" -Encoding UTF8

# Some Windows shells expose both Path and PATH in the current process.
# Normalize to a single PATH entry so Start-Process can inherit env vars safely.
$pathValue = [System.Environment]::GetEnvironmentVariable("PATH", "Process")
if (-not $pathValue) {
    $pathValue = [System.Environment]::GetEnvironmentVariable("Path", "Process")
}
[System.Environment]::SetEnvironmentVariable("Path", $null, "Process")
[System.Environment]::SetEnvironmentVariable("PATH", $null, "Process")
if ($pathValue) {
    [System.Environment]::SetEnvironmentVariable("PATH", $pathValue, "Process")
}

$proc = Start-Process `
    -FilePath $python `
    -ArgumentList "train_ppo_rl_baseline_entry.py" `
    -WorkingDirectory $root `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

[PSCustomObject]@{
    launched_at = (Get-Date).ToString("s")
    pid = $proc.Id
    stdout = $stdout
    stderr = $stderr
    status = $status
    ping = $ping
    tag = "PPO_RL_Goal_PushMerge_E300"
    epochs = 300
} | ConvertTo-Json -Depth 2 | Set-Content -Path $status -Encoding UTF8

Write-Output ("[launcher] pid={0}" -f $proc.Id)
Write-Output ("[launcher] stdout={0}" -f $stdout)
Write-Output ("[launcher] stderr={0}" -f $stderr)
Write-Output ("[launcher] status={0}" -f $status)
Write-Output ("[launcher] ping={0}" -f $ping)
