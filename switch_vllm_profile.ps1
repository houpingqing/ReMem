param([ValidateSet('ReMem32', 'ReMem8B', 'Sentrix')][string]$Profile = 'ReMem32')

$ErrorActionPreference = 'Stop'
$drive = $PSScriptRoot.Substring(0, 1).ToLowerInvariant()
$relative = $PSScriptRoot.Substring(2).Replace('\', '/')
$wslScript = "/mnt/$drive$relative/vllm_profile.py"
& wsl.exe -d Ubuntu-22.04-vllm-backup -u hpq -- python3 $wslScript $Profile
if ($LASTEXITCODE -ne 0) {
    throw "Failed to switch vLLM to $Profile (exit $LASTEXITCODE)"
}
