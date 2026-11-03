param(
    [Parameter(Mandatory=$true)][string]$Workspace,
    [Parameter(Mandatory=$true)][string]$Credentials,
    [Parameter(Mandatory=$true)][string]$ModelRoot,
    [Parameter(Mandatory=$true)][string]$InferencePython,
    [string]$Python = 'python',
    [string]$Bind = '127.0.0.1',
    [int]$Port = 8767,
    [string[]]$AllowNetwork = @('127.0.0.0/8')
)
$ErrorActionPreference = 'Stop'
$entry = Join-Path $PSScriptRoot 'scripts/test-tools/Yolo11VietnamPlatePoseTraining/pose_model_test_server.py'
$serverArgs = @('-X', 'utf8', $entry, '--workspace', $Workspace, '--credentials', $Credentials,
    '--model-root', $ModelRoot, '--inference-python', $InferencePython, '--bind', $Bind, '--port', $Port)
foreach ($network in $AllowNetwork) { $serverArgs += @('--allow-network', $network) }
& $Python @serverArgs
exit $LASTEXITCODE
