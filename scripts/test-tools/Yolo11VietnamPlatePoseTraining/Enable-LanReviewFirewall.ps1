param(
    [string]$PythonExe = 'C:\Python313\python.exe',
    [int]$Port = 8765,
    [string[]]$LocalAddress = @('10.32.0.180', '10.32.8.171'),
    [string]$RemoteSubnet = '10.32.0.0/16'
)
$ErrorActionPreference = 'Stop'
$principal = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this script in PowerShell as Administrator to allow inbound LAN access.'
}
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { throw 'Python executable not found.' }
$ruleName = "nATime-AnprReview-LAN-$Port"
if (Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue) {
    Get-NetFirewallRule -Name $ruleName | Select-Object Name, Enabled, Direction, Action
    return
}
New-NetFirewallRule -Name $ruleName -DisplayName 'nATime ANPR Review - private LAN' `
    -Direction Inbound -Action Allow -Protocol TCP -LocalPort $Port `
    -Program $PythonExe -LocalAddress $LocalAddress -RemoteAddress $RemoteSubnet -Profile Any |
    Select-Object Name, Enabled, Direction, Action
