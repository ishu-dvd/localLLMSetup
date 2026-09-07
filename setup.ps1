# Requires -Version 5.1
<#
.SYNOPSIS
  Turns this laptop into the model server, from nothing, in one command.

.DESCRIPTION
  Everything past this script is `localllm setup`, which is idempotent and
  resumable. This script's only job is the part that has to happen before
  Python code can run at all: making sure there is a Python to run it with,
  and putting `localllm` on the path.

  It is deliberately thin. The interesting decisions - which llama.cpp build,
  which model, how much context each laptop gets - are made by code that is
  tested, not by PowerShell that is not.

.PARAMETER Devices
  How many laptops will use this server. The context window is a pool divided
  across them, so this changes which model fits.

.PARAMETER Dir
  Where llama.cpp, the model and the service definition go.

.PARAMETER Model
  A catalogue id to install instead of the recommended one, e.g.
  'gpt-oss-20b:MXFP4'. Refused if it cannot fit.

.PARAMETER DryRun
  Show the plan and the download size, then stop. Nothing is written.

.PARAMETER NoService
  Skip the Administrator step at the end. The server will not start on boot,
  and the laptop will still sleep - use this only if you intend to run
  `localllm service install` yourself later.

.EXAMPLE
  .\setup.ps1 -Devices 2 -DryRun
  .\setup.ps1 -Devices 2
#>
[CmdletBinding()]
param(
    [int]    $Devices = 2,
    [string] $Dir     = 'C:\ai\localllm',
    [string] $Model   = '',
    [int]    $Context = 32768,
    [string] $Url     = '',
    [switch] $DryRun,
    [switch] $NoService
)

$ErrorActionPreference = 'Stop'

function Say([string] $Message) { Write-Host "==> $Message" -ForegroundColor Cyan }
function Warn([string] $Message) { Write-Host "!!  $Message" -ForegroundColor Yellow }

function Get-Python {
    <#
      Returns a command that runs Python 3.11+, or $null.

      The Windows `py` launcher is preferred over `python` because a stock
      Windows has a `python.exe` App Execution Alias on PATH that is not
      Python at all - it opens the Microsoft Store. Running that in a script
      looks like a hang, not an error.
    #>
    foreach ($candidate in @(
        @{ Exe = 'py';     Args = @('-3') },
        @{ Exe = 'python'; Args = @() },
        @{ Exe = 'python3'; Args = @() }
    )) {
        $exe = Get-Command $candidate.Exe -ErrorAction SilentlyContinue
        if (-not $exe) { continue }
        try {
            $version = & $candidate.Exe @($candidate.Args + @('-c', 'import sys; print("%d.%d" % sys.version_info[:2])')) 2>$null
        } catch { continue }
        if ($LASTEXITCODE -ne 0 -or -not $version) { continue }
        $parts = "$version".Trim().Split('.')
        if ($parts.Count -ge 2 -and [int]$parts[0] -eq 3 -and [int]$parts[1] -ge 11) {
            return $candidate
        }
    }
    return $null
}

Say 'Looking for Python 3.11 or newer'
$python = Get-Python
if (-not $python) {
    Warn 'No suitable Python found.'
    Write-Host ''
    Write-Host '  Install it with:'
    Write-Host '    winget install Python.Python.3.12'
    Write-Host ''
    Write-Host '  Then close this window, open a new one, and run this script again.'
    Write-Host '  (A new window is needed so PATH picks up the new install.)'
    exit 1
}
$pythonCmd = $python.Exe
$pythonArgs = $python.Args
Write-Host "    using $pythonCmd $($pythonArgs -join ' ')"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Say 'Installing localllm'
& $pythonCmd @($pythonArgs + @('-m', 'pip', 'install', '--quiet', '--upgrade', 'pip'))
& $pythonCmd @($pythonArgs + @('-m', 'pip', 'install', '--quiet', '-e', $repoRoot))
if ($LASTEXITCODE -ne 0) {
    Warn 'pip install failed. The output above says why.'
    exit 1
}

$setupArgs = @('-m', 'localllm', 'setup', '--devices', "$Devices", '--dir', $Dir, '--context', "$Context")
if ($Model)  { $setupArgs += @('--model', $Model) }
if ($Url)    { $setupArgs += @('--url', $Url) }
if ($DryRun) { $setupArgs += '--dry-run' }

Say 'Running the setup'
Write-Host ''
& $pythonCmd @($pythonArgs + $setupArgs)
$code = $LASTEXITCODE

if ($code -ne 0) {
    Write-Host ''
    Warn "Setup stopped with exit code $code. Nothing above this line was undone -"
    Warn 're-running this script picks up where it left off.'
    exit $code
}

if ($DryRun -or $NoService) { exit 0 }

# The half that actually makes the machine serve: power settings, the service,
# and the watchdog. It needs Administrator, so this either runs it directly
# (already elevated) or asks for consent through UAC once. Declining leaves a
# working install that simply does not start on boot - and says so.
Say 'Installing the service (needs Administrator)'
$deploy = Join-Path $Dir 'deploy'
$identity = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
$elevated = $identity.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if ($elevated) {
    & $pythonCmd @($pythonArgs + @('-m', 'localllm', 'service', 'install', '--dir', $deploy))
    $svc = $LASTEXITCODE
} else {
    Write-Host '    A UAC prompt is about to ask for Administrator. Decline it and'
    Write-Host '    nothing is changed - the rest of the setup is already done.'
    $inner = "-m localllm service install --dir `"$deploy`""
    try {
        $p = Start-Process -FilePath $pythonCmd `
            -ArgumentList (@($pythonArgs) + @($inner)) `
            -Verb RunAs -Wait -PassThru -ErrorAction Stop
        $svc = $p.ExitCode
    } catch {
        Warn 'Administrator was declined, so the service was not installed.'
        Write-Host ''
        Write-Host '  Everything else is done. To finish later, run this in an'
        Write-Host '  Administrator PowerShell:'
        Write-Host ''
        Write-Host "    localllm service install --dir `"$deploy`""
        exit 0
    }
}

if ($svc -ne 0) {
    Write-Host ''
    Warn 'The service scripts did not all succeed. The output above says which.'
    Warn "Re-run just that part with:  localllm service install --dir `"$deploy`""
    exit $svc
}

exit 0
