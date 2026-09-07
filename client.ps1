# Requires -Version 5.1
<#
.SYNOPSIS
  Points this laptop at the model server, from one pasted invite.

.DESCRIPTION
  The counterpart to setup.ps1, for the laptops that consume the model rather
  than serve it. Everything it needs travels in the invite: the URL, this
  laptop's own key, the model id, and the number of tokens the server actually
  gives each client.

  What it does beyond `localllm join`: installs a Python if there is none,
  installs localllm, and then reports which coding agent was chosen and what
  is still missing before that agent can run. A fresh laptop usually has no
  Node and no VS Code, and the resulting errors come from the agent's own
  installer and mention nothing about this project.

.PARAMETER Invite
  The token printed by `localllm setup` or `localllm invite`. It starts with
  `llmi1_`. It contains an API key - treat it as a password.

.PARAMETER Client
  Which coding agent to configure. Defaults to one chosen from the model in
  the invite.

.EXAMPLE
  .\client.ps1 llmi1_eyJj...._1a2b3c4d
  .\client.ps1 llmi1_eyJj...._1a2b3c4d -Client continue
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string] $Invite,

    [ValidateSet('', 'opencode', 'continue', 'aider', 'octofriend', 'qwen', 'cline')]
    [string] $Client = '',

    [string] $Out = '.'
)

$ErrorActionPreference = 'Stop'

function Say([string] $Message) { Write-Host "==> $Message" -ForegroundColor Cyan }
function Warn([string] $Message) { Write-Host "!!  $Message" -ForegroundColor Yellow }

function Get-Python {
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

if (-not $Invite.StartsWith('llmi1_')) {
    Warn 'That does not look like an invite - one starts with llmi1_.'
    Write-Host '  Generate it on the server with:  localllm invite <this-laptop> --url http://<server>:8080'
    exit 1
}

Say 'Looking for Python 3.11 or newer'
$python = Get-Python
if (-not $python) {
    Warn 'No suitable Python found.'
    Write-Host ''
    Write-Host '  Install it with:'
    Write-Host '    winget install Python.Python.3.12'
    Write-Host ''
    Write-Host '  Then open a NEW window and run this script again.'
    exit 1
}
$pythonCmd = $python.Exe
$pythonArgs = $python.Args

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Say 'Installing localllm'
& $pythonCmd @($pythonArgs + @('-m', 'pip', 'install', '--quiet', '-e', $repoRoot))
if ($LASTEXITCODE -ne 0) {
    Warn 'pip install failed. The output above says why.'
    exit 1
}

$clientArgs = @('-m', 'localllm', 'client', $Invite, '--out', $Out)
if ($Client) { $clientArgs += @('--client', $Client) }

Say 'Joining this laptop to the server'
Write-Host ''
& $pythonCmd @($pythonArgs + $clientArgs)
exit $LASTEXITCODE
