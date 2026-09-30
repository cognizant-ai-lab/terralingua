# demo/local/run.ps1 — Build and launch the TerraLingua local demo container.
#
# Usage:
#   # Default: empty grid world demo
#   $env:ANTHROPIC_API_KEY="sk-ant-..."; ./demo/local/run.ps1
#
#   # Use the bundled Neuro-SAN HOCON network
#   $env:ANTHROPIC_API_KEY="sk-ant-..."; ./demo/local/run.ps1 -Hocon
#
#   # Use a custom HOCON file (relative to repo root, or absolute)
#   $env:NEURO_SAN_HOCON="/path/to/network.hocon"
#   $env:ANTHROPIC_API_KEY="sk-ant-..."; ./demo/local/run.ps1
#
# Stop:
#   docker stop terralingua-demo

param(
    [switch]$Hocon
)

$ErrorActionPreference = "Stop"

$IMAGE = "terralingua-demo"
$CONTAINER = "terralingua-demo"
$PORT = if ($env:API_PORT) { $env:API_PORT } else { "8765" }
$DEFAULT_HOCON = "demo/local/neuro_san_network.hocon"

if (-not $env:ANTHROPIC_API_KEY) {
    Write-Error "[demo] ERROR: ANTHROPIC_API_KEY is not set."
    exit 1
}

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent (Split-Path -Parent $SCRIPT_DIR)

$RUNNER_ARGS = if ($env:TERRALINGUA_RUNNER_ARGS) { $env:TERRALINGUA_RUNNER_ARGS } else { "" }
$DOCKER_EXTRA_ARGS = @()

# HOCON mode is enabled if either -Hocon was passed or $env:NEURO_SAN_HOCON is set.
# When both are present, $env:NEURO_SAN_HOCON wins (explicit path overrides bundled).
$HOCON_SOURCE = ""
if ($env:NEURO_SAN_HOCON) {
    $HOCON_SOURCE = $env:NEURO_SAN_HOCON
} elseif ($Hocon) {
    $HOCON_SOURCE = $DEFAULT_HOCON
}

if ($HOCON_SOURCE) {
    if ([System.IO.Path]::IsPathRooted($HOCON_SOURCE)) {
        $HOCON_PATH = $HOCON_SOURCE
    } else {
        $HOCON_PATH = Join-Path $PROJECT_ROOT $HOCON_SOURCE
    }

    if (-not (Test-Path $HOCON_PATH)) {
        Write-Error "[demo] ERROR: HOCON file not found: $HOCON_PATH"
        exit 1
    }

    $HOCON_BASENAME = Split-Path $HOCON_PATH -Leaf
    $HOCON_MOUNT = "/app/demo/local/$HOCON_BASENAME"

    $DOCKER_EXTRA_ARGS += @(
        "--mount", "type=bind,src=$HOCON_PATH,dst=$HOCON_MOUNT,readonly"
    )

    $RUNNER_ARGS += " --graph.agent_network_hocon_path $HOCON_MOUNT"
    $RUNNER_ARGS += " --graph.agent_network_bidirectional_edges"
    $RUNNER_ARGS += " --genome sentence_directed --no-food_mechanism"
    $RUNNER_ARGS += " --init_food 0 --reproduction_cost -1"
    $RUNNER_ARGS += " --max_message_length 400"

    Write-Host "[demo] Neuro-SAN HOCON import enabled: $HOCON_PATH"
    Write-Host "[demo] Container HOCON path: $HOCON_MOUNT"
    Write-Host "[demo] Runner args: $RUNNER_ARGS"
} else {
    Write-Host "[demo] Running default grid demo (pass -Hocon to use the bundled agent network)."
}

Write-Host "[demo] Building image..."
docker build -t $IMAGE -f "$SCRIPT_DIR\Dockerfile" $PROJECT_ROOT

Write-Host "[demo] Image built."

$existing = docker ps -a --filter "name=^/$CONTAINER$" --format "{{.Names}}"

if ($existing -eq $CONTAINER) {
    docker rm -f $CONTAINER | Out-Null
}

Write-Host "[demo] Starting container..."

$DOCKER_RUN_CMD = @(
    "run", "-d",
    "--name", $CONTAINER,
    "-p", "${PORT}:${PORT}",
    "-e", "ANTHROPIC_API_KEY=$env:ANTHROPIC_API_KEY",
    "-e", "TERRALINGUA_RUNNER_ARGS=$RUNNER_ARGS"
)

$DOCKER_RUN_CMD += $DOCKER_EXTRA_ARGS
$DOCKER_RUN_CMD += $IMAGE

docker @DOCKER_RUN_CMD

Write-Host "[demo] Waiting for demo to be ready..."

for ($i = 1; $i -le 90; $i++) {
    try {
        Invoke-WebRequest -Uri "http://localhost:$PORT/api/stats" -UseBasicParsing -TimeoutSec 2 | Out-Null
        Write-Host "[demo] Ready."
        break
    } catch {
        if ($i -eq 90) {
            Write-Error "[demo] ERROR: Demo did not become ready in time."
            Write-Host "[demo] Check logs: docker logs $CONTAINER"
            exit 1
        }
        Start-Sleep -Seconds 1
    }
}

$URL = "http://localhost:$PORT/auto-login"

Write-Host ""
Write-Host "[demo] ✓ Demo ready — open this URL in your browser:"
Write-Host "[demo]   $URL"
Write-Host ""

Start-Process $URL

Write-Host ""
Write-Host "[demo] Container running. Logs: docker logs -f $CONTAINER"
Write-Host "[demo] Stop with:          docker stop $CONTAINER"
