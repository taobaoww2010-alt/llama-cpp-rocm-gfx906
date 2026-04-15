#!/bin/bash
# Multi-GPU llama.cpp server launcher
# Starts one server instance per GPU and provides load balancing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH=""
N_PARALLEL=4
CTX_SIZE=4096
PORT_BASE=8080
GPUS=()

usage() {
    echo "Usage: $0 -m <model.gguf> [options]"
    echo ""
    echo "Options:"
    echo "  -m, --model <path>       Model file path (required)"
    echo "  -n, --parallel <N>       Total parallel requests (default: 4)"
    echo "  -c, --ctx <N>            Context size (default: 4096)"
    echo "  -p, --port <N>          Base port for servers (default: 8080)"
    echo "  -g, --gpus <ids>        GPU IDs to use, comma-separated (default: all)"
    echo "  -h, --help              Show this help"
    echo ""
    echo "Example:"
    echo "  $0 -m model.gguf -n 8 -g 0,1"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -n|--parallel)
            N_PARALLEL="$2"
            shift 2
            ;;
        -c|--ctx)
            CTX_SIZE="$2"
            shift 2
            ;;
        -p|--port)
            PORT_BASE="$2"
            shift 2
            ;;
        -g|--gpus)
            IFS=',' read -ra GPUS <<< "$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

if [ -z "$MODEL_PATH" ]; then
    echo "Error: Model path is required"
    usage
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

NUM_GPUS=${#GPUS[@]}
if [ $NUM_GPUS -eq 0 ]; then
    NUM_GPUS=$(rocminfo 2>/dev/null | grep -c "GPU" || echo 2)
    GPUS=($(seq 0 $((NUM_GPUS-1))))
fi

echo "=== Multi-GPU llama.cpp Server ==="
echo "Model: $MODEL_PATH"
echo "GPUs: ${GPUS[*]}"
echo "Parallel requests: $N_PARALLEL"
echo "Context size: $CTX_SIZE"
echo "Base port: $PORT_BASE"
echo ""

SLOTS_PER_GPU=$(( (N_PARALLEL + NUM_GPUS - 1) / NUM_GPUS ))
NP_PER_SERVER=$((SLOTS_PER_GPU))

declare -a PIDS
declare -a PORTS

for i in "${!GPUS[@]}"; do
    GPU_ID="${GPUS[$i]}"
    PORT=$((PORT_BASE + i))

    echo "Starting server on GPU $GPU_ID (port $PORT, $NP_PER_SERVER slots)..."

    HIP_VISIBLE_DEVICES=$GPU_ID \
    ./llama-server \
        -m "$MODEL_PATH" \
        -mg "$GPU_ID" \
        -c "$CTX_SIZE" \
        -np "$NP_PER_SERVER" \
        --port "$PORT" \
        &

    PIDS+=($!)
    PORTS+=($PORT)

    sleep 1
done

echo ""
echo "=== Servers Started ==="
echo "PIDs: ${PIDS[*]}"
echo "Ports: ${PORTS[*]}"
echo ""
echo "Use nginx or a proxy to load balance across these ports:"
for port in "${PORTS[@]}"; do
    echo "  http://localhost:$port"
done
echo ""
echo "Press Ctrl+C to stop all servers"

cleanup() {
    echo ""
    echo "Stopping servers..."
    for pid in "${PIDS[@]}"; do
        kill $pid 2>/dev/null || true
    done
    wait 2>/dev/null || true
    echo "All servers stopped"
}
trap cleanup SIGINT SIGTERM

wait
