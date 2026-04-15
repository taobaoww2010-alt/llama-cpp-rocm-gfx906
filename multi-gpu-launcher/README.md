# Multi-GPU Launcher for llama.cpp

This directory contains tools to run llama.cpp server on multiple GPUs for parallel inference.

## Files

- `multi-gpu-launcher.sh` - Shell script to launch multiple server instances
- `multi-gpu-proxy.py` - Python load balancer to distribute requests

## Usage

### Option 1: Using the launcher script

```bash
# Make sure llama-server is built and in the current directory
./multi-gpu-launcher.sh -m model.gguf -n 8 -g 0,1
```

This will:
- Start server on GPU 0 (port 8080)
- Start server on GPU 1 (port 8081)
- Each server handles 4 parallel requests

### Option 2: Using the load balancer

```bash
# First, start servers manually on each GPU
HIP_VISIBLE_DEVICES=0 ./llama-server -m model.gguf -np 4 --port 8080 &
HIP_VISIBLE_DEVICES=1 ./llama-server -m model.gguf -np 4 --port 8081 &

# Then start the load balancer
pip install aiohttp
python3 multi-gpu-proxy.py -b http://localhost:8080 http://localhost:8081 --port 8088
```

Now all requests go to `http://localhost:8088` which distributes them across GPUs.

## Arguments

### multi-gpu-launcher.sh

| Argument | Description | Default |
|----------|-------------|---------|
| `-m, --model` | Model file path | Required |
| `-n, --parallel` | Total parallel requests | 4 |
| `-c, --ctx` | Context size | 4096 |
| `-p, --port` | Base port | 8080 |
| `-g, --gpus` | GPU IDs (comma-separated) | All GPUs |

### multi-gpu-proxy.py

| Argument | Description | Default |
|----------|-------------|---------|
| `-b, --backends` | Backend server URLs | Required |
| `-p, --port` | Load balancer port | 8088 |
| `-h, --host` | Host to bind | 0.0.0.0 |

## Architecture

```
                    +-----------------+
                    |   Load Balancer |
                    |   (port 8088)  |
                    +--------+--------+
                             |
          +------------------+------------------+
          |                                     |
+---------v----------+              +------------v----------+
|  llama-server     |              |  llama-server         |
|  GPU 0           |              |  GPU 1                |
|  (port 8080)     |              |  (port 8081)          |
|  4 slots        |              |  4 slots             |
+-----------------+              +---------------------+
```

## Limitations

1. Each GPU runs a **complete copy** of the model
2. No tensor parallelism (no communication between GPUs)
3. Requires enough VRAM to hold full model on each GPU

## Requirements

- ROCm 6.1+
- Multiple AMD GPUs
- Python 3.8+ (for load balancer)
- aiohttp (for load balancer): `pip install aiohttp`
