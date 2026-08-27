"""Kalypso baseline tunables. No GPU/Modal side effects on import."""

APP_NAME = "quail-milestone1"

MODEL_NAMES = {
    "qwen3-4b-fp8": "Qwen/Qwen3-4B-FP8",
    "qwen3-32b-fp8": "Qwen/Qwen3-32B-FP8",
}

GPU_MEMORY_UTILIZATION = 0.92
ENABLE_PREFIX_CACHING = True
SERVER_PORT = 8003
HEALTH_TIMEOUT_S = 600
HEALTH_POLL_S = 5

DATA_DIR = "/results/quailb_data"
SF = 0.1
