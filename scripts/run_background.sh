#!/bin/bash

# Background prompt tuning execution script
# Usage: bash scripts/run_background.sh [dataset_name] [total_samples]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET=${1:-bbh}
TOTAL_SAMPLES=${2:-20}
ITERATION_SAMPLES=5
ITERATIONS=10
MODEL="solar"
EVALUATOR="solar"
META_MODEL="solar"
OUTPUT_DIR="${PROJECT_ROOT}/results"

# Generate log filename with current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_DIR}/background_${DATASET}_${TIMESTAMP}.log"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "=== Starting background prompt tuning ==="
echo "Dataset: ${DATASET}"
echo "Total samples: ${TOTAL_SAMPLES}"
echo "Samples per iteration: ${ITERATION_SAMPLES}"
echo "Iterations: ${ITERATIONS}"
echo "Model: ${MODEL}"
echo "Log file: ${LOG_FILE}"
echo "========================================"

# Execute in background using nohup
nohup python3 "${SCRIPT_DIR}/run_prompt_tuning.py" \
    --dataset "${DATASET}" \
    --total_samples "${TOTAL_SAMPLES}" \
    --iteration_samples "${ITERATION_SAMPLES}" \
    --iterations "${ITERATIONS}" \
    --model "${MODEL}" \
    --evaluator "${EVALUATOR}" \
    --meta_model "${META_MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    > "${LOG_FILE}" 2>&1 &

# Save process ID
PID=$!
echo "${PID}" > "${OUTPUT_DIR}/process_${DATASET}_${TIMESTAMP}.pid"

echo "Background process started (PID: ${PID})"
echo "Real-time log monitoring: tail -f ${LOG_FILE}"
echo "Stop process: kill ${PID}"
echo "Check process status: ps -p ${PID}" 