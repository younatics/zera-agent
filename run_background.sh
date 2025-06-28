#!/bin/bash

# 백그라운드에서 프롬프트 튜닝 실행 스크립트
# Usage: ./run_background.sh [dataset_name] [total_samples]

DATASET=${1:-bbh}
TOTAL_SAMPLES=${2:-20}
ITERATION_SAMPLES=5
ITERATIONS=10
MODEL="solar"
EVALUATOR="solar"
META_MODEL="solar"
OUTPUT_DIR="./results"

# 현재 시간으로 로그 파일명 생성
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_DIR}/background_${DATASET}_${TIMESTAMP}.log"

# 출력 디렉토리 생성
mkdir -p ${OUTPUT_DIR}

echo "=== 백그라운드에서 프롬프트 튜닝 시작 ==="
echo "데이터셋: ${DATASET}"
echo "전체 샘플: ${TOTAL_SAMPLES}"
echo "이터레이션당 샘플: ${ITERATION_SAMPLES}"
echo "이터레이션: ${ITERATIONS}"
echo "모델: ${MODEL}"
echo "로그 파일: ${LOG_FILE}"
echo "========================================"

# nohup을 사용하여 백그라운드 실행
nohup python3 run_prompt_tuning.py \
    --dataset ${DATASET} \
    --total_samples ${TOTAL_SAMPLES} \
    --iteration_samples ${ITERATION_SAMPLES} \
    --iterations ${ITERATIONS} \
    --model ${MODEL} \
    --evaluator ${EVALUATOR} \
    --meta_model ${META_MODEL} \
    --output_dir ${OUTPUT_DIR} \
    > ${LOG_FILE} 2>&1 &

# 프로세스 ID 저장
PID=$!
echo ${PID} > "${OUTPUT_DIR}/process_${DATASET}_${TIMESTAMP}.pid"

echo "백그라운드 프로세스 시작됨 (PID: ${PID})"
echo "로그 실시간 모니터링: tail -f ${LOG_FILE}"
echo "프로세스 종료: kill ${PID}"
echo "프로세스 상태 확인: ps -p ${PID}" 