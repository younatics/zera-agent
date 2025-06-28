# 프롬프트 자동 튜닝 CLI 가이드

이 가이드는 명령줄에서 프롬프트 자동 튜닝을 실행하는 방법을 설명합니다.

## 주요 파일

- `run_prompt_tuning.py`: 메인 CLI 스크립트
- `run_background.sh`: 백그라운드 실행용 bash 스크립트  
- `run_batch_experiments.py`: 여러 실험을 배치로 실행하는 스크립트

## 기본 사용법

### 1. 단일 실험 실행

```bash
# 기본 설정으로 BBH 데이터셋 실행
python run_prompt_tuning.py --dataset bbh --total_samples 20 --iteration_samples 5 --iterations 10

# 다양한 옵션 사용
python run_prompt_tuning.py \
    --dataset mmlu \
    --total_samples 50 \
    --iteration_samples 5 \
    --iterations 10 \
    --model solar \
    --evaluator solar \
    --meta_model solar \
    --output_dir ./results/mmlu_test \
    --use_meta_prompt \
    --evaluation_threshold 0.8
```

### 2. 백그라운드에서 실행

```bash
# 기본 설정 (BBH, 20 샘플)
./run_background.sh

# 커스텀 설정
./run_background.sh gsm8k 100

# 로그 실시간 모니터링
tail -f ./results/background_bbh_YYYYMMDD_HHMMSS.log

# 프로세스 상태 확인
ps -p $(cat ./results/process_bbh_YYYYMMDD_HHMMSS.pid)
```

### 3. 배치 실험 실행

```bash
# 기본 설정 파일 생성
python run_batch_experiments.py --create_config

# 설정 확인 (실제 실행 없이)
python run_batch_experiments.py --dry_run

# 배치 실험 실행
python run_batch_experiments.py --config experiments_config.json
```

## 파라미터 설명

### 필수 파라미터
- `--dataset`: 사용할 데이터셋 (bbh, mmlu, gsm8k, cnn, mbpp, xsum, etc.)

### 샘플링 설정
- `--total_samples`: 전체 데이터에서 샘플링할 개수 (5, 20, 50, 100, 200)
- `--iteration_samples`: 매 이터레이션마다 사용할 샘플 수 (기본값: 5)
- `--iterations`: 이터레이션 수 (기본값: 10)

### 모델 설정
- `--model`: 메인 모델 (solar, gpt4o, claude, local1, local2, solar_strawberry)
- `--evaluator`: 평가 모델 (기본값: solar)
- `--meta_model`: 메타 프롬프트 생성 모델 (기본값: solar)

### 튜닝 설정
- `--use_meta_prompt`: 메타 프롬프트 사용 (기본값: True)
- `--evaluation_threshold`: 평가 점수 임계값 (기본값: 0.8)
- `--score_threshold`: 평균 점수 임계값 (기본값: None)

### 출력 설정
- `--output_dir`: 결과 저장 디렉토리 (기본값: ./results)
- `--seed`: 랜덤 시드 (기본값: 42)

## 지원 데이터셋

| 데이터셋 | 설명 | 샘플 형태 |
|---------|------|----------|
| `bbh` | Big-Bench Hard | 추론 문제 |
| `mmlu` | Massive Multitask Language Understanding | 객관식 |
| `mmlu_pro` | MMLU Pro | 고급 객관식 |
| `gsm8k` | Grade School Math 8K | 수학 문제 |
| `cnn` | CNN/DailyMail | 요약 |
| `mbpp` | Mostly Basic Python Programming | 코딩 |
| `xsum` | Extreme Summarization | 요약 |
| `truthfulqa` | TruthfulQA | 진실성 평가 |
| `hellaswag` | HellaSwag | 상식 추론 |
| `humaneval` | HumanEval | 코딩 평가 |
| `samsum` | Samsung Summary | 대화 요약 |
| `meetingbank` | MeetingBank | 회의 요약 |

## 실행 예시

### 예시 1: 소규모 테스트
```bash
# BBH 데이터셋으로 빠른 테스트
python run_prompt_tuning.py \
    --dataset bbh \
    --total_samples 5 \
    --iteration_samples 3 \
    --iterations 3 \
    --output_dir ./results/quick_test
```

### 예시 2: 중간 규모 실험
```bash
# MMLU 데이터셋으로 표준 실험
python run_prompt_tuning.py \
    --dataset mmlu \
    --total_samples 50 \
    --iteration_samples 5 \
    --iterations 10 \
    --model solar \
    --evaluator solar \
    --meta_model solar \
    --output_dir ./results/mmlu_standard
```

### 예시 3: 대규모 실험 (백그라운드)
```bash
# GSM8K 데이터셋으로 대규모 실험
nohup python run_prompt_tuning.py \
    --dataset gsm8k \
    --total_samples 200 \
    --iteration_samples 10 \
    --iterations 20 \
    --output_dir ./results/gsm8k_large \
    > gsm8k_large.log 2>&1 &
```

## 결과 파일

실행 후 다음 파일들이 생성됩니다:

- `results_DATASET_TIMESTAMP.csv`: 전체 결과 데이터
- `cost_summary_DATASET_TIMESTAMP.csv`: 비용 요약
- `best_prompt_DATASET_TIMESTAMP.json`: 최고 성능 프롬프트
- `config_DATASET_TIMESTAMP.json`: 실험 설정
- `prompt_tuning_TIMESTAMP.log`: 실행 로그

## 배치 실험 설정 예시

`experiments_config.json` 파일 예시:

```json
{
  "experiments": [
    {
      "name": "BBH_Small_Test",
      "dataset": "bbh",
      "total_samples": 20,
      "iteration_samples": 5,
      "iterations": 10,
      "model": "solar",
      "evaluator": "solar",
      "meta_model": "solar",
      "output_dir": "./results/bbh_small",
      "enabled": true
    },
    {
      "name": "MMLU_Medium_Test",
      "dataset": "mmlu",
      "total_samples": 50,
      "iteration_samples": 5,
      "iterations": 10,
      "model": "solar",
      "evaluator": "solar",
      "meta_model": "solar",
      "output_dir": "./results/mmlu_medium",
      "enabled": true
    }
  ],
  "global_settings": {
    "use_meta_prompt": true,
    "evaluation_threshold": 0.8,
    "score_threshold": null,
    "seed": 42,
    "delay_between_experiments": 60
  }
}
```

## 주의사항

1. **API 키 설정**: 사용하는 모델에 맞는 API 키가 `.env` 파일에 설정되어 있어야 합니다.
   - `SOLAR_API_KEY`
   - `OPENAI_API_KEY`
   - `ANTHROPIC_API_KEY`
   - `SOLAR_STRAWBERRY_API_KEY`

2. **메모리 사용량**: 큰 데이터셋이나 많은 샘플을 사용할 때는 충분한 메모리가 필요합니다.

3. **실행 시간**: 이터레이션 수와 샘플 수에 따라 실행 시간이 크게 달라집니다.

4. **비용 관리**: API 기반 모델을 사용할 때는 토큰 사용량과 비용을 주의깊게 모니터링하세요.

## 문제 해결

### 일반적인 오류
- **ModuleNotFoundError**: `pip install -r requirements.txt`로 의존성 설치
- **API 키 오류**: `.env` 파일에서 API 키 확인
- **메모리 부족**: 샘플 수 줄이거나 더 작은 데이터셋 사용
- **권한 오류**: `chmod +x run_background.sh`로 실행 권한 부여

### 로그 확인
```bash
# 실시간 로그 모니터링
tail -f ./results/*.log

# 에러 로그만 확인
grep -i error ./results/*.log
``` 