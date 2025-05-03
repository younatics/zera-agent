# Zera Agent (Prompt Auto Tuning Agent)

## 개요

**Zera Agent**는 다양한 LLM(대형 언어 모델)에서 프롬프트를 자동으로 최적화하고, 평가하며, 반복적으로 개선하는 프롬프트 자동 튜닝 에이전트입니다.  
이 에이전트는 프롬프트의 품질을 체계적으로 평가하고, 메타 프롬프트를 활용해 더 나은 프롬프트를 생성하며, 다양한 데이터셋과 모델에 대해 반복적으로 실험할 수 있도록 설계되었습니다.

---

## 디렉토리 구조 및 역할

```
agent/
  app/           # Streamlit 기반 웹 UI 및 상태 관리
  common/        # API 클라이언트 등 공통 유틸리티
  core/          # 프롬프트 튜닝 및 반복 결과 관리 핵심 로직
  dataset/       # 다양한 벤치마크 데이터셋 및 데이터 로더
  prompts/       # 시스템/유저/메타 프롬프트 템플릿
  test/          # 유닛 테스트 코드
  __init__.py    # 패키지 초기화

evaluation/
  base/                # 평가 시스템의 공통 베이스 및 실행 스크립트
  dataset_evaluator/   # 데이터셋별 평가기 (LLM 기반)
    bert/              # BERTScore 기반 프롬프트 비교
    llm_judge/         # LLM Judge 기반 비교 결과
  llm_judge/           # LLM Judge 평가 결과 CSV 등
  examples/            # 평가 및 튜닝 예제 코드
  results/             # 평가 결과 저장
  samples/             # 샘플 데이터
```

### agent 디렉토리
- **app/**: Streamlit 기반 웹 인터페이스 및 상태 관리
- **common/**: 다양한 LLM API와 통신하는 공통 클라이언트
- **core/**: 프롬프트 자동 튜닝 및 반복 결과 관리 핵심 로직
- **dataset/**: 다양한 벤치마크 데이터셋 로더 및 데이터 폴더
- **prompts/**: 시스템/유저/메타/평가 프롬프트 템플릿
- **test/**: 프롬프트 튜너 테스트 코드

### evaluation 디렉토리
- **base/**: 평가 시스템의 공통 베이스 클래스(`BaseEvaluator`)와 실행 스크립트(`main.py`)
- **dataset_evaluator/**: 각 데이터셋별 LLM 평가기 (예: `gsm8k_evaluator.py`, `mmlu_evaluator.py` 등)
  - **bert/**: BERTScore를 활용한 프롬프트 비교 및 결과(`bert_compare_prompts.py`, `zera_score.json`, `base_score.json` 등)
  - **llm_judge/**: LLM Judge 기반 비교 결과 저장
- **llm_judge/**: LLM Judge로 생성된 비교 결과 CSV
- **examples/**: 데이터셋별 평가/튜닝 예제 코드 및 실행법
- **results/**: 평가 결과 저장 폴더
- **samples/**: 샘플 데이터

---

## 주요 기능

- **프롬프트 자동 튜닝**:  
  - LLM의 성능을 최대화하기 위해 시스템/유저 프롬프트를 반복적으로 개선
  - 메타 프롬프트를 활용해 프롬프트 자체를 LLM이 직접 개선하도록 유도

- **다양한 모델 및 데이터셋 지원**:  
  - OpenAI GPT, Anthropic Claude, Upstage Solar, 로컬 LLM 등 다양한 모델 지원
  - MMLU, GSM8K, CNN, MBPP, TruthfulQA 등 벤치마크 데이터셋 내장

- **출력 평가 자동화**:  
  - 8가지 평가 기준(정확성, 완전성, 표현, 신뢰성, 간결성, 정답성, 구조적 일치, 추론 품질)으로 LLM 출력을 자동 평가
  - 평가 결과를 바탕으로 프롬프트 개선

- **다양한 평가 방식**:
  - **LLM 기반 평가**: 각 데이터셋별로 LLM이 직접 정답 여부, 점수, 세부 평가를 수행
  - **BERTScore 기반 평가**: BERT 임베딩을 활용해 프롬프트별 출력의 유사도(F1, Precision, Recall 등) 비교
  - **LLM Judge 기반 평가**: 두 프롬프트의 출력을 LLM이 직접 비교하여 승자/패자 및 이유를 판별

- **웹 UI 제공**:  
  - Streamlit 기반의 직관적인 실험 관리 및 결과 시각화

---

## 평가 시스템 사용법

### 1. LLM 평가 실행

`evaluation/base/main.py`를 통해 다양한 데이터셋과 프롬프트로 LLM 평가를 실행할 수 있습니다.

```bash
python evaluation/base/main.py --dataset <데이터셋명> --model <모델명> --model_version <버전> \
  --base_system_prompt <기존시스템프롬프트> --base_user_prompt <기존유저프롬프트> \
  --zera_system_prompt <제라시스템프롬프트> --zera_user_prompt <제라우저프롬프트> \
  --num_samples <샘플수>
```

- 평가 결과는 `evaluation/results/`에 저장됩니다.
- 정확도, ROUGE 등 다양한 지표로 프롬프트 성능을 비교할 수 있습니다.

### 2. BERTScore 기반 프롬프트 비교

`evaluation/dataset_evaluator/bert/bert_compare_prompts.py`를 실행하면 ZERA 프롬프트와 기존 프롬프트의 출력 결과를 BERTScore로 비교할 수 있습니다.

```bash
python evaluation/dataset_evaluator/bert/bert_compare_prompts.py
```

- 결과는 `comparison_results.csv`로 저장됩니다.

### 3. LLM Judge 기반 비교

`evaluation/llm_judge/comparison_results.csv` 등에서 LLM이 두 프롬프트의 출력을 직접 비교한 결과(승자, 이유 등)를 확인할 수 있습니다.

### 4. 예제 실행

`evaluation/examples/` 디렉토리에는 각 데이터셋별 예제 코드가 포함되어 있습니다.

```bash
python evaluation/examples/<dataset>_example.py
```

- 예제 실행 전 `requirements.txt` 설치 및 `.env` 환경변수 설정 필요

---

## 설치 및 실행

1. 의존성 설치
   ```
   pip install -r requirements.txt
   ```

2. 환경 변수 설정  
   `.env` 파일에 OpenAI, Anthropic 등 API 키를 입력

3. 웹 UI 실행
   ```
   streamlit run agent/app/streamlit_app.py
   ```

---

## 활용 예시

- 새로운 태스크에 맞는 최적의 프롬프트 자동 생성
- LLM 벤치마크 실험 자동화 및 결과 비교
- 프롬프트 엔지니어링 연구 및 실험
- 다양한 평가 방식(LLM, BERT, LLM Judge)으로 프롬프트 성능 정량/정성 비교

---

## 기여 및 문의

- Pull Request 및 Issue 환영
- 문의: [프로젝트 관리자 이메일 또는 깃허브 이슈]

---

이 README는 실제 코드 구조와 주요 기능, 평가 시스템 전체를 바탕으로 작성되었습니다.  
추가적으로 궁금한 점이나 세부 설명이 필요하면 말씀해 주세요! 