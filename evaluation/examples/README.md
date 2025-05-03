# Prompt Autotuning Agent Examples

이 디렉토리에는 Prompt Autotuning Agent의 다양한 사용 예제가 포함되어 있습니다.

## 디렉토리 구조

각 데이터셋별로 별도의 디렉토리가 있으며, 각 디렉토리에는 해당 데이터셋을 사용한 예제 코드가 포함되어 있습니다.

- `gsm8k/`: 수학 문제 풀이 예제
- `mmlu/`: 다중 선택형 문제 풀이 예제
- `bbh/`: Big-Bench Hard 태스크 예제
- `cnn_dailymail/`: 뉴스 요약 예제
- `mbpp/`: 프로그래밍 문제 풀이 예제

## 사용 방법

각 예제는 다음과 같이 실행할 수 있습니다:

```bash
python examples/<dataset>/example.py
```

각 예제는 기본적인 사용법과 함께 다양한 옵션을 제공합니다. 자세한 내용은 각 예제 파일의 주석을 참조하세요.

## 환경 설정

예제를 실행하기 전에 다음 사항을 확인하세요:

1. 필요한 패키지가 설치되어 있는지 확인:
   ```bash
   pip install -r requirements.txt
   ```

2. `.env` 파일이 프로젝트 루트 디렉토리에 있는지 확인:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## 예제 추가하기

새로운 예제를 추가하려면:

1. 해당 데이터셋의 디렉토리에 `example.py` 파일을 생성합니다.
2. 코드에 적절한 주석과 설명을 추가합니다.
3. 이 README.md 파일을 업데이트하여 새로운 예제에 대한 정보를 추가합니다. 