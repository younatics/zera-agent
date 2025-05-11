from typing import List, Dict, Any, Optional
from evaluation.base.evaluator import BaseEvaluator
import re
import random
import csv

class BBHEvaluator(BaseEvaluator):
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """BBH 데이터셋을 로드합니다."""
        if not dataset_path or dataset_path.lower() == "bbh":
            dataset_path = "agent/dataset/bbh_data/test.csv"
        data = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['input'].strip() and row['target'].strip():
                    data.append({
                        'question': row['input'],
                        'answer': row['target']
                    })
        return data
    
    def format_question(self, item: Dict[str, Any]) -> str:
        """BBH 질문을 포맷팅합니다."""
        return f"{item['question']}"
    
    def evaluate_response(self, response: str, ground_truth: Dict[str, Any]) -> bool:
        """BBH 응답을 평가합니다. 다양한 패턴에서 정답을 robust하게 추출합니다."""
        def semantic_normalize(ans):
            ans = ans.strip().lower()
            # 부정형 우선 처리
            if any(x in ans for x in [
                "not plausible", "not possible", "not valid", "not correct", "not true", "not likely", "not feasible", "not reasonable", "not acceptable", "not accurate"]):
                return "no"
            if ans in ["no", "false", "implausible", "incorrect", "not plausible", "not valid", "not correct", "not possible"]:
                return "no"
            if ans in ["yes", "true", "plausible", "valid", "correct"]:
                return "yes"
            return ans
        def normalize_answer(ans):
            ans = re.sub(r'(?i)(final answer:|answer is|answer:|정답은|정답:)', '', ans)
            ans = ans.replace('\n', ' ').replace('\r', ' ')
            numbers = re.findall(r'-?\d+(?:\.\d+)?', ans)
            if numbers:
                return numbers[-1]
            result = re.sub(r'[^A-Z0-9-]', '', ans.upper())
            return result if result else ans.strip().upper()
        def normalize_list_answer(ans):
            ans = re.sub(r'(?i)(final answer:|answer is|answer:|정답은|정답:)', '', ans)
            ans = ans.replace('\n', ' ').replace('\r', ' ').replace(',', ' ')
            return [w for w in re.findall(r'[A-Z0-9-]+', ans.upper()) if w]
        response_clean = response.strip().upper()
        model_answer = ""
        # 디버깅용: 정규식 매칭 상태 출력
        print("[DEBUG] response_clean:", response_clean)
        print("[DEBUG] 괄호 매치:", re.findall(r'\(([A-Z0-9\.]+)\)', response_clean))
        # 1. 'Final Answer:' 이후에 괄호/숫자/문자/문장 추출
        final_answer_match = re.search(r'FINAL ANSWER[:\-\s]*(IS)?[\s]*([\(\[]?([A-Z0-9\.]+)[\)\]]?)', response_clean)
        if final_answer_match:
            # 'FINAL ANSWER IS (J)'와 같이 IS가 있으면, 괄호 안 알파벳 우선 추출
            if final_answer_match.group(2) and re.match(r'[\(\[]?[A-Z0-9\.]+[\)\]]?', final_answer_match.group(2)):
                # 괄호가 있으면 괄호 제거
                model_answer = re.sub(r'^[\(\[]|[\)\]]$', '', final_answer_match.group(2))
            else:
                model_answer = final_answer_match.group(3) or final_answer_match.group(2)
            print("[DEBUG] model_answer (final_answer_match):", model_answer)
        else:
            # 2. 맨 마지막 괄호 안에 있는 알파벳/숫자/문자열 (항상 최우선)
            matches = re.findall(r'\(([A-Z0-9\.]+)\)', response_clean)
            if matches:
                model_answer = matches[-1]
                print("[DEBUG] model_answer (괄호 매치):", model_answer)
            else:
                # 3. '**J. ...**' 또는 '**J**' 또는 'J. ...' 또는 'J ...' 등 다양한 패턴
                match = re.search(r'\*\*?([A-Z0-9\.]+)\*\*?[\s\.]', response_clean)
                if not match:
                    match = re.search(r'([A-Z0-9\.]+)\.[\s]', response_clean)
                if not match:
                    match = re.search(r'([A-Z0-9\.]+)[\s]', response_clean)
                if not match:
                    match = re.search(r'ANSWER IS[\s:]*\(?([A-Z0-9\.]+)\)?', response_clean)
                if not match:
                    match = re.search(r'정답[은는]?[\s:]*\(?([A-Z0-9\.]+)\)?', response_clean)
                if not match:
                    # 4. 마지막에 등장하는 한 글자/숫자/문자열 추출
                    matches = re.findall(r'([A-Z0-9\.]+)', response_clean)
                    if matches:
                        model_answer = matches[-1]
                        print("[DEBUG] model_answer (마지막 등장):", model_answer)
                    else:
                        # 5. 마지막 줄의 단답형(숫자, True/False 등)
                        lines = response_clean.splitlines()
                        for line in reversed(lines):
                            line = line.strip()
                            if line and not line.startswith('FINAL ANSWER'):
                                model_answer = line
                                print("[DEBUG] model_answer (마지막 줄):", model_answer)
                                break
                        else:
                            print("[DEBUG] model_answer (실패):", model_answer)
                            return False
                else:
                    model_answer = match.group(1)
                    print("[DEBUG] model_answer (패턴 매치):", model_answer)
                print("[DEBUG] model_answer (else 블록 마지막):", model_answer)
        # 마지막에라도 model_answer가 비어있으면 전체 response_clean 사용
        if not model_answer:
            model_answer = response_clean.strip()
            print("[DEBUG] model_answer (비어있을 때):", model_answer)
        print("[DEBUG] model_answer (정답 비교 직전):", model_answer)
        correct_answer = ground_truth['answer']
        # 디버깅용 로그
        print(f"모델 답변: {model_answer} / 실제 답변: {correct_answer}")
        print(f"normalize_answer(모델): {normalize_answer(model_answer)} / normalize_answer(정답): {normalize_answer(correct_answer)}")
        print(f"normalize_list_answer(모델): {normalize_list_answer(model_answer)} / normalize_list_answer(정답): {normalize_list_answer(correct_answer)}")
        # 정답 비교
        model_list = normalize_list_answer(model_answer)
        correct_list = normalize_list_answer(correct_answer)
        # 1. 숫자형: 실제 정답이 숫자이고, 모델 답변 내에 그 숫자가 있으면 정답
        if re.fullmatch(r'-?\d+(?:\.\d+)?', correct_answer.strip()):
            if correct_answer.strip() in re.findall(r'-?\d+(?:\.\d+)?', model_answer):
                return True
        # 2. 숫자가 포함된 경우: normalize_answer만 비교
        if re.search(r'-?\d+(?:\.\d+)?', model_answer) or re.search(r'-?\d+(?:\.\d+)?', correct_answer):
            if normalize_answer(model_answer) == normalize_answer(correct_answer):
                return True
        # 3. 리스트형(여러 단어): 단어 리스트가 같으면 정답
        if len(model_list) > 1 or len(correct_list) > 1:
            if model_list == correct_list:
                return True
            # 모델 답변의 마지막 줄만 normalize_list_answer로 비교
            last_line = response.strip().split('\n')[-1]
            if normalize_list_answer(last_line) == correct_list:
                return True
        # 4. 의미적 동치(불리언/판별형 등): 모델 답변 내 포함 여부까지
        if semantic_normalize(model_answer) == semantic_normalize(correct_answer):
            return True
        if semantic_normalize(response.strip().split('\n')[-1]) == semantic_normalize(correct_answer):
            return True
        # 모델 답변에 의미적 동치 키워드가 포함되어 있으면 정답
        if semantic_normalize(correct_answer) == "no":
            if any(x in model_answer.lower() for x in ["no", "false", "implausible", "incorrect", "not plausible", "not valid", "not correct", "not possible"]):
                return True
        if semantic_normalize(correct_answer) == "yes":
            if any(x in model_answer.lower() for x in ["yes", "true", "plausible", "valid", "correct"]):
                return True
        # 0. 모델 답변의 마지막 단어만 추출해서 실제 정답과 normalize 결과가 같으면 무조건 정답 처리
        last_line = response.strip().split('\n')[-1]
        last_word = re.findall(r'[A-Z0-9-]+', last_line.upper())
        if last_word and normalize_answer(last_word[-1]) == normalize_answer(correct_answer):
            return True
        # 6. 단일 답변: normalize_answer로 비교
        return normalize_answer(model_answer) == normalize_answer(correct_answer)

    def get_sample_indices(self, num_samples: int) -> list:
        """평가에 사용할 샘플의 인덱스를 반환합니다."""
        if not hasattr(self, 'dataset_cache') or self.dataset_cache is None:
            self.dataset_cache = self.load_dataset("")
        total_samples = len(self.dataset_cache)
        indices = random.sample(range(total_samples), min(num_samples, total_samples))
        return indices 