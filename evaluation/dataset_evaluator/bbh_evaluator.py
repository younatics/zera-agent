from typing import List, Dict, Any, Optional
from evaluation.base.evaluator import BaseEvaluator
import re
import random
import csv

class BBHEvaluator(BaseEvaluator):
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load BBH dataset."""
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
        """Format BBH question."""
        return f"{item['question']}"
    
    def evaluate_response(self, response: str, ground_truth: Dict[str, Any]) -> bool:
        """Evaluate BBH response. Robustly extract correct answers from various patterns."""
        def semantic_normalize(ans):
            ans = ans.strip().lower()
            # Handle negative forms first
            if any(x in ans for x in [
                "not plausible", "not possible", "not valid", "not correct", "not true", "not likely", "not feasible", "not reasonable", "not acceptable", "not accurate"]):
                return "no"
            if ans in ["no", "false", "implausible", "incorrect", "not plausible", "not valid", "not correct", "not possible"]:
                return "no"
            if ans in ["yes", "true", "plausible", "valid", "correct"]:
                return "yes"
            return ans
        def normalize_answer(ans):
            ans = re.sub(r'(?i)(final answer:|answer is|answer:|answer is|answer:)', '', ans)
            ans = ans.replace('\n', ' ').replace('\r', ' ')
            numbers = re.findall(r'-?\d+(?:\.\d+)?', ans)
            if numbers:
                return numbers[-1]
            result = re.sub(r'[^A-Z0-9-]', '', ans.upper())
            return result if result else ans.strip().upper()
        def normalize_list_answer(ans):
            ans = re.sub(r'(?i)(final answer:|answer is|answer:|answer is|answer:)', '', ans)
            ans = ans.replace('\n', ' ').replace('\r', ' ').replace(',', ' ')
            return [w for w in re.findall(r'[A-Z0-9-]+', ans.upper()) if w]
        response_clean = response.strip().upper()
        model_answer = ""
        # 1. Extract brackets/numbers/characters/sentences after 'Final Answer:'
        final_answer_match = re.search(r'FINAL ANSWER[:\-\s]*([\(\[]?([A-Z0-9\.]+)[\)\]]?)', response_clean)
        if final_answer_match:
            model_answer = final_answer_match.group(2)
        else:
            # 2. Alphabet/numbers/strings in the last parentheses (highest priority)
            matches = re.findall(r'\(([A-Z0-9\.]+)\)', response_clean)
            if matches:
                model_answer = matches[-1]
            else:
                # 3. Various patterns like '**J. ...**' or '**J**' or 'J. ...' or 'J ...'
                match = re.search(r'\*\*?([A-Z0-9\.]+)\*\*?[\s\.]', response_clean)
                if not match:
                    match = re.search(r'([A-Z0-9\.]+)\.[\s]', response_clean)
                if not match:
                    match = re.search(r'([A-Z0-9\.]+)[\s]', response_clean)
                if not match:
                    match = re.search(r'ANSWER IS[\s:]*\(?([A-Z0-9\.]+)\)?', response_clean)
                if not match:
                    match = re.search(r'ANSWER[\s:]*\(?([A-Z0-9\.]+)\)?', response_clean)
                if not match:
                    # 4. Extract the last single character/number/string
                    matches = re.findall(r'([A-Z0-9\.]+)', response_clean)
                    if matches:
                        model_answer = matches[-1]
                    else:
                        # 5. Short answer type (numbers, True/False, etc.) in the last line
                        lines = response_clean.splitlines()
                        for line in reversed(lines):
                            line = line.strip()
                            if line and not line.startswith('FINAL ANSWER'):
                                model_answer = line
                                break
                        else:
                            return False
                else:
                    model_answer = match.group(1)
        # If model_answer is still empty at the end, use the entire response_clean
        if not model_answer:
            model_answer = response_clean.strip()
        correct_answer = ground_truth['answer']
        # Debug logs
        print(f"Model answer: {model_answer} / Actual answer: {correct_answer}")
        print(f"normalize_answer(model): {normalize_answer(model_answer)} / normalize_answer(correct): {normalize_answer(correct_answer)}")
        print(f"normalize_list_answer(model): {normalize_list_answer(model_answer)} / normalize_list_answer(correct): {normalize_list_answer(correct_answer)}")
        # Compare answers
        model_list = normalize_list_answer(model_answer)
        correct_list = normalize_list_answer(correct_answer)
        # 1. Numeric type: If actual answer is numeric and that number exists in model answer, it's correct
        if re.fullmatch(r'-?\d+(?:\.\d+)?', correct_answer.strip()):
            if correct_answer.strip() in re.findall(r'-?\d+(?:\.\d+)?', model_answer):
                return True
        # 2. If numbers are included: Compare only with normalize_answer
        if re.search(r'-?\d+(?:\.\d+)?', model_answer) or re.search(r'-?\d+(?:\.\d+)?', correct_answer):
            if normalize_answer(model_answer) == normalize_answer(correct_answer):
                return True
        # 3. List type (multiple words): Correct if word lists are the same
        if len(model_list) > 1 or len(correct_list) > 1:
            if model_list == correct_list:
                return True
            # Compare only the last line of model answer with normalize_list_answer
            last_line = response.strip().split('\n')[-1]
            if normalize_list_answer(last_line) == correct_list:
                return True
        # 4. Semantic equivalence (boolean/discriminative, etc.): Including presence in model answer
        if semantic_normalize(model_answer) == semantic_normalize(correct_answer):
            return True
        if semantic_normalize(response.strip().split('\n')[-1]) == semantic_normalize(correct_answer):
            return True
        # Correct if model answer contains semantically equivalent keywords
        if semantic_normalize(correct_answer) == "no":
            if any(x in model_answer.lower() for x in ["no", "false", "implausible", "incorrect", "not plausible", "not valid", "not correct", "not possible"]):
                return True
        if semantic_normalize(correct_answer) == "yes":
            if any(x in model_answer.lower() for x in ["yes", "true", "plausible", "valid", "correct"]):
                return True
        # 0. Extract only the last word of model answer and treat as correct if normalize result matches actual answer
        last_line = response.strip().split('\n')[-1]
        last_word = re.findall(r'[A-Z0-9-]+', last_line.upper())
        if last_word and normalize_answer(last_word[-1]) == normalize_answer(correct_answer):
            return True
        # 6. Single answer: Compare with normalize_answer
        return normalize_answer(model_answer) == normalize_answer(correct_answer)

    def get_sample_indices(self, num_samples: int) -> list:
        """Return indices of samples to use for evaluation."""
        if not hasattr(self, 'dataset_cache') or self.dataset_cache is None:
            self.dataset_cache = self.load_dataset("")
        total_samples = len(self.dataset_cache)
        indices = random.sample(range(total_samples), min(num_samples, total_samples))
        return indices 