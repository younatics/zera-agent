from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import csv
import io

@dataclass
class TestCaseResult:
    """개별 테스트 케이스의 결과를 저장하는 데이터 클래스"""
    test_case: int
    question: str
    expected_output: str
    actual_output: str
    score: float
    evaluation_details: Dict

@dataclass
class IterationResult:
    """각 반복에 대한 결과를 저장하는 데이터 클래스"""
    iteration: int
    system_prompt: str
    user_prompt: str
    avg_score: float
    std_dev: float
    top3_avg_score: float
    best_avg_score: float
    best_sample_score: float
    test_case_results: List[TestCaseResult]
    meta_prompt: Optional[str]
    task_type: str
    task_description: str
    created_at: datetime = field(default_factory=datetime.now)

    def add_test_case_result(self, 
                         test_case: int,
                         question: str,
                         expected_output: str,
                         actual_output: str,
                         score: float,
                         evaluation_details: Dict) -> None:
        """테스트 케이스 결과를 추가합니다."""
        test_case_result = TestCaseResult(
            test_case=test_case,
            question=question,
            expected_output=expected_output,
            actual_output=actual_output,
            score=score,
            evaluation_details=evaluation_details
        )
        self.test_case_results.append(test_case_result)

        # 최고 점수 업데이트
        if score > self.best_sample_score:
            self.best_sample_score = score