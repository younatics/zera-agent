from typing import List, Dict, Any, Optional
from datasets import load_dataset
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetLoader:
    DATASET_INFO = {
        "gsm8k": {
            "path": "gsm8k",
            "split": "test",
            "local_path": "datasets/gsm8k/test.json"
        },
        "mmlu": {
            "path": "cais/mmlu",
            "split": "test",
            "local_path": "datasets/mmlu/test.json"
        },
        "mmlu_pro": {
            "path": "TIGER-Lab/MMLU-Pro",
            "split": "test",
            "local_path": "datasets/mmlu_pro/test.json"
        },
        "bbh": {
            "path": "BAAI/big-bench-hard",
            "split": "test",
            "local_path": "datasets/bbh/test.json"
        },
        "cnn_dailymail": {
            "path": "cnn_dailymail",
            "split": "test",
            "local_path": "datasets/cnn_dailymail/test.json"
        },
        "samsum": {
            "path": "samsum",
            "split": "test",
            "local_path": "datasets/samsum/test.json"
        },
        "mbpp": {
            "path": "mbpp",
            "split": "test",
            "local_path": "datasets/mbpp/test.json"
        }
    }
    
    @staticmethod
    def load_dataset(dataset_name: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """허깅페이스에서 데이터셋을 로드하거나 로컬 파일에서 로드합니다."""
        if dataset_name not in DatasetLoader.DATASET_INFO:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        info = DatasetLoader.DATASET_INFO[dataset_name]
        local_path = Path(info["local_path"])
        
        # 로컬 파일이 있으면 로컬에서 로드
        if local_path.exists():
            logger.info(f"Loading {dataset_name} from local file: {local_path}")
            with open(local_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            # 허깅페이스에서 다운로드
            logger.info(f"Downloading {dataset_name} from Hugging Face")
            dataset = load_dataset(info["path"], split=info["split"])
            data = [item for item in dataset]
            
            # 로컬에 저장
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        # 샘플링
        if num_samples:
            data = data[:num_samples]
            
        return data 