from datasets import load_dataset
import os
import json

def download_mmlu_pro():
    """
    Downloads the MMLU-Pro dataset from Hugging Face and saves it locally.
    Dataset source: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
    """
    print("Downloading MMLU-Pro dataset...")
    
    # Load the dataset
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    
    # Create directories if they don't exist
    mmlu_pro_dir = os.path.join(os.path.dirname(__file__), "mmlu_pro_data")
    os.makedirs(mmlu_pro_dir, exist_ok=True)
    
    # Process and save each split
    for split in dataset.keys():
        split_data = dataset[split]
        
        # Group questions by category
        category_data = {}
        for item in split_data:
            category = item['category']
            if category not in category_data:
                category_data[category] = []
            
            # Convert the item to a dictionary and append
            question_data = {
                'question_id': item['question_id'],
                'question': item['question'],
                'options': item['options'],
                'answer': item['answer'],
                'answer_index': item['answer_index'],
                'cot_content': item['cot_content'],
                'category': item['category'],
                'src': item['src']
            }
            category_data[category].append(question_data)
        
        # Save each category to a separate file
        for category, questions in category_data.items():
            category_dir = os.path.join(mmlu_pro_dir, category.lower())
            os.makedirs(category_dir, exist_ok=True)
            
            output_file = os.path.join(category_dir, f"{split}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(questions, f, ensure_ascii=False, indent=2)
        
        print(f"Processed {split} split: {len(split_data)} questions across {len(category_data)} categories")
    
    print("MMLU-Pro dataset download and processing complete!")

if __name__ == "__main__":
    download_mmlu_pro() 