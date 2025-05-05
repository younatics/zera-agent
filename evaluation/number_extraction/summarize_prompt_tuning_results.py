import pandas as pd
import argparse

# 인자 파서 설정
def parse_args():
    parser = argparse.ArgumentParser(description='Prompt tuning results summarizer')
    parser.add_argument('--input', type=str, default='evaluation/number_extraction/prompt_tuning_results_bbh_2.csv', help='Input CSV file path')
    parser.add_argument('--output', type=str, default='evaluation/number_extraction/prompt_tuning_results_bbh_2_summary.csv', help='Output CSV file path')
    return parser.parse_args()

def main():
    args = parse_args()
    input_path = args.input
    output_path = args.output

    # CSV 읽기
    # encoding 오류가 있을 경우 encoding='utf-8' 또는 'utf-8-sig' 등으로 변경
    df = pd.read_csv(input_path)

    # 평가항목별 점수 컬럼 추출 (Average Score, Standard Deviation, Top3 Average Score 등은 제외)
    exclude_cols = {'Average Score', 'Standard Deviation', 'Top3 Average Score', 'Best Average Score', 'Best Sample Score'}
    score_columns = [col for col in df.columns if col.endswith('_Score') and col not in exclude_cols]

    # 집계할 항목: 대표 점수 + 평가항목별 점수
    agg_items = ['Average Score', 'Standard Deviation', 'Top3 Average Score'] + score_columns

    # 이터레이션별로 평균값 구하기
    pivot_df = df.groupby('Iteration')[agg_items].mean().T

    # 결과 저장
    pivot_df.to_csv(output_path)

    print(f'완료! -> {output_path}')

if __name__ == '__main__':
    main() 