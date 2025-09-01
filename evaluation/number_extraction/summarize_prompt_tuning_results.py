import pandas as pd
import argparse

# Argument parser setup
def parse_args():
    parser = argparse.ArgumentParser(description='Prompt tuning results summarizer')
    parser.add_argument('--input', type=str, default='evaluation/number_extraction/prompt_tuning_results_bbh_2.csv', help='Input CSV file path')
    parser.add_argument('--output', type=str, default='evaluation/number_extraction/prompt_tuning_results_bbh_2_summary.csv', help='Output CSV file path')
    return parser.parse_args()

def main():
    args = parse_args()
    input_path = args.input
    output_path = args.output

    # Read CSV
    # Change encoding to 'utf-8' or 'utf-8-sig' if there are encoding errors
    df = pd.read_csv(input_path)

    # Extract score columns by evaluation item (exclude Average Score, Standard Deviation, Top3 Average Score, etc.)
    exclude_cols = {'Average Score', 'Standard Deviation', 'Top3 Average Score', 'Best Average Score', 'Best Sample Score'}
    score_columns = [col for col in df.columns if col.endswith('_Score') and col not in exclude_cols]

    # Items to aggregate: representative score + score by evaluation item
    agg_items = ['Average Score', 'Standard Deviation', 'Top3 Average Score'] + score_columns

    # Calculate average values by iteration
    pivot_df = df.groupby('Iteration')[agg_items].mean().T

    # Save results
    pivot_df.to_csv(output_path)

    print(f'Completed! -> {output_path}')

if __name__ == '__main__':
    main() 