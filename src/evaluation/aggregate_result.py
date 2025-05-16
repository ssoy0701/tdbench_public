import pandas as pd
import ast
import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument('--tasktype', '-t', choices=['basic', 'temp_align', 'join'], default='join')
parser.add_argument('--task',  default='culture')

args = parser.parse_args()

if args.tasktype == 'join':
    def parse_txt_to_table_join(file_path, model):
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as file:
            data_dict = json.load(file)

        metrics_2hop = data_dict['2hop']['metrics']
        metrics_3hop = data_dict['3hop']['metrics']

        category = [f'2hop_{key}'for key in metrics_2hop.keys()] \
            + [f'3hop_{key}'for key in metrics_3hop.keys()]

        table_data = {"Category": category, 
                      f"{model}": list(metrics_2hop.values()) \
                        + list(metrics_3hop.values())}
 
        # Create a DataFrame
        df = pd.DataFrame(table_data)
        return df


    model_list = ['gpt35', 'gpt4',  'gpt4o', 'llama', 'mixtral', 'gemma', 'qwen', 'granite']
    # model_list = ['gpt35', 'gpt4']
    df = pd.DataFrame()

    for model in model_list:
        print(f"Processing {model}...")
        # Read the file
        file_path = f'/home/v-kimsoyeon/temp_eval/results/olympic/{model}_join_result.json'

        # Parse the text file and create the table
        table = parse_txt_to_table_join(file_path, model)
        df = pd.concat([df, table], ignore_index=True).groupby('Category').max().reset_index()


    total_result_path = file_path.replace(model, 'total').replace('.json', '.csv')
    print(df)
    df.to_csv(total_result_path, index=False)



elif args.tasktype == 'basic':
    def parse_txt_to_table(file_path, model):
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as file:
            data_dict = json.load(file)
        
        # Extract the relevant data for the table
        table_data = {
            "Category": [
                "unique_correct", "unique_time", "unique_both",
                "none_correct", "none_time", "none_both",
                "multiple_correct", "multiple_time", "multiple_both",
            ],
            f"{model}": [
                data_dict['unique']['answer']['percentage']['correct'],
                data_dict['unique']['time']['percentage']['loose'],
                data_dict['unique']['both']['percentage']['consistent_and_correct'],
                data_dict['none']['answer']['percentage']['correct'],
                data_dict['none']['time']['percentage']['loose'],
                data_dict['none']['both']['percentage']['consistent_and_correct'],
                data_dict['multiple']['answer']['percentage']['correct'],
                data_dict['multiple']['time']['percentage']['loose'],
                data_dict['multiple']['both']['percentage']['consistent_and_correct'],
            ]
        }
        
        # Create a DataFrame
        df = pd.DataFrame(table_data)
        return df


    model_list = ['gpt35', 'gpt4',  'gpt4o', 'llama', 'mixtral', 'gemma', 'qwen', 'granite']
    # model_list = ['gpt35', 'gpt4']
    df = pd.DataFrame()

    for model in model_list:
        print(f"Processing {model}...")
        # Read the file
        file_path = f'/home/soyeon/temp_eval/results/dyknow/leaders/{model}_basic_result_verified_result.json'

        # Parse the text file and create the table
        table = parse_txt_to_table(file_path, model)
        # print(table)
        df = pd.concat([df, table], ignore_index=True).groupby('Category').max().reset_index()


    total_result_path = file_path.replace(model, 'total').replace('.json', '.csv')
    print(df)
    df.to_csv(total_result_path, index=False)



elif args.tasktype == 'temp_align':

    output_rows = []

    model_list = ['gpt35', 'gpt4',  'gpt4o', 'llama', 'mixtral', 'gemma', 'qwen', 'granite']

    for model in model_list:
        # Read the file
        model_log_path = f'/home/soyeon/temp_eval/results/alignment/{args.task}/{model}_now_result.txt'
        with open(model_log_path, 'r') as f:
            content = f.read()

        # percentage dict만 추출
        if 'percentage' not in content:
            continue
        try:
            percentage_str = content.split('percentage:')[1].strip()
            percentage_dict = ast.literal_eval(percentage_str)
        except Exception as e:
            print(f"Error parsing {model_log_path}: {e}")
            continue

        # 계산
        correct_A = percentage_dict.get('correct', 0) + percentage_dict.get('partial_correct', 0)
        correct_AT = percentage_dict.get('correct', 0)
        correct_diff = correct_A - correct_AT

        outdated_A = percentage_dict.get('outdated', 0) + percentage_dict.get('partial_outdated', 0)
        outdated_AT = percentage_dict.get('outdated', 0)
        outdated_diff = outdated_A - outdated_AT

        incorrect_A = percentage_dict.get('incorrect', 0)
        incorrect_AT = incorrect_A + percentage_dict.get('partial_correct', 0) + percentage_dict.get('partial_outdated', 0)
        incorrect_diff = incorrect_AT - incorrect_A

        output_rows.append({
            'Model': model,
            'correct_A': correct_A,
            'correct_AT': correct_AT,
            'correct_diff': correct_diff,
            'outdated_A': outdated_A,
            'outdated_AT': outdated_AT,
            'outdated_diff': outdated_diff,
            'incorrect_A': incorrect_A,
            'incorrect_AT': incorrect_AT,
            'incorrect_diff': incorrect_diff
        })

    df = pd.DataFrame(output_rows)

    if not df.empty:
        avg_row = df.drop(columns=['Model']).mean(numeric_only=True)
        avg_row['Model'] = 'Average'
        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    df.update(df.select_dtypes(include='number').round(1))

    df.to_csv(model_log_path.replace(model, 'total').replace('txt', 'csv'), index=False)
    print("Saved to total_result.csv")




