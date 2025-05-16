import pandas as pd
import argparse
import ast

# Argument parser to get CSV file path
parser = argparse.ArgumentParser(description='Interactively update rows in a CSV with incorrect analysis.')
parser.add_argument('--file', '-f', type=str, help='Path to the CSV file')
parser.add_argument('--answer', '-a', type=str)
parser.add_argument('--time', '-t', type=str)
parser.add_argument('--qtype', '-q', type=str)
parser.add_argument('--sample', '-s', type=int)
parser.add_argument('--op', '-o', type=str)
parser.add_argument('--task', type=str, choices=['answer', 'time', 'both'], default='answer', help='Task type (answer or time)')
parser.add_argument('--no_update', '-n', action='store_true', help='Do not update the CSV file')
args = parser.parse_args()


# Load CSV file into a DataFrame
csv_file = args.file
df_orig = pd.read_csv(csv_file)

new_df = df_orig.copy()

if args.answer:
    new_df = new_df[new_df['analysis_answer'] == args.answer]

if args.time:
    new_df = new_df[new_df['analysis_time'] == args.time]

if args.qtype:
    new_df = new_df[new_df['qtype'] == args.qtype]

if args.op:
    new_df = new_df[new_df['op'] == args.op]

if args.sample:
    
    # new_df = new_df[~new_df['ssoy_time'].isin(['partial_consistent', 'consistent', 'inconsistent'])]
    new_df = new_df.sample(args.sample, random_state=1116)





pd.set_option('display.max_colwidth', None)

# # heuristics for entity resolution errors
# def get_answer_cand_name(answer):
#     cand = []

#     answer = answer.lower().strip()
#     cand.append(answer) # ideal case
#     cand.append(unidecode.unidecode(answer)) # for unicode errors

#     split_answer = answer.split(' ')
#     if len(split_answer) == 3: # probably names with middle name
#         cand.append(split_answer[0] + ' ' + split_answer[2])
#         cand.append(split_answer[0] + ' ' + split_answer[1])
#         cand.append(split_answer[0])

#     return cand

# Interactive function to update rows with 'incorrect' analysis
def update_incorrect_rows(df, col_name):

    save_file = csv_file.split('.csv')[0] + '_updated.csv'
    for index, row in df.iterrows():
        # if row['analysis_answer'] == 'correct' and row['analysis_time'] == 'none' and \
        #     row['qtype'] == 'basic_none':
        row_dict = row.to_dict()
        print('========================================')
        for key, value in row_dict.items():

            if key == 'analysis':
                continue

            if key == 'outdated':
                value = ast.literal_eval(value)
                find = False
                # for _, r in enumerate(value):
                #     if r[0].lower().strip() in row['model_response'].lower():
                #         print("[outdated rows (hit)]")
                #         print(r)
                #     find = True
                #     break

                for i in range(len(value)):
                    if value[i][0].lower() in row['model_response'].lower():
                        find = True
                        print("[outdated rows (hit)]")
                        print(value[i])
                        break
            
                if not find:
                    value.sort(key=lambda x: x[1], reverse=True)
                    print("[outdated rows]")
                    print(value[:3])
                print("\n")

            else:
                print(f"[{key}]\n{value}\n")

        print('========================================')
        inst = 'What is your analysis of model response?\n\tcorrect(c): answer (o) time (o)\n\tpartial correct(pc): answer (o) time (x)\n\toutdated(o): answer(outdated) time(consistent)\n\tpartially outdated(po): answer(outdated) time(not consistent)\n\tincorrect(i): answer(x) or time(x)\n:'
        new_value = input(inst).strip().lower()
        

        if new_value == 'c':
            new_value = 'correct'
        elif new_value == 'pc':
            new_value = 'partial_correct'
        elif new_value == 'o':
            new_value = 'outdated'
        elif new_value == 'po':
            new_value = 'partial_outdated'
        elif new_value == 'i':
            new_value = 'incorrect'
        elif new_value == 'u':
            new_value = 'unsure'
        else:
            print("No value.")

        df.at[index, col_name] = new_value
        df.to_csv(csv_file, index=False)
        print(f"Updated 'analysis' for Row {index} to: {new_value}")



    # results_answer = ['correct', 'incorrect', 'ambiguous', 'unsure']
    # results_time = ['consistent', 'inconsistent', 'none']


def show_rows(df): 
    for index, row in df.iterrows():
        row_dict = row.to_dict()
        print('========================================')
        for key, value in row_dict.items():

            if key in ['qtype', 'op', 'sql_fir_db']:
                continue

            if key == 'incorrect_rows':
                value = ast.literal_eval(value)
                find = False

                for i in range(len(value)):
                    if value[i]['Name'].lower() in row['model_response'].lower():
                        find = True
                        print("[Incorrect rows (hit)]")
                        print(value[i])
                        break
            
                if not find:
                    value.sort(key=lambda x: x['Start'], reverse=True)
                    print("[Incorrect rows]")
                    print(value[:5])
                print("\n")

            else:
                print(f"[{key}]\n{value}\n")

        print('========================================')
        inst = 'Press enter to proceed'
        new_value = input(inst).strip().lower()



def update_incorrect_rows_basic(df, col_name):
    for index, row in df.iterrows():
        # if row['analysis_answer'] == 'correct' and row['analysis_time'] == 'none' and \
        #     row['qtype'] == 'basic_none':
        row_dict = row.to_dict()



        print('========================================')
        for key, value in row_dict.items():

            if key in ['qtype', 'op', 'sql_fir_db']:
                continue

            if key == 'incorrect_rows':
                value = ast.literal_eval(value)
                find = False
                # for _, r in enumerate(value):
                #     if r[0].lower().strip() in row['model_response'].lower():
                #         print("[outdated rows (hit)]")
                #         print(r)
                #     find = True
                #     break

                for i in range(len(value)):
                    if value[i]['Name'].lower() in row['model_response'].lower():
                        find = True
                        print("[Incorrect rows (hit)]")
                        print(value[i])
                        break
            
                if not find:
                    value.sort(key=lambda x: x['Start'], reverse=True)
                    print("[Incorrect rows]")
                    print(value[:5])
                print("\n")

            else:
                print(f"[{key}]\n{value}\n")

        print('========================================')

        if args.task == 'answer':
            inst = 'What is your analysis of model response (answer, c/i/a/u)?:'

        elif args.task == 'time':
            inst = 'What is your analysis of model response (time, c/pc/i/a)?:'

        new_value = input(inst).strip().lower()

        if args.task == 'answer':

            if new_value == 'c':
                new_value = 'correct'
            elif new_value == 'i':
                new_value = 'incorrect'
            elif new_value == 'a':
                new_value = 'ambiguous'
            elif new_value == 'u':
                new_value = 'unsure'
            else:
                print("No value.")

            df_orig.at[index, f'{col_name}_answer'] = new_value
            df_orig.to_csv(csv_file, index=False)
            print(f"Updated 'analysis_answer' for Row {index} to: {new_value}")


            inst_time = 'What is your analysis of model response (time, c/i/n)?:'
            new_value = input(inst_time).strip().lower()

            if new_value == 'c':
                new_value = 'consistent'
            elif new_value == 'i':
                new_value = 'inconsistent'
            elif new_value == 'n':
                new_value = 'none'
            else:
                print("No value.")

            df_orig.at[index, f'{col_name}_time'] = new_value
            df_orig.to_csv(csv_file, index=False)
            print(f"Updated 'analysis_time' for Row {index} to: {new_value}")


        elif args.task == 'time':
            

        
            if new_value == 'c':
                new_value = 'consistent'
            elif new_value == 'pc':
                new_value = 'partially_consistent'
            elif new_value == 'i':
                new_value = 'inconsistent'
            elif new_value == 'a':
                new_value = 'ambiguous'
            else:
                print("No value.")

            df_orig.at[index, f'{col_name}_time'] = new_value
            df_orig.to_csv(csv_file, index=False)
            print(f"Updated 'analysis_time' for Row {index} to: {new_value}")



if args.no_update:
    show_rows(new_df)
else:
    # Run the interactive update function
    col_name = input("What is your name?: ").strip().lower()
    update_incorrect_rows_basic(new_df, col_name)