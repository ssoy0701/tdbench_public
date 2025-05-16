import sys

sys.path.append('/home/soyeon/temp_eval/src')

import pandas as pd
import argparse
import ast
import utils.utils_llm as lutils
import re
import signal
import time
import json


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Code execution timed out!")


signal.signal(signal.SIGALRM, timeout_handler)


# criteria dict
criteria_dict = {
    'both': ['overlap', 'overlapped-by', 'equal', 'start', 'finish',  'during', 'contain'],
    'start': ['after', 'met-by', 'started-by'],
    'end': ['before', 'meet', 'finished-by']
}


def get_criteria(relation):
    """
    Get criteria based on the relation
    :param relation: relation string
    :return: criteria list
    """
    for key, values in criteria_dict.items():
        if relation in values:
            return key
    raise ValueError(f"Unknown relation: {relation}")
    

def calc_results(args, df):
    df['analysis_answer_time'] = df.apply( lambda row: 'correct' \
            if row['analysis_answer'] == 'correct' and row['analysis_time'] == 'consistent' else 'incorrect', \
                axis=1)


    # save the result
    final_dict = {}

    # for answer cardinality
    for qtype in ['unique', 'none', 'multiple']:
        qtype_df = df[df['qtype'] == 'basic' + '_' + qtype]
        num_questions = len(qtype_df)
        if num_questions == 0:
            continue

        # 개수 계산
        correct_answer = len(qtype_df[qtype_df['analysis_answer'] == 'correct'])
        incorrect_answer = len(qtype_df[qtype_df['analysis_answer'] == 'incorrect'])
        ambiguous_answer = len(qtype_df[qtype_df['analysis_answer'] == 'ambiguous'])
        unsure_answer = len(qtype_df[qtype_df['analysis_answer'] == 'unsure'])

        consistent_time = len(qtype_df[qtype_df['analysis_time'] == 'consistent'])
        partial_time = len(qtype_df[qtype_df['analysis_time'] == 'partial_consistent'])
        inconsistent_time = len(qtype_df[qtype_df['analysis_time'] == 'inconsistent'])

        both_correct = len(qtype_df[qtype_df['analysis_answer_time'] == 'correct'])

        # 딕셔너리 구성
        final_dict[qtype] = {
            'answer': {
                'total': num_questions ,
                'correct': correct_answer,
                'incorrect': incorrect_answer,
                'ambiguous': ambiguous_answer,
                'unsure': unsure_answer,
                'percentage': {
                    'correct': round(correct_answer / num_questions * 100, 1) if num_questions  else 0,
                    'incorrect': round(incorrect_answer / num_questions * 100, 1) if num_questions  else 0,
                    'ambiguous': round(ambiguous_answer / num_questions * 100, 1) if num_questions  else 0,
                    'unsure': round(unsure_answer / num_questions   * 100, 1) if num_questions    else 0,
                }
            },
            'time': {
                'total': num_questions,
                'consistent': consistent_time,
                'partial_consistent': partial_time,
                'inconsistent': inconsistent_time,
                'loose': consistent_time + (partial_time // 2),
                'percentage': {
                    'consistent': round(consistent_time / num_questions * 100, 1) if num_questions else 0,
                    'partial_consistent': round(partial_time / num_questions * 100, 1) if num_questions else 0,
                    'inconsistent': round(inconsistent_time / num_questions * 100, 1) if num_questions else 0,
                    'loose': round((consistent_time + (partial_time // 2)) / num_questions * 100, 1) if num_questions else 0
                }
            },
            'both': {
                'consistent_and_correct': both_correct,
                'percentage': {
                    'consistent_and_correct': round(both_correct / num_questions * 100, 1) if num_questions else 0
                }
            }
        }

    # for operations
    basic_op_lists = ['after', 'before', 'equal', 'contain', 'finish', 'start', 'meet', 'met-by', 'overlap', 'during', 'finished-by', 'started-by', 'overlapped-by']
    
    op_dict = {op: 0 for op in basic_op_lists}
    for op in basic_op_lists:
        op_df = df[df['op'].str.contains(op)]

        total_op_num = len(op_df)
        correct_percentage = round(len(op_df[op_df['analysis_answer'] == 'correct']) / total_op_num * 100, 1)
        consistent_percentage = round(len(op_df[op_df['analysis_time'] == 'consistent']) / \
                                                    (len(op_df[op_df['analysis_time'] == 'consistent']) + len(op_df[op_df['analysis_time'] == 'inconsistent'])) * 100, 1)
        correct_and_consistent_percentage = round(len(op_df[(op_df['analysis_answer_time'] == 'correct')]) / total_op_num * 100, 1)

        op_result = {'total': total_op_num, \
                     'correct': correct_percentage,\
                    'consistent': consistent_percentage,\
                    'correct_and_consistent': correct_and_consistent_percentage
                    }

        op_dict[op] = op_result

    final_dict['op'] = op_dict

    final_dict['total'] = {
        'total': len(df),
    }

    # dump with json
    result_file_path = args.file_path.replace('.csv', '_result.json')
    with open(result_file_path, 'w') as f:
        json.dump(final_dict, f, indent=4)
        
    df.to_csv(args.file_path.replace(".csv", "_final.csv"), index=False)
    print('Calculated result and updated CSV file saved successfully.')    
    print(f"TXT result saved at {result_file_path}.")
    


def verify_with_llm(args, df):

    for row_id, row in df.iterrows():
        print('Processing row:', row['idx'])


        # get model_response
        response = row['model_response'].lower()

        # get criteria
        criteria = get_criteria(row['op'])



        # evaluate w.r.t. question type
        if row['qtype'] == 'basic_none': # if answer is 'no answer'
            if 'no answer' in response:
                answer = 'consistent'

            else:
                answer = 'inconsistent'

        else:
            # get time and answer

            correct_rows = ast.literal_eval(row['correct_rows'])
            key_list = list(correct_rows[0].keys())
            answer_entity_key = [key for key in key_list if key not in ['Start', 'End']][0]


            # write prompt to evaluate using LLM
            if criteria == 'both':

                # create entity date
                entity_date_list = []
                for correct_row in correct_rows:
                    start_time = pd.to_datetime(correct_row['Start']).strftime('%Y-%m-%d')
                    end_time = pd.to_datetime(correct_row['End']).strftime('%Y-%m-%d')
                    answer_entity = correct_row[answer_entity_key]

                    entity_date_list.append(f"For the entity `{answer_entity}`:\n**Start date:** {start_time}\n**End date:** {end_time}\n")

                entity_date = "\n".join(entity_date_list)
                
                # create prompt
                prompt = f"""You are given a reference **start date** and **end date**. Check whether the response correctly includes both dates, even if they are expressed in a different but equivalent format (e.g., `26 Jan 2025`, `January 26, 2025`, `2025/01/26`, etc.).

    - If **both** of the two dates is are correctly mentioned with the intended meaning (i.e., the start date is described as the start date, and the end date as the end date), respond with **"Yes"**.  
    - If **one** of the two dates is correctly mentioned with the intended meaning, respond with **"Half"**.
    - If **neither** date is correctly mentioned with the correct meaning, respond with **"No"**.  
    Your answer must be one of: `Yes`, `Half`, or `No`. Be concise.

    {entity_date}

    **Response:**  
    {response}

    **Answer:"""


            else:

                # create entity date
                entity_date_list = []
                for correct_row in correct_rows:
                    start_time = pd.to_datetime(correct_row['Start']).strftime('%Y-%m-%d')
                    end_time = pd.to_datetime(correct_row['End']).strftime('%Y-%m-%d')
                    answer_entity = correct_row[answer_entity_key]
                    date = start_time if criteria == 'start' else end_time

                    entity_date_list.append(f"For the entity `{answer_entity}`:\n**{criteria.capitalize()} date:** {date}\n*")

                entity_date = "\n".join(entity_date_list)

                start_time = pd.to_datetime(correct_rows[0]['Start']).strftime('%Y-%m-%d')
                end_time = pd.to_datetime(correct_rows[0]['End']).strftime('%Y-%m-%d')
                answer_entity = correct_rows[0][answer_entity_key]

                date = start_time if criteria == 'start' else end_time

                prompt = f"""You are given a reference date, which is either a start date or an end date. Check whether the response correctly includes this specific date, even if it is expressed in a different but equivalent format (e.g., `26 Jan 2025`, `January 26, 2025`, `2025/01/26`, etc.).

    - If the correct {criteria} date is mentioned with the correct meaning, respond with **"Yes"**.  
    - If the date is incorrect, missing, or referred to incorrectly (e.g., an end date mentioned as a start date), respond with **"No"**.  
    Your answer must be one of: `Yes` or `No`. Be concise.

    {entity_date}

    **Response:**  
    "{response}"

    **Answer:"""

            print('prompt:', prompt)
            
            # call model APIs and save responses
            fail_cnt = 0
            while (fail_cnt < 2):
            # call model
                try:
                    signal.alarm(15)
                    result = lutils.run_llm(prompt=prompt, \
                                            tasktype='no_system_msg', \
                                            model='deepseek')
                    print('result:', result)
                    signal.alarm(0)
                    break

                except Exception as error:
                    fail_cnt +=1
                    # print(f"{model} ERROR with idx: {id} (fail_cnt: {fail_cnt}) : {error}")
                    
                    if fail_cnt == 2:
                        print(f"Too many errors. Skipping idx: {row_id}")
                        response = "RESPONSE ERROR"

                except TimeoutException as error:
                    print(f"Timeout!")
                    result = "RESPONSE ERROR"
            

            # check result
            last_line = result.split('\n')[-1].lower()
            if 'yes' in last_line:
                answer = 'consistent'
            elif 'half' in last_line:
                answer = 'partial_consistent'
            elif 'no' in last_line:
                answer = 'inconsistent'
            else:
                answer = 'ambiguous'

        print('!!!!! Detected:', answer)

        # update the df with the result
        df.at[row_id, 'analysis_time'] = answer
        
    # save the updated df
    df.to_csv(args.file_path.replace(".csv", "_verified.csv"), index=False)
    print('Updated CSV file saved successfully.')


if __name__ == "__main__":

    # read csv file
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", "-f", type=str, default=None, help='direct result file path')
    parser.add_argument("--calc", action='store_true', help='calculate the verified results')
    args = parser.parse_args()

    # file_path = '/home/soyeon/temp_eval/results/dyknow/gpt4_basic_result.csv'

    df = pd.read_csv(args.file_path)
    
    # verify with LLM
    if args.calc:
        calc_results(args, df)
    else:
        verify_with_llm(args, df)
