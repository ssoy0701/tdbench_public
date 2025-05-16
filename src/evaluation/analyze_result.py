'''
Given a QA log file, this script will analyze the result and save it as a txt file.
'''



import argparse
import random
import ast
import os
import pandas as pd
import json
import unidecode
import re


import utils.utils_llm as lutils
import utils.utils_dataset as dutils

NUM_TO_MONTH = {1: 'january', 2: 'february', 3: 'march', 4: 'april', 5: 'may', 6: 'june', 7: 'july', 8: 'august', 9: 'september', 10: 'october', 11: 'november', 12: 'december'}

def parse_args():

    '''
    Config
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices = dutils.DATASET_LIST, default='olympic')
    parser.add_argument("--model", type=str, choices = lutils.MODEL_LIST)
    parser.add_argument('--tasktype', "-t", type=str, choices=['join', 'basic', 'temp_align_carpe_ours', 'temp_align_carpe', 'temp_align_dyknow', 'temp_align', 'validate', 'gen', 'disc'], \
                        default= 'join')
    # parser.add_argument('--rag', type=str, choices = ['rag_gold', 'rag_question'], default=None)
    parser.add_argument("--random_seed", "-s", type=int, default=1116)
    parser.add_argument("--recalc", '-r', action='store_true', help="Recalculate the result")
    parser.add_argument("--file", "-f", type=str, default=None, help='direct result file path')
    parser.add_argument("--overwrite", action='store_true', help="Overwrite the result file")
    parser.add_argument("--folder", "-fd", type=str, default=None, help='folder containing result files')
    parser.add_argument("--index", "-i", type=int, default=0, help='start index for analysis')
    
    args = parser.parse_args()

    return args



def get_const_result_by_op(op, start_hit, end_hit):
    if op in ['after', 'start']:
        return 'consistent' if start_hit else 'inconsistent'
    
    elif op in ['before', 'finish']:
        return 'consistent' if end_hit else 'inconsistent'

    elif op in ['equal']:
        return 'consistent' if start_hit and end_hit else 'inconsistent'
    
    else:
        return 'consistent' if start_hit or end_hit else 'inconsistent'

# heuristics for entity resolution errors
def get_answer_cand(answer):
    cand = []

    answer = answer.lower().strip()
    cand.append(answer) # ideal case
    cand.append(unidecode.unidecode(answer)) # for unicode errors

    if ',' in answer:
        split_answer = answer.split(',')
        for splitted in split_answer:
            cand.append(splitted.strip())

    if answer == 'kostas stephanopoulos':
        cand.append('konstantinos stephanopoulos')
    if answer == 'shōwa':
        cand.append('hirohito')

    split_answer = answer.split(' ')
    if len(split_answer) == 2:
        # cand.append(split_answer[0])
        # cand.append(split_answer[1])
        pass

    elif len(split_answer) == 3: # probably names with middle name
        cand.append(split_answer[0] + ' ' + split_answer[2])
        cand.append(split_answer[0] + ' ' + split_answer[1])
        cand.append(split_answer[0])

    if answer == 'ilegal':
        cand.append('illegal')

    return cand

# heuristics for entity resolution errors
def get_answer_cand_name(answer):
    cand = []

    answer = answer.lower().strip()
    cand.append(answer) # ideal case
    cand.append(unidecode.unidecode(answer)) # for unicode errors

    split_answer = answer.split(' ')
    if len(split_answer) == 3: # probably names with middle name
        cand.append(split_answer[0] + ' ' + split_answer[2])
        cand.append(split_answer[0] + ' ' + split_answer[1])
        # cand.append(split_answer[0])

    return cand
        

def get_gold_date(gold_date):
    if pd.isna(gold_date):
        return 'now', 'now'
    gold_year_start = str(pd.to_datetime(gold_date).year)
    gold_month_start = NUM_TO_MONTH[pd.to_datetime(gold_date).month]

    return gold_year_start, gold_month_start


def analyze_result_join(task, tasktype, model_log_path, overwrite = False, index = 0):

    # read original df
    dataset_df = dutils.get_dataset(task).df

    # helper func
    def get_city_name(row):
        start = row['correct_rows'][0]['Start']
        city = dataset_df[dataset_df['Start'] == start]['City'].values[0]
        return city

    # read qa result
    qa_result = []
    with open(model_log_path, 'r') as file:
        for line in file:
            # Parse each line as a JSON object
            qa_result.append(json.loads(line))


    # iterate through the qa results
    df = pd.read_json(model_log_path, lines=True)
    for row_id, row in enumerate(qa_result):

        if index > row_id:
            continue


        # parse model answer 
        model_answer = str(row['model_response']).strip().lower()

        # preprocess gold answer 
        answer_col = row['op']
        gold_answer_single = row['correct_rows'][0][answer_col].lower()
        gold_answer = get_answer_cand(gold_answer_single)


        # preprocess incorrect answer
        inc_name_list = dataset_df[dataset_df[answer_col].str.lower() != gold_answer_single][answer_col].unique().tolist()
   

        # initialize all results
        result_middle = 'init'
        result_time = 'init'

        # analyze the result
        if 'unsure' in model_answer:
            result_answer = 'unsure'
            result_middle = 'none'
            result_time = 'none'

        elif any(no_resp in model_answer for no_resp in ['no answer', 'no one', 'none']):
            result_answer = 'incorrect' # because there is answer
            result_middle = 'none'
            result_time = 'none'

        # if there are any incorrect names, then it is incorrect
        elif any(inc_name in model_answer for inc_name in inc_name_list):
            result_answer = 'incorrect'

        # if there is only a correct answer
        elif any(gold in model_answer for gold in gold_answer):
            result_answer = 'correct'

        else: # there is no gold answer nor incorrect answers
            result_answer = 'incorrect'
        
        # now lets check middle answer
        # (i.e., country for both 2-hop and 3-hop)
        if result_middle == 'init': # no unsure, no "No answer"
            gold_middle = row['correct_rows'][0]['Country'].lower().strip()
            inc_middle = dataset_df[dataset_df['Country'].str.lower() != gold_middle]['Country'].unique().tolist()

            # analyze middle
            if gold_middle in model_answer:
                result_middle = 'correct'
            elif any(inc in model_answer for inc in inc_middle):
                result_middle = 'incorrect'
            else:            
                result_middle = 'ambiguous'
       

        # now check time
        if result_time == 'init':
            
            gold_date_start = row['correct_rows'][0]['Start']
            gold_date_end = row['correct_rows'][0]['End']

            gold_year_start, gold_month_start = get_gold_date(gold_date_start)
            gold_year_end, gold_month_end = get_gold_date(gold_date_end)

            # first check the year info
            year_hit = gold_year_start in model_answer
            month_hit = gold_month_start in model_answer or gold_month_end in model_answer

            if year_hit:
                #check if there is month info
                if any(mon in model_answer for mon in NUM_TO_MONTH.values()):
                    if month_hit:
                        result_time = 'consistent'
                    else:
                        result_time = 'inconsistent'
                else:
                    result_time = 'ambiguous'
                # result_time = 'consistent'
            else:
                result_time = 'ambiguous'

        else: 
            year_hit = 'none'


        if row['qtype'] == 'join_3hop': # we need additional city info
            gold_middle_city = get_city_name(row)
            row['correct_rows'][0]['City'] = gold_middle_city
            inc_middle_city = dataset_df[dataset_df['City'].str.lower() != gold_middle_city]['City'].unique().tolist()

            if gold_middle_city.lower() in model_answer:
                result_middle_3hop = 'correct'
            elif any(inc.lower() in model_answer for inc in inc_middle_city):
                result_middle_3hop = 'incorrect'
            else:
                result_middle_3hop = 'ambiguous'

        else:
            result_middle_3hop = 'none'



        df.loc[row_id, 'analysis_answer'] = result_answer
        df.loc[row_id, 'analysis_middle'] = result_middle 
        df.loc[row_id, 'analysis_middle_3hop'] = result_middle_3hop
        
        df.loc[row_id, 'analysis_time'] = result_time
        df.loc[row_id, 'analysis_time_3hop'] = year_hit




    final_dict = {}

    for hops in ['2hop', '3hop']:

        curr_df = df[df['qtype'] == f'join_{hops}']

        final_dict[hops] = {
            
            'answer':{
                'total': len(curr_df),
                'correct': len(curr_df[curr_df['analysis_answer'] == 'correct']),
                'incorrect': len(curr_df[curr_df['analysis_answer'] == 'incorrect']), 
                'ambiguous': len(curr_df[curr_df['analysis_answer'] == 'ambiguous']),
                'unsure': len(curr_df[curr_df['analysis_answer'] == 'unsure']),
                # 'percentage': {r: round(result_dict_answer[r]/num_questions * 100, 1) for r in result_dict_answer.keys()},
            }, 

            'middle_ans': {
                'total': len(curr_df),
                'correct': len(curr_df[curr_df['analysis_middle'] == 'correct']),
                'incorrect': len(curr_df[curr_df['analysis_middle'] == 'incorrect']),
                'ambiguous': len(curr_df[curr_df['analysis_middle'] == 'ambiguous']),
                'none': len(curr_df[curr_df['analysis_middle'] == 'none']),
                # 'percentage': {r: round(result_dict_answer[r]/num_questions * 100, 1) for r in result_dict_middle.keys()},
            },

            'middle_time': {
                'total': len(curr_df),
                'consistent': len(curr_df[curr_df['analysis_time'] == 'consistent']),
                'inconsistent': len(curr_df[curr_df['analysis_time'] == 'inconsistent']), 
                'ambiguous': len(curr_df[curr_df['analysis_time'] == 'ambiguous']),
                'none': len(curr_df[curr_df['analysis_time'] == 'none']),
                # 'percentage': {r: round(result_dict_time[r]/num_questions * 100, 1) for r in result_dict_time.keys()}
                # 'percentage': {'consistent': round(len(curr_df[curr_df['analysis_time'] == 'consistent']) / \
                                                    # (len(curr_df[curr_df['analysis_time'] == 'consistent']) + len(df[df['analysis_time'] == 'inconsistent'])) * 100, 1),},
            },

            'middle_ans_3hop': {  
                'total': len(curr_df),
                'correct': len(curr_df[curr_df['analysis_middle_3hop'] == 'correct']),
                'incorrect': len(curr_df[curr_df['analysis_middle_3hop'] == 'incorrect']),
                'ambiguous': len(curr_df[curr_df['analysis_middle_3hop'] == 'ambiguous']),
                'none': len(curr_df[curr_df['analysis_middle_3hop'] == 'none']),
            },
        }

        final_dict[hops]['metrics'] = {

            'final_answer_correct': round(final_dict[hops]['answer']['correct'] / final_dict[hops]['answer']['total'] * 100, 1),
            'final_answer_correct_and_consistent': round(len(curr_df[(curr_df['analysis_answer'] == 'correct') & (curr_df['analysis_middle'] != 'incorrect') & (curr_df['analysis_time']!= 'inconsistent')]) / final_dict[hops]['answer']['total'] * 100, 1),
            'wrong_name_given_correct_country': round(len(curr_df[(curr_df['analysis_answer'] == 'incorrect') & (curr_df['analysis_middle'] == 'correct')]) / final_dict[hops]['answer']['incorrect'] * 100, 1),
            'wrong_name_given_correct_country_and_time': round(len(curr_df[(curr_df['analysis_answer'] == 'incorrect') & (curr_df['analysis_time'] == 'consistent') & (curr_df['analysis_middle'] == 'correct')]) / final_dict[hops]['answer']['incorrect'] * 100, 1),
        }

        if hops == '2hop':
            final_dict[hops]['metrics']['wrong_country_given_correct_city_and_time'] = \
                round(len(curr_df[(curr_df['analysis_middle'] == 'incorrect')]) / final_dict[hops]['middle_ans']['total'] * 100, 1)

        if hops == '3hop': # add hallucination rate with city too
            final_dict[hops]['metrics']['wrong_city_given_edition'] = \
                round(len(curr_df[(curr_df['analysis_middle_3hop'] == 'incorrect')]) / len(curr_df) * 100, 1)
            final_dict[hops]['metrics']['wrong_city_given_edition_and_time'] = \
                round(len(curr_df[(curr_df['analysis_middle_3hop'] == 'incorrect') & (curr_df['analysis_time_3hop'] == True)]) / len(curr_df) * 100, 1)
                
            if final_dict[hops]['middle_ans']['incorrect'] > 0:
                final_dict[hops]['metrics']['wrong_country_given_correct_city'] = \
                    round(len(curr_df[(curr_df['analysis_middle'] == 'incorrect') & (curr_df['analysis_middle_3hop'] == 'correct')]) / (final_dict[hops]['middle_ans']['incorrect']) * 100, 1)
                final_dict[hops]['metrics']['wrong_country_given_correct_city_and_time'] = \
                    round(len(curr_df[(curr_df['analysis_middle'] == 'incorrect') & (curr_df['analysis_time'] == 'consistent') & (curr_df['analysis_middle_3hop'] == 'correct')]) / (final_dict[hops]['middle_ans']['incorrect']) * 100, 1)
            
            else: 
                final_dict[hops]['metrics']['wrong_country_given_correct_city'] = 0.0
                final_dict[hops]['metrics']['wrong_country_given_correct_city_and_time'] = 0.0

    print(final_dict)

    # dump with json
    result_file_path = model_log_path.replace('.jsonl', '_result.json')
    with open(result_file_path, 'w') as f:
        json.dump(final_dict, f, indent=4)

    result_csv_path = model_log_path.replace('.jsonl', '_result.csv')
    


    
    df.to_csv(result_csv_path, index = False)

    print(f"TXT result saved at {result_file_path}.")
    print(f"CSV result saved at {result_csv_path}.\n")
    
    # return final_dict, df
    return
    

def analyze_result_basic(tasktype, model_log_path, overwrite = False, index = 0):


    qa_result = []
    with open(model_log_path, 'r') as file:
        for line in file:
            # Parse each line as a JSON object
            qa_result.append(json.loads(line))

    # get the number of questions
    df = pd.read_json(model_log_path, lines=True)
    num_questions = len(df)
    results_answer = ['correct', 'partial_correct', 'incorrect', 'ambiguous', 'unsure']
    results_time = ['consistent', 'inconsistent', 'none']

    result_dict_answer = {r: 0 for r in results_answer}
    result_dict_time = {r: 0 for r in results_time}

    result_dict_all = {}
    for qtype in ['unique', 'none', 'multiple']:
        result_dict_all[qtype] = {'answer': result_dict_answer.copy(), \
                                    'time': result_dict_time.copy()}
        
    # for multiple answers
    result_dict_all['multiple']['answer']['correct_among_hits'] = []
    result_dict_all['multiple']['time']['consistent_among_hits'] = []
    

    # iterate through the qa results
    for row_id, row in enumerate(qa_result):

        if index > row_id:
            continue

        # parse model answer 
        model_answer = str(row['model_response']).strip().lower()


        # first check answer
        if 'unique' in row['qtype']:
            result_dict_answer = result_dict_all['unique']['answer']
            result_dict_time = result_dict_all['unique']['time']

            # preprocess gold answer 
            gold_answer = row['correct_rows'][0]['Name']
            gold_answer = get_answer_cand(gold_answer)

            # preprocess incorrect answer
            inc_name_list = []
            for inc in row['incorrect_rows']:
                inc_name_list.extend(get_answer_cand(inc['Name']))   


            # analyze the result
            if 'unsure' in model_answer:
                result_answer = 'unsure'

            elif any(no_resp in model_answer for no_resp in ['no answer', 'no one', 'none']):
                result_answer = 'incorrect' # because there is answer

            # if there are any incorrect names, then it is incorrect
            elif (row['op'] != 'during' or ('between' not in row['question'] and row['op'] == 'during')) and any(inc_name in model_answer for inc_name in inc_name_list):
                result_answer = 'incorrect'

            # if there is only a correct answer
            elif any(gold in model_answer for gold in gold_answer):
                result_answer = 'correct'

            else: # there is no gold answer nor incorrect answers
                result_answer = 'incorrect'


        elif 'none' in row['qtype']:
            # get the result dict
            result_dict_answer = result_dict_all['none']['answer']
            result_dict_time = result_dict_all['none']['time']

            # preprocess gold answer
            gold_answer = ['no answer', 'no one', 'none']

            # analyze the result
            if 'unsure' in model_answer:
                result_answer = 'unsure'

            elif any(gold in model_answer for gold in gold_answer):
                result_answer = 'correct'
            
            else:
                result_answer = 'incorrect'


        elif 'multiple' in row['qtype']:
            # get the result dict
            result_dict_answer = result_dict_all['multiple']['answer']
            result_dict_time = result_dict_all['multiple']['time']


            # analyze the result
            if 'unsure' in model_answer:
                result_answer = 'unsure'

            elif any(no_resp in model_answer for no_resp in ['no answer', 'no one', 'none']):
                result_answer = 'incorrect' # because there is answer

            # if there are any incorrect names or correct names
            else: 
                result_answer_list = []

                # preprocess incorrect answer
                inc_name_list = []
                for inc in row['incorrect_rows']:
                    inc_name_list.append(get_answer_cand(inc['Name'])) 

                # check any incorrect names
                for inc_name in inc_name_list:
                    if (row['op'] != 'during' or ('between' not in row['question'] and row['op'] == 'during')) and any(inc_n in model_answer for inc_n in inc_name):
                        result_answer_list.append('incorrect')
                
                # preprocess correct answer
                gold_name_list = []
                for correct_row in row['correct_rows']:
                    gold_name_list.append(get_answer_cand(correct_row['Name']))

                # check any correct names
                for gold_name in gold_name_list:
                    if any(gold_n in model_answer for gold_n in gold_name):
                        result_answer_list.append('correct')
                
                # check results
                if row['op'] != 'during' and 'incorrect' in result_answer_list: 
                    result_answer = 'incorrect'

                elif 'correct' in result_answer_list:
                    if len(row['correct_rows']) >=5:
                        # check if count of correct is 5
                        if len([r for r in result_answer_list if r == 'correct']) >= 5:
                            result_answer = 'correct'
                        else:
                            result_answer = 'partial_correct'

                    else: 
                        # check if count of correct is 1
                        if len([r for r in result_answer_list if r == 'correct']) == len(row['correct_rows']):
                            result_answer = 'correct'
                        else:
                            result_answer = 'partial_correct'

                    # result_answer = 'correct'
                else: # none of cor/incor answers are included
                    result_answer = 'incorrect'

                if len(result_answer_list) > 0:
                    correct_hit_cnt = len([r for r in result_answer_list if r == 'correct'])
                    correct_among_hits = correct_hit_cnt/len(result_answer_list)
                    result_dict_answer['correct_among_hits'].append(correct_among_hits)

        else:
            raise ValueError(f"Qtype {row['qtype']} is not implemented yet.")
                                
        
        # now check time consistency for mentioned names
        result_time = 'none' # default value
        row['correct_rows'].extend(row['incorrect_rows'])  # this is all info    
        

        # for non-multiple answers, check with total string
        # for multiple answers, check with each line
        if not 'multiple' in row['qtype']:

            for inc in row['correct_rows']:
                inc_name_list = get_answer_cand(inc['Name'])

                # we got the name
                if any(inc_name in model_answer for inc_name in inc_name_list):

                    gold_date_start = inc['Start']
                    gold_date_end = inc['End']

                    gold_year_start, gold_month_start = get_gold_date(gold_date_start)
                    gold_year_end, gold_month_end = get_gold_date(gold_date_end)

                    # first check the year info
                    start_hit = gold_year_start in model_answer
                    end_hit = gold_year_end in model_answer

                    if re.search(r'\d{4}', model_answer):    # if contains year info                
                        result_time = get_const_result_by_op(row['op'], start_hit, end_hit)
                        if result_time == 'inconsistent':
                            break # we don't need to check further


                    # now check if there is month info
                    if any(mon in model_answer for mon in NUM_TO_MONTH.values()):
                        start_hit = gold_year_start in model_answer and gold_month_start in model_answer
                        end_hit = gold_year_end in model_answer and gold_month_end in model_answer

                        result_time = get_const_result_by_op(row['op'], start_hit, end_hit)
                        if result_time == 'inconsistent':
                            break

        else:
            
            result_time_list = []

            model_answer_list = model_answer.split('\n')
            for model_answer in model_answer_list:
                for inc in row['correct_rows']:
                    inc_name_list = get_answer_cand(inc['Name'])

                    # we got the name
                    if any(inc_name in model_answer for inc_name in inc_name_list):

                        gold_date_start = inc['Start']
                        gold_date_end = inc['End']

                        gold_year_start, gold_month_start = get_gold_date(gold_date_start)
                        gold_year_end, gold_month_end = get_gold_date(gold_date_end)

                        # first check the year info
                        start_hit = gold_year_start in model_answer
                        end_hit = gold_year_end in model_answer

                        if re.search(r'\d{4}', model_answer):    # if contains year info
                            result_time = get_const_result_by_op(row['op'], start_hit, end_hit)
                            # if result_time == 'inconsistent':
                            #     break
                            # else:
                            #     cons_hit += 1

                        # now check if there is month info
                        if any(mon in model_answer for mon in NUM_TO_MONTH.values()):
                            start_hit = gold_year_start in model_answer and gold_month_start in model_answer
                            end_hit = gold_year_end in model_answer and gold_month_end in model_answer

                            result_time = get_const_result_by_op(row['op'], start_hit, end_hit)
                            # if result_time == 'inconsistent':
                            #     break
                            # else:
                            #     cons_hit += 1

                        result_time_list.append(result_time)

                # if 'inconsistent' in result_time_list:
                #     result_time = 'inconsistent'
                #     break
                # else:
                #     continue

            if 'inconsistent' in result_time_list:
                result_time = 'inconsistent'
            elif 'consistent' in result_time_list:
                result_time = 'consistent'
            else:
                result_time = 'none'

            incons_cnt = len([r for r in result_time_list if r == 'inconsistent'])
            cons_cnt = len([r for r in result_time_list if r == 'consistent'])

            if (incons_cnt + cons_cnt) > 0:
                consistent_among_hits = cons_cnt/(cons_cnt + incons_cnt) 
                result_dict_time['consistent_among_hits'].append(consistent_among_hits)


        # update result counts and df
        result_dict_answer[result_answer] += 1
        result_dict_time[result_time] += 1

        df.loc[row_id, 'analysis_answer'] = result_answer
        df.loc[row_id, 'analysis_time'] = result_time

        
    final_dict = {}
    # add metadata 
    for qtype in ['unique', 'none', 'multiple']:


        result_dict_answer = result_dict_all[qtype]['answer']
        result_dict_time = result_dict_all[qtype]['time']

        qtype_df = df[df['qtype'] == tasktype + '_' + qtype]
        num_questions = len(qtype_df)
        if num_questions == 0:
            continue

        correct_among_hits = result_dict_answer.pop('correct_among_hits', [])
        consistent_among_hits = result_dict_time.pop('consistent_among_hits', [])

        final_dict[qtype]= {
            
            'answer':{
                'total': len(qtype_df),
                'correct': len(qtype_df[qtype_df['analysis_answer'] == 'correct']),
                'incorrect': len(qtype_df[qtype_df['analysis_answer'] == 'incorrect']), 
                'ambiguous': len(qtype_df[qtype_df['analysis_answer'] == 'ambiguous']),
                'unsure': len(qtype_df[qtype_df['analysis_answer'] == 'unsure']),
                'percentage': {r: round(result_dict_answer[r]/num_questions * 100, 1) for r in result_dict_answer.keys()},
            }, 
            'time': {
                'total': len(qtype_df),
                'consistent': len(qtype_df[qtype_df['analysis_time'] == 'consistent']),
                'inconsistent': len(qtype_df[qtype_df['analysis_time'] == 'inconsistent']), 
                'none': len(qtype_df[qtype_df['analysis_time'] == 'none']),
                # 'percentage': {r: round(result_dict_time[r]/num_questions * 100, 1) for r in result_dict_time.keys()}
                'percentage': {'consistent': round(len(qtype_df[qtype_df['analysis_time'] == 'consistent']) / \
                                                   (len(qtype_df[qtype_df['analysis_time'] == 'consistent']) + len(qtype_df[qtype_df['analysis_time'] == 'inconsistent'])) * 100, 1),},
            },
            'both': {
                'consistent_and_correct': round(len(qtype_df[(qtype_df['analysis_answer'] == 'correct') & \
                                                             (qtype_df['analysis_time'] != 'inconsistent')]) / len(qtype_df) * 100, 1)
                                                },
        }

        if qtype == 'multiple':
            correct_among_hits_mean = round(sum(correct_among_hits) / num_questions * 100, 1)
            consistent_among_hits_mean = round(sum(consistent_among_hits) / len(qtype_df[qtype_df['analysis_time'] != 'none']) * 100, 1)

            final_dict[qtype]['answer']['correct_among_hits'] = correct_among_hits_mean
            final_dict[qtype]['time']['consistent_among_hits'] = consistent_among_hits_mean

    basic_op_lists = ['after', 'before', 'equal', 'contain', 'finish', 'start', 'meet', 'met-by', 'overlap', 'during', 'finished-by', 'started-by', 'overlapped-by']
    
    op_dict = {op: 0 for op in basic_op_lists}
    for op in basic_op_lists:
        op_df = df[df['op'].str.contains(op)]

        total_op_num = len(op_df)
        correct_percentage = round(len(op_df[op_df['analysis_answer'] == 'correct']) / total_op_num * 100, 1)
        consistent_percentage = round(len(op_df[op_df['analysis_time'] == 'consistent']) / \
                                                    (len(op_df[op_df['analysis_time'] == 'consistent']) + len(op_df[op_df['analysis_time'] == 'inconsistent'])) * 100, 1)
        correct_and_consistent_percentage = round(len(op_df[(op_df['analysis_answer'] == 'correct') & \
                                                            (op_df['analysis_time'] != 'inconsistent')]) / total_op_num * 100, 1)

        op_result = {'total': total_op_num, \
                     'correct': correct_percentage,\
                    'consistent': consistent_percentage,\
                        'correct_and_consistent': correct_and_consistent_percentage
                    }

        op_dict[op] = op_result

    final_dict['op'] = op_dict

    final_dict['total'] = {
        'total': len(df),
        'correct_and_consistent': round(len(df[(df['analysis_answer'] == 'correct') & \
                                                             (df['analysis_time'] != 'inconsistent')]) / len(df) * 100, 1),
    }

    print(result_dict_all)

    # dump with json
    result_file_path = model_log_path.replace('.jsonl', '_result.json')
    with open(result_file_path, 'w') as f:
        json.dump(final_dict, f, indent=4)

    result_csv_path = model_log_path.replace('.jsonl', '_result.csv')
    df.to_csv(result_csv_path, index = False)

    print(f"TXT result saved at {result_file_path}.")
    print(f"CSV result saved at {result_csv_path}.\n")
    
    # return final_dict, df
    return


def analyze_result_tempalign(tasktype, model_log_path, granularity, overwrite = False):

    # read log file
    df = pd.read_json(model_log_path, lines=True)

    


    # get the number of questions
    num_questions = len(df)
    results = ['total', 'correct', 'partial_correct', 'outdated', 'partial_outdated', 'incorrect', 'ambiguous', 'unsure']
    result_dict = {r: 0 for r in results}
    result_dict['total'] = num_questions


    

    # iterate through the qa results
    for i in range(num_questions):

        # parse model answer 
        model_answer = str(df['model_response'][i]).strip().lower()
        # print(model_answer)
        
        # analyze the result
        if tasktype == 'temp_align_dyknow':

            # preprocess gold answer
            gold_answer = get_answer_cand(df['gold_answer'][i])

            outdated_list = []
            for outdated in df['outdated'][i]:
                outdated_list.extend(get_answer_cand(outdated))

            # analyze the result
            if 'unsure' in model_answer:
                result = 'unsure'
            elif any(gold in model_answer for gold in gold_answer):
                result = 'correct'
            elif any(outdated in model_answer for outdated in outdated_list):
                result = 'outdated'
            else:
                result = 'incorrect'


        elif tasktype == 'temp_align':


            # get gold answer
            answer_dict = df['correct_rows'][i][0]
            answer_col_key =  [k for k in answer_dict.keys() if k not in ['Start', 'End']][0]
            gold_answer, gold_date = answer_dict[answer_col_key], answer_dict['Start'] # we analyze start
            
            # preprocess gold answer and time
            gold_answer = get_answer_cand(gold_answer)
            gold_year = str(pd.to_datetime(gold_date).year)

            if granularity == 'month':
                gold_month = NUM_TO_MONTH[pd.to_datetime(gold_date).month] # check depend on granularity

            # print(gold_answer, gold_year)

            # make outdated answer list
            outdated_list = []
            if len(df['incorrect_rows'][i]) >= 0:
                for outdated in df['incorrect_rows'][i]:
                    if granularity == 'month':
                        outdated_list.append([get_answer_cand(outdated[answer_col_key]), [str(pd.to_datetime(outdated['Start']).year), NUM_TO_MONTH[pd.to_datetime(outdated['Start']).month]]])
                    
                    else:
                        outdated_list.append([get_answer_cand(outdated[answer_col_key]), [str(pd.to_datetime(outdated['Start']).year)]])



            # analyze the result
            result = 'incorrect'


            if 'unsure' in model_answer:
                result = 'unsure'

            elif any(gold in model_answer for gold in gold_answer):

                if gold_year in model_answer:
                    result = 'correct'

                    # if i < 78*3: # presidents
                    if granularity == 'month':
                        if any(mon in model_answer for mon in NUM_TO_MONTH.values()) and gold_month not in model_answer:
                            result = 'partial_correct'

                else:
                    result = 'partial_correct'

            else:
                if len(outdated_list) >= 0:
                    for outdated in outdated_list:
                        if any(out in model_answer for out in outdated[0]):

                            if granularity == 'month':
                                if any(mon in model_answer for mon in NUM_TO_MONTH.values()) and outdated[1][1] not in model_answer:
                                    result = 'partial_outdated'
                                else:
                                    result = 'outdated'
                                break

                            else:
                                if outdated[1][0] in model_answer:
                                    result = 'outdated'
                                else:
                                    result = 'partial_outdated'
                                break

            # print(result) 

        elif tasktype == 'temp_align_carpe' or tasktype == 'temp_align_carpe_ours':
            
            # preprocess gold answer
            gold_answer = df['gold_answer'][i].strip().lower() # 'option a' or 'option b'

            if tasktype == 'temp_align_carpe':
                options = ['option a', 'option b']
                options.remove(gold_answer)
                outdated = options[0]

                # analyze the result
                if 'unsure' in model_answer:
                    result = 'unsure'
                elif gold_answer in model_answer:
                    result = 'correct'
                elif outdated in model_answer:
                    result = 'outdated'
                else:
                    result = 'incorrect'

            else:
                if gold_answer == 'option a':
                    part_correct = 'option b'
                    outdated = 'option c'
                    part_outdated = 'option d'
                else: # gold answer is option c
                    part_correct = 'option d'
                    outdated = 'option a'
                    part_outdated = 'option b'

                # analyze the result
                if 'unsure' in model_answer:
                    result = 'unsure'
                elif gold_answer in model_answer:
                    result = 'correct'
                elif part_correct in model_answer:
                    result = 'partial_correct'
                elif outdated in model_answer:
                    result = 'outdated'
                elif part_outdated in model_answer:
                    result = 'partial_outdated'
                else:
                    result = 'incorrect'

        else:
            raise NotImplementedError(f"Task type {tasktype} is not implemented yet.")


        # update result counts and df
        result_dict[result] += 1
        df.loc[i, 'analysis'] = result
        
    

    # add metadata 
    result_dict['percentage'] = {r: round(result_dict[r]/num_questions * 100, 2) for r in result_dict.keys()}


    # save result and df
    result_file_path = model_log_path.replace('.jsonl', '_result.txt')
    with open(result_file_path, 'w' if overwrite else'a') as f:
        for k, v in result_dict.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")

    result_csv_path = model_log_path.replace('.jsonl', '_result.csv')
    df.to_csv(result_csv_path, index = False)

    print(f"TXT result saved at {result_file_path}.")
    print(f"CSV result saved at {result_csv_path}.\n")

    return





def main():
    '''
    Main function to analyze QA result
    '''

    # config
    print("================= Config ==========================")
    args = parse_args()
    task = args.task.strip()
    model = args.model
    tasktype = args.tasktype
    # rag = args.rag
    random_seed = args.random_seed
    overwrite = args.overwrite
    index = args.index

    for k, v in args.__dict__.items():
        print(f"{k}: {v}")
    print("===================================================")
    
    # save_dir = '/home/soyeon/temp_eval/results'

    # set random seed
    random.seed(random_seed)

 
    # prepare log file
    print(f"Reading output jsonl file...\n")
    if args.file: # analyze single file
        model_log_path = args.file 

    elif args.folder: # analyze all files in the folder
        model_log_path = args.folder 


    if not os.path.exists(model_log_path):
        raise ValueError(f"QA Log path does not exist: {model_log_path}")




    # recalculation mode
    if args.recalc:
        print("Recalculating the result...\n")

        # read csv with manual correction
        if not os.path.exists(args.file):
            raise ValueError(f"In recalc mode, csv file is required. CSV file does not exist: {args.file}")
        df = pd.read_csv(args.file)

        # recalculate the result
        result_dict = {}
        for result in df['analysis'].unique():
            result_dict[result] = len(df[df['analysis'] == result])
 
        # add total and percentage
        num_questions = len(df)
        result_dict['percentage'] = {r: round(result_dict[r]/num_questions * 100, 2) for r in result_dict.keys()}
        result_dict2 = result_dict.copy()
        result_dict2.pop('percentage')
        if 'partial_correct' in result_dict.keys():
            result_dict2['correct'] = result_dict['correct'] + result_dict['partial_correct']
            result_dict2.pop('partial_correct')
        
        if 'partial_outdated' in result_dict.keys():
            result_dict2['outdated'] = result_dict['outdated'] + result_dict['partial_outdated']
            result_dict2.pop('partial_outdated')
        result_dict['percentage_coarse'] = {r: round(result_dict2[r]/num_questions * 100, 2) for r in result_dict2.keys()}

        
        # save recalculated result
        result_file_path =args.file.replace('.csv', '.txt')
        with open(result_file_path, 'a') as f:
            f.write("====== Recalculated ======\n")
            for k, v in result_dict.items():
                f.write(f"{k}: {v}\n")
        print("Done recalculating the result.")
        return

    

    # analyze result and save
    print("Analyzing result...\n")
    
    if 'temp_align' in tasktype:

        # read original df
        granularity = dutils.get_dataset(task).granularity[0]
        
        if args.folder:
            log_path_list = [lpath for lpath in os.listdir(model_log_path) if lpath.endswith('.jsonl')]
            for log_path in log_path_list:
                curr_log_path = os.path.join(model_log_path, log_path)
                print("Analyzing: ", curr_log_path)
                analyze_result_tempalign(tasktype, curr_log_path, granularity, overwrite = overwrite)

        else:   
            analyze_result_tempalign(tasktype, model_log_path, granularity, overwrite = overwrite)

    elif tasktype == 'basic':

        analyze_result_basic(tasktype, model_log_path, overwrite, index)
    
    elif tasktype == 'join':
        
        # we need task for join
        analyze_result_join(task, tasktype, model_log_path, overwrite, index)
    
    else:
        raise NotImplementedError(f"Task type {tasktype} is not implemented yet.")



if __name__  =="__main__":
    main()