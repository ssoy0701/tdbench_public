'''
Given SQL queries, this script will convert them to natural language questions & according answers to json file.
'''

import json
import argparse
import random
import os
import re
import time
import pandas as pd

import utils.utils_llm as lutils
import utils.utils_dataset as dutils


# argparser
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, choices = dutils.DATASET_LIST, default='dyknow')
parser.add_argument("--file_name", type=str, default=None)
parser.add_argument("--qtype", '-q', type=str, choices = ['now', 'basic', 'join'], default='basic')
parser.add_argument("--sample", type=int, default=150, help="Number of samples to generate from each category (None/Unique/Multiple).")
parser.add_argument("--random_seed", type=int, default='1116')
parser.add_argument("--print", action="store_true", help="Print the generated questions.")
parser.add_argument("--balance_op", action="store_true", help="Balance the number of operations.")
parser.add_argument("--do_sample", action="store_true", help="Sample questions from each category.")
args = parser.parse_args()

RESULT_DIR = '/home/soyeon/temp_eval/qa_pairs'


# config
task = args.task
qtype = args.qtype
sample_num_config = args.sample
do_sample = args.do_sample
print_qa = args.print
file_name = args.file_name if args.file_name else f'{qtype}_sql.json'
dpath = os.path.join(RESULT_DIR, task, file_name)
spath = dpath.replace('.json', '_gpt4o.jsonl')
random.seed(args.random_seed)



# read data
with open(dpath, "r", encoding="utf-8") as f:
    target_json = json.load(f)
    print("Total number of queries: ", len(target_json))


# system message for database description
if task == 'dyknow':
    # system_msg = "The following SQL query is about a relational database Leader(country, role, name, start, end) where start, end are date information."
    # system_msg = "The following SQL query is about a relational database Organization(organization_name, organization_type, role, person_name, start, end) where start, end are date information."
    system_msg = "The following SQL query is about a relational database Sport(name, team, start, end) where start, end are date information."
elif task == 'legal':
    system_msg = "The following SQL query is about a relational database SameSexLaw(country, law_type, legality, start, end) where start, end are date information and legality is either 'legal' or 'illegal'."
elif task == 'environ':
    system_msg = "The following SQL query is about a relational database CarbonMechanism(jurisdiction, type, name, status, start, end) where start, end are date information and status is either 'yes (exist)' or 'no (does not exist)'."
elif task == 'culture':
    system_msg = "The following SQL query is about a relational database CulturalHeritage(member_state, heritage_element, status, start, end) where start, end are date information and status is either 'proclaimed' or 'inscribed'."
elif task == 'movie':
    system_msg = "The following SQL query is about a relational database Movie(title, director, cast, release_year, start, end) where start (release date), end are date information. Here, focus on ORDER BY clause."
# elif task == 'carbon':
#     system_msg = "The following SQL query is about a relational database CarbonMechanism(Jurisdiction, Type, Name, Implemented, Start, End) where start, end are date information, and Implemented denotes whether the mechanism has been implemented or not."
elif task == 'olympic':
    system_msg = "The following SQL query is about a relational database OlympicGames(Game_edition, Game_name, Country, City, Name, Role) where 'Name' and 'Role' are about leaders of the **host country**."
else: 
    raise NotImplementedError
system_msg += "\n\nTranslate the provided SQL query to a natural language question. Do not include any artificial phrases like 'according to the database', 'from the table', or 'based on the query'. Also, do not describe the FROM clause or mention the table name. Focus only on the selected fields and filtering conditions. Generate 3 different questions each starting with 'Q: '. Only return the generated questions."


if args.balance_op:
    # read converted data
    # generated_qa = pd.read_json(spath, lines=True)
    
    generated_qa = pd.read_json(spath, lines=True)
    all_qa = pd.read_json(dpath)
    print("Total number of generated questions: ", len(generated_qa))

    # categorize w.r.t. ops
    ops = generated_qa['op'].unique().tolist()
    print("Operations: ", ops)

    new_idx = len(generated_qa)
    for op in ops:
               
        op_df = generated_qa[generated_qa['op'] == op]
        print(f"!!!!!!!Operation: {op}, Number of questions: {len(op_df)}")

        
        
        if len(op_df) < 40:
            print("Insufficient number of questions. Sampling more...")

            op_all_qa = all_qa[all_qa['op'] == op]
            print("Number of all op questions: ", len(op_all_qa))

            op_all_qa = op_all_qa[~op_all_qa['sql_for_db'].isin(op_df['sql_for_db'])]
            print("Number of available questions: ", len(op_all_qa))

            sample_num = min(40-len(op_df), len(op_all_qa))
            sampled = op_all_qa.sample(sample_num, random_state=args.random_seed)
            sampled = sampled.to_dict(orient='records')

            
            for q in sampled:
                print("processing: ", q['idx'])

                # generate prompt for SQL conversion
                prompt = q['sql_for_db']
            
                if task == 'olympic':
                    prompt_nlq = prompt.replace('country, start, end', '').lower()
                else: 
                    prompt_nlq = prompt.replace(', start, end', '').replace('start', 'start_date').replace('end', 'end_date')

                # generate question
                nlq = lutils.run_gpt4o(prompt_nlq, system_msg, temp=0.3) # 0.5 for dyknow, 0.3 for carbon
                
                questions = re.findall(r"Q: (.*?)(?=\nQ:|\n*$)", nlq, re.DOTALL)

                print(questions)
                q['questions'] = questions
                print('questions: ', len(questions))

                if print_qa:
                    print("SQL: ", prompt)
                    for i, question in enumerate(questions):
                        print(f"Q{i+1}: ", question)
                        time.sleep(2)

                # give new id
                q['idx'] = new_idx  
        
                # write to file
                with open(spath.replace('.jsonl', '_balanced.jsonl'), "a", encoding="utf-8") as f:
                    json.dump(q, f, ensure_ascii=False)
                    f.write('\n') 

                new_idx += 1
            
        else:
            print("Sufficient number of questions.\n")
    

    finished = pd.read_json(spath.replace('.jsonl', '_balanced.jsonl'), lines=True)
    print("Total number of generated questions: ", len(finished))
    # count ops and its number
    ops = finished['op'].unique().tolist()
    for op in ops:
        op_df = finished[finished['op'] == op]
        print(f"Operation: {op}, Number of questions: {len(op_df)}")
    exit()




if qtype == 'basic':
    # categorize w.r.t. the number of answers
    none_of_above = [v for v in target_json if v['qtype'] == 'basic_none']
    unique = [v for v in target_json if v['qtype'] == 'basic_unique']
    multiple = [v for v in target_json if v['qtype'] == 'basic_multiple']
    all_q = {'none_of_above': none_of_above, 'unique': unique, 'multiple': multiple}

    # for qa_type, qa_items in all_q.items():
    #     print('qa_type: ', qa_type)

    #     qa_ops = [v['op'] for v in qa_items]
    #     Counter = dict((x,qa_ops.count(x)) for x in set(qa_ops))
    #     print(Counter)
    # exit()
        
    # print given SQL info
    print("=====================================")
    print("Task: ", task)
    print("qtype: ", qtype)
    print("Total questions: ", len(target_json))
    print("\tNone of above: ", len(none_of_above))
    print("\tUnique: ", len(unique))
    print("\tMultiple: ", len(multiple))
    print("=====================================")


elif qtype == 'join':
    all_q = {'join': target_json}

    # print given SQL info
    print("=====================================")
    print("Task: ", task)
    print("qtype: ", qtype)
    print("Total questions: ", len(target_json))
    print("=====================================")



elif qtype == 'now':
    all_q = {'now': target_json}

    # print given SQL info
    print("=====================================")
    print("Task: ", task)
    print("qtype: ", qtype)
    print("Total questions: ", len(target_json))
    print("=====================================")

    if not do_sample:
        sample_num_config = len(target_json)
        print(f"We will use all {sample_num_config} questions.")
        print("=====================================")
    else:
        print(f"We will sample {sample_num_config} questions.")
        print("=====================================")



new_idx = 0

for q_type, q_items in all_q.items(): 
    
    print(f"Processing Query with {q_type} answers")
    print("=====================================")
 

    # random sample 10 question from each category
    sample_num = min(sample_num_config, len(q_items))
    sampled = random.sample(q_items, sample_num)

    if not qtype == 'now':
        print(f"We will sample {sample_num_config} questions from each category.")
        print("=====================================")


    for q in sampled:
        print("processing: ", q['idx'])

        # generate prompt for SQL conversion
        prompt = q['sql_for_db']
    
        if task == 'olympic':
            prompt_nlq = prompt.replace('country, start, end', '').lower()
        else: 
            prompt_nlq = prompt.replace(', start, end', '').replace('start', 'start_date').replace('end', 'end_date')



        if qtype == 'now':
            if task == 'movie':
                prompt_nlq = prompt.replace('order by', 'ORDER BY').replace(', start, end', '')
            else:
                prompt_nlq = prompt.replace(pd.Timestamp.now().strftime("%Y-%m-%d"), 'now()').replace(', start, end', '')
        
        print(prompt_nlq)
        print(q['correct_rows'])
        
        # generate question
        nlq = lutils.run_gpt4o(prompt_nlq, system_msg, temp=0.3) # 0.5 for dyknow, 0.3 for carbon
        # print(nlq)
        # parse questions
        # questions = re.findall(r"Q: (.*?[.?])", nlq)
        # questions = re.findall(r"Q: (.*?)(?= Q:|$)", nlq)
        questions = re.findall(r"Q: (.*?)(?=\nQ:|\n*$)", nlq, re.DOTALL)

        # print(questions)
        q['questions'] = questions
        print('questions: ', len(questions))

        if print_qa:
            print("SQL: ", prompt)
            for i, question in enumerate(questions):
                print(f"Q{i+1}: ", question)
                time.sleep(2)

        # give new id
        q['idx'] = new_idx  
  
        # write to file
        with open(spath, "a", encoding="utf-8") as f:
            json.dump(q, f, ensure_ascii=False)
            f.write('\n') 

        new_idx += 1
        

