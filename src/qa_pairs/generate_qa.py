'''
This script generates the QA pairs for the given dataset and task.

Return:
    - For disc task: json file with (entity_id, question, answer)
    - For gen task: json file with (entity_id, question, answer)
'''

import pandas as pd
import random
import argparse
import os
import json
from sqlite3 import connect
import itertools


import utils_llm as lutils
import utils_dataset as dutils


def parse_args():
    '''
    Config
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices = ['same_sex', 'ai_models', 'carbon', 'fashion', 'heritage', 'dyknow_pm'], default='fashion')
    parser.add_argument('--tasktype', "-t", type=str, choices=['validate', 'gen', 'disc', 'order'], default= 'disc')
    parser.add_argument("--index", type=int, default = 0)
    parser.add_argument("--random_seed", "-s", type=int, default=1116)
    parser.add_argument("--all", "-a", action='store_true')
    parser.add_argument("--sample", type=int, default=500)

    args = parser.parse_args()
    
    return args



def main():
    '''
    Main function to run QA task
    '''

    # config
    print("================= Config ==========================")
    args = parse_args()
    task = args.task.strip()
    tasktype = args.tasktype
    idx = args.index
    random_seed = args.random_seed
    all = args.all
    sample = args.sample

    for k, v in args.__dict__.items():
        print(f"{k}: {v}")
    print("===================================================")
    data_dir = './dataset'
    save_dir = os.path.join('./qa_pairs', task)
    os.makedirs(save_dir, exist_ok=True)

    # remove existing file
    save_path = os.path.join(save_dir, f'qa_{tasktype}.jsonl')
    if os.path.exists(save_path):
        os.remove(save_path)

    

    # set random seed
    random.seed(random_seed)
    dutils.set_random_seed(random_seed)


    # get dataset
    dataset = dutils.get_dataset(task, data_dir)
    df = dataset.df


    # skip until given index# skip until given index
    # this is useful as API calls can crash and we can resume from the last index
    
    if tasktype == 'validate' or tasktype == 'disc':
        if idx > len(df):
            raise ValueError("Index is out of range")
    
        for entity_id in range(len(df)):
            if(idx > entity_id):
                print(f"Skipping idx: {entity_id}...")
                continue
            else:
                print(f"Running idx: {entity_id}...")


            # get an entity
            entity_list = df.iloc[entity_id].tolist()[1:] # we exclude the first column which is index
            if None in entity_list:
                raise ValueError("Value of Entity is Strange")

            
            if tasktype == 'validate':
                # get validation prompt for the row
                
                question = dataset.get_val_prompt(entity_list)
                qa_dict = {'entity_id': entity_id, 'question': question}
                


            elif tasktype == 'disc':
                # get disc prompt for the row
                question, answer = dataset.get_disc_prompt(entity_list)
                qa_dict = {'entity_id': entity_id, 'question': question, 'answer': answer}

            
            with open(save_path, 'a') as f:
                f.write(json.dumps(qa_dict) + '\n')

    
    # for generate task, we have to write SQL queries
    elif tasktype == 'gen':

        #just for now 
        if task == 'same_sex':
            df = df[df['legality'] != 'Established']

        # write sql query
        conn = connect(':memory:')
        df = df.drop(columns=['index'])
        df.to_sql(task, conn, if_exists='replace', index=False)
        print("Connected to in-memory database.\n")
        print("Number of rows: ", len(df))
        for col in df.columns:
            print(f"Columns: {col}\tNumber of unique values: {len(df[col].unique())}")


        # generate select query and where query
        
        # target_col = dataset.target[0]
        select_query_list = []
        where_query_dict = {}
        # theo = 1

        for col in df.columns:

            if col == 'year':
                continue

            # elif col == target_col:
            select_query = 'select distinct ' +  col + f' from {task}'
            select_query_list.append(select_query)
            col_values = df[col].unique()

            # else:
                # col_values = df[col].unique()
                # theo *= (len(col_values)+1)

            where_query_col = [' ']
            for value in col_values:
                where_query_col.append(col + ' = ' + f"'{str(value)}'")
            # where_query_list.append(where_query_col)
            where_query_dict[col] = where_query_col

        # print(where_query_dict)         
        

        
        # calculate theoretical number of queries
        # theo = theo - 1
        theo_list = []
        final_query_list = []
        for select_query in select_query_list:
            if select_query.split(" ")[2] not in dataset.target:
                continue

            print("current select query: ", select_query)
            cols_except_values = [v for k, v in where_query_dict.items() if k not in select_query]
            
            theo = 1
            for cond in cols_except_values:
                theo *= (len(cond))
            theo_list.append(theo-1)
            

            # make cartesian product of where queries
            
            where_query_list = []
            for element in itertools.product(*cols_except_values):
                
                # check cartesian product
                element = [e for e in element if e.strip()] # remove ' ' from element

                # construct where conditions
                if len(element) == 0: # no conditions
                    continue

                elif len(element) == 1: # one conditions, no "and"
                    where_query_list.append('where ' + element[0])
                    
                else: # multiple conditions, add "and"
                    except_orderby = [e for e in element if 'order by' not in e]
                    orderby = [e for e in element if 'order by' in e]

                    where_query = 'where ' + ' and '.join(except_orderby)
                    if len(orderby) > 0:
                        where_query += ' ' + orderby[0]

                    where_query_list.append(where_query)

        



        # make final query
            for where_query in where_query_list:
                final_query_list.append(select_query + ' ' + where_query)

                if tasktype == 'order':
                    final_query_list.append(select_query + ' ' + where_query + ' order by year limit 1')
                    final_query_list.append(select_query + ' ' + where_query + ' order by year desc limit 1')

            print(f"Constructing queries...\n")


        # iterate each query and analyze the answer count
        qa_list = []
        # query_id = 0
        for query in final_query_list:
            query_dict = {'query': query}

            # execute query
            answer = pd.read_sql(query, conn)
            query_dict['answer_cnt'] = len(answer)
            # query_id += 1
            qa_list.append(query_dict)


        
        # categorize queries as none/unique/multiple
        none_list = [query_dict for query_dict in qa_list if query_dict['answer_cnt'] == 0]
        unique_list = [query_dict for query_dict in qa_list if query_dict['answer_cnt'] == 1]
        multiple_list = [query_dict for query_dict in qa_list if query_dict['answer_cnt'] > 1]


        # check if the number of queries is correct
        print("Total number of queries: ", len(final_query_list))
        print("Theoretical number: ", theo-1 if tasktype == 'gen' else theo*3)
        print("\tNone: ", len(none_list))
        print("\tUnique: ", len(unique_list))
        print("\tMultiple: ", len(multiple_list))


        # sample 500 for each categories
        if not all:
            final_qa_list = []
            for category in [none_list, unique_list, multiple_list]:
                if len(category) > sample:
                    print("Sampling...")
                    # TODO: check random 
                    sampled = random.sample(category, sample)
                    final_qa_list.extend(sampled)
                else:
                    final_qa_list.extend(category)
            qa_list = final_qa_list



        # save generated QAs to jsonl file
        if all:
            save_path.replace(".log", "_all.log")


        # generate NL questions using LLMs
        query_id = 0
        for query_dict in final_qa_list:
            print("Processing: ", query_id)
            # print("Query: ", query_dict['query'])
            prompt_sql = query_dict['query']
            prompt_sql_noyear =  prompt_sql.replace('year, ', '')
            nlq = lutils.run_llm(prompt_sql_noyear, \
                                tasktype='convert_sql', \
                                model='gpt4o', \
                                temp = 0)
            
            # update query dict with question
            query_dict['query_id'] = query_id
            query_dict['question'] = 'Q: ' + nlq + '\n'

            # execute query to get answer
            answer = pd.read_sql(prompt_sql_noyear, conn)
            query_dict['answer'] = answer[answer.columns[0]].unique().tolist()
            assert(len(query_dict['answer']) == query_dict['answer_cnt'])

            with open(save_path, 'a') as f:
                f.write(json.dumps(query_dict) + '\n')
            query_id += 1


    print("QA pairs generated successfully!")
    return

if __name__ == "__main__":
    main()
