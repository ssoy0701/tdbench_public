'''
Run QA task for given model and task, and save LLM responses to log file.
'''


import random
import argparse
import os
import time
import json
import signal

import src.utils.utils_llm as lutils
import src.utils.utils_dataset as dutils



def parse_args():
    '''
    Config
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, \
                        choices = dutils.DATASET_LIST, default='olympic')
    parser.add_argument('--tasktype', "-t", type=str, \
                        choices=['now', 'join', 'basic', 'temp_align_carpe_ours', 'temp_align_carpe', 'temp_align_dyknow', 'temp_align', 'validate', 'gen', 'disc'], default= 'join')
    parser.add_argument("--index", type=int, default = 0)
    parser.add_argument("--rag", action='store_true', default=False)
    parser.add_argument("--cot", action='store_true', default=False)
    parser.add_argument('--timecot', action='store_true', default=False)
    parser.add_argument("--random_seed", "-s", type=int, default=1116)
    parser.add_argument("--model", type=str, choices = lutils.MODEL_LIST, required=True)
  
    args = parser.parse_args()
    
    return args


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Code execution timed out!")


def run_qa_now(log_file_path, qa_path, model, start_idx=0, new_prompt=None):
    systype = 'temp_align'

    with open(qa_path, 'r') as file:
        qa_data = [json.loads(line) for line in file] 

    # Set the timeout signal handler
    signal.signal(signal.SIGALRM, timeout_handler)

   
    for q_items in qa_data:

        # skip until given start_index
        if(start_idx > q_items['idx']):
            print(f"Skipping idx: {q_items['idx']}...")
            continue

        # get qa prompt for the row
        print(f"Running idx: {q_items['idx']}...")     

        # we have multiple questions
        for prompt_idx, question in enumerate(q_items['questions']):
            id = str(q_items['idx']) + '-' + str(prompt_idx)    
            question = 'Q: ' + question + '\nA:'


            # call model APIs and save responses
            fail_cnt = 0
            while (fail_cnt < 5):
                try:
                    signal.alarm(15)

                    print(question)
                    response = lutils.run_llm(question, \
                                            tasktype=systype, \
                                            model=model, \
                                            new_prompt=new_prompt)
                    print(response)

                    signal.alarm(0)
                    break

                except Exception as error:
                    fail_cnt +=1
                    print(f"{model} ERROR with idx: {id} (fail_cnt: {fail_cnt}) : {error}")
                    
                    if fail_cnt == 5:
                        print(f"Too many errors. Skipping idx: {id}")
                        response = "RESPONSE ERROR"

                except TimeoutException as e:
                    response = "RESPONSE ERROR"



            # this is to avoid API rate limit
            time.sleep(1)

            # save responses to log file
            qa_result_dict = {'idx': id, \
                            "qtype": q_items['qtype'], \
                            "sql_for_db": q_items['sql_for_db'], \
                            'question': question, \
                            'model_response': response, \
                            'correct_rows': q_items['correct_rows'],\
                            'incorrect_rows': q_items['incorrect_rows']}

            # save as jsonl file
            with open(log_file_path, 'a') as file:
                file.write(json.dumps(qa_result_dict) + '\n')


def run_qa_basic(log_file_path, dataset, task, tasktype, rag, cot, timecot, model, start_idx = 0):

    # read jsonl file
    file_path = f"./qa_pairs/{task}/{tasktype}_org_sport_sql_gpt4o.jsonl"

    if tasktype == 'join':
        systype = 'simple_gen'
    elif tasktype == 'basic':
        systype = 'basic'
    else:
        raise NotImplementedError(f"Tasktype {tasktype} is not implemented yet.")

    if rag:
        file_path = f"./qa_pairs/{task}/{tasktype}_sql_gpt4o_rag.jsonl"

        log_file_path = log_file_path.replace('.jsonl', '_rag.jsonl')

    with open(file_path, 'r') as file:
        qa_data = [json.loads(line) for line in file]

    # Set the timeout signal handler
    signal.signal(signal.SIGALRM, timeout_handler)


    if cot:
        if task != 'olympic':
            raise NotImplementedError(f"Task {task} is not implemented yet.")
        
        with open(f'/home/v-kimsoyeon/temp_eval/qa_pairs/olympic/cot.txt') as f:
            cot_prompt = f.read()

        log_file_path = log_file_path.replace('.jsonl', '_cot.jsonl')

    if timecot:
        if task != 'olympic':
            raise NotImplementedError(f"Task {task} is not implemented yet.")        
        
        qa_data = [qa for qa in qa_data if qa['qtype'] == 'join_2hop']

        with open(f'/home/v-kimsoyeon/temp_eval/qa_pairs/olympic/time_cot.txt') as f:
            cot_prompt = f.read()

        log_file_path = log_file_path.replace('.jsonl', '_timecot.jsonl')


        

    
    for q_items in qa_data:

        # skip until given start_index
        if(start_idx > q_items['idx']):
            print(f"Skipping idx: {q_items['idx']}...")
            continue

        # just for now
        if q_items['op'] == 'City':
            continue

        # get qa prompt for the row
        print(f"Running idx: {q_items['idx']}...")     


        # we have multiple questions
        for prompt_idx, question in enumerate(q_items['questions']):

            if rag:
                rag_prompt = 'The provided passage may not provide all information. In this case, use your own knowledge to answer the question.\n\n'
                for rag_type in ['rag_gold']:
                    id = str(q_items['idx']) + '-' + str(prompt_idx) + '-' + rag_type

                    
                    rag_text = q_items[rag_type] + '\n\n'
                    question = rag_prompt + rag_text + 'Q: ' + question + '\nA:'


                    # call model APIs and save responses
                    fail_cnt = 0
                    while (fail_cnt < 5):
                        try:
                            signal.alarm(15)

                            response = lutils.run_llm(question, tasktype=systype, model=model)
                            print(response)
                            signal.alarm(0)
                            break

                        except Exception as error:
                            fail_cnt +=1
                            print(f"{model} ERROR with idx: {id} (fail_cnt: {fail_cnt}) : {error}")
                            
                            if fail_cnt == 5:
                                print(f"Too many errors. Skipping idx: {id}")
                                response = "RESPONSE ERROR"

                        except TimeoutException as e:
                            response = "RESPONSE ERROR"


                    # this is to avoid API rate limit
                    time.sleep(1)

                    # save responses to log file
                    qa_result_dict = {'idx': id, \
                                    "qtype": q_items['qtype'], \
                                    "op": q_items['op'], \
                                    "sql_for_db": q_items['sql_for_db'], \
                                    'question': question, \
                                    'model_response': response, \
                                    'correct_rows': q_items['correct_rows'],\
                                    'incorrect_rows': q_items['incorrect_rows']}

                    # save as jsonl file
                    with open(log_file_path, 'a') as file:
                        file.write(json.dumps(qa_result_dict) + '\n')



            else:

                # get context

                context_str = dataset.get_context(q_items['correct_rows'], \
                                q_items['incorrect_rows'], id=2)
                



                id = str(q_items['idx']) + '-' + str(prompt_idx)    
                question = context_str + '\n\n'+ 'Q: ' + question + '\nA:'


                if cot or timecot:
                    question = cot_prompt + '\n\n\n' + question


                # call model APIs and save responses
                fail_cnt = 0
                while (fail_cnt < 5):
                    try:
                        signal.alarm(15)

                        response = lutils.run_llm(question, tasktype=systype, model=model)
                        print(response)

                        signal.alarm(0)
                        break

                    except Exception as error:
                        fail_cnt +=1
                        print(f"{model} ERROR with idx: {id} (fail_cnt: {fail_cnt}) : {error}")
                        
                        if fail_cnt == 5:
                            print(f"Too many errors. Skipping idx: {id}")
                            response = "RESPONSE ERROR"

                    except TimeoutException as e:
                        response = "RESPONSE ERROR"



                # this is to avoid API rate limit
                time.sleep(1)

                # save responses to log file
                qa_result_dict = {'idx': id, \
                                "qtype": q_items['qtype'], \
                                "op": q_items['op'], \
                                "sql_for_db": q_items['sql_for_db'], \
                                'question': question, \
                                'model_response': response, \
                                'correct_rows': q_items['correct_rows'],\
                                'incorrect_rows': q_items['incorrect_rows']}

                # save as jsonl file
                with open(log_file_path, 'a') as file:
                    file.write(json.dumps(qa_result_dict) + '\n')



def run_qa(save_dir, task, tasktype, rag, cot, timecot, dataset, model, start_idx = 0):
    '''
    Run QA task for given model and task, and save LLM responses to log file.

    Args:
    - mixed: float, the probability of mixed question (i.e. all true question)
    - demo: bool, whether to use pre-defined demos for few-shot setting
    - rag: bool, whether to run RAG setting
    - index: int, the entity index to start running QA task.
    '''

    # get log file path and dataset
    log_file_path = lutils.get_savename(save_dir= save_dir, \
                                            task= task, \
                                            model= model, \
                                            tasktype= tasktype, \
                                            endswith= '.jsonl')
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)


    # run tasktype
    if 'temp_align' in tasktype:
        pass
        # run_qa_temp_align(log_file_path, tasktype, dataset, model, start_idx)

    elif 'basic' in tasktype or 'join' in tasktype: 
        run_qa_basic(log_file_path, dataset, task, tasktype, rag, cot, timecot, model, start_idx)

    elif tasktype == 'now':
        qa_path = f"./qa_pairs/{task}/{tasktype}_sql_gpt4o.jsonl"
        
        new_prompt = None
        if dataset.binary is not None: # new sysprompt for binary questions
            new_prompt = dataset.get_binary_sysprompt()
            
        run_qa_now(log_file_path, qa_path, model, start_idx, new_prompt)
    
    else:
        raise NotImplementedError(f"Tasktype {tasktype} is not implemented yet.")
    

    return



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
    rag = args.rag
    cot = args.cot
    timecot = args.timecot
    random_seed = args.random_seed
    model = args.model

    for k, v in args.__dict__.items():
        print(f"{k}: {v}")
    print("===================================================")
    save_dir = './results'

    # set random seed
    random.seed(random_seed)
    dutils.set_random_seed(random_seed)


    # get dataset
    dataset = dutils.get_dataset(task)


    # start QA
    print(f"Starting main QA...\n")
    run_qa(save_dir=save_dir, task=task, tasktype=tasktype, rag=rag, cot=cot, timecot=timecot, dataset=dataset, model=model, start_idx=idx)

    return


if __name__ == "__main__":
    main()

