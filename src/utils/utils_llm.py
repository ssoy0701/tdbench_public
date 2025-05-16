from openai import AzureOpenAI
import google.generativeai as genai
from google.ai import generativelanguage as glm
import anthropic
import ollama

MODEL_LIST = ['gpt35', 'gpt4',  'gpt4o', 'llama', 'mixtral', 'gemma', 'qwen', 'granite']
MODEL_PRETTY_LIST = ['GPT-3.5', 'GPT4', 'GPT-4o', 'Llama3', 'Mixtral', 'Gemma2', 'Qwen2', 'Granite3']
MODEL_TO_PRETTY = dict(zip(MODEL_LIST, MODEL_PRETTY_LIST))


def get_savename(save_dir, task=None, tasktype='', model='', endswith = '.log'):
    '''
    Given save_dir and task, return savename of gpt-3.5 and gpt-4 log.
    num_choices, all_true, hard is optional.

    Returns:
    gpt35_log: save_dir/task/gpt35.log
    gpt4_log: save_dir/task/gpt4.log
    '''
    
    if task is None:
        model_log = f'{save_dir}/{model}'

    else:
        model_log = f'{save_dir}/{task}/{model}'

    if tasktype:
        model_log += f'_{tasktype}'

    model_log += endswith

    return model_log



def write_to_file(savename, index="", prompt="", response="", answer=None, entity=None, sep=False):
    '''
    Log file format.
    '''
    with open(savename,'a') as f:

        if sep:
            f.write("========================================\n")
            return
    
        f.write(f"{index}th question\n")
        f.write("<Question> \n")
        f.write(prompt)
        f.write("\n\n")
        f.write("<Answer> \n")
        try:
            f.write(response)
        except:
            f.write("")
        f.write("\n\n")

        if answer:
            f.write('<Gold Answer> \n')
            f.write(str(answer))
            f.write("\n\n")

        if entity:
            f.write('<Gold Entity> \n')
            f.write(str(entity))
            f.write('\n\n')

    return



def run_llm(prompt, tasktype, model, temp=0, cot=None, new_prompt=None):

    # system message
    if tasktype=='simple_gen':
        system_msg = "Answer the following question. If you are unsure, say 'unsure'. Be concise."
        # system_msg = "Answer the following question in yes or no, and then say the related year information with a word 'since'. Say 'unsure' if you don't know. Be concise."
    elif tasktype=='simple_disc':
        system_msg = "Answer the following multiple question. Say 'unsure' if you don't know. Be concise and say your answer with a word 'Option'."
    elif tasktype =='validate':
        system_msg = "Answer the following question in yes or no. Be concise"
    elif tasktype == 'convert_sql':
        system_msg = 'Convert the following SQL query to a natural language question. Only return the generated question.'
    # elif tasktype == 'temp_align':
        # system_msg = "Answer the following question. Provide the date information (month and year) with the answer starting with 'since'. Say 'unsure' if you don't know. Be concise."
    elif tasktype == 'temp_align':

        system_msg = "Answer the following question. Provide the short direct answer that includes both the factual answer and the date information (month and year) of the fact, prefixed with the word 'since'. Say 'unsure' if you don't know. Be concise."

        if new_prompt: # the answer can be binary
            system_msg = system_msg.replace('Answer the following question.', new_prompt)
        # system_msg = "You are a concise factual question-answering assistant. Given a natural language question, provide a short and direct answer that includes both the factual answer and the start date (month and year) of the fact, prefixed with the word 'since'. Your response should be a single, natural-sounding sentence. If you don't know the answer, respond with “unsure.”"
    elif tasktype == 'basic':
        # system_msg = "Answer the following question by including date information (month and year) in your answer. Present your answers in bullet points. There may be multiple answers or none. If there are no answers, say 'no answer'. If you are uncertain, say 'unsure'. Be concise."
        system_msg = "Answer the following question. Provide the short direct answer that includes both the factual answer and rationale with the date information of the fact. If there is no valid answer, respond with 'No answer'. If there is one correct answer, return it as a short sentence. If there are multiple valid answers, present them clearly as bullet points. If you are unsure, respond with 'Unsure'. Be concise."
    elif tasktype == 'simple_gen_dyknow':
        system_msg = "Answer with the name only. If you are unsure, say 'unsure'. Be concise."
    elif tasktype == 'no_system_msg':
        system_msg = None

    # run model
    if model == 'gpt35':
        return run_gpt35(prompt, system_msg, cot)
    elif model == 'gpt4':
        return run_gpt4(prompt, system_msg)
    elif model == 'gemini':
        return run_gemini(prompt, system_msg)
    elif model == 'gpt4o':
        return run_gpt4o(prompt, system_msg, temp=temp)
    elif model == 'llama':
        return run_ollama(prompt, system_msg, model='llama3.1:70b', temp=temp)
    elif model == 'mixtral':
        return run_ollama(prompt, system_msg, model='mixtral:8x7b', temp=temp)
    elif model == 'gemma':
        return run_ollama(prompt, system_msg, model='gemma2:27b', temp=temp)
    elif model == 'qwen':
        return run_ollama(prompt, system_msg, model='qwen2:72b', temp=temp)
    elif model == 'granite':
        return run_ollama(prompt, system_msg, model='granite3.1-dense:8b', temp=temp)
    elif model == 'deepseek':
        return run_ollama(prompt, system_msg, model='deepseek-r1:14b', temp=temp)
    else:
        raise ValueError(f"Unknown model: {model}")

def run_ollama(prompt, system_msg=None, model='llama3.1:70b', temp=0):

    if system_msg is None:
        msg = [
            {"role": "user", "content":prompt},
        ]
    
    else:
        msg = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content":prompt},
        ]

    response=  ollama.chat(model=model, \
                           messages = msg,  \
                            options={'temperature':temp})

    return response['message']['content']



def run_gpt4(prompt, system_msg):
    '''
    Run GPT-4 and return response.
    System message is predefined for validate, multiqa, multiqa_hint (used for all true questions), and rag.
    Temperature is set to 0 to get deterministic response.
    '''

    client = AzureOpenAI( 
        api_version = "...",
        azure_endpoint= "...",
        api_key= "..."
    )



    msg = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content":prompt},
    ]

    completion = client.chat.completions.create(
        model="gpt-4",
        messages = msg, 
        temperature = 0
    )

    return completion.choices[0].message.content



def run_gemini(prompt, system_msg):

    msg = [system_msg, prompt]

    genai.configure(api_key="...")

    model = genai.GenerativeModel('gemini-1.5-pro')
    config = {"temperature" : 0, "max_output_tokens" : 200}
    safety_settings = {
        glm.HarmCategory.HARM_CATEGORY_HARASSMENT: glm.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        glm.HarmCategory.HARM_CATEGORY_HATE_SPEECH: glm.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        glm.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: glm.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        glm.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: glm.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    }
    response = model.generate_content(msg, generation_config = config, safety_settings = safety_settings)

    return response.text



def run_claude(prompt, system_msg):
    client = anthropic.Anthropic(
        api_key="...",
    )

    msg = [
            {"role": "user", "content":prompt},
        ]
    
    message = client.messages.create(
        model="claude-3-sonnet-20240229",
        system=system_msg,
        temperature=0.0,
        max_tokens = 1024,
        messages= msg
    )

    return message.content[0].text



def run_gpt35(prompt, system_msg, cot=None):
    '''
    Run GPT-3.5 and return response.
    System message is predefined for validate, multiqa, multiqa_hint (used for all true questions), and rag.
    Temperature is set to 0 to get deterministic response.
    '''

    client = AzureOpenAI( 
        api_version = "...",
        azure_endpoint= "...",
        api_key="..."
    )
   
    if cot: 
        cot_q, cot_a = cot
        msg = [
            {"role": "system", "content": system_msg}
        ]
        for q,a in zip(cot_q, cot_a):
            msg.append({'role': 'user', 'content': q})
            msg.append({'role': 'assistant', 'content': a})

        msg.append({"role": "user", "content":prompt})

    else:

        msg = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content":prompt},
        ]
 
    completion = client.chat.completions.create(
        model="gpt-35-turbo",
        messages = msg,
        temperature = 0)
    
    return completion.choices[0].message.content



def run_gpt4o(prompt, system_msg, temp=0, gpu_idx=0):
    if gpu_idx==0:
        client = AzureOpenAI( 
            api_version = "...",
            azure_endpoint=  "...",
            api_key="..."
        )
    elif gpu_idx==1:
        client = AzureOpenAI( 
        api_version ="...",
        azure_endpoint = "...", 
        api_key="..."
        )
    else:
        raise ValueError("GPU STRANGE")
    
    if system_msg is None:
        msg = [
            {"role": "user", "content":prompt},
        ]
    else: 
        msg = [            
            {"role": "system", "content": system_msg},
            {"role": "user", "content":prompt},      
        ]

    completion = client.chat.completions.create(
        model="gpt4o" if gpu_idx ==0 else "gpt4o_sweden",
        messages = msg, 
        temperature = temp
    )
    return completion.choices[0].message.content


