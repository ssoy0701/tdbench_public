'''
Dataset configuration for the temporal evaluation task
'''

import re
import random
import pandas as pd
import os

# add this dataset list before you define one
DATASET_LIST = ['olympic', 'dyknow', 'legal', 'environ', 'culture', 'movie']


def set_random_seed(seed):
    random.seed(seed)


def get_dataset(task, data_dir='/home/soyeon/temp_eval/dataset/crafted'):
    '''
    Return task specific dataset, reading from data_dir.
    '''

    # Provide dataset based on the task
    if task == 'legal':
        return SameSexLaw(os.path.join(data_dir, 'legal', 'legal.csv'))

    elif task == 'environ':
        return Carbon(os.path.join(data_dir, 'environ', 'environ.csv'))

    elif task == 'culture':
        return Heritage(os.path.join(data_dir, 'culture', 'culture.csv'))
    
    elif task == 'movie':
        return Movie(os.path.join(data_dir, 'movie', 'movie.csv'))
    
    elif task == 'olympic':
        df = pd.read_csv(\
            os.path.join(data_dir, 'olympic', 'olympic_dyknow_leaders.csv'))
        return Olympic(df)
    
    elif task == 'dyknow':
        df_name_list = ['dyknow_leaders.csv', 'dyknow_sports.csv', 'dyknow_organizations.csv']
        df_path_list = [os.path.join(data_dir, 'dyknow', df) for df in df_name_list]

        return Dyknow(df_path_list)
    else:
        raise ValueError(f"Invalid task: {task}")
    

def df_to_str(df):
    '''
    Convert a DataFrame to a string.
    '''
    headers = "| " + " | ".join(df.columns) + " |"
    separators = "| " + " | ".join(['---'] * len(df.columns)) + " |"
    
    rows = [
        "| " + " | ".join(str(cell) for cell in row) + " |"
        for _, row in df.iterrows()
    ]
    
    markdown = "\n".join([headers, separators] + rows)
    return markdown


def get_false_year(gold_year):
    '''
    Sample false year given the gold year.
    Random samples within [gold year+1, gold year+3].
    '''
    return int(gold_year) + random.randint(1, 3)


def get_false_value(df, from_col, subject, to_col):
    '''
    Return a false value from to_col, w.r.t. subject in from_col.
    '''
    subject_list = df[df[from_col] == subject][to_col].values
    false_list = list(set(df[to_col]) - set(subject_list))
    return random.choice(false_list)


def get_outdated_values(df, fd_lhs_list, fd_rhs_list, row):
    
    # get rows that agree on the lhs
    for fd_lhs in fd_lhs_list:
        df = df[df[fd_lhs] == row[fd_lhs]]

    # get outdated rows (i.e., except current rows)
    df = df[df['End'].notna()]

    # sort by year and get unique values
    outdated = df.sort_values(by='Start', ascending=False)[[fd_rhs_list[0], 'Start']].values.tolist()
    return outdated


class Olympic():   
    def __init__(self, df):
        self.df = df
        self.col_info = {
            0: 'Country', \
            1: 'Role', \
            2: 'Name', \
            3: 'Game_edition', \
            4: 'City', \
            5: 'Season', \
            6: 'Game_name', \
            7: 'Start', \
            8: 'End', \
        }
        # self.fd = [(['Game_name', 'Role'], ['Name']),
        #            (['Name', 'Role', 'Season'], ['City'])]

        self.fd = [(['Game_edition', 'Season', 'Role'], ['Name'])]

        self.fd = {
            '2hop': [(['Game_name', 'Role'], ['Name'])],
            '3hop': [(['Game_edition', 'Season', 'Role'], ['Name'])],
        }

        self.granularity = 'month'



class Mydataset:

    df = []
    df_curr = []
    fd = []
    granularity = []
    binary = None

    def get_length(self, curr = False):
        '''
        Get the length of the dataset.
        If curr = True, return the length of the current rows (i.e., self.df_curr).
        '''

        if len(self.df) == 0:
            return 0
        else: 

            target_df = self.df if not curr else self.df_curr
            length = 0
            for curr_df in target_df:
                length += len(curr_df)
            return length
        
    def get_current_rows(self):
        '''
        Get only current rows from the dataset. 
        This should be called before temporal alignment task.
        '''

        if self.get_length() == 0:
            raise ValueError("Dataset is empty!!")
            
        # get current rows
        print("Creating current rows...")
        new_df_list = []
        for curr_df in self.df: 
            curr_df = curr_df[curr_df['End'].isna()].copy()
            new_df_list.append(curr_df)
            print(f"Current df length: {len(curr_df)}")

        self.df_curr = new_df_list
        print(f"Created current df with total length: {self.get_length(curr=True)}\n")
        return new_df_list

    def set_binary(self, binary_ans_list):
        '''
        Set the binary answer list.
        '''
        self.binary = binary_ans_list

    def get_binary_sysprompt(self):
        '''
        Get the system prompt for binary classification.
        '''
        if self.binary is None:
            raise ValueError("Binary answer list is not set!!")
        
        return f"Answer the following question with {self.binary[0]} or {self.binary[1]}."


    def get_context(self, correct_rows, incorrect_rows, return_str=True, id=0):
        '''
        Get the context for the given rows.
        if return_str is True, return the context as a string.
        else, return the context as a DataFrame.
        '''

        df = self.df[id].copy()  # Assuming df[0] is the main DataFrame
        correct_names = set(row["Name"] for row in correct_rows)
        incorrect_names = set(row["Name"] for row in incorrect_rows)

        df_correct = df[df['Name'].isin(correct_names)]
        df_incorrect = df[df['Name'].isin(incorrect_names)]

        excluded_names = correct_names.union(incorrect_names)
        df_random_candidates = df[~df['Name'].isin(excluded_names)]
        df_random = df_random_candidates.sample(n=3, random_state=42)  # random_state는 재현 가능성용

        df_final = pd.concat([df_correct, df_incorrect, df_random], ignore_index=True)
        df_final_shuffled = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

        if return_str:
            return df_to_str(df_final_shuffled)
        else:
            return df_final_shuffled
    


class Dyknow(Mydataset):
    def __init__(self, df_path_list):
        print(f"Loading DyKnow dataset from...")
        for df_path in df_path_list:
            print(f"\t{df_path}")
        self.df_leaders = pd.read_csv(df_path_list[0])
        self.fd_leaders = (['Country', 'Role'], ['Name'])
        self.gran_leaders = 'month'

        self.df_sports = pd.read_csv(df_path_list[1])
        self.fd_sports = (['Name', 'Sport'], ['Team'])
        self.gran_sports = 'year'

        self.df_orgs = pd.read_csv(df_path_list[2])
        self.fd_orgs = (['Organization_name', 'Organization_type', 'Role'], ['Name'])
        self.gran_orgs = 'year'

        self.df = [self.df_leaders, self.df_sports, self.df_orgs]
        self.fd = [self.fd_leaders, self.fd_sports, self.fd_orgs]
        self.granularity = [self.gran_leaders, self.gran_sports, self.gran_orgs]

        print(f"Total length of the dataset: {self.get_length()}\n")


class SameSexLaw(Mydataset):
    def __init__(self, df_path):
        print(f"Loading Legal dataset from...")
        print(f"\t{df_path}")
        self.df = [pd.read_csv(df_path)] # Country,Law_type,Legality,Start
        self.fd = [(['Country', 'Law_type'], ['Legality'])]
        self.granularity = ['year']
        print(f"Total length of the dataset: {self.get_length()}\n")

        self.set_binary(['Yes/Legal', 'No/Illegal'])


class Carbon(Mydataset):
    def __init__(self, df_path):
        print(f"Loading Carbon dataset from...")
        print(f"\t{df_path}")
        self.df = [pd.read_csv(df_path)] 
        self.fd = [(['Jurisdiction', 'Type'], ['Status'])]
        self.granularity = ['year']
        print(f"Total length of the dataset: {self.get_length()}\n")

        self.set_binary(['Yes/Implemented', 'No/Not Implemented'])

class Heritage(Mydataset):
    def __init__(self, df_path):
        print(f"Loading Heritage dataset from...")
        print(f"\t{df_path}")
        self.df = [pd.read_csv(df_path)] # Member_state,Heritage_element,Status_Inscribed_or_Proclaimed,Region,Year
        self.fd = [(['Heritage_element'], ['Status_Inscribed_or_Proclaimed'])]
        self.granularity = ['year']
        print(f"Total length of the dataset: {self.get_length()}\n")

        self.set_binary(['Inscribed', 'Proclaimed'])


class Movie(Mydataset):
    def __init__(self, df_path):
        print(f"Loading Movie dataset from...")
        print(f"\t{df_path}")
        self.df = [pd.read_csv(df_path)] # Country,Movie_name,Year
        self.fd = [(['Title'], ['Director'])]
        self.granularity = ['year']
        print(f"Total length of the dataset: {self.get_length()}\n")
