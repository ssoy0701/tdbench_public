from langchain_community.document_loaders import WikipediaLoader
import pandas as pd
import json


def get_related_docs(retriever, entity):
    entity_doc = retriever.get_relevant_documents(entity)
    if len(entity_doc) > 0:
        return entity_doc[0].metadata['summary']
    else: 
        return 'None'


print("Preprocessing RAG dataset...\n")


df = pd.read_json('/home/v-kimsoyeon/temp_eval/qa_pairs/olympic/join_sql_gpt4o.jsonl', lines=True)


import re

def parse_query(query, pattern):
    # Extract the Game_name value from the SQL query
    if pattern == '2hop':
        match = re.search(r'Game_name\s*=\s*["\']([^"\']+)["\']', query)
        if match:
            return f'{match.group(1)} Olympics'

    elif pattern == '3hop':

        game_edition_match = re.search(r'Game_edition\s*=\s*["\'](\d+)["\']', query)
        season_match = re.search(r'Season\s*=\s*["\']([^"\']+)["\']', query)

        game_edition = game_edition_match.group(1) if game_edition_match else None
        season = season_match.group(1) if season_match else None


        if game_edition:
            return f'{game_edition}th {season} Olympics'
        else:
            return None


df['2hop_query'] = df['sql_for_db'].apply(lambda x: parse_query(x, '2hop'))
df['3hop_query'] = df['sql_for_db'].apply(lambda x: parse_query(x, '3hop'))





def get_rag(query):
    retriever = WikipediaLoader(query=query, load_max_docs=1)
    docs = retriever.load()

    if len(docs) == 0:
        print(f"No documents found for query: {query}")
        return None

    elif len(docs) >= 1:
        print(f"Found {len(docs)} documents for query: {query}")
        first = docs[0].metadata['summary']
        return first


df['rag_2hop_gold'] = df['2hop_query'].apply(lambda x: get_rag(x) if x else None)
df.to_csv('/home/v-kimsoyeon/temp_eval/qa_pairs/olympic/join_sql_gpt4o_rag.csv', index=False)

df['rag_2hop_question'] = df['questions'].apply(lambda x: get_rag(x[0]) if x else None)
df.to_csv('/home/v-kimsoyeon/temp_eval/qa_pairs/olympic/join_sql_gpt4o_rag.csv', index=False)

df['rag_3hop_gold'] = df['3hop_query'].apply(lambda x: get_rag(x) if x else None)
df.to_csv('/home/v-kimsoyeon/temp_eval/qa_pairs/olympic/join_sql_gpt4o_rag.csv', index=False)

df['rag_3hop_question'] = df['questions'].apply(lambda x: get_rag(x[1]) if x else None)
df.to_csv('/home/v-kimsoyeon/temp_eval/qa_pairs/olympic/join_sql_gpt4o_rag.csv', index=False)

