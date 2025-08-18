import dotenv

if not dotenv.load_dotenv():
    raise Exception("dotenv não foi carregado corretamente")

from google.adk.agents import Agent
from rapidfuzz import process, fuzz
import pandas as pd

df = pd.read_parquet("base_2.parquet")

def tool_search_file_suffix(file_suffix: str) -> list[str]:
    """
    file_suffix (str): the file suffix we want to filter, it's ALWAYS in uppercase.

    User query might contain file suffixes in the following regex format: IO\.\d+ or POP\.\d+ or MA.CTO\.\d+
    You must provide ONLY the file suffix and the return will tell if it's possible to provide information about it.
    

    return: a list with strings with the file names that we can search more about the file suffix
    or it returns a empty list (did not find any files.)
    """
    return df[df['filename'].str.startswith(file_suffix)]['filename'].unique().tolist()

def get_information_about_filename(file_name: str) -> list[dict]:
    """
    file_name (str): the filename.

    get more information about a specific filename.
    When this is returned you should provide an answer to the user question or a summary of all pages.
    return: dict with the following format: {page: int, content: string} where page is the page where the content is at.
    or it returns an empty list (could not find information about the filename)
    """
    content = df[df['filename']==file_name][['page', 'content']].to_dict('records')
    return content

def search_general_information_based_on_user_query(query: str) -> str:
    """
    Search the user query among many files to provide the one that contains the best answers.
    This might return multiple (to a maximum of 5) top results.
    return: dict that contains the following:
    position: int - the position (1 is the best match) for the user query
    document: str - document name (filename)
    content: str - document content
    page: int - document's page for the extracted content
    """

    matches = process.extract(query, df['content'], limit=5, scorer=fuzz.partial_ratio)
    result = {}
    for en, (content, score, index) in enumerate(matches):
        result[en] = {'document': df.iloc[index]['filename'], "page": df.iloc[index]['page'], 'content': df.iloc[index]['content']}
    return str(result)


root_agent = Agent(
    name="FileSearch",
    model="gemini-2.0-flash-lite",
    description="Answer the user",
    instruction="""
    Any instructions as you want 
    """,
    tools=[tool_search_file_suffix, get_information_about_filename, search_general_information_based_on_user_query]
)
