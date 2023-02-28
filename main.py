from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
import requests
import openai

# openai.api_key = ('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')
API='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'

# OPENAI_API_KEY="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"

def get_wiki_data(title, first_paragraph_only):
    url = f"https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&explaintext=1&titles={title}"
    if first_paragraph_only:
        url += "&exintro=1"
    data = requests.get(url).json()
    return Document(
        page_content=list(data["query"]["pages"].values())[0]["extract"],
        metadata={"source": f"https://en.wikipedia.org/wiki/{title}"},
    )

sources = [
    get_wiki_data("Unix", True),
    get_wiki_data("Microsoft_Windows", True),
    get_wiki_data("Linux", True),
    get_wiki_data("Seinfeld", True),
]

# chain = load_qa_with_sources_chain(OpenAI(temperature=0, openai_api_key=API), chain_type="map_reduce")

# def print_answer(question):
#     print(
#         chain(
#             {
#                 "input_documents": sources,
#                 "question": question,
#             },
#             return_only_outputs=True,
#         )["output_text"]
#     )

# from langchain_bot import print_answer
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
search_index = FAISS.from_documents(sources, OpenAIEmbeddings(openai_api_key=API))


chain = load_qa_with_sources_chain(OpenAI(temperature=0, openai_api_key=API))

def print_answer(question):
    print(
        chain(
            {
                "input_documents": search_index.similarity_search(question, k=4),
                "question": question,
            },
            return_only_outputs=True,
        )["output_text"]
    )
print_answer("Who were the writers of Seinfeld?")

print_answer("reanswer the last question, but elaborate more thsi time")