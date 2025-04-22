from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# df = pd.read_csv("jeh_resume.txt")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./reference_db"
add_documents = not os.path.exists(db_location) # check if database exists

# here we add one or more documents to our vector database
# we can add the contents of a file, the records in a dataframe or 
#   from some other source 

# we can associate some metadata with the document

# for 

if add_documents:
    documents = []
    ids = []

    # add the contents of a file
    file_path = "/Users/eric/projects/code_assistant/jeh_resume.txt"

    with open(file_path, 'r') as file:
        resume_content = file.read()
        # print(content)
        
    # for i, row in df.iterrows():
    document = Document(
        page_content=resume_content,
        metadata={"name":"function_name_tbd", "date": "2025-04-19"},
        id="1"
    )
    ids.append("1")
    documents.append(document)
        
vector_store = Chroma(
    collection_name="function_code",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 1}
)