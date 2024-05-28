from langchain_community.document_loaders import NotionDBLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from groq import Groq
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def load_document(notion_api_key, notion_database_id):
    try:
        loader = NotionDBLoader(notion_api_key, notion_database_id)
    except Exception as e:
        print(e)

    # data = loader.load()
    notion_document = loader.load()

    document_lst = []
    for document in notion_document:
        # print(document.metadata)
        document_lst.append(document.metadata)
    # print(document_lst)
    try:
        document_string = ""
        for i in range(len(document_lst)):
            # print(document_lst[i])
            document_string += document_lst[i]["name"] + " "
            document_string += document_lst[i]["branch"] + " "
            document_string += document_lst[i]["id"] + "\n"
            # document_string += document_lst[i]
        print(document_string)
    except Exception as e:
        print(e)
    return document_string


def split_document(document_string):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        text = text_splitter.create_documents([document_string])
    except Exception as e:
        print("Error in text splitting : {str(e)}")
    return text


def generate_embedding(text):
    try:
        embedding = HuggingFaceEmbeddings()
        db = Chroma.from_documents(text, embedding)
    except Exception as e:
        print("Error in storing and generating embedding")
    return db
