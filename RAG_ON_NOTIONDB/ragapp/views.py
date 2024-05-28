from django.shortcuts import render
from django.http import HttpResponse

from langchain_community.document_loaders import NotionDBLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from groq import Groq
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
import os

from .functions import load_document
from .functions import split_document 
from .functions import generate_embedding

load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

notion_api_key = os.environ.get("notion_api_key")
notion_database_id = os.environ.get("notion_database_id")

document_string = load_document(notion_api_key, notion_database_id)
text = split_document(document_string) 
db = generate_embedding(text)  


# Create your views here.
def home(request):
    if request.method=="POST":
        user_question = request.POST.get("question")

        try:
            embedding = HuggingFaceEmbeddings()
            query_embedding = embedding.embed_query(user_question)
            # print("Query Embedding : ", query_embedding)

            retriever = db.as_retriever(search_kwargs={"k":2})

        except Exception as e:
            print("Error in chroma db and retrieveing : ", e)

        try:
            # result_doc = db.similarity_search(user_question)
            # print(result_doc)

            result_doc = retriever.invoke(user_question)
            # result_doc = retriever.invoke(str(query_embedding))
            # print((result_doc))

            combined_result = ""

            for result in result_doc:
                combined_result += result.page_content

            print(combined_result)
        except Exception as e:
            print("Error in retrieving ",e)

        try:
            chat = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

            system = f"""You will be provided a context and you will be asked question. You have to answer that question based on that context.In the context name of Student, his department and the unique id number will be present. Context is {combined_result}. """
            human = "{text}"
            prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

            chain = prompt | chat
            result = chain.invoke(
                {
                    "text": f"""Here is my question. Answer the following question : {user_question}. If the answer is not available in the context then simply say I don't know.
                    Just answer the question. Don't write any other sentences in the response.
                    """
                }
            )
        except Exception as e:
            print("Exceptino in chain invoking : ",e)

        # try:
        return render(
            request,
            "home.html",
            {"result": f"{result.content}"},
        )

        # return render(request, 'home.html')
    else:
        return render(request, "home.html")
    # return HttpResponse("Home Page")
