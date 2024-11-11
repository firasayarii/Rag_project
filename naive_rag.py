import os
from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from langchain_groq import ChatGroq
import chromadb
from chromadb.utils import embedding_functions
from utilities import Token
from load_documents import document


load_dotenv()
# Embedding function :
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction()

# Initialize The Chroma Client :
chroma_client = chromadb.PersistentClient(path='chroma_storage')
chroma_collection = chroma_client.get_or_create_collection(
    "medical-collect", embedding_function=embedding_function
)
# The LLM Model and the Prompt Template
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    groq_api_key=Token,
    # other params...
)





# Generate Embeddings for document

ids = [str(i) for i in range(len(document))]

chroma_collection.add(ids=ids, documents=document)



def query_documents(question , n_results=2) :
    n_results = chroma_collection.query(
        query_texts=question, n_results=n_results, include=["documents", "embeddings"])
    
    relevant_chunks = n_results["documents"][0]
    return relevant_chunks

def generate_response(question, context,llm=llm):
    
    context="\n\n".join(context)

    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""you are an assistant for question-answering tasks      
        You are a knowledgeable Medical assistant. 
        Your users are inquiring about some medical information.
    Based on the following context:\n\n{context}\n\nAnswer the query: {question} , if you dont know please answer i dont know"""
    )
    
    chain = LLMChain(llm=llm,prompt = prompt)

    response = chain.run({'question': question, 'context': context})

    return response

question = "Number of newly diagnosed cancer cases and deathes in the UK for all cancers ?"
context = query_documents(question)
res=generate_response(question,context)
print(res)

