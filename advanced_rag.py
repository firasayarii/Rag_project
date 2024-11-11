from helper_utils import project_embeddings, word_wrap
from pypdf import PdfReader
import os
from pypdf import PdfReader
import umap
import chromadb
from langchain_groq import ChatGroq
from utilities import Token
from langchain import PromptTemplate, LLMChain


# Read the pdf document
reader = PdfReader("data/microsoft-annual-report.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

pdf_texts = [text for text in pdf_texts if text]

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)

character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

#print(character_split_texts[1])
#print(f"\nTotal chunks: {len(character_split_texts)}")

token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

#print(word_wrap(token_split_texts[0]))
#print(f"\nTotal chunks: {len(token_split_texts)}")

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

embedding_function = SentenceTransformerEmbeddingFunction()
#print(embedding_function([token_split_texts[10]]))

chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(
    "microsoft-collection", embedding_function=embedding_function
)
# extract the embeddings of the token_split_texts
ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)
chroma_collection.count()


#query='what was the revenu of the last year'
#results = chroma_collection.query(query_texts=[query], n_results=5)
#retrieved_documents = results["documents"][0]

"""for document in retrieved_documents:
     print(word_wrap(document))
     print("\n")"""

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    groq_api_key=Token,
    # other params...
)

def augment_query_generated(query, model=llm):
    prompt = PromptTemplate(
        input_variables=["query"],
        template=""""You are a helpful expert financial research assistant. 
   Provide an example answer to the given question : {query}, that might be found in a document like an annual report."""
    )
    
    chain = LLMChain(llm=llm,prompt = prompt)

    response = chain.run({'query': query})

    return response

original_query = "What was the total profit for the year, and how does it compare to the previous year?"
hypothetical_answer = augment_query_generated(original_query)

joint_query = f"{original_query} {hypothetical_answer}"
results = chroma_collection.query(
    query_texts=joint_query, n_results=5, include=["documents", "embeddings"]
)
retrieved_documents = results["documents"][0]

def generate_response(joint_query, retrieved_documents,llm=llm):
    
    context="\n\n".join(context)

    prompt = PromptTemplate(
        input_variables=[joint_query, retrieved_documents],
        template=""""You are a helpful expert financial research assistant. 
   Provide an answer to the given question and those given answers exemples : {joint_query}, based in those informations{retrieved_documents} """
    )
    
    chain = LLMChain(llm=llm,prompt = prompt)

    response = chain.run({'joint_query': joint_query, 'retrieved_documents': retrieved_documents})

    return response
    