import os
import time
import pickle
import numpy as np
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.llms import Ollama  
from typing import Any, List, Optional
import fitz
import gradio as gr


llm = OllamaLLM(model="llama3.2:1b")

PROMPT_TEMPLATE = """You are an expert assistant tasked with providing accurate, precise, short and relevant responses based on the given document content. The answer should be short, precise, accurate and should be correct always. Focus on answering questions directly related to the specified sections or tables from the document provided. strictly do not include DOCUMENT NUMBER, SECTION NUMBER, REVIEW, etc. EVERY TIME NEW QUERY/PROMPT IS ASKED THE RESPONSE SHOULD BE SHORT, PRECISE AND ACCURATE ALWAYS.

Context: {context} (Include relevant, short, accurate and precise responses)

Question: {question}

Your answer:"""

prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["question", "context"])

pdf_paths = [
    r"pdf1",
    r"PDF2",
    r"PDF3",
    r"pdf4"
]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

split_configs = [
    {"chunk_size": 1500, "chunk_overlap": 200},  
    {"chunk_size": 600, "chunk_overlap": 100}    
]

def load_pdf_text(pdf_paths):
    text = ""
    for pdf_path in pdf_paths:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
    return text

raw_text = load_pdf_text(pdf_paths)

indices = []
for i, config in enumerate(split_configs):
    chunks_path = f"document_chunks_{i}.pkl"
    embeddings_path = f"embeddings_{i}.pkl"
    index_path = f"faiss_index_{i}"

    from langchain.text_splitter import CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        length_function=len
    )
    documents = text_splitter.create_documents([raw_text])

    if not os.path.exists(chunks_path):
        with open(chunks_path, 'wb') as f:
            pickle.dump(documents, f)
    else:
        with open(chunks_path, 'rb') as f:
            documents = pickle.load(f)

    texts = [doc.page_content for doc in documents]

    if not os.path.exists(embeddings_path):
        batch_size = 32
        embeddings_list = []
        for j in range(0, len(texts), batch_size):
            batch = texts[j:j + batch_size]
            embeddings_list.extend(embeddings.embed_documents(batch))

        with open(embeddings_path, 'wb') as f:
            pickle.dump(embeddings_list, f)
    else:
        with open(embeddings_path, 'rb') as f:
            embeddings_list = pickle.load(f)

    if not os.path.exists(index_path):
        dimension = len(embeddings_list[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings_list).astype('float32'))

        from langchain.docstore.document import Document
        docstore = InMemoryDocstore({str(k): doc for k, doc in enumerate(documents)})
        index_to_docstore_id = {k: str(k) for k in range(len(documents))}

        db = FAISS(
            embedding_function=embeddings.embed_query,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )

        db.save_local(index_path)
    else:
        db = FAISS.load_local(index_path, embeddings.embed_query, allow_dangerous_deserialization=True)

    indices.append(db)

def get_ensemble_rag_response(input_text):
    responses = []
    for db in indices:
        retriever = db.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt}
        )
        
        response = qa_chain.run(input_text)
        responses.append(response)

    final_response = " ".join(set(responses))
    return final_response

def query_ensemble(input_text):
    start_time = time.time()
    
    # Optionally reset or reinitialize the LLM here if possible.
    # llm.reset()  # Uncomment if your LLM has a reset method

    response = get_ensemble_rag_response(input_text)
    
    end_time = time.time()
    
    return response, f"Execution time: {end_time - start_time} seconds"

iface = gr.Interface(
    fn=query_ensemble,
    inputs="text",
    outputs=["text", "text"],
    title="Precise RAG",
    description="Dual context Chatbot"
)

iface.launch()
