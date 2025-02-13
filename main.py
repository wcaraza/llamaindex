from fastapi import FastAPI,Request,Query
from pydantic import BaseModel

#from civitas_poc8 import main
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate, StorageContext, load_index_from_storage
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from prompts import context

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello, world!"}

class ChatInput(BaseModel):
    query: str
    university_name: str

@app.post("/query/")
async def query_llm(chat: ChatInput):
    """
    Endpoint para hacer consultas al LLM.
    """
    try:
        llm = Ollama(model="mistral", request_timeout=30.0)

        Settings.llm = llm
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-mpnet-base-v2"  # all-MiniLM-L6-v2
        )
        storage_dir = "storage_pdf"

        embedding = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

        prompt_template = PromptTemplate(context)
        prompt_context = prompt_template.format(university_name=chat.university_name)

        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        vector_index = load_index_from_storage(storage_context)

        query_engine = vector_index.as_query_engine(
            llm=llm,
            embed_model=embedding,
            context=prompt_context)
        response = query_engine.query(chat.query)
        #response = main().query(prompt)
        #response = llm.complete(prompt)  # Llama al modelo LLM
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

