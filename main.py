from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PandasCSVReader
from llama_index.llms.ollama import Ollama

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from prompts import context

from llama_index.core import Settings

import os, time


def main():

    llm = Ollama(model="mistral", request_timeout=300.0)

    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    #if "OPENAI_API_KEY" in os.environ:
    #    del os.environ["OPENAI_API_KEY"]


    storage_dir = "storage"

    parser=PandasCSVReader()
    fe={".csv":parser}


    embedding = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    start_time2 = time.time()
    if os.path.exists(storage_dir):
        print("🔄 Cargando índice guardado...")
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        index = load_index_from_storage(storage_context)
    else:
        print("📥 Cargando archivos desde el directorio...")
        loader = SimpleDirectoryReader("./data/civitas", file_extractor=fe)
        docs = loader.load_data(num_workers=4)

        index = VectorStoreIndex.from_documents(
            docs,
            embed_model=embedding,
            show_progress=True,
            num_workers=4
        )

        index.storage_context.persist(persist_dir=storage_dir)

    #retriever = index.as_retriever(similarity_top_k=3)
    #retrieved_docs = retriever.retrieve("Documents are separated according kind of historical performance of students and it can serve as reference to give recommendations a students with similar performance")
    #content = "\n\n".join([doc.text for doc in retrieved_docs])

    query_engine = index.as_query_engine(llm=llm,embed_model=embedding,context=context)#,content=f"{content}"

    end_time2 = time.time()
    elapsed_time2 = end_time2 - start_time2
    print(f"Elapsed time: {elapsed_time2} seconds (Indexing Manage)")

    while (prompt := input("Enter a prompt (q to quit): ")) != "q":
        retries = 0

        while retries < 3:
            try:
                start_time3 = time.time()
                #query = "¿Puedes enumerar 4 recomendaciones para el alumno con codigo 244277?"
                response = query_engine.query(prompt)
                print(response)

                end_time3 = time.time()
                elapsed_time3 = end_time3 - start_time3
                print(f"Elapsed time: {elapsed_time3} seconds (Model Response Query)")

                break
            except Exception as e:
                retries += 1
                print(f"Error occured, retry #{retries}:", e)

        if retries >= 3:
            print("Unable to process request, try again...")
            continue

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")



#for doc in docs:
#    if doc.metadata['file_type']=="text/csv":
#        print(doc)
