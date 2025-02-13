from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate, StorageContext, load_index_from_storage
from llama_index.core.embeddings import resolve_embed_model
from dotenv import load_dotenv
from llama_index.core import Settings,PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from prompts import context

#from llama_index.core.prompts import LangchainPromptTemplate
#from langchain import hub



import os, time


load_dotenv()





def main():

    llm = Ollama(model="mistral", request_timeout=30.0)

    Settings.llm = llm
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-mpnet-base-v2"  # all-MiniLM-L6-v2
    )
    storage_dir = "storage_pdf"
    parser = LlamaParse(result_type="markdown")
    file_extractor = {".pdf": parser}


    embedding = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    if os.path.exists(storage_dir):
        print("🔄 Cargando índice guardado...")
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        vector_index = load_index_from_storage(storage_context)
    else:
        print("📥 Cargando archivos desde el directorio...")
        loader = SimpleDirectoryReader("./data", file_extractor=file_extractor)
        docs = loader.load_data(num_workers=5)


        vector_index = VectorStoreIndex.from_documents(
            docs,
            embed_model=embedding,
            show_progress=True,
            num_workers=5
        )

        vector_index.storage_context.persist(persist_dir=storage_dir)

    prompt_template = PromptTemplate(context)
    prompt_context = prompt_template.format(university_name="Chicago")

    #langchain_prompt = hub.pull("rlm/rag-prompt")
    #res_dict = self.pull_repo("rlm/rag-prompt")

    # lc_prompt_tmpl = LangchainPromptTemplate(
    #      template=langchain_prompt,
    #      template_var_mappings={"query_str": "question", "context_str": "context"},
    # )

    query_engine = vector_index.as_query_engine(
        llm=llm,
        embed_model=embedding,
        context=prompt_context)

    while (prompt := input("Enter a prompt (q to quit): ")) != "q":
        retries = 0

        while retries < 3:
            try:
                start_time2 = time.time()
                result = query_engine.query(prompt)
                print(result)
                end_time2 = time.time()
                elapsed_time2 = end_time2 - start_time2
                print(f"Elapsed time: {elapsed_time2} seconds (Model Response Query)")
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