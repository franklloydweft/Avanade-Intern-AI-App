from llama_index import LangchainEmbedding, VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext
from llama_index.storage.docstore import DocumentStore
from llama_index.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
import openai

def prep_data():
    #Set Azure OpenAI Credentials
    openai.api_type = "azure"
    openai.api_base = "https://new-ai-tool.openai.azure.com/"
    openai.api_version = "2023-03-15-preview"
    openai.api_key = 'e6924cb9a78946bea8b6ad0916b544a0'

    #Create variables for the Large Language Model and Embedding LLM from Azure
    llm = AzureOpenAI(engine="new-tool-gpt", model="gpt-35-turbo", temperature=0.7)
    embedding_llm = LangchainEmbedding(
        OpenAIEmbeddings(
            model="text-embedding-ada-002",
            deployment="embedding",
            openai_api_key=openai.api_key,
            openai_api_base=openai.api_base,
            openai_api_type=openai.api_type,
            openai_api_version=openai.api_version,
        ),
        embed_batch_size=1,
    )
    data = SimpleDirectoryReader(input_dir='./data').load_data()
    document_store = DocumentStore(data)

    #Create Global service context.
    from llama_index import set_global_service_context
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embedding_llm,
    )
    set_global_service_context(service_context)

    #Create Index
    index = VectorStoreIndex.from_documents(data, embedding_model = embedding_llm)
    index.storage_context.persist('./indexes')
    
    return