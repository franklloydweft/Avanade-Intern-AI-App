from llama_index.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index import LangchainEmbedding, ServiceContext, StorageContext, load_index_from_storage, set_global_service_context
from llama_index.memory import ChatMemoryBuffer
from index_create import *
import openai
import logging
import sys
import os

# enable logging
logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# azure information
openai.api_type = "azure"
openai.api_base = "https://new-ai-tool.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = 'e6924cb9a78946bea8b6ad0916b544a0'

# set up LLM & embedding
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

# fetch and index data
doesExist = os.path.exists("indexes")

if not doesExist:
    prep_data()

# create global service context
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embedding_llm,
)

set_global_service_context(service_context)

# fetch index
storage_context = StorageContext.from_defaults(persist_dir="./indexes")
loaded_index = load_index_from_storage(storage_context)

# chatbot
memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
chat_engine = loaded_index.as_chat_engine(chat_mode="context", memory=memory, system_prompt="You are a chatbot that answers questions and summarizes documentation for Dow Chemical Company polyurethane products. In your answer always cite from which document you obtained information.")
message = ""
system_message = "Hello! I am a chatbot that summarizes information and answers questions about Dow Chemical Company polyurethane products. Ask me anything!"
print(system_message)
print("Press q to quit chatting with me\n")
exitState = False
while not exitState:
    message = input()
    if message == 'q' or message == 'Q':
        exitState = True
        break
    response = chat_engine.chat(message)
    print(response)
print("Goodbye!")