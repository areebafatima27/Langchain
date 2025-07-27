from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='HuggingFaceH4/zephyr-7b-beta',
    task='text-generation'  
)

model = ChatHuggingFace(llm=llm)
system_prompt = "<|system|>\nYou are a helpful, conversational AI assistant.\n"
messages= [
    SystemMessage(content='You are a helpful assistant.'),
    HumanMessage(content="Tell me about Langchain.")
]
result = model.invoke(messages)
messages.append(AIMessage(result.content ))
print(messages)