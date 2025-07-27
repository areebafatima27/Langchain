from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# Initialize the Hugging Face LLM using Mistral 7B Instruct
llm = HuggingFaceEndpoint(
    repo_id='mistralai/Mistral-7B-Instruct-v0.2',
    task='text-generation',
    # model_kwargs={"temperature": 0.7, "max_new_tokens": 512, "return_full_text": False}
)

chat_model = ChatHuggingFace(llm=llm)

# Setup system message and chat history
chat_history = [SystemMessage(content="You are a helpful assistant.")]

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Add user message
    chat_history.append(HumanMessage(content=user_input))

    # Get AI response
    ai_response = chat_model.invoke(chat_history)
    print("AI:", ai_response.content.strip())

    # Save AI response in history
    chat_history.append(AIMessage(content=ai_response.content.strip()))

# Print full chat history
print("\n=== Full Chat History ===")
for line in chat_history:
    role = "User" if isinstance(line, HumanMessage) else "AI" if isinstance(line, AIMessage) else "System"
    print(f"{role}: {line.content}")
