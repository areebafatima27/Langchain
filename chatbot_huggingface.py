from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='HuggingFaceH4/zephyr-7b-beta',
    task='text-generation'  
)

model = ChatHuggingFace(llm=llm)
chat_history = []
system_prompt = "<|system|>\nYou are a helpful, conversational AI assistant.\n"

# while True:
#     user_input = input('You: ')
#     chat_history.append(user_input)
#     if user_input == 'exit':
#        break
#     result =  model.invoke(user_input)
#     chat_history.append(result.content)
#     print("AI: ", result.content)
# print(chat_history)

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Add user input to history
    chat_history.append(f"<|user|>\n{user_input}")

    # Combine into full prompt
    full_prompt = system_prompt + "\n".join(chat_history) + "\n<|assistant|>"

    # Generate response
    response = model.invoke(full_prompt)
    answer = response.content.strip()

    # Add response to history
    chat_history.append(f"<|assistant|>\n{answer}")

    print("AI:", answer)

# Optional: show full chat history at the end
print("\n=== Full Chat History ===")
for line in chat_history:
    print(line)