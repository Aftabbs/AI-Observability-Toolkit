from langchain_groq import ChatGroq
from src import ObservabilityCallback

# Initialize the callback
callback = ObservabilityCallback(
    session_id="my_session_123",
    metadata={"user_id": "user_456"}
)

# Use with any LangChain LLM
llm = ChatGroq(model="openai/gpt-oss-120b", callbacks=[callback])
response = llm.invoke("What is observability?")
print(response)