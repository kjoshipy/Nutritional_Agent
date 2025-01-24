from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_open_ai_functions_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

# Configuration: Loader, Splitter, and Retriever
def setup_retriever():
    """Sets up the retriever for calorie-related information."""
    loader = WebBaseLoader("https://example.com/nutrition-data")  # Replace with an actual source
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    split_docs = splitter.split_documents(docs)

    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embedding=embedding)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# Initialize Language Model
def setup_model():
    """Initializes the OpenAI model for the assistant."""
    return ChatOpenAI(
        model='gpt-3.5-turbo-1106',  # Replace with your preferred model
        temperature=0.7
    )

# Prompt Template for Calorie Assistant
def setup_prompt():
    """Defines the system instructions and user interaction prompts."""
    return ChatPromptTemplate(
        ("system", "You are a friendly calorie-counting assistant named CalBot. Your goal is to estimate calorie counts from images, provide nutritional advice, and answer food-related questions."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    )

# Tools for Searching and Retrieval
def setup_tools(retriever):
    """Creates tools for retrieving information."""
    search = TavilySearchResults()
    retriever_tool = create_retriever_tool(
        retriever,
        "nutrition_search",
        "Use this tool for retrieving calorie and nutritional information."
    )
    return [search, retriever_tool]

# Create the Chat Agent
def create_agent(model, prompt, tools):
    """Creates the agent with the specified model, prompt, and tools."""
    return create_open_ai_functions_agent(llm=model, prompt=prompt, tools=tools)

# Create the Agent Executor
def create_agent_executor(agent, tools):
    """Creates the agent executor."""
    return AgentExecutor(agent=agent, tools=tools)

# Chat Process Function
def process_chat(agent_executor, user_input, chat_history):
    """Processes the user input and returns the assistant's response."""
    response = agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    return response["output"]

# Main Application Loop
if __name__ == '__main__':
    retriever = setup_retriever()
    model = setup_model()
    prompt = setup_prompt()
    tools = setup_tools(retriever)
    agent = create_agent(model, prompt, tools)
    agent_executor = create_agent_executor(agent, tools)

    chat_history = []

    print("Welcome to CalBot! Type 'exit' to end the session.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        response = process_chat(agent_executor, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

        print(f"CalBot: {response}")
