from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver  
from langgraph.graph import add_messages
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_DIRECTORY = ".\chroma_db2"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
parser = StrOutputParser()

class ChatBotState(TypedDict):
    # 'add_messages' ensures new messages are appended to history
    messages: Annotated[list[BaseMessage], add_messages] 
    retri_chunks: list

def retriever_node(state: ChatBotState):
    # 1. Load existing DB (don't use from_documents here, that creates a new one)
    db = Chroma(persist_directory=CHROMA_DB_DIRECTORY, embedding_function=embeddings)
    
    retriever = db.as_retriever(search_kwargs={"k": 2})
    
    # 2. Extract the query text from the last message in the list
    last_message = state["messages"][-1]
    query_text = last_message.content
    
    # 3. Retrieve
    retriever_chunks = retriever.invoke(query_text)
    
    # 4. Return a dict to update the state key 'retri_chunks'
    return {"retri_chunks": retriever_chunks}

    
def chat_node(state: ChatBotState):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash") 
    
    # Get retrieved docs and last user message
    chunks = state["retri_chunks"]
    last_message = state["messages"][-1]
    query_text = last_message.content

    prompt = PromptTemplate(
       template = """
            You are the Tara InfoTech Help ChatBot, a professional customer support assistant.

            Instructions:
            1. Answer the user's question strictly based ONLY on the provided context below.
            2. Do not use outside knowledge or make up information.
            3. If the answer is not found in the context, politely state: "I'm sorry, but I don't have that information in my current records."
            4. Keep your answer concise and helpful.
            5. provide summary of the context if relevant.
            
            Context Information:
            {chunks}

            User Query:
            {Query}
            """,
        input_variables=["Query", "chunks"]
    )
    
    chain = prompt | llm | parser
    
    response = chain.invoke({"Query": query_text, "chunks": chunks})
    
    # Return dict to update 'messages'
    return {"messages": [AIMessage(content=response)]}


checkpointer = InMemorySaver()

graph = StateGraph(ChatBotState)

graph.add_node("retriever_node", retriever_node)
graph.add_node("chat_node", chat_node)

graph.add_edge(START, "retriever_node")
graph.add_edge("retriever_node", "chat_node")
graph.add_edge("chat_node", END)

chatBot = graph.compile(checkpointer=checkpointer)
