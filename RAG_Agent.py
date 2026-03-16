from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, HumanMessage
from operator import add as add_messages
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0
    )



# Create Embedding Model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)

# Path to the PDF document
pdf_path = "Stock_Market_Performance_2024.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

# Load PDF
pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise


# Chunkking Process
text_splitter= RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


pages_split =text_splitter.split_documents(pages) #apply to all of pages

persist_directory=r"D:\Studying\LangGraph"
collection_name= "stock_market"

if  not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    vectorstore= Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print("Created ChromaDB vector store!")

except Exception as e:
    print(f"Error setting up ChomaDB: {str(e)}")
    raise

# Retriever
retreiver =vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":5} # amount of chanks return
)

@tool
def retrieve_tool(query:str)->str:
    """Tool searches for relevant information in the PDF based on the user's query."""
    docs = retreiver.invoke(query)
    
    if not docs:
        return "I found no relevant information in the stock market performance document."
    
    results=[]
    
    for i,doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
        
    return "\n\n".join(results)
tools =[retrieve_tool]
llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
def should_continue(state:AgentState):
    """Check if the last message contains tool calls."""
    result=state["messages"][-1]
    return hasattr(result,"tool_calls") and len(result.tool_calls)>0


system_prompt="".join([
    "You are am intelligent AI assistance who answers questions about Stock Market Performance in 2024 based on the provided PDF document.",
    "Use the 'retrieve_tool' to search for relevant information in the PDF document to answer the user's questions.",
    "If you need to look up some information before asking a follow up question, you are allowed to do that",
    "Please always cite the specific parts of the documents you use in answers",
    "If you don't know the answer, say you don't know and don't try to make up an answer"
                      ])

tools_dict= {our_tool.name: our_tool for our_tool in tools}


# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state and system prompt."""
    messages = list(state["messages"])
    messages= [SystemMessage(content=system_prompt)]+messages
    message = llm.invoke(messages)
    return {'messages': [message]}

# Retriever Agent
def take_action(state: AgentState)->AgentState:
    """Execute tool calls from the LLM's response."""
    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query','No query provided')}")
        
        if not t['name'] in tools_dict:
            print(f"Tool {t['name']} not found in tools_dict")
            result = "Incorrect tool name. Please check the tool name and try again."
            
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query',''))
            print(f"Result Length: {len(str(result))}")
            
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        
    print("Tools Exection Completed. Returning results to LLM.")
    return {'messages':results}

# Graph Construction
graph = StateGraph(AgentState)

graph.add_node("llm",call_llm)
graph.add_node("retriever_agent",take_action)

graph.set_entry_point("llm")

graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        True:"retriever_agent",
        False:END
    }
)
graph.add_edge("retriever_agent","llm")


app =graph.compile()

# Running the Agent in a loop
def running_agent():
    """Run the agent in a loop until it decides to end the conversation."""
    
    print("\n=== RAG Agent ===")
    
    while True:
        user_input= input("\nWhat is your question? ")
        if user_input.lower() in ["exit","quit"]:
            break
        
        messages = [HumanMessage(content=user_input)]
        result = app.invoke({"messages": messages})
        print("\n=== Agent Answer ===")
        
        final_message = result["messages"][-1]

        content = final_message.content

        if isinstance(content, list):
            print(content[0]["text"])
        else:
            print(content)
                
if __name__ == "__main__":
    running_agent()
        