import streamlit as st
from langchain_core.messages import HumanMessage
from RAG_Agent import app   # import your compiled LangGraph app


st.set_page_config(page_title="Stock Market RAG Agent")

st.title("📈 Stock Market RAG Assistant")
st.write("Ask questions about Stock Market Performance 2024")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Ask a question...")

if user_input:

    # Show user message
    st.chat_message("user").markdown(user_input)

    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    # Call LangGraph agent
    result = app.invoke({
        "messages": [HumanMessage(content=user_input)]
    })

    final_message = result["messages"][-1].content

    if isinstance(final_message, list):
        answer = final_message[0]["text"]
    else:
        answer = final_message

    # Show assistant answer
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )