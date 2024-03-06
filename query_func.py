import streamlit as st
from embedding_function import generate_embeddings, find_most_similar
import pandas as pd
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("CIE 2024 A Level Psychology Query App (Consumer & Clinical)")

def generate_embeddings(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-large")
    return response.data[0].embedding

def find_most_similar(query_embedding, embeddings, top_n=3):
    similarities = np.dot(embeddings, query_embedding)
    most_similar_indices = similarities.argsort()[-top_n:][::-1]
    return most_similar_indices

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4-turbo-preview"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load the CSV file with embeddings
df = pd.read_csv("psychology_embeddings.csv")
embeddings = df['embedding'].apply(eval).tolist()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
query = st.chat_input("enter your query")

if query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)
    
    # Step 4: Query the embeddings using a GPT-based LLM API
    query_embedding = generate_embeddings(query)
    most_similar_indices = find_most_similar(query_embedding, embeddings)
    
    most_similar_subjects = df.loc[most_similar_indices, 'topic'].tolist()
    most_similar_contents = df.loc[most_similar_indices, 'content'].tolist()
    
    prompt = f"""
    You are a world-class CIE A-Level Psychology Tutor. Your task is to provide a comprehensive and well-structured answer to a student's question about psychology. Use the provided context, which is based on the syllabus content of the A-Level Psychology Exams, to formulate your response.

    I will provide the question and the top 3 most relevant syllabus contexts of CIE A level Psychology. When answering the question, keep the following in mind:
    - Ensure your response is clear, coherent, and easy to understand.
    - Use relevant psychological theories, concepts, and studies from the provided exact syllabus context to support your explanation.
    - If necessary, provide examples to illustrate complex ideas.
    - Use appropriate terminology and explain any technical terms that the student may not be familiar with.
    - Structure your answer in a logical manner, using paragraphs to separate main points.

    ### Syllabus Context ###
    1. {most_similar_subjects[0]}
    {most_similar_contents[0]}

    2. {most_similar_subjects[1]} 
    {most_similar_contents[1]}

    3. {most_similar_subjects[2]}
    {most_similar_contents[2]}

    ### Question ###
    {query}

    Answer:
    """
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            stream=True,
            temperature=0.3,
            top_p=0.2,
        )
        response = st.write_stream(stream)
        
        st.session_state.messages.append({"role": "assistant", "content": response})