import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("CIE 2024 A Level Psychology Query App (Consumer & Clinical)")

def generate_embeddings(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-large")
    return response.data[0].embedding

def find_most_similar(query_embedding, embeddings, top_n=2):
    similarities = np.dot(embeddings, query_embedding)
    most_similar_indices = similarities.argsort()[-top_n:][::-1]
    return most_similar_indices

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4-0125-preview"

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
    You are an expert CIE A-Level Psychology Tutor. Your task is to provide a comprehensive, well-structured, and engaging answer to a student's question about psychology. Use the provided context, which is based on the most relevant syllabus content of the CIE A-Level Psychology Exams, to formulate your response.

    When answering the question, you must adhere to the following guidelines:

    Clarity: Ensure your response is clear, coherent, and easy to understand for an A-Level student.
    Relevance: Use relevant psychological theories, concepts, and studies from the provided syllabus context to support your explanation. Stay focused on the question and avoid irrelevant information.
    Examples: If necessary, provide real-world examples or analogies to illustrate complex ideas and help the student grasp the concepts more easily.
    Terminology: Use appropriate psychological terminology and briefly explain any technical terms that the student may not be familiar with.
    Structure: Organize your answer in a logical manner, using paragraphs to separate main points and ensure a smooth flow of information.
    Engagement: Make your response engaging and interesting to read, encouraging the student's curiosity and enthusiasm for psychology.
    Syllabus Context:

    1. {most_similar_subjects[0]} {most_similar_contents[0]}

    2. {most_similar_subjects[1]} {most_similar_contents[1]}
    Question:
    {query}

    Answer:
    """

    # Write the completed prompt to a file
    with open("one_shot_example.txt", "w") as file:
        file.write(prompt)

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
            top_p=1,
        )
        response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})