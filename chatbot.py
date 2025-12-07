# Import necessary libraries for building the chatbot API
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import sqlite3
from datetime import datetime

# Create a new FastAPI application instance
app = FastAPI()

# Load a pre-trained question-answering model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Establish a connection to the SQLite database
conn = sqlite3.connect('chatbot_logs.db', check_same_thread=False)
cursor = conn.cursor()

# Create a table to store chatbot logs if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_question TEXT,
        bot_answer TEXT,
        timestamp TEXT
    )
''')
conn.commit()

# Define a sample context for the chatbot to answer questions from
faq_context = """
    Welcome to our customer support. 
    We offer 24/7 support on all products. 
    For shipping information, orders take 3-5 business days. 
    Returns are accepted within 30 days. 
    Contact email: support@example.com.
"""

# Define a Pydantic model for the query input
class Query(BaseModel):
    question: str

# Define an endpoint to ask the chatbot a question
@app.post("/ask")
async def ask_bot(query: Query):
    # Use the NLP model to answer the question based on the context
    result = qa_pipeline(question=query.question, context=faq_context)
    answer = result['answer']

    # Log the interaction to the SQLite database
    cursor.execute("INSERT INTO logs (user_question, bot_answer, timestamp) VALUES (?, ?, ?)", 
                   (query.question, answer, datetime.now().isoformat()))
    conn.commit()

    # Return the chatbot's answer
    return {"answer": answer}
