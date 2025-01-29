from flask import Flask, request, jsonify, session
import os
import secrets
import fitz  # PyMuPDF
import openai
import faiss
import numpy as np

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Configure the OpenAI client to use the Azure endpoint and API key
openai.api_type = "azure"
openai.api_base = "https://myopenai-1345825.openai.azure.com/"
openai.api_version = "2024-05-01-preview"
openai.api_key = "8hJtwvqoeEePqE2lsMc4lyKBj64NeaUBu7MjBEorX7VZbOapQuFbJQQJ99BAACYeBjFXJ3w3AAABACOGlnz1"

def extract_texts_from_directory(pdf_directory):
    pdf_texts = {}
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            text = extract_text_from_pdf(pdf_path)
            pdf_texts[filename] = text
    return pdf_texts

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def generate_embeddings(text):
    response = openai.Embedding.create(
        input=text,
        engine="text-embedding-3-large"
    )
    return response['data'][0]['embedding']

def store_embeddings_in_faiss(embeddings):
    dimension = len(next(iter(embeddings.values())))
    index = faiss.IndexFlatL2(dimension)
    embedding_list = list(embeddings.values())
    index.add(np.array(embedding_list).astype('float32'))
    return index, list(embeddings.keys())

def query_faiss_index(index, query_embedding, k=1):
    D, I = index.search(np.array([query_embedding]).astype('float32'), k)
    return I[0]

def generate_answer_from_azure_openai(prompt, max_tokens=2048):
    response = openai.ChatCompletion.create(
        engine="compare-data-with-rules",  # Use a compatible model for chat completion
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens
    )
    return response.choices[0].message['content'].strip()

# Example usage:
pdf_directory = "/Users/venkatasaimaruthi.n/Downloads/makeathon_code/rag_inputs"
pdf_texts = extract_texts_from_directory(pdf_directory)

# Generate embeddings for the extracted text
pdf_embeddings = {filename: generate_embeddings(text) for filename, text in pdf_texts.items()}
faiss_index, filenames = store_embeddings_in_faiss(pdf_embeddings)

@app.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data['question']
    
    # Generate the query embedding
    query_embedding = generate_embeddings(question)
    
    # Query the vector database to find the most relevant documents
    nearest_neighbors_indices = query_faiss_index(faiss_index, query_embedding)
    nearest_neighbors_filenames = [filenames[i] for i in nearest_neighbors_indices]
    
    # Generate the answer based on the retrieved documents
    context_texts = [pdf_texts[filename] for filename in nearest_neighbors_filenames]
    context = " ".join(context_texts)  # Combine texts from nearest neighbors
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    answer = generate_answer_from_azure_openai(prompt)
    
    # Maintain chat history in session
    if 'chat_history' not in session:
        session['chat_history'] = []
    session['chat_history'].append({'role': 'user', 'content': question})
    session['chat_history'].append({'role': 'assistant', 'content': answer})
    
    return jsonify({'answer': answer, 'chat_history': session['chat_history']})

if __name__ == '__main__':
    app.run(debug=True)
