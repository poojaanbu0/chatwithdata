# chatwithdata

# Product-Description-Bot
The Product Description Retrieval Bot is an innovative solution designed for e-commerce platforms focused on technical products. This chatbot revolutionizes the way businesses interact with customers by providing instant access to detailed product descriptions extracted from comprehensive PDF documents. By leveraging advanced natural language processing techniques, the bot intelligently processes user queries and efficiently navigates through the document's content, delivering precise and relevant information in real time. This capability not only streamlines the shopping experience for customers but also empowers businesses to present their product offerings more effectively. The bot enhances customer engagement by answering inquiries related to specifications, features, and comparisons, ultimately contributing to higher sales and improved customer satisfaction through informed and timely responses.

```
pip install sentence-transformers faiss-cpu PyMuPDF


# Step 1: Extract Text from PDF File
def extract_text_from_pdf(pdf_file):
    text = ""
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()  # Extract text from each page
    return text

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()
def split_text(text):
    return sent_tokenize(text)  # Split text into sentences using NLTK


def create_embeddings(text_chunks, model):
    embeddings = model.encode(text_chunks, convert_to_numpy=True)  # Create embeddings for each text chunk
    return embeddings


def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # Get the dimensionality of the embeddings
    index = faiss.IndexFlatIP(dimension)  # Using Inner Product (Cosine Similarity)
    faiss.normalize_L2(embeddings)  # Normalize embeddings for cosine similarity
    index.add(embeddings)  # Add the embeddings to the index
    return index


def retrieve_similar_sentences(query, model, index, text_chunks, threshold=0.5):
    query_embedding = model.encode([query], convert_to_numpy=True)  # Create an embedding for the query
    faiss.normalize_L2(query_embedding)  # Normalize the query embedding for cosine similarity
    distances, indices = index.search(query_embedding, 3)  # Search for the top 3 nearest embeddings
    results = [text_chunks[i] for i, distance in zip(indices[0], distances[0]) if distance > threshold]  # Apply threshold
    return results if results else ["No relevant sentences found."]

def curate_response(similar_sentences, system_message):
    if system_message == "summarize":
        summarizer = pipeline("summarization")  # Use a summarization model
        summarized_text = summarizer(' '.join(similar_sentences), max_length=130, min_length=30, do_sample=False)
        return summarized_text[0]['summary_text']

    elif system_message == "detailed response":
        return '\n\n'.join(similar_sentences)

    elif system_message == "insights only":
        insights = [sentence for sentence in similar_sentences if sentence.startswith('-')]  # Extract bullet points
        return '\n'.join(insights) if insights else "No key insights found."

    else:
        return '\n\n'.join(similar_sentences)

!pip install PyPDF2
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import re  # Import for regular expressions
from nltk.tokenize import sent_tokenize  # Import for sentence tokenization

import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import re  # Import for regular expressions
from nltk.tokenize import sent_tokenize  # Import for sentence tokenization
import nltk  # Import NLTK to download resources

# Download the 'punkt' tokenizer
nltk.download('punkt')

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    text = ""
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()  # Extract text from each page
    return text

# Function to clean and preprocess text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()

# Function to split text into sentences
def split_text(text):
    return sent_tokenize(text)  # Split text into sentences using NLTK

# Other necessary functions (create_embeddings, build_faiss_index, retrieve_similar_sentences, curate_response)

def main():
    # Step 1: Extract text from a PDF file
    pdf_file = '/content/PD.pdf'
    raw_text = extract_text_from_pdf(pdf_file)

    # Step 2: Clean and preprocess the text
    cleaned_text = clean_text(raw_text)

    # Step 3: Split the text into sentences
    text_chunks = split_text(cleaned_text)

    # Step 4: Load the embedding model and create embeddings for each sentence
    model = SentenceTransformer('paraphrase-mpnet-base-v2')
    embeddings = create_embeddings(text_chunks, model)

    # Step 5: Build the FAISS index
    index = build_faiss_index(embeddings)

    print("Welcome to the RAG Bot!")
    print("Ask your question, or type 'exit' to quit.")
    system_message = 'detailed response'

    while True:
        query = input("\nYou: ")
        if query.lower() == 'exit':
            break

        # Step 6: Retrieve relevant sentences based on the query
        similar_sentences = retrieve_similar_sentences(query, model, index, text_chunks)

        # Step 7: Curate the response based on system message
        curated_response = curate_response(similar_sentences, system_message)

        # Display result
        print(f"Bot:\n{curated_response}\n")

if __name__ == "__main__":
    main()

```

## output:
![image](https://github.com/user-attachments/assets/b010a1e4-b886-445b-a8a1-478c6a1be600)

