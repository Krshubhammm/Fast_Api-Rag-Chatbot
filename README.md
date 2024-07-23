# Fast_Api-Rag-Chatbot
conversational PDF chatbot using Open AI and FastApi
# PDF Chatbot

## Description

PDF Chatbot is an intelligent conversational AI system that allows users to upload PDF documents and engage in question-answering interactions based on the document's content. Built with FastAPI and leveraging OpenAI's powerful language models, this application demonstrates the practical implementation of Retrieval-Augmented Generation (RAG) in a real-world scenario.

## Features

- PDF Upload: Users can upload PDF documents to the system.
- Text Extraction: Automatically extracts text content from uploaded PDFs.
- Intelligent Chunking: Splits extracted text into manageable chunks for processing.
- Vector Embedding: Creates and stores vector embeddings of text chunks for efficient retrieval.
- Conversational AI: Enables users to ask questions about the uploaded document and receive contextually relevant answers.
- RAG Implementation: Utilizes Retrieval-Augmented Generation to provide accurate and context-aware responses.

## Technology Stack

- FastAPI: For creating robust and high-performance API endpoints.
- OpenAI API: Leverages GPT-3.5-turbo for natural language understanding and generation.
- LangChain: Facilitates the creation of the conversational retrieval chain.
- FAISS: Efficient similarity search and clustering of dense vectors.
- PyPDF: For extracting text from PDF documents.
- Pydantic: Data validation and settings management using Python type annotations.
- Python-dotenv: Management of environment variables.

## Installation

1. Clone this repository:
2. Install the required dependencies:
pip install -r requirements.txt
3. Set up your OpenAI API key in a `.env` file:
OPENAI_API_KEY=your_api_key_here

## Usage

1. Start the FastAPI server:

3. Access the API documentation at `http://localhost:8000/docs`

4. Use the `/upload` endpoint to upload a PDF file.

5. Use the `/query` endpoint to ask questions about the uploaded PDF.

## API Endpoints

- `GET /`: Root endpoint, returns a welcome message.
- `POST /upload`: Uploads a PDF file and processes it for querying.
- `POST /query`: Accepts a query about the uploaded PDF and returns an AI-generated response.

## Testing

Run the unit tests using pytest:

