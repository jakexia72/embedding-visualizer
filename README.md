# Embeddings Visualizer

A simple application to visualize text embeddings from CSV files using various clustering techniques.

## Features

- Upload CSV files containing text data
- Generate embeddings using OpenAI API or sentence-transformers library
- Visualize embeddings using different dimensionality reduction techniques (PCA, t-SNE, UMAP)
- Apply clustering algorithms (K-Means, DBSCAN, HDBSCAN) to identify patterns
- Interactive visualization with Plotly

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Start the application:
   ```
   python app.py
   ```
2. Open your browser and navigate to `http://localhost:5000`
3. Upload a CSV file containing text data
4. Select an embedding model (OpenAI or sentence-transformers) and clustering parameters
5. Explore the visualizations

## Available Embedding Models

### OpenAI Models

- text-embedding-3-small
- text-embedding-3-large
- text-embedding-ada-002

### Sentence-Transformers Models

- all-MiniLM-L6-v2
- all-mpnet-base-v2

## CSV Format

Your CSV file should contain at least one column with text data. The application will allow you to select which column to use for generating embeddings.

## Requirements

- Python 3.8+
- OpenAI API key
- Dependencies listed in requirements.txt
