import os
import pandas as pd
import numpy as np
import json
import pickle
from flask import Flask, render_template, request, jsonify, session
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
import hdbscan
import umap
from werkzeug.utils import secure_filename
import uuid
import httpx
from openai import OpenAI
from dotenv import load_dotenv
import glob
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.secret_key = os.urandom(24)
app.config["UPLOAD_FOLDER"] = "data"
app.config["EMBEDDINGS_FOLDER"] = "embeddings"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload

# Create necessary folders if they don't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["EMBEDDINGS_FOLDER"], exist_ok=True)
os.makedirs(os.path.join(app.static_folder or "static"), exist_ok=True)

# Initialize OpenAI client with a custom HTTP client to avoid the proxies issue
api_key = os.getenv("OPENAI_API_KEY")
http_client = httpx.Client()
client = OpenAI(api_key=api_key, http_client=http_client)

# Available embedding models
EMBEDDING_MODELS = {
    "text-embedding-3-small": "OpenAI text-embedding-3-small",
    "text-embedding-3-large": "OpenAI text-embedding-3-large",
    "text-embedding-ada-002": "OpenAI text-embedding-ada-002",
    "all-MiniLM-L6-v2": "Sentence-Transformers all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "Sentence-Transformers all-mpnet-base-v2",
}

# Available dimensionality reduction techniques
DIM_REDUCTION = {"PCA": "PCA", "TSNE": "t-SNE", "UMAP": "UMAP"}

# Available clustering algorithms
CLUSTERING = {"kmeans": "K-Means", "dbscan": "DBSCAN", "hdbscan": "HDBSCAN"}


@app.route("/")
def index():
    # Get list of saved embeddings
    saved_embeddings = get_saved_embeddings()

    return render_template(
        "index.html",
        embedding_models=EMBEDDING_MODELS,
        dim_reduction=DIM_REDUCTION,
        clustering=CLUSTERING,
        saved_embeddings=saved_embeddings,
    )


def get_saved_embeddings():
    """Get list of saved embeddings"""
    embedding_files = glob.glob(os.path.join(app.config["EMBEDDINGS_FOLDER"], "*.pkl"))
    saved_embeddings = []

    for file_path in embedding_files:
        try:
            with open(file_path, "rb") as f:
                metadata = pickle.load(f)

            # Extract filename without path and extension
            filename = os.path.basename(file_path).replace(".pkl", "")

            saved_embeddings.append(
                {
                    "id": filename,
                    "name": metadata.get("name", filename),
                    "model": metadata.get("model", "unknown"),
                    "date": metadata.get("date", "unknown"),
                    "num_documents": metadata.get("num_documents", 0),
                    "columns": metadata.get("columns", []),
                }
            )
        except Exception as e:
            print(f"Error loading embedding file {file_path}: {str(e)}")

    # Sort by date (newest first)
    saved_embeddings.sort(
        key=lambda x: x["date"] if isinstance(x["date"], str) else "", reverse=True
    )

    return saved_embeddings


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith(".csv"):
        # Generate a unique ID for this session
        session_id = str(uuid.uuid4())
        session["session_id"] = session_id

        # Save the file with a secure filename
        filename = secure_filename(file.filename)
        file_path = os.path.join(
            app.config["UPLOAD_FOLDER"], f"{session_id}_{filename}"
        )
        file.save(file_path)

        # Read the CSV to get column names
        df = pd.read_csv(file_path)
        columns = df.columns.tolist()

        # Store the file path and original filename in the session
        session["file_path"] = file_path
        session["original_filename"] = filename
        session["columns"] = columns
        # Don't store the DataFrame in the session anymore
        # session["df"] = df.to_json(orient="records")

        return jsonify({"success": True, "columns": columns})

    return jsonify({"error": "File must be a CSV"}), 400


@app.route("/load_embedding", methods=["POST"])
def load_embedding():
    data = request.json
    embedding_id = data.get("embedding_id")

    if not embedding_id:
        return jsonify({"error": "No embedding ID provided"}), 400

    embedding_path = os.path.join(
        app.config["EMBEDDINGS_FOLDER"], f"{embedding_id}.pkl"
    )

    if not os.path.exists(embedding_path):
        return jsonify({"error": "Embedding file not found"}), 404

    try:
        with open(embedding_path, "rb") as f:
            embedding_data = pickle.load(f)

        # Store the embedding data in the session
        session["embedding_data"] = embedding_id
        session["columns"] = embedding_data.get("columns", [])

        # Store the original dataframe if available
        if "original_data" in embedding_data:
            session["df"] = json.dumps(embedding_data["original_data"])

        return jsonify(
            {
                "success": True,
                "columns": embedding_data.get("columns", []),
                "text_column": embedding_data.get("text_column", ""),
                "model": embedding_data.get("model", ""),
            }
        )
    except Exception as e:
        return jsonify({"error": f"Error loading embedding: {str(e)}"}), 500


def get_openai_embeddings(texts, model_name):
    """Generate embeddings using OpenAI API"""
    embeddings = []

    # Process in batches to avoid API limits
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        try:
            # Use the OpenAI API as documented
            response = client.embeddings.create(
                model=model_name, input=batch, encoding_format="float"
            )
            # Extract embeddings from response
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            raise

    return np.array(embeddings)


def save_embeddings(
    embeddings,
    texts,
    model_name,
    text_column,
    columns,
    original_data=None,
    original_filename=None,
):
    """Save embeddings to disk"""
    # Generate a unique ID if not provided
    embedding_id = str(uuid.uuid4())

    # Create metadata
    metadata = {
        "name": original_filename or f"Embeddings {embedding_id[:8]}",
        "model": model_name,
        "date": datetime.now().isoformat(),
        "num_documents": len(texts),
        "text_column": text_column,
        "columns": columns,
        "embeddings": embeddings,
        "texts": texts,
        "original_data": original_data,
    }

    # Save to disk
    embedding_path = os.path.join(
        app.config["EMBEDDINGS_FOLDER"], f"{embedding_id}.pkl"
    )
    with open(embedding_path, "wb") as f:
        pickle.dump(metadata, f)

    return embedding_id


@app.route("/process", methods=["POST"])
def process_data():
    data = request.json
    text_column = data.get("text_column")
    model_name = data.get("model")
    dim_reduction = data.get("dim_reduction")
    clustering_method = data.get("clustering")

    # Handle empty values with defaults
    try:
        n_clusters = int(data.get("n_clusters", 5))
    except (ValueError, TypeError):
        n_clusters = 5

    try:
        eps = float(data.get("eps", 0.5))
    except (ValueError, TypeError):
        eps = 0.5

    try:
        min_samples = int(data.get("min_samples", 5))
    except (ValueError, TypeError):
        min_samples = 5

    save_embedding = data.get("save_embedding", False)
    embedding_id = data.get("embedding_id")

    # Check if we're using a saved embedding or generating new ones
    if embedding_id:
        # Load saved embedding
        embedding_path = os.path.join(
            app.config["EMBEDDINGS_FOLDER"], f"{embedding_id}.pkl"
        )

        if not os.path.exists(embedding_path):
            return jsonify({"error": "Embedding file not found"}), 404

        try:
            with open(embedding_path, "rb") as f:
                embedding_data = pickle.load(f)

            embeddings = embedding_data.get("embeddings")
            texts = embedding_data.get("texts")
            original_data = embedding_data.get("original_data")

            if embeddings is None or texts is None:
                return jsonify({"error": "Invalid embedding data"}), 500

            # Convert original_data to DataFrame if available
            if original_data:
                df = pd.DataFrame(original_data)
            else:
                # Create a simple DataFrame with just the text
                df = pd.DataFrame({"text": texts})
        except Exception as e:
            return jsonify({"error": f"Error loading embedding: {str(e)}"}), 500
    else:
        # Generate new embeddings
        if "file_path" not in session:
            return jsonify({"error": "No file uploaded"}), 400

        # Load the data
        file_path = session["file_path"]
        df = pd.read_csv(file_path)

        if text_column not in df.columns:
            return jsonify({"error": f"Column {text_column} not found in CSV"}), 400

        # Get the text data
        texts = df[text_column].fillna("").tolist()

        # Generate embeddings based on the selected model
        try:
            if model_name.startswith("text-embedding"):
                # Use OpenAI embeddings
                embeddings = get_openai_embeddings(texts, model_name)
            else:
                # Use sentence-transformers
                model = SentenceTransformer(model_name)
                embeddings = model.encode(texts, show_progress_bar=True)
        except Exception as e:
            return jsonify({"error": f"Error generating embeddings: {str(e)}"}), 500

        # Save embeddings if requested
        if save_embedding:
            original_filename = session.get("original_filename", "Untitled")
            columns = session.get("columns", [])
            # Convert DataFrame to list of records for serialization
            original_data = df.to_dict(orient="records")
            embedding_id = save_embeddings(
                embeddings,
                texts,
                model_name,
                text_column,
                columns,
                original_data,
                original_filename,
            )

    # Apply dimensionality reduction
    n_components = 2

    if dim_reduction == "PCA":
        reducer = PCA(n_components=n_components)
        reduced_data = reducer.fit_transform(embeddings)
    elif dim_reduction == "TSNE":
        reducer = TSNE(n_components=n_components, random_state=42)
        reduced_data = reducer.fit_transform(embeddings)
    elif dim_reduction == "UMAP":
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        reduced_data = reducer.fit_transform(embeddings)
    else:
        return jsonify({"error": "Invalid dimensionality reduction method"}), 400

    # Apply clustering
    if clustering_method == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = clusterer.fit_predict(embeddings)
    elif clustering_method == "dbscan":
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = clusterer.fit_predict(embeddings)
    elif clustering_method == "hdbscan":
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_samples)
        clusters = clusterer.fit_predict(embeddings)
    else:
        return jsonify({"error": "Invalid clustering method"}), 400

    # Create a DataFrame with the results
    result_df = pd.DataFrame(
        {
            "x": reduced_data[:, 0],
            "y": reduced_data[:, 1],
            "cluster": clusters.tolist(),
            "text": texts,
        }
    )

    # Merge with original data for hover information
    # First, ensure the indices match
    if len(df) == len(result_df):
        # Add all columns from original data for hover information
        for col in df.columns:
            if col not in result_df.columns:
                result_df[col] = df[col].values

    # Get cluster summaries for the table
    unique_clusters = sorted(list(set(clusters.tolist())))
    cluster_summaries = []

    for cluster_id in unique_clusters:
        cluster_items = result_df[result_df["cluster"] == cluster_id]
        cluster_size = len(cluster_items)
        # Get a sample of items from this cluster (up to 5)
        sample_items = (
            cluster_items.sample(min(5, cluster_size))
            if cluster_size > 0
            else pd.DataFrame()
        )

        cluster_summaries.append(
            {
                "cluster_id": cluster_id,
                "size": cluster_size,
                "sample_items": sample_items.to_dict(orient="records"),
            }
        )

    # Prepare data for D3.js visualization
    visualization_data = {
        "points": result_df.to_dict(orient="records"),
        "is_3d": False,
        "clusters": unique_clusters,
        "cluster_summaries": cluster_summaries,
        "model": model_name,
        "dim_reduction": DIM_REDUCTION[dim_reduction],
        "clustering": CLUSTERING[clustering_method],
    }

    response_data = {
        "success": True,
        "visualization_data": visualization_data,
        "stats": {
            "num_documents": len(texts),
            "num_clusters": len(set(clusters)) - (1 if -1 in clusters else 0),
            "noise_points": (
                sum(1 for c in clusters if c == -1) if -1 in clusters else 0
            ),
        },
    }

    # Add embedding_id to response if saved
    if save_embedding and embedding_id:
        response_data["embedding_id"] = embedding_id

    return jsonify(response_data)


@app.route("/saved_embeddings", methods=["GET"])
def saved_embeddings():
    """Get list of saved embeddings"""
    saved_embeddings = get_saved_embeddings()
    return jsonify({"success": True, "embeddings": saved_embeddings})


if __name__ == "__main__":
    app.run(debug=True)
