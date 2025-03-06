#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Ensure the OpenAI API key is set from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded environment variables from .env file"
else
    echo "Warning: .env file not found"
fi

# Run the Flask application with the Python interpreter from the virtual environment
python3 app.py 