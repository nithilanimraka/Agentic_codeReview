# Stage 1: Base Image with the specific Python version you requested
FROM python:3.12.10-slim

# Set the working directory inside the container
WORKDIR /app

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies that might be needed for your Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# --- Model Caching Stage ---
# Copy and run the model download script
# We need to pass the HF_TOKEN at build time.
COPY download_model.py .
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
ARG GOOGLE_API_KEY
ENV GOOGLE_API_KEY=${GOOGLE_API_KEY}
RUN python download_model.py

# Copy the rest of your application source code into the container
COPY . .

# Initialize git submodules to fetch the parser source code
RUN git submodule update --init --recursive

# Build the tree-sitter parsers inside the container for the correct architecture
RUN python src/duplicate_check/build_parsers.py

# Build the FAISS index inside the container for the correct architecture
RUN python build_faiss_index.py


# Copy the startup script and make it executable
COPY start.sh .
RUN chmod +x start.sh

# Expose the port your FastAPI app runs on (as seen in your main.py)
EXPOSE 8000

# The command to run your application when the container starts
# Note: --host 0.0.0.0 is crucial to make it accessible from outside the container
# Set the startup script as the command to run
CMD ["./start.sh"]