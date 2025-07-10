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
RUN python download_model.py

# Copy the rest of your application source code into the container
COPY . .

# Expose the port your FastAPI app runs on (as seen in your main.py)
EXPOSE 8000

# The command to run your application when the container starts
# Note: --host 0.0.0.0 is crucial to make it accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]