# Use official lightweight Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install build dependencies (needed for compiling some python packages if necessary)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app, source code, and pre-trained models
COPY app/ app/
COPY src/ src/
COPY outputs/models/ outputs/models/

# Expose default port
EXPOSE 8000

# Run FastAPI using uvicorn (binds to 0.0.0.0 and reads port dynamically from environment)
CMD ["sh", "-c", "uvicorn app.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
