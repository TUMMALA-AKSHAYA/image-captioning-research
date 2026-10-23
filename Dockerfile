FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies (libgl1 replaces deprecated libgl1-mesa-glx)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code files
COPY app.py model_pipeline.py ./

# Expose FastAPI default port
EXPOSE 8000

# Command to launch production Uvicorn server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]