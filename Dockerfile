# Lightweight Python 3.12 base image
FROM python:3.12-slim

# Prevent bytecode generation and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install runtime system dependencies required for video/audio processing and SQLite
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python project dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Copy application code
COPY . .

# Ensure persistent directories exist
RUN mkdir -p database chromadb downloads

# Expose Gradio default port
EXPOSE 7860

# Launch interactive chat server bound to all interfaces for container networking
CMD ["python", "main.py", "chat", "--server-name", "0.0.0.0", "--server-port", "7860"]
