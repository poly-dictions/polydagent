FROM python:3.12-slim

WORKDIR /app

# Install build dependencies for native packages (lru-dict)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY api_server.py .
COPY features/ ./features/

# Expose port
EXPOSE 5777

# Run (PORT is set by Railway)
CMD ["python", "api_server.py"]
