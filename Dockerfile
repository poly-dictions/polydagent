FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY launchpad_server.py .

# Expose port
EXPOSE 5777

# Run (PORT is set by Railway)
CMD ["sh", "-c", "uvicorn launchpad_server:app --host 0.0.0.0 --port ${PORT:-5777}"]
