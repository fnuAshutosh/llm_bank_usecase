# Banking LLM API - Production Ready
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements/base.txt requirements/prod.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r base.txt && \
    pip install --no-cache-dir -r prod.txt

# Download spaCy model for PII detection
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY src/ /app/src/
COPY config/ /app/config/

# Create logs directory
RUN mkdir -p /app/logs

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
