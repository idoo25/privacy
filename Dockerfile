# PrivacyFlow Dockerfile
# Privacy-preserving person detection system

# Base image - use slim for smaller size
FROM python:3.9-slim-bullseye

# Metadata
LABEL maintainer="PrivacyFlow"
LABEL description="Privacy-preserving person detection and flow analysis"
LABEL version="1.0.0"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Video capture
    libv4l-dev \
    v4l-utils \
    # Build tools (for some pip packages)
    gcc \
    g++ \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash privacyflow

# Set working directory
WORKDIR /app

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=privacyflow:privacyflow . .

# Create necessary directories
RUN mkdir -p /app/data/exports /app/models /app/logs && \
    chown -R privacyflow:privacyflow /app

# Download model placeholders (replace with actual models in production)
RUN python scripts/download_models.py --lightweight

# Switch to non-root user
USER privacyflow

# Expose API port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/v1/health || exit 1

# Default command
CMD ["python", "main.py", "--config", "config.yaml", "--station-id", "station_001", "--api-port", "5000"]
