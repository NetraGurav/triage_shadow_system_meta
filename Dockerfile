FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv/bin/uv

# Set up a non-root user for Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:/uv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRADIO_ANALYTICS_ENABLED="False" \
    HF_HUB_DISABLE_TELEMETRY="1"

WORKDIR /app

# Install python dependencies using uv
COPY --chown=user requirements.txt .
RUN uv pip install --no-cache -r requirements.txt

# Copy the rest of the application
COPY --chown=user . .

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
