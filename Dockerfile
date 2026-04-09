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
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    GRADIO_ANALYTICS_ENABLED="False" \
    HF_HUB_DISABLE_TELEMETRY="1"

WORKDIR /app

# Create a virtual environment owned by user 1000 to safely store packages
ENV VIRTUAL_ENV=/home/user/venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install python dependencies using uv
COPY --chown=user requirements.txt .
RUN uv pip install --no-cache -r requirements.txt

# Copy the rest of the application
COPY --chown=user . .

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
