FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --timeout=60 --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt

COPY src/ src/
COPY models/ models/

# Set environment variable to prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose a port (if we later add a FastAPI/Flask API)
EXPOSE 8000

# Default command to run inference
CMD ["python", "src/inference.py"]
