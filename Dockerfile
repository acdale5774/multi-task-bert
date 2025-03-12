# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependencies and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code and models
COPY src/ src/
COPY models/ models/

# Copy the entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose FastAPI port (if needed later)
EXPOSE 8000

# Set the entrypoint script as the default command
ENTRYPOINT ["/app/entrypoint.sh"]
