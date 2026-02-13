# Use slim base for smaller image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (cache layer)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire src + other needed files
COPY src/ ./src/
COPY data/ ./data/

# Default command: run main.py (override with args in K8s)
CMD ["python", "src/main.py"]
