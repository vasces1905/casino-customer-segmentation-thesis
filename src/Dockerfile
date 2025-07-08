# Use Python 3.10 base image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements.txt from parent directory
COPY ../requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files from ./src
COPY . .

# Default command to run main module
CMD ["python", "main.py"]
