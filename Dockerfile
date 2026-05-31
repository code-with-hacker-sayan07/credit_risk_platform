# Use official Python lightweight image
FROM python:3.10-slim

# Set environment variable to ensure logs are printed instantly
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies needed for compiling packages if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Run model training to initialize the database and save the pipeline
RUN python -m src.ml.train

# Expose port 8000
EXPOSE 8000

# Start the FastAPI application via uvicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
