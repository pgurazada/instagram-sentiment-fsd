FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Install system dependencies only if needed, then clean up
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY . .

# Create a non-root user and switch to it
RUN useradd -m appuser
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]