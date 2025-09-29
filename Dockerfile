# Use official Python image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and models
COPY app /app/app
COPY models /app/models

# Expose port for FastAPI
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
