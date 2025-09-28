# Use official Python 3.12 image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

# Create app directory
WORKDIR $APP_HOME

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY app.py .
COPY 36120-25SP-25731542-experiment-rain.pkl .
COPY 36120-25SP-25731542-experiment-3dayprep-best.pkl .

# Expose the port for FastAPI
EXPOSE 8000

# Command to run the app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
