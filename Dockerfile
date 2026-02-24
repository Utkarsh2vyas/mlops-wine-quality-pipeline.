# Start with a standard Python environment
FROM python:3.9

# Create a working directory
WORKDIR /app

# Copy the requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your FastAPI app code
COPY app.py .

# Tell the server to run the FastAPI app on port 7860 (Hugging Face's default port)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
