FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY src/ src/
COPY models/ models/

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "src/app.py"]
