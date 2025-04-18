# Use official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
#RUN pip install --no-cache-dir -r requirements/test_requirements.txt
#RUN pip install --no-cache-dir build

# Train the model
#RUN python loanprediction/train_pipeline.py

# Run tests
#RUN pytest

# Copy .whl file 
COPY loanprediction-0.0.1-py3-none-any.whl .
#RUN python -m build && \
    #mv dist/*.whl .

# Install FastAPI dependencies
RUN pip install --no-cache-dir -r loanprediction_api/requirements.txt

# Expose port for FastAPI
EXPOSE 8000

# Start FastAPI app
CMD ["python", "loanprediction_api/app/main.py"]

