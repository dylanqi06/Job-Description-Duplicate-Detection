# Use an official Python runtime as a parent image
FROM python:3.9-slim

WORKDIR /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 19530
EXPOSE 9091

ENV MILVUS_HOST=milvus
ENV MILVUS_PORT=19530

# Run the application
CMD ["python", "src.py"]
