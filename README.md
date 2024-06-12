# Job Description Duplicate Detection

## Setup and Running

### Prerequisites

- Docker and Docker Compose installed

### Build and Run the Docker Containers

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourrepo/job-description-duplicate-detection.git
    cd job-description-duplicate-detection
    ```

2. **Build and run the Docker containers:**

    ```sh
    docker-compose up --build
    ```

3. **Access the application:**

    The application should be running and accessible. Check the logs for any specific URLs or ports if necessary.

### Environment Variables

- `MILVUS_HOST`: The hostname for the Milvus server (default: `milvus`)
- `MILVUS_PORT`: The port for the Milvus server (default: `19530`)

### Notes

- For large datasets, mount the dataset directory as a volume to avoid copying large files into the Docker image.
