# Deployment Guide for fluence.network

## Prerequisites

1. Docker installed on your local machine
2. fluence.network account and CLI tools
3. Your hackathon data files in the `hackathon_data/` directory

## Local Testing

### Using Docker Compose (Recommended)

1. **Build and run the application:**
   ```bash
   docker-compose up --build
   ```

2. **Access the application:**
   - Open your browser and go to `http://localhost:8501`

3. **Stop the application:**
   ```bash
   docker-compose down
   ```

### Using Docker directly

1. **Build the image:**
   ```bash
   docker build -t hackathon-evaluator .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8501:8501 \
     -v $(pwd)/hackathon_data:/app/hackathon_data:ro \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/chroma_db:/app/chroma_db \
     hackathon-evaluator
   ```

## Deploying to fluence.network

### 1. Prepare your data

Ensure your `hackathon_data/` directory contains your JSON files:
```bash
ls hackathon_data/
# Should show your JSON files like EthGlobal_New-Delhi_2025_September.json
```

### 2. Build and push to Docker registry

1. **Tag your image:**
   ```bash
   docker tag hackathon-evaluator your-registry/hackathon-evaluator:latest
   ```

2. **Push to registry:**
   ```bash
   docker push your-registry/hackathon-evaluator:latest
   ```

### 3. Deploy using fluence CLI

1. **Install fluence CLI** (if not already installed):
   ```bash
   npm install -g @fluencelabs/cli
   ```

2. **Login to fluence:**
   ```bash
   fluence login
   ```

3. **Deploy your service:**
   ```bash
   fluence deploy --service hackathon-evaluator --image your-registry/hackathon-evaluator:latest
   ```

### 4. Environment Variables

If you need to set environment variables for your deployment:

```bash
fluence deploy \
  --service hackathon-evaluator \
  --image your-registry/hackathon-evaluator:latest \
  --env STREAMLIT_SERVER_PORT=8501 \
  --env STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

## Configuration

### Port Configuration
- The application runs on port 8501 by default
- This is configurable via the `STREAMLIT_SERVER_PORT` environment variable

### Data Persistence
- `hackathon_data/`: Read-only mount for your JSON data files
- `data/`: Read-write mount for scraped data
- `chroma_db/`: Read-write mount for the vector database

### Memory Requirements
- Minimum: 1GB RAM
- Recommended: 2GB+ RAM (for sentence-transformers model loading)

## Troubleshooting

### Common Issues

1. **Port already in use:**
   ```bash
   # Kill process using port 8501
   lsof -ti:8501 | xargs kill -9
   ```

2. **Permission issues with volumes:**
   ```bash
   # Fix ownership
   sudo chown -R $USER:$USER data/ chroma_db/
   ```

3. **Out of memory errors:**
   - Increase Docker memory limit
   - Consider using a smaller sentence transformer model

### Health Check

The application includes a health check endpoint:
- URL: `http://your-domain:8501/_stcore/health`
- Returns 200 OK when healthy

### Logs

View application logs:
```bash
# Using docker-compose
docker-compose logs -f

# Using docker
docker logs <container-id>
```

## Security Considerations

1. **Non-root user**: The container runs as a non-root user for security
2. **Environment variables**: Store sensitive data in environment variables, not in the image
3. **Data volumes**: Use read-only mounts for static data when possible

## Performance Optimization

1. **Model caching**: The sentence transformer model is downloaded on first run
2. **Vector database**: ChromaDB data persists in the `chroma_db/` volume
3. **Memory usage**: Monitor memory usage, especially during model loading

## Monitoring

Monitor your deployment:
- Health check endpoint: `/health`
- Streamlit metrics: Available in the Streamlit interface
- Container logs: Use `docker logs` or fluence CLI monitoring tools
