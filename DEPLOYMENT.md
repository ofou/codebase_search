# 🚀 Deployment Guide

## Quick Start

### 1. Build and Deploy
```bash
# Build the Docker image
docker build -t codebase-search .

# Start the service
docker-compose up -d

# Check logs
docker-compose logs -f
```

### 2. Verify Deployment
```bash
# Check if the service is running
docker-compose ps

# Test health endpoint
curl http://localhost:8086/health

# Check index status
# Use the get_index_status tool in your MCP client
```

### 3. Development Mode
```bash
# Run with faster file watching for development
docker-compose -f docker-compose.dev.yml up -d
```

## Testing

### Run All Tests
```bash
python3 run_all_tests.py
```

### Individual Tests
```bash
# Logic tests
python3 test_standalone.py

# Integration tests
python3 test_integration.py

# File watching tests
python3 test_file_watching.py
```

## Monitoring

### Health Checks
- **Endpoint**: `http://localhost:8086/health`
- **Script**: `python3 health_check.py`
- **Docker**: `docker-compose ps`

### Logs
```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f codebase-search
```

### Index Status
Use the `get_index_status` tool to monitor:
- Number of indexed files
- Cache size
- Background indexing status
- File watching status

## Troubleshooting

### Common Issues

1. **Ollama not starting**
   ```bash
   # Check if Ollama is running
   docker-compose exec codebase-search ollama list
   
   # Pull the model manually
   docker-compose exec codebase-search ollama pull nomic-embed-text
   ```

2. **Port conflicts**
   ```bash
   # Check what's using port 8086
   lsof -i :8086
   
   # Change port in docker-compose.yml if needed
   ```

3. **File watching not working**
   ```bash
   # Check if volumes are mounted correctly
   docker-compose exec codebase-search ls -la /app
   
   # Restart with development compose
   docker-compose -f docker-compose.dev.yml up -d
   ```

### Performance Tuning

1. **Increase batch size** for faster indexing
2. **Adjust cache settings** for memory usage
3. **Modify file size limits** for your use case
4. **Configure ignored directories** to skip unnecessary files

## Production Deployment

### Environment Variables
```bash
# Set in docker-compose.yml or .env file
OLLAMA_HOST=0.0.0.0
ENABLE_BACKGROUND_INDEXING=true
ENABLE_FILE_WATCHING=true
INDEX_UPDATE_DELAY=2.0
```

### Resource Limits
```yaml
# Add to docker-compose.yml
services:
  codebase-search:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
```

### Persistence
```yaml
# Add to docker-compose.yml
volumes:
  ollama_data:
    driver: local
  cache_data:
    driver: local
```

## Security Considerations

1. **Network isolation** - Use internal networks
2. **Volume permissions** - Set appropriate file permissions
3. **Model security** - Ensure Ollama models are from trusted sources
4. **Access control** - Limit access to the MCP server

## Backup and Recovery

### Backup Index
```bash
# Backup the cache directory
docker-compose exec codebase-search tar -czf /tmp/cache_backup.tar.gz .cache/
docker cp codebase-search:/tmp/cache_backup.tar.gz ./cache_backup.tar.gz
```

### Restore Index
```bash
# Restore the cache directory
docker cp ./cache_backup.tar.gz codebase-search:/tmp/
docker-compose exec codebase-search tar -xzf /tmp/cache_backup.tar.gz -C /
```
