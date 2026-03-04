# Docker Quick Start Guide

This guide will help you run the Momentum Trading Dashboard using Docker.

## Prerequisites

- Docker installed on your system ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose (usually comes with Docker Desktop)

## Quick Start

### Method 1: Using Docker Compose (Easiest)

```bash
# Navigate to project directory
cd Momentum_trading

# Start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Access the dashboard
# Open browser: http://localhost:8501
```

### Method 2: Using Docker CLI

```bash
# Build the image
docker build -t momentum-trading-dashboard .

# Run the container
docker run -d \
  --name momentum-dashboard \
  -p 8501:8501 \
  --restart unless-stopped \
  momentum-trading-dashboard

# View logs
docker logs -f momentum-dashboard

# Access the dashboard
# Open browser: http://localhost:8501
```

## Common Commands

### Docker Compose

```bash
# Start containers
docker-compose up -d

# Stop containers
docker-compose down

# View logs
docker-compose logs -f

# Restart containers
docker-compose restart

# Rebuild and restart
docker-compose up -d --build

# Check status
docker-compose ps
```

### Docker CLI

```bash
# List running containers
docker ps

# Stop container
docker stop momentum-dashboard

# Start container
docker start momentum-dashboard

# Remove container
docker rm momentum-dashboard

# View logs
docker logs -f momentum-dashboard

# Execute command in container
docker exec -it momentum-dashboard bash

# Check container health
docker inspect --format='{{.State.Health.Status}}' momentum-dashboard
```

## Customization

### Change Port

**Docker Compose:**
Edit `docker-compose.yml`:
```yaml
ports:
  - "8502:8501"  # Change 8502 to your preferred port
```

**Docker CLI:**
```bash
docker run -d -p 8502:8501 momentum-trading-dashboard
```

### Development Mode (Live Reload)

Uncomment the volumes section in `docker-compose.yml`:
```yaml
volumes:
  - ./momentum_dashboard.py:/app/momentum_dashboard.py
  - ./METADATA.csv:/app/METADATA.csv
```

Then restart:
```bash
docker-compose up -d --build
```

Now changes to local files will reflect in the container.

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 8501
lsof -i :8501  # macOS/Linux
netstat -ano | findstr :8501  # Windows

# Kill the process or use a different port
docker-compose down
# Edit docker-compose.yml to use different port
docker-compose up -d
```

### Container Fails to Start

```bash
# Check logs
docker-compose logs

# Check container status
docker-compose ps

# Rebuild from scratch
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Image Build Fails

```bash
# Clean Docker cache
docker system prune -a

# Rebuild
docker-compose build --no-cache
```

### Connection Issues

```bash
# Check if container is running
docker ps

# Check container health
docker inspect momentum-trading-dashboard | grep Health

# Test from inside container
docker exec -it momentum-dashboard curl http://localhost:8501/_stcore/health
```

## Performance Optimization

### Resource Limits

Add to `docker-compose.yml`:
```yaml
services:
  momentum-dashboard:
    # ... existing config ...
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 512M
```

### Clean Up Old Data

```bash
# Remove unused images
docker image prune -a

# Remove all stopped containers
docker container prune

# Remove unused volumes
docker volume prune

# Complete cleanup
docker system prune -a --volumes
```

## Deployment

### Production Deployment

1. **Use Environment Variables** (instead of hardcoded secrets):
   ```yaml
   environment:
     - SHEET_ID=${SHEET_ID}
     - GCP_CREDENTIALS=${GCP_CREDENTIALS}
   ```

2. **Use Docker Secrets** (for sensitive data):
   ```yaml
   secrets:
     - gcp_credentials
   ```

3. **Enable SSL/TLS** (use reverse proxy like nginx):
   ```yaml
   services:
     nginx:
       image: nginx:alpine
       ports:
         - "443:443"
       volumes:
         - ./nginx.conf:/etc/nginx/nginx.conf
   ```

### Cloud Deployment Options

**Google Cloud Run:**
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/momentum-dashboard
gcloud run deploy --image gcr.io/PROJECT_ID/momentum-dashboard --platform managed
```

**AWS ECS/Fargate:**
```bash
docker tag momentum-trading-dashboard:latest YOUR_ECR_REPO:latest
docker push YOUR_ECR_REPO:latest
# Deploy via ECS console or CLI
```

**Heroku:**
```bash
heroku container:push web -a your-app-name
heroku container:release web -a your-app-name
```

## Health Checks

The application includes a health check endpoint:
```bash
curl http://localhost:8501/_stcore/health
```

Docker will automatically restart the container if health checks fail.

## Monitoring

### View Resource Usage

```bash
# Real-time stats
docker stats momentum-dashboard

# Container details
docker inspect momentum-dashboard
```

### Export Logs

```bash
# Save logs to file
docker logs momentum-dashboard > app.log 2>&1

# With Docker Compose
docker-compose logs > app.log 2>&1
```

## Security Best Practices

1. **Don't hardcode secrets** - Use environment variables or Docker secrets
2. **Run as non-root user** - Add to Dockerfile:
   ```dockerfile
   RUN useradd -m -u 1000 appuser
   USER appuser
   ```
3. **Scan for vulnerabilities**:
   ```bash
   docker scan momentum-trading-dashboard
   ```
4. **Keep base image updated**:
   ```bash
   docker pull python:3.11-slim
   docker-compose build --no-cache
   ```

## Support

For issues or questions:
- Check logs: `docker-compose logs -f`
- Rebuild: `docker-compose up -d --build`
- Clean start: `docker-compose down && docker-compose up -d --build`
