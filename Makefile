.PHONY: help build up down restart logs clean rebuild test

# Default target
help:
	@echo "Momentum Trading Dashboard - Docker Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  build       Build the Docker image"
	@echo "  up          Start the application (detached)"
	@echo "  down        Stop and remove containers"
	@echo "  restart     Restart the application"
	@echo "  logs        View application logs (follow mode)"
	@echo "  clean       Remove containers, images, and volumes"
	@echo "  rebuild     Clean build and start"
	@echo "  test        Test if the application is running"
	@echo "  shell       Open bash shell in container"
	@echo "  status      Show container status"

# Build the Docker image
build:
	docker-compose build

# Start the application
up:
	docker-compose up -d
	@echo "✅ Application started! Visit http://localhost:8501"

# Stop the application
down:
	docker-compose down

# Restart the application
restart:
	docker-compose restart

# View logs
logs:
	docker-compose logs -f

# Clean everything
clean:
	docker-compose down -v
	docker system prune -f

# Rebuild from scratch
rebuild: clean build up
	@echo "✅ Rebuild complete! Visit http://localhost:8501"

# Test if application is running
test:
	@curl -f http://localhost:8501/_stcore/health && echo "✅ Application is healthy!" || echo "❌ Application is not responding"

# Open shell in container
shell:
	docker-compose exec momentum-dashboard bash

# Show status
status:
	docker-compose ps
	@echo ""
	docker stats --no-stream momentum-trading-dashboard
