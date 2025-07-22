.PHONY: help build run stop logs test clean dev

help: ## Show this help message
	@echo "Codebase Search MCP Server - Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

build: ## Build the Docker image
	docker build -t codebase-search .

run: ## Run the container
	docker-compose up -d

stop: ## Stop the container
	docker-compose down

logs: ## View container logs
	docker-compose logs -f

dev: ## Run in development mode with file watching
	docker-compose -f docker-compose.dev.yml up -d

test: ## Run system tests
	python3 test_system.py

clean: ## Clean up containers and images
	docker-compose down -v
	docker rmi codebase-search 2>/dev/null || true

health: ## Check system health
	python3 health_check.py

status: ## Show container status
	docker-compose ps

restart: ## Restart the service
	docker-compose restart

rebuild: ## Rebuild and restart
	docker-compose down
	docker-compose up -d --build
