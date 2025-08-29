#!/bin/bash

echo "=== Ollama Modell Warmup ==="

# Modelle vorab laden
echo "Loading embedding model..."
docker compose exec ollama ollama run nomic-embed-text <<< "test"

echo "Loading chat model..."
docker compose exec ollama ollama run phi3:mini <<< "Hello"

echo "Models loaded and ready!"

# Erste API Query zum Warmup
echo "API Warmup..."
sleep 2
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"test"}' > /dev/null 2>&1

echo "=== Warmup complete ==="
echo "Try your query now - should be much faster!"
