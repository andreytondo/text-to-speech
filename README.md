# text-to-speech

This project boots a minimal FastAPI service that uses Coqui TTS to synthesize audio locally.

Endpoints:
- POST /tts -> returns audio (mp3 or wav)
- GET /voices -> model info
- GET /health -> health check

Quick start
1. Build and run with docker-compose:
  ```bash
   docker compose up --build
  ```
2. POST:
  ```bash
   curl -X POST "http://localhost:8000/tts" -H "Content-Type: application/json" -d '{"text":"Olá, tudo bem?","format":"mp3"}' --output out.mp3
  ```

Notes
- The first run may download model weights; this can take time and disk.
- Coqui TTS (TTS package) will pull PyTorch. Make sure your environment has enough RAM/disk.
