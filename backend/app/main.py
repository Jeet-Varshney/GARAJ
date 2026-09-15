from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import stream

app = FastAPI(
    title="SIH26104 Real-Time Voice Cloning Detection Pipeline (Phase 0)",
    description="Backend audio streaming WebSocket service with rolling buffer & pluggable DetectionEngine interface.",
    version="0.1.0"
)

# Enable CORS for local Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(stream.router)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "SIH26104 Phase 0 Audio Backend",
        "audio_config": {
            "sample_rate": 16000,
            "channels": 1,
            "encoding": "pcm_s16le"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
