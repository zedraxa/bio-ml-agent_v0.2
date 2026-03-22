import logging
from fastapi import FastAPI
from bio_ml_agent.routers.platform_routes import router as platform_router
from bio_ml_agent.utils.config import get_config

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gateway")

# Config
config = get_config()

# Gateway App
app = FastAPI(
    title="Bio-ML Agent Gateway",
    description="Public-facing API Gateway for Bio-ML Swarm",
    version="1.0.0"
)

# Platform Routes (Publicly exposed through Gateway)
app.include_router(platform_router, prefix="/api/v1/platform", tags=["Platform"])

@app.get("/health")
def health():
    return {"status": "ok", "service": "gateway"}

def main():
    import uvicorn
    # Gateway port 8000 is defined in docker-compose.yml
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
