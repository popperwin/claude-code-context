from fastapi import FastAPI
from typing import Dict

app = FastAPI(title="API Server", version="1.0.0")

@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Hello World"}

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "healthy"}
