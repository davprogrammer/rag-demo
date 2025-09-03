from fastapi import FastAPI
from services.routes import router
from services.logging import setup_logging

app = FastAPI(title="RAG Demo API")
app.include_router(router)

setup_logging("INFO")

@app.get("/")
def root():
    return {"ok": True, "service": "rag-demo-api"}
