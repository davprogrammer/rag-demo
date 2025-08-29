from fastapi import FastAPI
from services.routes import router as api_router

app = FastAPI(title="RAG Demo API")
app.include_router(api_router)

@app.get("/")
def root():
    return {"ok": True, "service": "rag-demo-api"}
