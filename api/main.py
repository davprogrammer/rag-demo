from fastapi import FastAPI
from services.routes import router

app = FastAPI(title="RAG Demo API")
app.include_router(router)

@app.get("/")
def root():
    return {"ok": True, "service": "rag-demo-api"}
