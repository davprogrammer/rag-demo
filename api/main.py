from fastapi import FastAPI
from services.routes import router

app = FastAPI(title="RAG FAQ API")
app.include_router(router)