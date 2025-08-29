from fastapi import FastAPI
import threading, time
import requests
from services.routes import router
from services import config

app = FastAPI(title="RAG FAQ API")
app.include_router(router)

