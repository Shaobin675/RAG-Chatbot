from fastapi import FastAPI
from app.api import router

app = FastAPI(title="Upload Service")
app.include_router(router)
