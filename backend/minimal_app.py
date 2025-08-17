from fastapi import FastAPI

app = FastAPI(title="Minimal Test API")

@app.get("/")
def root():
    return {"message": "Welcome to the Minimal Test API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}