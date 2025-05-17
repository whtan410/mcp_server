# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import learningstyle_router, sketchfab_qwen, learningstyle_determiner

app = FastAPI(title="Sketchfab 3D Model API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
#app.include_router(sketchfab_router.router)
app.include_router(learningstyle_router.router)
app.include_router(sketchfab_qwen.router)
app.include_router(learningstyle_determiner.router)


@app.get("/")
async def root():
    return {"message": "Welcome to Sketchfab 3D Model API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
        # uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)