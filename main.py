"""
Entry point for running the FastAPI server from the repo root.
Allows `uvicorn main:app --reload --port 8000` to work without `cd server_py`.
"""
from server_py.main import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server_py.main:app", host="0.0.0.0", port=8000, reload=True)
