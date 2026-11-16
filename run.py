import os

from src.api.rest_api import app


if __name__ == "__main__":
    import uvicorn

    profile = os.getenv("INFERENCE_PROFILE", "realtime")
    print(f"[CampusSecurityAgent] active profile={profile}, runtime=gpu, backend=onnxruntime")
    uvicorn.run(app, host="0.0.0.0", port=8000)
