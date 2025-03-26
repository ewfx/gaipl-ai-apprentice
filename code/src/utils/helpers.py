import os

def ensure_upload_dir():
    """Ensure upload directory exists"""
    os.makedirs("./uploads", exist_ok=True)
