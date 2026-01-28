import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    VIDEO_DIR = os.path.join(DATA_DIR, "videos")
    INDEX_DIR = os.path.join(DATA_DIR, "index")
    
    INDEX_FILE = os.path.join(INDEX_DIR, "faiss_index.bin")
    METADATA_FILE = os.path.join(INDEX_DIR, "metadata.pkl")
    
    # Settings
    FRAME_INTERVAL = 1  # Process 1 frame every second
    EMBEDDING_DIM = 1536 # For text-embedding-3-small

    @staticmethod
    def ensure_dirs():
        os.makedirs(Config.VIDEO_DIR, exist_ok=True)
        os.makedirs(Config.INDEX_DIR, exist_ok=True)