import faiss
import pickle
import numpy as np
import os
from .config import Config

class VectorStore:
    def __init__(self):
        self.index = None
        self.metadata = {}
        self.load_index()

    def load_index(self):
        if os.path.exists(Config.INDEX_FILE) and os.path.exists(Config.METADATA_FILE):
            self.index = faiss.read_index(Config.INDEX_FILE)
            with open(Config.METADATA_FILE, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(Config.EMBEDDING_DIM)
            self.metadata = {}

    def save_index(self):
        faiss.write_index(self.index, Config.INDEX_FILE)
        with open(Config.METADATA_FILE, "wb") as f:
            pickle.dump(self.metadata, f)

    def add_record(self, vector, meta_info):
        """
        vector: list or np array of floats
        meta_info: dict containing video_path, timestamp, description
        """
        vector_np = np.array([vector]).astype('float32')
        self.index.add(vector_np)
        
        # ID is the current count minus 1 (since we just added 1)
        idx_id = self.index.ntotal - 1
        self.metadata[idx_id] = meta_info

    def search(self, query_vector, k=3):
        vector_np = np.array([query_vector]).astype('float32')
        distances, indices = self.index.search(vector_np, k)
        
        results = []
        for idx in indices[0]:
            if idx != -1:
                results.append(self.metadata[idx])
        return results