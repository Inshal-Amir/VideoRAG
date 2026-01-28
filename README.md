# 🎥 Video RAG: Chat with Your Video

A **Video Retrieval Augmented Generation (Video RAG)** application that allows users to upload videos, index them semantically, and "chat" with the content using natural language.

Built with **Python**, **Streamlit**, **OpenAI (GPT-4o-mini)**, and **FAISS**.

---

## 🚀 Features

* **Video Ingestion**: Upload MP4 files directly via the UI.
* **Smart Indexing**: Automatically extracts frames (1 per second), generates descriptive captions using Multimodal AI, and creates vector embeddings.
* **Semantic Search**: Search for specific events ("red car turning left") rather than just keywords.
* **Chat Interface**: Ask questions naturally ("Did anyone enter the shop?"). The system classifies intent to distinguish between casual chat and video search.
* **Precision Playback**: Returns the **Top 3** most relevant distinct events and plays specific 4-second clips of the exact moment found.
* **Modular Architecture**: Clean separation of concerns (Processor, LLM, Vector Store) for scalability.

---

## 🛠️ Tech Stack

* **Frontend**: Streamlit
* **Video Processing**: OpenCV (extraction), MoviePy (clipping)
* **AI Models**:
    * *Vision*: GPT-4o-mini (Frame Captioning)
    * *Embeddings*: text-embedding-3-small (Vectorization)
    * *Chat*: GPT-4o-mini (Response Generation)
* **Vector Database**: FAISS (Local, high-performance similarity search)

---

## 📂 Project Structure

```text
video_rag_project/
│
├── main.py                  # Entry point (Streamlit UI & Logic)
├── requirements.txt         # Python dependencies
├── .env                     # API Keys (Not committed to git)
│
├── modules/                 # Core Logic
│   ├── __init__.py
│   ├── config.py            # Central configuration & paths
│   ├── llm.py               # OpenAI client & Router logic
│   ├── processor.py         # OpenCV frame extraction & MoviePy clipping
│   └── vector_store.py      # FAISS index management
│
└── data/                    # Local Storage (Auto-generated)
    ├── videos/              # Uploaded raw video files
    ├── clips/               # Generated temporary clips for playback
    └── index/               # FAISS index file & metadata pickle
