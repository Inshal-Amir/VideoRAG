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

```

---

## ⚙️ Installation

Follow these steps to set up the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/Inshal-Amir/VideoRAG.git
cd VideoRag

```

### 2. Create a Virtual Environment (venv)

It is recommended to use a virtual environment to keep dependencies isolated.

**For Windows:**

```bash
python -m venv venv
venv\Scripts\activate

```

**For macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate

```

*(You will see `(venv)` appear at the start of your terminal line once activated.)*

### 3. Install Dependencies

Install all required Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt

```

### 4. Set Up Environment Variables

Create a file named `.env` in the root directory and add your OpenAI API key.

**Create `.env` file:**

```text
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx

```

*(Replace `sk-proj-xxxx` with your actual OpenAI API Key)*

---

## ▶️ How to Run

Once installation is complete, you can launch the application using Streamlit.

1. Ensure your virtual environment is active.
2. Run the main script:

```bash
python -m streamlit run main.py

```

3. The application will automatically open in your default web browser at:
`http://localhost:8000`

---

## 🧩 Usage Guide

1. **Upload**: Click the "Browse files" button and select an MP4 video.
2. **Wait for Indexing**: The system will process the video frame-by-frame. This may take a moment depending on video length.
3. **Chat**: Type a query in the chat box (e.g., *"Show me the red car"*).
4. **View Results**: The AI will return the most relevant clips. Click the video player to watch the specific event.
