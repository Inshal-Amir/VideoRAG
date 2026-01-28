import streamlit as st
import os

from modules.config import Config
from modules.processor import VideoProcessor
from modules.llm import LLMClient
from modules.vector_store import VectorStore

# Initialize system
Config.ensure_dirs()
llm = LLMClient()
store = VectorStore()
processor = VideoProcessor()

st.set_page_config(layout="wide", page_title="Video Chat Assistant")

# --- SIDEBAR: Video Processing ---
with st.sidebar:
    st.header("📂 Video Manager")
    uploaded_file = st.file_uploader("Upload New Video", type=["mp4"])
    
    if uploaded_file and st.button("Process & Index"):
        # Save file
        save_path = os.path.join(Config.VIDEO_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        progress_bar = st.progress(0)
        status = st.empty()
        
        # Indexing Loop
        frames_gen = processor.extract_frames(save_path, interval=Config.FRAME_INTERVAL)
        count = 0
        
        for timestamp, b64_frame in frames_gen:
            status.caption(f"Indexing frame at {timestamp:.1f}s...")
            caption = llm.get_image_caption(b64_frame)
            vector = llm.get_embedding(caption)
            
            store.add_record(vector, {
                "video_path": save_path,
                "timestamp": timestamp,
                "description": caption
            })
            count += 1
            
        store.save_index()
        progress_bar.progress(100)
        status.success(f"✅ Ready! Indexed {count} frames.")

# --- MAIN CHAT ---
st.title("💬 Chat with your Video")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me to find something in your videos."}]

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "media" in msg:
            for clip_path in msg["media"]:
                if os.path.exists(clip_path):
                    st.video(clip_path)

# Handle User Input
if prompt := st.chat_input("Ex: Find the red car"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            # 1. ROUTER: Check if this is a search or just chat
            intent = llm.check_intent(prompt)
            
            media_paths = []
            answer_text = ""

            if intent == "CHAT":
                # Just chat, no video search
                answer_text = llm.general_chat(prompt)
                st.markdown(answer_text)
            
            else:
                # 2. PERFORM SEARCH
                # Strategy: Fetch MORE candidates (k=10) first, then deduplicate, then limit to Top 3.
                # This prevents getting 3 frames of the exact same second.
                query_vector = llm.get_embedding(prompt)
                raw_results = store.search(query_vector, k=10)
                
                # Filter duplicates (keep events 2 seconds apart)
                unique_results = []
                seen_times = []
                
                for r in raw_results:
                    is_duplicate = False
                    for t in seen_times:
                        # If a new result is within 2.0 seconds of an existing one, ignore it
                        if abs(r['timestamp'] - t) < 2.0:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        unique_results.append(r)
                        seen_times.append(r['timestamp'])
                
                # 3. PRIORITIZE TOP 3
                # We simply slice the list. Since FAISS returns results ordered by relevance (best first),
                # taking the first 3 unique items gives us the "Top 3 Match Searches".
                top_3_results = unique_results[:3]

                # 4. GENERATE ANSWER
                answer_text = llm.get_chat_response(prompt, top_3_results)
                st.markdown(answer_text)
                
                # 5. CREATE CLIPS (Only if relevant results found)
                if "couldn't find" not in answer_text.lower() and top_3_results:
                    st.markdown("---")
                    st.caption(f"Showing Top {len(top_3_results)} Matches:")
                    
                    clips_dir = os.path.join(Config.DATA_DIR, "clips")
                    os.makedirs(clips_dir, exist_ok=True)

                    for res in top_3_results:
                        start_t = max(0, res['timestamp'] - 2)
                        end_t = res['timestamp'] + 2
                        
                        video_name = os.path.basename(res['video_path'])
                        clip_name = f"clip_{video_name}_{int(start_t)}_{int(end_t)}.mp4"
                        clip_path = os.path.join(clips_dir, clip_name)
                        
                        # Create clip if missing
                        if not os.path.exists(clip_path):
                            processor.create_clip(res['video_path'], start_t, end_t, clip_path)
                        
                        if os.path.exists(clip_path):
                            st.video(clip_path)
                            media_paths.append(clip_path)

            # Save to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer_text,
                "media": media_paths
            })