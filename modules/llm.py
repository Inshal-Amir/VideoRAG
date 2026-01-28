from openai import OpenAI
from .config import Config

class LLMClient:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)

    # ... [Keep get_image_caption and get_embedding exactly as they were] ...

    def get_image_caption(self, base64_image):
        """Sends image to GPT-4o-mini for description."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": "Describe this video frame in detail. Mention objects, actions, and text."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                max_tokens=100
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting caption: {e}")
            return "Error generating description."

    def get_embedding(self, text):
        """Converts text to vector."""
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return [0.0] * Config.EMBEDDING_DIM

    # --- NEW FUNCTION: ROUTER ---
    def check_intent(self, user_query):
        """
        Decides if the user wants to SEARCH the video or just CHAT.
        Returns: 'SEARCH' or 'CHAT'
        """
        system_prompt = (
            "You are an intent classifier. Analyze the user's input. "
            "If the user is asking to find, look for, spot, or describe something in a video, return 'SEARCH'. "
            "If the user is asking for your name, greeting, or general knowledge, return 'CHAT'. "
            "Output ONLY the word 'SEARCH' or 'CHAT'."
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content.strip().upper()
        except:
            return "SEARCH" # Default to search if unsure

    def get_chat_response(self, user_query, context_results):
        """
        Generates a natural answer. Now includes a check for IRRELEVANT results.
        """
        if not context_results:
            # Fallback for general chat
            return self.general_chat(user_query)

        context_str = "\n".join([
            f"- Timestamp {r['timestamp']:.1f}s: {r['description']}" 
            for r in context_results
        ])

        system_prompt = (
            "You are a video analysis assistant. "
            "1. Use the provided 'Video Findings' to answer the user's question. "
            "2. IF the findings are NOT relevant to the question (e.g., user asks for a car, findings are about a cat), "
            "ignore the findings and tell the user you found nothing matching their request. "
            "3. If multiple events are relevant, mention all of them with their timestamps."
        )

        user_prompt = f"User Question: {user_query}\n\nVideo Findings:\n{context_str}"

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content

    def general_chat(self, user_query):
        """Handles questions like 'What is your name?' without using video context."""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful Video RAG Assistant. You help users find events in their videos."},
                {"role": "user", "content": user_query}
            ]
        )
        return response.choices[0].message.content