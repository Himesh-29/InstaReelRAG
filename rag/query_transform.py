import os
from openai import OpenAI
from dotenv import load_dotenv
from config import get_config
from config.logger import setup_logger

load_dotenv()
logger = setup_logger("QueryTransformer")

class QueryTransformer:
    def __init__(self, use_openrouter=None):
        from config import get_config, get_llm_client_and_model
        self.config = get_config()["llm"]
        self.client, self.model = get_llm_client_and_model()

    def rephrase_query(self, current_query: str, chat_history: list) -> str:
        """
        Rewrites the query using chat history for context (coreference resolution)
        and expands it with keywords suitable for vector/BM25 retrieval.
        """
        # If there's no history, just return the original query or a slightly expanded version
        if not chat_history:
            return current_query
            
        # Format history robustly for both Gradio tuples and Gradio dicts
        history_text = ""
        for item in chat_history:
            if isinstance(item, dict):
                role = item.get("role", "User").capitalize()
                msg = item.get("content", "")
                if msg:
                    history_text += f"{role}: {msg}\n"
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                user_msg, bot_msg = item
                if user_msg:
                    history_text += f"User: {user_msg}\n"
                if bot_msg:
                    history_text += f"Assistant: {bot_msg}\n"
            else:
                history_text += f"{str(item)}\n"
            
        system_prompt = self.config["query_rephraser_system_prompt"]
        
        user_prompt = f"Conversation History:\n{history_text}\n\nFollow-up Query: {current_query}\n\nRephrased Search Query:"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.get("temperature", 0.1)
            )
            rephrased = response.choices[0].message.content.strip()
            if rephrased != current_query:
                logger.info(f"Rephrased Query: '{rephrased}' (Original: '{current_query}')")
            return rephrased
        except Exception as e:
            logger.error(f"Error rephrasing query: {e}")
            return current_query
