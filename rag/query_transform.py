import os
from config import get_config, get_llm_chat_completion
from config.logger import setup_logger

logger = setup_logger("QueryTransformer")

class QueryTransformer:
    def __init__(self, *args, **kwargs):
        self.config = get_config()["llm"]

    def rephrase_query(self, current_query: str, chat_history: list) -> str:
        """
        Rewrites the query using chat history for context (coreference resolution)
        and expands it with keywords suitable for vector/BM25 retrieval.
        """
        # If there's no history, just return the original query or a slightly expanded version
        if not chat_history:
            return current_query
            
        # Format history
        history_text = ""
        for role, msg in chat_history:
            history_text += f"{role.capitalize()}: {msg}\n"
            
        system_prompt = self.config["query_rephraser_system_prompt"]
        
        user_prompt = f"Conversation History:\n{history_text}\n\nFollow-up Query: {current_query}\n\nRephrased Search Query:"
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            rephrased = get_llm_chat_completion(
                messages=messages,
                temperature=self.config.get("temperature", 0.1)
            ).strip()
            if rephrased != current_query:
                logger.info(f"Rephrased Query: '{rephrased}' (Original: '{current_query}')")
            return rephrased
        except Exception as e:
            logger.error(f"Error rephrasing query: {e}")
            return current_query

