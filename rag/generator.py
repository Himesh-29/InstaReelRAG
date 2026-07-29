import os
from openai import OpenAI
from dotenv import load_dotenv
from config import get_config
from config.logger import setup_logger

load_dotenv()
logger = setup_logger("ContextAnswerGenerator")

class ContextAnswerGenerator:
    """
    Takes retrieved context information and the user question, formats them into a structured prompt,
    and calls the configured LLM (OpenAI / OpenRouter / Groq / Gemini) to generate a grounded answer.
    """
    def __init__(self, use_openrouter=None):
        from config import get_config, get_llm_client_and_model
        self.config = get_config()["llm"]
        self.client, self.model = get_llm_client_and_model()
            
    def generate_answer(self, query: str, context_docs: list[dict]) -> str:
        """
        Synthesizes an answer using the LLM by combining the user query with retrieved context documents.
        """
        logger.info(f"Synthesizing answer using model '{self.model}' with {len(context_docs)} context documents...")
        # Construct the context block
        context_text = ""
        for i, doc in enumerate(context_docs):
            content = doc.get("content", "")
            meta = doc.get("metadata", {})
            url = meta.get("url", "No URL provided")
            context_text += f"\n--- Source {i+1} (URL: {url}) ---\n{content}\n"
            
        system_prompt = self.config["generator_system_prompt"]
        
        user_prompt = f"Context Information:\n{context_text}\n\nQuestion: {query}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.get("temperature", 0.1),
                max_tokens=self.config.get("max_tokens", 512)
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating answer with LLM: {e}")
            return f"Error generating answer: {e}"