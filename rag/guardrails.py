import logging
import json
import numpy as np
from config.config import get_llm_chat_completion

logger = logging.getLogger("instareelrag")

class SemanticFilter:
    """
    Evaluates text safety by computing cosine similarity against clean, 
    professional policy descriptions (no hardcoded hate text).
    """
    def __init__(self, threshold: float = 0.15, policies: list[str] = None):
        # We import here to avoid circular imports if any, and only load model when needed
        from database.vector_store import LocalEmbeddingFunction
        self.embed_fn = LocalEmbeddingFunction()
        
        # We use a very low threshold (0.15) because all-MiniLM-L6-v2 generates 
        # heavily normalized vectors where even distinct concepts have low absolute cosine distance.
        # As seen in testing, safe queries score ~0.03-0.12, while unsafe queries score ~0.16-0.35.
        self.threshold = threshold
        self.policies = policies or []
        # Precompute the vector embeddings of our clean policy rules
        if self.policies:
            self.policy_embeddings = np.array(self.embed_fn(self.policies))
        else:
            self.policy_embeddings = []
        logger.info(f"Initialized SemanticFilter with {len(self.policies)} policies.")
        
    def _cosine_similarity(self, a, b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    def is_unsafe(self, text: str) -> tuple[bool, float, str]:
        """
        Returns (is_unsafe, max_similarity, matched_policy).
        """
        if not text or not text.strip():
            return False, 0.0, ""
            
        text_emb = np.array(self.embed_fn([text])[0])
        max_sim = 0.0
        matched_policy = ""
        
        for i, policy_emb in enumerate(self.policy_embeddings):
            sim = self._cosine_similarity(text_emb, policy_emb)
            if sim > max_sim:
                max_sim = sim
                matched_policy = self.policies[i]
                
        if max_sim >= self.threshold:
            return True, max_sim, matched_policy
            
        return False, max_sim, ""


class LLMSafetyFilter:
    """
    Evaluates text safety using the configured LLM API.
    """
    def is_unsafe(self, text: str, is_output: bool = False) -> tuple[bool, str]:
        if not text or not text.strip():
            return False, ""
            
        try:
            from config.config import CONFIG
            prompt_template = CONFIG.get("guardrails", {}).get(
                "llm_guardrail_prompt",
                "You are a strict safety guardrail. Analyze the {target} for safety violations. Output JSON with key 'result' set to 'SAFE' or 'VIOLATION'."
            )
            target = "generated AI output" if is_output else "user input"
            system_prompt = prompt_template.format(target=target)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Text to check: {text}"}
            ]
            res = get_llm_chat_completion(
                messages=messages, 
                temperature=0.0, 
                max_tokens=20
            )
            
            if res is None:
                logger.warning("LLM returned None, failing open.")
                return False, ""
                
            try:
                parsed_res = json.loads(res)
                result_val = parsed_res.get("result", "").upper()
            except json.JSONDecodeError:
                # Fallback to string matching if JSON parsing fails
                result_val = res.upper()
                
            if "VIOLATION" in result_val:
                return True, "Violated LLM sentiment and content guardrail."
        except Exception as e:
            logger.warning(f"LLM guardrail check encountered error (failing open): {e}")
            
        return False, ""


class Guardrails:
    """
    Aggregate Guardrail system: 
    Layer 1: Semantic Embedding Similarity (MiniLM)
    Layer 2: LLM-as-a-Judge API Check
    Dynamically enabled/disabled via config.json.
    """
    def __init__(self):
        from config.config import CONFIG
        guardrails_config = CONFIG.get("guardrails", {})
        
        self.use_semantic_check = guardrails_config.get("use_semantic_check", True)
        self.use_llm_check = guardrails_config.get("use_llm_check", True)
        semantic_threshold = guardrails_config.get("semantic_threshold", 0.15)
        self.refusal_message = guardrails_config.get(
            "refusal_message", 
            "⚠️ **Guardrail Alert:** We cannot provide an answer to your request as it may violate safety, sentiment, or content guidelines."
        )
        safety_policies = guardrails_config.get("safety_policies", [])
        
        self.semantic_filter = SemanticFilter(threshold=semantic_threshold, policies=safety_policies) if self.use_semantic_check else None
        self.llm_filter = LLMSafetyFilter() if self.use_llm_check else None

    def check_input_safety(self, query: str) -> tuple[bool, str]:
        """Checks user input using enabled filters."""
        if not query or not query.strip():
            return True, "Safe"

        # 1. Semantic Filter (MiniLM)
        if self.use_semantic_check and self.semantic_filter:
            is_unsafe, sim_score, policy = self.semantic_filter.is_unsafe(query)
            if is_unsafe:
                logger.warning(f"Semantic Guardrail tripped (score: {sim_score:.2f}) on policy: '{policy}'")
                return False, "Input semantically matched an unsafe policy concept."

        # 2. LLM Guardrail (OpenRouter API)
        if self.use_llm_check and self.llm_filter:
            is_unsafe, reason = self.llm_filter.is_unsafe(query, is_output=False)
            if is_unsafe:
                logger.warning("LLM input guardrail flagged query.")
                return False, f"Input {reason}"

        return True, "Safe"

    def check_output_safety(self, response_text: str) -> tuple[bool, str]:
        """Checks AI output using enabled filters."""
        if not response_text or not response_text.strip():
            return True, "Safe"

        # 1. Semantic Filter (MiniLM)
        if self.use_semantic_check and self.semantic_filter:
            is_unsafe, sim_score, policy = self.semantic_filter.is_unsafe(response_text)
            if is_unsafe:
                logger.warning(f"Output Semantic Guardrail tripped (score: {sim_score:.2f}).")
                return False, "Output semantically matched an unsafe policy concept."

        # 2. LLM Guardrail (OpenRouter API)
        if self.use_llm_check and self.llm_filter:
            is_unsafe, reason = self.llm_filter.is_unsafe(response_text, is_output=True)
            if is_unsafe:
                logger.warning("LLM output guardrail flagged AI response.")
                return False, f"Output {reason}"

        return True, "Safe"
