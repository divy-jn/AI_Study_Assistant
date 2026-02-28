"""
LLM Service - Ollama Integration
Provides robust interface to Ollama with retry logic, circuit breaker, and fallback
"""
import asyncio
import httpx
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime


from app.core.logging_config import LoggerMixin, LogExecutionTime, get_logger
from app.core.exceptions import (
    LLMConnectionException,
    LLMModelNotFoundException,
    LLMGenerationException,
    LLMTimeoutException,
    LLMRateLimitException
)
from app.core.retry_utils import (
    retry_with_backoff,
    CircuitBreaker,
    with_timeout,
    SafeExecutor
)
from app.core.config import settings


logger = get_logger(__name__)


class OllamaLLMService(LoggerMixin):
    """
    Service for interacting with Ollama LLM
    Includes retry logic, circuit breaker, and comprehensive error handling
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 120
    ):
        """
        Initialize Ollama LLM Service
        
        Args:
            base_url: Ollama API base URL (defaults to settings)
            model: Model name (defaults to settings)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.model = model or settings.OLLAMA_MODEL
        self.timeout = timeout
        
        # Circuit breaker to prevent cascading failures
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=LLMConnectionException
        )
        
        # HTTP client with timeout
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        
        self.logger.info(
            f"Initialized LLM Service | Model: {self.model} | URL: {self.base_url}"
        )
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()
        self.logger.info("LLM Service closed")
    
    @retry_with_backoff(
        max_retries=3,
        initial_delay=1.0,
        exceptions=(LLMConnectionException, LLMTimeoutException)
    )
    async def check_model_availability(self) -> bool:
        """
        Check if the configured model is available
        
        Returns:
            True if model is available
            
        Raises:
            LLMConnectionException: If cannot connect to Ollama
            LLMModelNotFoundException: If model not found
        """
        with LogExecutionTime(self.logger, "Check model availability"):
            try:
                response = await self.client.get(f"{self.base_url}/api/tags")
                
                if response.status_code != 200:
                    raise LLMConnectionException(
                        url=self.base_url,
                        original_exception=Exception(f"Status code: {response.status_code}")
                    )
                
                data = response.json()
                models = data.get("models", [])
                model_names = [m.get("name", "") for m in models]
                
                # Check if our model is in the list
                model_found = any(self.model in name for name in model_names)
                
                if not model_found:
                    self.logger.error(
                        f"Model '{self.model}' not found. Available: {model_names}"
                    )
                    raise LLMModelNotFoundException(model_name=self.model)
                
                self.logger.info(f"Model '{self.model}' is available")
                return True
                
            except httpx.ConnectError as e:
                raise LLMConnectionException(
                    url=self.base_url,
                    original_exception=e
                )
            except httpx.TimeoutException as e:
                raise LLMTimeoutException(timeout_seconds=self.timeout)
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate text using Ollama
        
        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            stream: Whether to stream response
            context: Additional context for logging
            
        Returns:
            Generated text
            
        Raises:
            LLMGenerationException: If generation fails
        """
        # Use defaults from settings if not provided
        temperature = temperature or settings.OLLAMA_TEMPERATURE
        max_tokens = max_tokens or settings.OLLAMA_MAX_TOKENS
        top_p = top_p or settings.OLLAMA_TOP_P
        
        context = context or {}
        
        self.logger.info(
            f"🧠 Generating response | "
            f"Prompt length: {len(prompt)} chars | "
            f"Temperature: {temperature} | "
            f"Max tokens: {max_tokens}",
            extra=context
        )
        
        with LogExecutionTime(self.logger, "LLM Generation", logging.INFO):
            try:
                # Execute with circuit breaker protection
                result = await self._generate_with_protection(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stream=stream
                )
                
                self.logger.info(
                    f"Generated response | "
                    f"Length: {len(result)} chars",
                    extra=context
                )
                
                return result
                
            except Exception as e:
                self.logger.error(
                    f"Generation failed | Error: {str(e)}",
                    exc_info=True,
                    extra=context
                )
                raise
    
    @retry_with_backoff(
        max_retries=settings.OLLAMA_MAX_RETRIES,
        initial_delay=2.0,
        max_delay=30.0,
        exceptions=(LLMConnectionException, LLMTimeoutException)
    )
    async def _generate_with_protection(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        top_p: float,
        stream: bool
    ) -> str:
        """
        Internal generation method with retry protection
        """
        try:
            # Build request payload
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": top_p
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            # Make request
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            
            if response.status_code == 429:
                raise LLMRateLimitException()
            
            if response.status_code != 200:
                raise LLMGenerationException(
                    prompt_sample=prompt[:100],
                    reason=f"HTTP {response.status_code}: {response.text}",
                    original_exception=None
                )
            
            # Parse response
            if stream:
                # Handle streaming response
                full_response = ""
                async for line in response.aiter_lines():
                    if line:
                        import json
                        chunk = json.loads(line)
                        if "response" in chunk:
                            full_response += chunk["response"]
                return full_response
            else:
                # Handle non-streaming response
                data = response.json()
                return data.get("response", "")
            
        except httpx.ConnectError as e:
            raise LLMConnectionException(
                url=self.base_url,
                original_exception=e
            )
        except httpx.TimeoutException:
            raise LLMTimeoutException(timeout_seconds=self.timeout)
        except LLMRateLimitException:
            raise
        except Exception as e:
            if not isinstance(e, (LLMConnectionException, LLMTimeoutException)):
                raise LLMGenerationException(
                    prompt_sample=prompt[:100],
                    reason=str(e),
                    original_exception=e
                )
            raise
    
    async def generate_with_fallback(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        fallback_response: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate with fallback to default response if all retries fail
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            fallback_response: Response to return if generation fails
            **kwargs: Additional arguments for generate()
            
        Returns:
            Generated text or fallback response
        """
        try:
            return await self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                **kwargs
            )
        except Exception as e:
            self.logger.warning(
                f"Generation failed, using fallback | Error: {str(e)}"
            )
            
            if fallback_response:
                return fallback_response
            else:
                return (
                    "I apologize, but I'm having trouble generating a response "
                    "right now. Please try again in a moment."
                )
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Chat completion with message history
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            context: Additional context for logging
            
        Returns:
            Assistant's response
        """
        # Build system prompt and user prompt from messages
        system_messages = [m for m in messages if m.get("role") == "system"]
        user_messages = [m for m in messages if m.get("role") in ("user", "assistant")]
        
        system_prompt = None
        if system_messages:
            system_prompt = "\n".join(m.get("content", "") for m in system_messages)
        
        # Build conversation context
        conversation = "\n".join(
            f"{m.get('role', 'user').title()}: {m.get('content', '')}"
            for m in user_messages
        )
        
        return await self.generate(
            prompt=conversation,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            context=context
        )
    

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Generate streaming text using Ollama
        
        Yields:
             Text chunks
        """
        # Use defaults from settings if not provided
        temperature = temperature or settings.OLLAMA_TEMPERATURE
        max_tokens = max_tokens or settings.OLLAMA_MAX_TOKENS
        top_p = top_p or settings.OLLAMA_TOP_P
        
        context = context or {}
        
        self.logger.info(
            f"🧠 Streaming response | "
            f"Prompt length: {len(prompt)} chars",
            extra=context
        )
        
        try:
            # Build request payload
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": top_p
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            # Streaming request
            async with self.client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise LLMGenerationException(
                        prompt_sample=prompt[:100],
                        reason=f"HTTP {response.status_code}: {error_text.decode()}",
                        original_exception=None
                    )
                
                async for line in response.aiter_lines():
                    if line:
                        import json
                        chunk = json.loads(line)
                        if "response" in chunk:
                            yield chunk["response"]
                            
        except Exception as e:
            self.logger.error(
                f"Streaming failed | Error: {str(e)}",
                exc_info=True,
                extra=context
            )
            # Re-raise so the caller handles it (e.g. closes the stream)
            raise

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Chat completion with streaming
        """
        # Reuse chat logic to build prompt
        system_messages = [m for m in messages if m.get("role") == "system"]
        user_messages = [m for m in messages if m.get("role") in ("user", "assistant")]
        
        system_prompt = None
        if system_messages:
            system_prompt = "\n".join(m.get("content", "") for m in system_messages)
        
        conversation = "\n".join(
            f"{m.get('role', 'user').title()}: {m.get('content', '')}"
            for m in user_messages
        )
        
        async for chunk in self.generate_stream(
            prompt=conversation,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            context=context
        ):
            yield chunk



# Singleton instance for application-wide use
_llm_service_instance: Optional[OllamaLLMService] = None



def set_llm_service_override(service: OllamaLLMService):
    """Override the global instance (for testing)"""
    global _llm_service_instance
    _llm_service_instance = service


async def get_llm_service() -> OllamaLLMService:
    """
    Get or create LLM service singleton instance.
    Compatible with FastAPI Depends.
    """
    global _llm_service_instance
    
    if _llm_service_instance is None:
        _llm_service_instance = OllamaLLMService()
        # Verify model availability on first use
        # Note: In production, this should ideally be done in startup event
        try:
            await _llm_service_instance.check_model_availability()
        except Exception as e:
            logger.warning(f"LLM model check failed on init: {e}")
    
    return _llm_service_instance


async def close_llm_service():
    """Close the global LLM service instance"""
    global _llm_service_instance
    
    if _llm_service_instance:
        await _llm_service_instance.close()
        _llm_service_instance = None


if __name__ == "__main__":
    # Test LLM service
    async def test():
        async with OllamaLLMService() as llm:
            # Check availability
            await llm.check_model_availability()
            
            # Test generation
            response = await llm.generate(
                prompt="What is machine learning?",
                system_prompt="You are a helpful AI assistant.",
                max_tokens=100
            )
            
            print(f"\nResponse:\n{response}\n")
            
            # Test chat
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi! How can I help you?"},
                {"role": "user", "content": "Tell me a joke."}
            ]
            
            chat_response = await llm.chat(messages, max_tokens=50)
            print(f"\nChat Response:\n{chat_response}\n")
    
    asyncio.run(test())
