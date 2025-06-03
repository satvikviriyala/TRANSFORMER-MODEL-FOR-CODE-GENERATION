# src/serving/api_llm.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import time
import os
from src.serving.inference_llm import LLMGenerator, get_llm_generator
from src.utils.logging_utils import setup_logger
from src.utils.config_loader import load_config

logger = setup_logger("LLM_FastAPIApp")
config = load_config()
serving_cfg = config['serving']

app = FastAPI(title="LLM Code Generation Service API")

# Dependency Injection
def get_generator() -> LLMGenerator:
    return get_llm_generator()

class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = serving_cfg['max_new_tokens']
    temperature: float = serving_cfg['temperature']
    top_k: int = serving_cfg['top_k']

class GenerationResponse(BaseModel):
    prompt: str
    generated_text: str

@app.on_event("startup")
async def startup_event():
    logger.info("LLM API Startup: Initializing generator instance...")
    get_generator() # Trigger loading model/tokenizer
    logger.info("LLM API Startup complete.")

@app.get("/health_llm", status_code=200) # Different health endpoint
async def health_check_llm():
    generator = get_generator()
    if generator.model is not None and generator.tokenizer is not None:
         return {"status": "ok", "message": "LLM Service is running and model/tokenizer appear loaded."}
    else:
         logger.warning("Health check LLM: Model or tokenizer not loaded.")
         return {"status": "error", "message": "LLM Service running but model/tokenizer failed to load."}

@app.post("/generate", response_model=GenerationResponse)
async def generate_code(
    request: GenerationRequest,
    generator: LLMGenerator = Depends(get_generator)
):
    """Generates text based on the input prompt."""
    start_time = time.time()
    logger.debug(f"Received generation request for prompt: {request.prompt[:50]}...")
    try:
        generated_text = generator.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k
        )
        duration = time.time() - start_time
        logger.info(f"Generation processing took {duration:.4f} seconds.")

        return GenerationResponse(
            prompt=request.prompt,
            generated_text=generated_text
        )
    except Exception as e:
        logger.error(f"Error processing generation request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during generation.")

# To run locally: uvicorn src.serving.api_llm:app --reload --port 8001