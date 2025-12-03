import modal
import os
from typing import Optional

app = modal.App("vibetune-inference")

image = (
    modal.Image.from_registry("python:3.11-slim")
    .apt_install("git", "build-essential", "gcc", "g++", "make")
    .pip_install(
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.35.0",
        "peft>=0.7.0",
        "accelerate>=0.24.0",
        "fastapi",
        "bitsandbytes",
    )
)

model_volume = modal.Volume.from_name("trained-models")

@app.cls(
    image=image,
    volumes={"/models": model_volume},
    gpu="T4",
    scaledown_window=1800,  # Keep container warm for 30 minutes
    max_containers=1,  # Only allow 1 container at a time
    # min_containers=1,  # Always keep 1 warm container ready
)
class InferenceService:
    """High-performance inference service with preloaded model."""
    
    @modal.enter()
    def load_model(self):
        """Preload base model when container starts - eliminates cold start for requests."""
        from unsloth import FastLanguageModel
        
        print("[Inference] 🚀 Container starting - preloading base model...")
        
        self.base_model_name = "unsloth/Qwen3-1.7B"
        self.max_seq_length = 2048
        self.current_model_id = "base"
        
        # Preload base model
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=False,
        )
        self.model = FastLanguageModel.for_inference(self.model)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("[Inference] ✅ Base model preloaded and ready!")
        print(f"[Inference] 📊 Model: {self.base_model_name}")
    
    def _load_trained_model(self, model_id: str) -> bool:
        """Load a trained model from volume. Returns True if successful."""
        from unsloth import FastLanguageModel
        import json
        
        if self.current_model_id == model_id:
            print(f"[Inference] ✅ Model already loaded: {model_id}")
            return True
        
        model_volume.reload()
        model_path = f"/models/{model_id}"
        
        # Try fuzzy matching if exact path doesn't exist
        if not os.path.exists(model_path):
            if os.path.exists("/models"):
                dirs = [d for d in os.listdir("/models") if os.path.isdir(os.path.join("/models", d))]
                matching = [d for d in dirs if model_id.split("-")[-1] in d]
                if matching:
                    model_path = f"/models/{matching[0]}"
                    print(f"[Inference] 🔍 Fuzzy matched to: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"[Inference] ❌ Model not found: {model_id}")
            return False
        
        try:
            print(f"[Inference] 📦 Loading trained model from: {model_path}")
            
            # Clean quantization config if needed
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                keys_to_remove = [
                    'bnb_4bit_compute_dtype', 'bnb_4bit_quant_type',
                    'bnb_4bit_use_double_quant', 'load_in_4bit',
                    'load_in_8bit', 'quantization_config'
                ]
                modified = False
                for key in keys_to_remove:
                    if key in config:
                        del config[key]
                        modified = True
                
                if modified:
                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=2)
                    print(f"[Inference] 🔧 Cleaned quantization config")
            
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=False,
            )
            self.model = FastLanguageModel.for_inference(self.model)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.current_model_id = model_id
            print(f"[Inference] ✅ Trained model loaded: {model_id}")
            return True
            
        except Exception as e:
            print(f"[Inference] ❌ Error loading trained model: {e}")
            return False
    
    @modal.method()
    def generate(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 250,
        top_p: float = 0.9,
    ) -> dict:
        """Generate text using preloaded or requested model."""
        import torch
        import time
        
        start_time = time.time()
        
        print(f"[Inference] 📥 Request received")
        print(f"[Inference]   - modelId: {model_id or 'base'}")
        print(f"[Inference]   - prompt: {prompt[:50]}...")
        print(f"[Inference]   - current loaded: {self.current_model_id}")
        
        # Determine target model
        target_model_id = None
        if model_id and (model_id.startswith("qwen-finetuned-") or model_id.startswith("training-")):
            target_model_id = model_id
        
        # Load trained model if requested and different from current
        model_type = "base"
        if target_model_id:
            if self._load_trained_model(target_model_id):
                model_type = "finetuned"
            else:
                print(f"[Inference] ⚠️ Falling back to base model")
        
        try:
            print(f"[Inference] 📝 Generating: temp={temperature}, max_tokens={max_tokens}")
            
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
            input_length = inputs["input_ids"].shape[1]
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    length_penalty=1.0,
                )
            
            generated_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Clean up thinking tags
            generated_text = generated_text.replace("<think>", "").replace("</think>", "").strip()
            
            if not generated_text or len(generated_text) < 3:
                generated_text = "I couldn't generate a meaningful response. Please try rephrasing."
            
            inference_time = time.time() - start_time
            print(f"[Inference] ✅ Generated {len(generated_tokens)} tokens in {inference_time:.2f}s")
            print(f"[Inference] Model type: {model_type}")
            
            return {
                "text": generated_text,
                "tokens": len(generated_tokens),
                "finish_reason": "stop",
                "model_type": model_type,
                "inference_time": inference_time,
            }
            
        except Exception as e:
            print(f"[Inference] ❌ Generation error: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return {
                "text": "",
                "tokens": 0,
                "finish_reason": "error",
                "error": str(e),
            }
    
    @modal.asgi_app()
    def serve(self):
        """Serve FastAPI app with preloaded model."""
        # Import FastAPI and Pydantic here (only needed at runtime in Modal container)
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field, validator
        
        # Define Pydantic models here (only needed at runtime)
        class InferenceRequest(BaseModel):
            prompt: str = Field(..., min_length=1, max_length=5000, description="Prompt text for inference")
            modelId: Optional[str] = Field(None, pattern=r'^[a-zA-Z0-9\-_]*$', description="Trained model ID or None for base model")
            temperature: float = Field(0.7, ge=0.0, le=2.0, description="Generation temperature (0-2)")
            max_tokens: int = Field(250, ge=1, le=4096, description="Max tokens to generate")
            top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
            
            @validator('temperature')
            def temperature_reasonable(cls, v):
                if v > 1.5:
                    raise ValueError('Temperature > 1.5 may cause unstable generation')
                return v

        class InferenceResponse(BaseModel):
            text: str
            tokens: int
            finish_reason: str = "stop"
        
        web_app = FastAPI(title="VibeTune Inference API", version="2.0")
        
        @web_app.post("/inference", response_model=InferenceResponse)
        async def inference_endpoint(request: InferenceRequest):
            """Run inference with preloaded model - ultra-fast responses."""
            print(f"[Inference] 📥 Endpoint request: {request.prompt[:50]}...")
            
            result = self.generate.local(
                prompt=request.prompt,
                model_id=request.modelId,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
            )
            
            print(f"[Inference] Result: inference_time={result.get('inference_time', 'N/A')}s")
            
            if "error" in result:
                print(f"[Inference] ❌ Error: {result['error']}")
                raise HTTPException(status_code=500, detail=result.get("error", "Inference failed"))
            
            if not result.get("text"):
                return InferenceResponse(
                    text="Model generated no response. Please try again.",
                    tokens=0,
                    finish_reason="empty",
                )
            
            print(f"[Inference] 📤 Success")
            return InferenceResponse(
                text=result["text"],
                tokens=result.get("tokens", 0),
                finish_reason=result.get("finish_reason", "stop"),
            )
        
        @web_app.get("/health")
        async def health():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "vibetune-inference",
                "model_volume": "trained-models",
                "backend": "unsloth-optimized",
                "current_model": self.current_model_id,
            }
        
        @web_app.get("/")
        async def root():
            """API info endpoint."""
            return {
                "message": "VibeTune Inference API v2.0",
                "endpoints": ["/inference", "/health", "/docs"],
                "backend": "unsloth-optimized",
                "features": ["preloaded-model", "ultra-fast-inference"],
            }
        
        return web_app

