# import modal
# import os
# from typing import Optional

# app = modal.App("vibetune-inference")

# image = (
#     modal.Image.debian_slim()
#     .pip_install(
#         "torch>=2.0.0",
#         "transformers>=4.35.0",
#         "peft>=0.7.0",
#         "accelerate>=0.24.0",
#         "fastapi",  # Required for web endpoints
#     )
# )

# # Store trained models in a volume (shared with training)
# model_volume = modal.Volume.from_name("trained-models", create_if_missing=True)

# @app.function(
#     image=image,
#     volumes={"/models": model_volume},  # Mount volume to access trained models
#     gpu="T4",  # Use GPU for inference
#     timeout=600,  # Increased to 10 minutes to handle cold starts
#     container_idle_timeout=300,  # Keep container warm for 5 minutes
# )
# @modal.fastapi_endpoint(method="POST")
# def infer(item: dict):
#     """
#     HTTP endpoint: POST /inference
#     Run inference with trained model or base model
    
#     Request body:
#     {
#         "prompt": "User: What is AI?\nAssistant:",
#         "modelId": "qwen-finetuned-abc123",  # Optional: trained model ID
#         "temperature": 0.7,
#         "max_tokens": 250,
#         "top_p": 0.9
#     }
#     """
#     from transformers import AutoModelForCausalLM, AutoTokenizer
#     import torch
    
#     prompt = item.get("prompt", "")
#     model_id = item.get("modelId")  # Optional: trained model ID
#     temperature = item.get("temperature", 0.7)
#     max_tokens = item.get("max_tokens", 250)
#     top_p = item.get("top_p", 0.9)
    
#     print(f"[Inference] 📥 Received request:")
#     print(f"[Inference]   - modelId: {model_id or 'None (using base model)'}")
#     print(f"[Inference]   - prompt length: {len(prompt)} chars")
#     print(f"[Inference]   - temperature: {temperature}, max_tokens: {max_tokens}")
    
#     if not prompt:
#         return {"error": "Prompt is required"}, 400
    
#     base_model_name = "Qwen/Qwen2.5-3B-Instruct"
    
#     # If modelId is provided, try to load trained model from volume
#     if model_id and (model_id.startswith("qwen-finetuned-") or model_id.startswith("training-")):
#         import os
#         print(f"[Inference] Reloading volume to see latest models...")
#         model_volume.reload()
        
#         # Models are saved as: /models/{trained_model_id}
#         # Handle both old format (qwen-finetuned-training-176) and new format (qwen-finetuned-1763651478088)
#         model_path = f"/models/{model_id}"
        
#         print(f"[Inference] Attempting to load trained model: {model_id}")
#         print(f"[Inference] Model path: {model_path}")
        
#         # First, try the exact path
#         if os.path.exists(model_path):
#             try:
#                 print(f"[Inference] ✅ LOADING TRAINED MODEL from: {model_path}")
#                 from peft import PeftModel
                
#                 # Load base model first
#                 base_model_name = "Qwen/Qwen2.5-3B-Instruct"
#                 print(f"[Inference] Loading base model: {base_model_name}")
#                 tokenizer = AutoTokenizer.from_pretrained(model_path)
#                 base_model = AutoModelForCausalLM.from_pretrained(
#                     base_model_name,
#                     dtype=torch.float16,
#                     device_map="auto"
#                 )
                
#                 # Load LoRA adapters on top of base model
#                 print(f"[Inference] Loading LoRA adapters from: {model_path}")
#                 model = PeftModel.from_pretrained(base_model, model_path)
#                 print(f"[Inference] ✅✅✅ SUCCESSFULLY LOADED TRAINED MODEL with LoRA adapters: {model_id}")
#                 print(f"[Inference] ✅ Model path exists and LoRA adapters loaded correctly")
#             except Exception as e:
#                 print(f"[Inference] ❌ ERROR loading trained model: {e}")
#                 print(f"[Inference] ⚠️  Falling back to base model")
#                 model_path = None
#         else:
#             # Model not found at exact path - try to find it by searching
#             print(f"[Inference] Trained model not found at exact path: {model_path}")
            
#             # List available models for debugging
#             available_models = []
#             if os.path.exists("/models"):
#                 try:
#                     dirs = [d for d in os.listdir("/models") if os.path.isdir(os.path.join("/models", d))]
#                     print(f"[Inference] Available model directories: {dirs}")
#                     available_models = dirs
                    
#                     # Try to find a matching model (fuzzy match)
#                     # If model_id is "qwen-finetuned-training-176", look for directories containing "training-176"
#                     if "training-" in model_id:
#                         # Extract the key part (e.g., "training-176" from "qwen-finetuned-training-176")
#                         key_part = model_id.split("-")[-2] + "-" + model_id.split("-")[-1]  # e.g., "training-176"
#                         matching_dirs = [d for d in dirs if key_part in d]
#                         if matching_dirs:
#                             model_path = f"/models/{matching_dirs[0]}"
#                             print(f"[Inference] 🔍 Found matching model: {model_path}")
#                             try:
#                                 print(f"[Inference] ✅ LOADING TRAINED MODEL from: {model_path}")
#                                 from peft import PeftModel
                                
#                                 # Load base model first
#                                 base_model_name = "Qwen/Qwen2.5-3B-Instruct"
#                                 tokenizer = AutoTokenizer.from_pretrained(model_path)
#                                 base_model = AutoModelForCausalLM.from_pretrained(
#                                     base_model_name,
#                                     dtype=torch.float16,
#                                     device_map="auto"
#                                 )
                                
#                                 # Load LoRA adapters
#                                 model = PeftModel.from_pretrained(base_model, model_path)
#                                 print(f"[Inference] ✅✅✅ SUCCESSFULLY LOADED TRAINED MODEL with LoRA: {matching_dirs[0]}")
#                                 print(f"[Inference] ✅ Model loaded using fuzzy match with LoRA adapters")
#                             except Exception as e:
#                                 print(f"[Inference] ❌ ERROR loading matched model: {e}")
#                                 model_path = None
#                         else:
#                             print(f"[Inference] ⚠️  No matching model found. Available: {dirs}")
#                             model_path = None
#                     else:
#                         # Try exact match with different formats
#                         # If looking for "qwen-finetuned-1763651478088", also try "training-1763651478088-..."
#                         timestamp_match = model_id.replace("qwen-finetuned-", "")
#                         matching_dirs = [d for d in dirs if timestamp_match in d]
#                         if matching_dirs:
#                             model_path = f"/models/{matching_dirs[0]}"
#                             print(f"[Inference] 🔍 Found matching model by timestamp: {model_path}")
#                             try:
#                                 print(f"[Inference] ✅ LOADING TRAINED MODEL from: {model_path}")
#                                 from peft import PeftModel
                                
#                                 # Load base model first
#                                 base_model_name = "Qwen/Qwen2.5-3B-Instruct"
#                                 tokenizer = AutoTokenizer.from_pretrained(model_path)
#                                 base_model = AutoModelForCausalLM.from_pretrained(
#                                     base_model_name,
#                                     dtype=torch.float16,
#                                     device_map="auto"
#                                 )
                                
#                                 # Load LoRA adapters
#                                 model = PeftModel.from_pretrained(base_model, model_path)
#                                 print(f"[Inference] ✅✅✅ SUCCESSFULLY LOADED TRAINED MODEL with LoRA: {matching_dirs[0]}")
#                             except Exception as e:
#                                 print(f"[Inference] ❌ ERROR loading matched model: {e}")
#                                 model_path = None
#                         else:
#                             print(f"[Inference] ⚠️  No matching model found. Available: {dirs}")
#                             model_path = None
#                 except Exception as e:
#                     print(f"[Inference] Error listing models: {e}")
#                     model_path = None
            
#             if not model_path or not os.path.exists(model_path):
#                 print(f"[Inference] ⚠️  Falling back to base model")
#                 print(f"[Inference] 💡 Tip: Check Modal training logs for the exact model ID that was saved")
#                 model_path = None
    
#     # Load base model if trained model not found or not requested
#     if 'model' not in locals() or 'tokenizer' not in locals():
#         print(f"[Inference] 🔵 LOADING BASE MODEL: {base_model_name}")
#         print(f"[Inference] ⚠️  No trained model will be used")
#         tokenizer = AutoTokenizer.from_pretrained(base_model_name)
#         model = AutoModelForCausalLM.from_pretrained(
#             base_model_name,
#             dtype=torch.float16,
#             device_map="auto"
#         )
    
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
    
#     print(f"[Inference] Model loaded successfully")
    
#     try:
#         # Tokenize and generate
#         print(f"[Inference] Tokenizing prompt...")
#         inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#         input_length = inputs["input_ids"].shape[1]
        
#         print(f"[Inference] Input prompt length: {len(prompt)} chars, {input_length} tokens")
#         print(f"[Inference] Prompt preview: {prompt[:200]}...")
        
#         print(f"[Inference] Starting generation with max_new_tokens={max_tokens}, temperature={temperature}...")
#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=max_tokens,
#                 temperature=temperature,
#                 top_p=top_p,
#                 do_sample=True,
#                 pad_token_id=tokenizer.eos_token_id,
#                 eos_token_id=tokenizer.eos_token_id,
#             )
        
#         print(f"[Inference] Generation complete. Output shape: {outputs.shape}")
        
#         # Decode the full generated sequence
#         generated_tokens = outputs[0][input_length:]  # Only the newly generated tokens
#         generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
#         # Clean up the response
#         response_text = generated_text.strip()
        
#         # Remove any remaining prompt artifacts
#         # Sometimes the model repeats parts of the prompt
#         if prompt in response_text:
#             response_text = response_text.replace(prompt, "").strip()
        
#         print(f"[Inference] Generated tokens: {len(generated_tokens)}")
#         print(f"[Inference] Response text length: {len(response_text)}")
#         print(f"[Inference] Response preview: {response_text[:200]}")
        
#         # If response is empty or too short, return a helpful message
#         if not response_text or len(response_text) < 3:
#             print(f"[Inference] WARNING: Empty or very short response generated!")
#             print(f"[Inference] Full generated text: {generated_text}")
#             response_text = "I apologize, but I couldn't generate a meaningful response. Please try rephrasing your question."
        
#         return {
#             "text": response_text,
#             "tokens": len(outputs[0]),
#             "finish_reason": "stop"
#         }
#     except Exception as e:
#         print(f"[Inference] ERROR during generation: {str(e)}")
#         print(f"[Inference] Error type: {type(e).__name__}")
#         import traceback
#         print(f"[Inference] Traceback: {traceback.format_exc()}")
#         raise

import modal
import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

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

