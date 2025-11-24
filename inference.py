import modal
import os
from typing import Optional

app = modal.App("vibetune-inference")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "peft>=0.7.0",
        "accelerate>=0.24.0",
        "fastapi",  # Required for web endpoints
    )
)

# Store trained models in a volume (shared with training)
model_volume = modal.Volume.from_name("trained-models", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/models": model_volume},  # Mount volume to access trained models
    gpu="T4",  # Use GPU for inference
    timeout=600,  # Increased to 10 minutes to handle cold starts
    container_idle_timeout=300,  # Keep container warm for 5 minutes
)
@modal.fastapi_endpoint(method="POST")
def infer(item: dict):
    """
    HTTP endpoint: POST /inference
    Run inference with trained model or base model
    
    Request body:
    {
        "prompt": "User: What is AI?\nAssistant:",
        "modelId": "qwen-finetuned-abc123",  # Optional: trained model ID
        "temperature": 0.7,
        "max_tokens": 250,
        "top_p": 0.9
    }
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    prompt = item.get("prompt", "")
    model_id = item.get("modelId")  # Optional: trained model ID
    temperature = item.get("temperature", 0.7)
    max_tokens = item.get("max_tokens", 250)
    top_p = item.get("top_p", 0.9)
    
    print(f"[Inference] 📥 Received request:")
    print(f"[Inference]   - modelId: {model_id or 'None (using base model)'}")
    print(f"[Inference]   - prompt length: {len(prompt)} chars")
    print(f"[Inference]   - temperature: {temperature}, max_tokens: {max_tokens}")
    
    if not prompt:
        return {"error": "Prompt is required"}, 400
    
    base_model_name = "Qwen/Qwen2.5-3B-Instruct"
    
    # If modelId is provided, try to load trained model from volume
    if model_id and (model_id.startswith("qwen-finetuned-") or model_id.startswith("training-")):
        import os
        print(f"[Inference] Reloading volume to see latest models...")
        model_volume.reload()
        
        # Models are saved as: /models/{trained_model_id}
        # Handle both old format (qwen-finetuned-training-176) and new format (qwen-finetuned-1763651478088)
        model_path = f"/models/{model_id}"
        
        print(f"[Inference] Attempting to load trained model: {model_id}")
        print(f"[Inference] Model path: {model_path}")
        
        # First, try the exact path
        if os.path.exists(model_path):
            try:
                print(f"[Inference] ✅ LOADING TRAINED MODEL from: {model_path}")
                from peft import PeftModel
                
                # Load base model first
                base_model_name = "Qwen/Qwen2.5-3B-Instruct"
                print(f"[Inference] Loading base model: {base_model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    dtype=torch.float16,
                    device_map="auto"
                )
                
                # Load LoRA adapters on top of base model
                print(f"[Inference] Loading LoRA adapters from: {model_path}")
                model = PeftModel.from_pretrained(base_model, model_path)
                print(f"[Inference] ✅✅✅ SUCCESSFULLY LOADED TRAINED MODEL with LoRA adapters: {model_id}")
                print(f"[Inference] ✅ Model path exists and LoRA adapters loaded correctly")
            except Exception as e:
                print(f"[Inference] ❌ ERROR loading trained model: {e}")
                print(f"[Inference] ⚠️  Falling back to base model")
                model_path = None
        else:
            # Model not found at exact path - try to find it by searching
            print(f"[Inference] Trained model not found at exact path: {model_path}")
            
            # List available models for debugging
            available_models = []
            if os.path.exists("/models"):
                try:
                    dirs = [d for d in os.listdir("/models") if os.path.isdir(os.path.join("/models", d))]
                    print(f"[Inference] Available model directories: {dirs}")
                    available_models = dirs
                    
                    # Try to find a matching model (fuzzy match)
                    # If model_id is "qwen-finetuned-training-176", look for directories containing "training-176"
                    if "training-" in model_id:
                        # Extract the key part (e.g., "training-176" from "qwen-finetuned-training-176")
                        key_part = model_id.split("-")[-2] + "-" + model_id.split("-")[-1]  # e.g., "training-176"
                        matching_dirs = [d for d in dirs if key_part in d]
                        if matching_dirs:
                            model_path = f"/models/{matching_dirs[0]}"
                            print(f"[Inference] 🔍 Found matching model: {model_path}")
                            try:
                                print(f"[Inference] ✅ LOADING TRAINED MODEL from: {model_path}")
                                from peft import PeftModel
                                
                                # Load base model first
                                base_model_name = "Qwen/Qwen2.5-3B-Instruct"
                                tokenizer = AutoTokenizer.from_pretrained(model_path)
                                base_model = AutoModelForCausalLM.from_pretrained(
                                    base_model_name,
                                    dtype=torch.float16,
                                    device_map="auto"
                                )
                                
                                # Load LoRA adapters
                                model = PeftModel.from_pretrained(base_model, model_path)
                                print(f"[Inference] ✅✅✅ SUCCESSFULLY LOADED TRAINED MODEL with LoRA: {matching_dirs[0]}")
                                print(f"[Inference] ✅ Model loaded using fuzzy match with LoRA adapters")
                            except Exception as e:
                                print(f"[Inference] ❌ ERROR loading matched model: {e}")
                                model_path = None
                        else:
                            print(f"[Inference] ⚠️  No matching model found. Available: {dirs}")
                            model_path = None
                    else:
                        # Try exact match with different formats
                        # If looking for "qwen-finetuned-1763651478088", also try "training-1763651478088-..."
                        timestamp_match = model_id.replace("qwen-finetuned-", "")
                        matching_dirs = [d for d in dirs if timestamp_match in d]
                        if matching_dirs:
                            model_path = f"/models/{matching_dirs[0]}"
                            print(f"[Inference] 🔍 Found matching model by timestamp: {model_path}")
                            try:
                                print(f"[Inference] ✅ LOADING TRAINED MODEL from: {model_path}")
                                from peft import PeftModel
                                
                                # Load base model first
                                base_model_name = "Qwen/Qwen2.5-3B-Instruct"
                                tokenizer = AutoTokenizer.from_pretrained(model_path)
                                base_model = AutoModelForCausalLM.from_pretrained(
                                    base_model_name,
                                    dtype=torch.float16,
                                    device_map="auto"
                                )
                                
                                # Load LoRA adapters
                                model = PeftModel.from_pretrained(base_model, model_path)
                                print(f"[Inference] ✅✅✅ SUCCESSFULLY LOADED TRAINED MODEL with LoRA: {matching_dirs[0]}")
                            except Exception as e:
                                print(f"[Inference] ❌ ERROR loading matched model: {e}")
                                model_path = None
                        else:
                            print(f"[Inference] ⚠️  No matching model found. Available: {dirs}")
                            model_path = None
                except Exception as e:
                    print(f"[Inference] Error listing models: {e}")
                    model_path = None
            
            if not model_path or not os.path.exists(model_path):
                print(f"[Inference] ⚠️  Falling back to base model")
                print(f"[Inference] 💡 Tip: Check Modal training logs for the exact model ID that was saved")
                model_path = None
    
    # Load base model if trained model not found or not requested
    if 'model' not in locals() or 'tokenizer' not in locals():
        print(f"[Inference] 🔵 LOADING BASE MODEL: {base_model_name}")
        print(f"[Inference] ⚠️  No trained model will be used")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.float16,
            device_map="auto"
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"[Inference] Model loaded successfully")
    
    try:
        # Tokenize and generate
        print(f"[Inference] Tokenizing prompt...")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs["input_ids"].shape[1]
        
        print(f"[Inference] Input prompt length: {len(prompt)} chars, {input_length} tokens")
        print(f"[Inference] Prompt preview: {prompt[:200]}...")
        
        print(f"[Inference] Starting generation with max_new_tokens={max_tokens}, temperature={temperature}...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        print(f"[Inference] Generation complete. Output shape: {outputs.shape}")
        
        # Decode the full generated sequence
        generated_tokens = outputs[0][input_length:]  # Only the newly generated tokens
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Clean up the response
        response_text = generated_text.strip()
        
        # Remove any remaining prompt artifacts
        # Sometimes the model repeats parts of the prompt
        if prompt in response_text:
            response_text = response_text.replace(prompt, "").strip()
        
        print(f"[Inference] Generated tokens: {len(generated_tokens)}")
        print(f"[Inference] Response text length: {len(response_text)}")
        print(f"[Inference] Response preview: {response_text[:200]}")
        
        # If response is empty or too short, return a helpful message
        if not response_text or len(response_text) < 3:
            print(f"[Inference] WARNING: Empty or very short response generated!")
            print(f"[Inference] Full generated text: {generated_text}")
            response_text = "I apologize, but I couldn't generate a meaningful response. Please try rephrasing your question."
        
        return {
            "text": response_text,
            "tokens": len(outputs[0]),
            "finish_reason": "stop"
        }
    except Exception as e:
        print(f"[Inference] ERROR during generation: {str(e)}")
        print(f"[Inference] Error type: {type(e).__name__}")
        import traceback
        print(f"[Inference] Traceback: {traceback.format_exc()}")
        raise

