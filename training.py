# import modal
# import os
# from typing import List, Dict
# import requests

# # Create Modal app
# app = modal.App("vibetune-training")

# # Define the image with all dependencies
# image = (
#     modal.Image.debian_slim()
#     .pip_install(
#         "torch>=2.0.0",
#         "transformers>=4.35.0",
#         "datasets>=2.14.0",
#         "peft>=0.7.0",  # For LoRA fine-tuning (more efficient)
#         "accelerate>=0.24.0",
#         "requests",
#         "fastapi",  # Required for web endpoints
#     )
# )

# # Create a persistent volume for storing trained models
# model_volume = modal.Volume.from_name("trained-models", create_if_missing=True)

# @app.function(
#     image=image,
#     timeout=3600,  # 1 hour timeout
#     gpu="A10G",  # Use GPU for training
#     volumes={"/models": model_volume},  # Mount volume for persistent storage
# )
# def train_model(
#     project_id: str,
#     training_data: List[Dict],
#     training_job_id: str,
#     callback_url: str
# ):
#     """
#     Fine-tune a model on the provided training data.
    
#     Args:
#         project_id: Your project identifier
#         training_data: List of {input, output, messages, synthetic?} examples
#         training_job_id: Unique job identifier
#         callback_url: URL to call when training completes
    
#     Returns:
#         dict with trainedModelId and modelEndpoint
#     """
#     from transformers import (
#         AutoModelForCausalLM,
#         AutoTokenizer,
#         TrainingArguments,
#         Trainer,
#         DataCollatorForLanguageModeling,
#         TrainerCallback
#     )
#     from peft import LoraConfig, get_peft_model, TaskType
#     from datasets import Dataset
#     import torch
    
#     print(f"[Training] Starting training job: {training_job_id}")
#     print(f"[Training] Training data examples: {len(training_data)}")
    
#     # NEW: Log data composition (original vs synthetic)
#     synthetic_count = sum(1 for item in training_data if item.get('synthetic', False))
#     original_count = len(training_data) - synthetic_count
    
#     print(f"[Training] {'='*60}")
#     print(f"[Training] 📊 Dataset Composition:")
#     print(f"[Training]   - Original examples: {original_count}")
#     print(f"[Training]   - Synthetic examples: {synthetic_count}")
#     print(f"[Training]   - Total: {len(training_data)}")
#     if original_count > 0:
#         print(f"[Training]   - Augmentation ratio: {synthetic_count/original_count:.1f}x")
#     print(f"[Training] {'='*60}")
    
#     # 1. Load base model
#     base_model_name = "Qwen/Qwen2.5-3B-Instruct"  # Your base model
#     print(f"[Training] Loading base model: {base_model_name}")
    
#     tokenizer = AutoTokenizer.from_pretrained(base_model_name)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
    
#     model = AutoModelForCausalLM.from_pretrained(
#         base_model_name,
#         dtype=torch.float16,  # Fixed: use dtype instead of torch_dtype
#         device_map="auto"
#     )
    
#     # 2. Setup LoRA for efficient fine-tuning
#     lora_config = LoraConfig(
#         task_type=TaskType.CAUSAL_LM,
#         r=32,  # LoRA rank - increased from 16 for better learning capacity
#         lora_alpha=64,  # Scaled with r
#         lora_dropout=0.05,
#         target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # More target modules
#     )
#     model = get_peft_model(model, lora_config)
    
#     # 3. Prepare training data
#     # First, let's check the structure of training_data
#     print(f"[Training] First training example structure: {training_data[0] if training_data else 'No data'}")
    
#     def tokenize_function(examples):
#         """Tokenize the formatted examples
#         When batched=True, examples is a dict-like object (LazyBatch) with keys as column names
#         """
#         print("INSIDE THE TOKENIZE FUNCTION? ",examples)
#         # Extract input and output - LazyBatch supports dict-like access
#         inputs = examples['input']
#         outputs = examples['output']
        
#         # Ensure they're lists (they should already be lists when batched=True)
#         if not isinstance(inputs, list):
#             inputs = [inputs]
#         if not isinstance(outputs, list):
#             outputs = [outputs]
        
#         # Format each example
#         texts = [format_example(str(inp), str(out)) for inp, out in zip(inputs, outputs)]
        
#         # Tokenize
#         tokenized = tokenizer(
#             texts,
#             truncation=True,
#             max_length=512,
#             padding="max_length"
#         )
#         tokenized["labels"] = tokenized["input_ids"].copy()
#         return tokenized    


#     def format_example(input_text, output_text):
#         """Format training example using Qwen's chat template"""
#         # Use the same format as inference to ensure the model learns correctly
#         # This matches the Qwen chat template format
#         messages = [
#             {"role": "user", "content": input_text},
#             {"role": "assistant", "content": output_text}
#         ]
        
#         # Apply Qwen's chat template
#         prompt = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=False
#         )
#         return prompt
    
#     # NEW: Clean training data - remove metadata fields before creating dataset
#     clean_training_data = [
#         {k: v for k, v in item.items() if k not in ['synthetic', 'originalExampleIndex', 'generatedAt']}
#         for item in training_data
#     ]
    
#     # Convert to dataset format
#     # Ensure training_data has the right structure
#     print(f"[Training] Training data sample (cleaned): {clean_training_data[0] if clean_training_data else 'No data'}")
#     dataset = Dataset.from_list(clean_training_data)
#     print(f"[Training] Dataset columns: {dataset.column_names}")
#     print(f"[Training] Dataset size: {len(dataset)}")
    
#     tokenized_dataset = dataset.map(
#         tokenize_function,
#         batched=True,
#         remove_columns=dataset.column_names
#     )
    
#     # 4. Setup training arguments
#     training_args = TrainingArguments(
#         output_dir=f"/tmp/results/{training_job_id}",
#         num_train_epochs=5,  # Increased from 3 for better convergence
#         per_device_train_batch_size=2,
#         gradient_accumulation_steps=4,
#         learning_rate=1e-4,  # Lowered from 2e-4 for more stable training
#         warmup_steps=50,  # Reduced warmup for small dataset
#         logging_steps=10,
#         save_strategy="epoch",
#         save_total_limit=2,
#         fp16=True,  # Use mixed precision
#         remove_unused_columns=False,
#     )
    
#     # 5. Create trainer and train
#     data_collator = DataCollatorForLanguageModeling(
#         tokenizer=tokenizer,
#         mlm=False
#     )
    
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_dataset,
#         data_collator=data_collator,
#     )
    
#     print(f"[Training] 🚀 Starting fine-tuning...")
#     print(f"[Training] Configuration:")
#     print(f"[Training]   - Dataset size: {len(tokenized_dataset)} examples")
#     print(f"[Training]   - Epochs: {training_args.num_train_epochs}")
#     print(f"[Training]   - Batch size: {training_args.per_device_train_batch_size}")
#     print(f"[Training]   - Learning rate: {training_args.learning_rate}")
#     print(f"[Training]   - LoRA rank: {lora_config.r}")
#     print(f"[Training]   - LoRA alpha: {lora_config.lora_alpha}")
#     print(f"[Training] {'='*60}")
    
#     train_result = trainer.train()
    
#     # 6. Print training results
#     print(f"[Training] {'='*60}")
#     print(f"[Training] ✅ Training Complete!")
#     print(f"[Training] {'='*60}")
    
#     # Get final metrics
#     metrics = train_result.metrics
#     final_loss = metrics.get('train_loss', 0)
#     print(f"[Training] 📈 Final Training Metrics:")
#     print(f"[Training]   - Total training time: {metrics.get('train_runtime', 0):.2f}s")
#     print(f"[Training]   - Final loss: {final_loss:.4f}")
#     print(f"[Training]   - Samples per second: {metrics.get('train_samples_per_second', 0):.2f}")
#     print(f"[Training]   - Total epochs: {metrics.get('epoch', 0)}")
#     print(f"[Training]   - Original examples used: {original_count}")
#     print(f"[Training]   - Synthetic examples used: {synthetic_count}")
    
#     # Quality assessment
#     if final_loss < 1.0:
#         print(f"[Training] ✅ Excellent! Loss < 1.0 indicates strong learning")
#     elif final_loss < 2.0:
#         print(f"[Training] ✅ Good! Loss < 2.0 indicates decent learning")
#     elif final_loss < 3.0:
#         print(f"[Training] ⚠️  Moderate. Consider more epochs or better data")
#     else:
#         print(f"[Training] ❌ High loss ({final_loss:.4f}). Model may need more training or data quality check")
    
#     print(f"[Training] {'='*60}")
    
#     # 7. Create model identifier first (before saving)
#     import re
#     timestamp_match = re.search(r'training-(\d+)', training_job_id)
#     if timestamp_match:
#         timestamp = timestamp_match.group(1)
#         trained_model_id = f"qwen-finetuned-{timestamp}"
#     else:
#         trained_model_id = f"qwen-finetuned-{training_job_id[:12]}"
    
#     # 8. Save the fine-tuned model to persistent volume
#     # Use model_id as directory name for easy lookup in inference
#     model_save_path = f"/models/{trained_model_id}"
#     trainer.save_model(model_save_path)
#     tokenizer.save_pretrained(model_save_path)
    
#     # Also save with training_job_id for reference
#     backup_path = f"/models/{training_job_id}"
#     trainer.save_model(backup_path)
#     tokenizer.save_pretrained(backup_path)
    
#     # Commit the volume to persist the model
#     model_volume.commit()
    
#     print(f"[Training] 💾 Model saved to: {model_save_path}")
#     print(f"[Training] 💾 Backup saved to: {backup_path}")
#     print(f"[Training] 💾 Model committed to volume: trained-models")
    
#     # 9. Get inference endpoint URL
#     # Option 1: Pass it in the training request (recommended)
#     # Option 2: Use environment variable
#     # Option 3: Construct from Modal's automatic URL (we'll get this from the request)
#     # For now, we'll use the inference endpoint that should be passed in the request
#     # or fall back to environment variable
#     inference_endpoint = "https://adamssamuel9955--vibetune-inference-infer.modal.run"
    
#     # If not in env, we can construct it, but it's better to pass it from Next.js
#     # Modal URLs follow pattern: https://{username}--{app-name}-{function-name}.modal.run
#     # But we don't know username here, so best to pass it from Next.js
#     if not inference_endpoint:
#         # Fallback: construct from known pattern (you'll need to update this)
#         # Or better: pass inferenceEndpointUrl in the training request
#         inference_endpoint = "https://your-username--vibetune-inference-infer.modal.run"
#         print(f"[Training] WARNING: MODAL_INFERENCE_URL not set, using fallback")
    
#     model_endpoint = inference_endpoint
    
#     # 9. Call callback URL to notify your app (non-blocking)
#     # If callback fails, training still succeeded - user can manually update status
#     try:
#         callback_payload = {
#             "trainingJobId": training_job_id,
#             "projectId": project_id,
#             "status": "completed",
#             "trainedModelId": trained_model_id,
#             "modelEndpoint": model_endpoint,
#             "metrics": {
#                 "finalLoss": final_loss,
#                 "originalExamples": original_count,
#                 "syntheticExamples": synthetic_count,
#                 "totalExamples": len(training_data)
#             }
#         }
        
#         response = requests.post(
#             callback_url,
#             json=callback_payload,
#             headers={"Content-Type": "application/json"},
#             timeout=10
#         )
        
#         if response.ok:
#             print(f"[Training] Callback sent successfully")
#         else:
#             print(f"[Training] Callback failed: {response.status_code} - {response.text}")
#             print(f"[Training] Training completed successfully. Use manual update endpoint if callback failed.")
#     except Exception as e:
#         print(f"[Training] Error sending callback: {e}")
#         print(f"[Training] Training completed successfully. Use manual update endpoint:")
#         print(f"[Training] POST /api/training/manual-update with:")
#         print(f"[Training]   projectId: {project_id}")
#         print(f"[Training]   trainedModelId: {trained_model_id}")
#         print(f"[Training]   modelEndpoint: {model_endpoint}")
    
#     return {
#         "trainedModelId": trained_model_id,
#         "modelEndpoint": model_endpoint,
#         "modelPath": model_save_path,
#     }


# # Create HTTP endpoint for training
# @app.function(image=image)
# @modal.fastapi_endpoint(method="POST")
# def train_endpoint(item: dict):
#     """
#     HTTP endpoint: POST /train
#     Receives training request and starts async training
    
#     Expected request body:
#     {
#         "projectId": "project-123",
#         "trainingData": [...],
#         "trainingJobId": "job-123",
#         "callbackUrl": "https://your-app.com/api/training/callback",
#         "inferenceEndpointUrl": "https://your-username--vibetune-inference-infer.modal.run"  # Optional
#     }
#     """
#     project_id = item.get("projectId")
#     training_data = item.get("trainingData", [])
#     training_job_id = item.get("trainingJobId")
#     callback_url = item.get("callbackUrl")
#     inference_endpoint_url = item.get("inferenceEndpointUrl")  # Optional: passed from Next.js
    
#     if not all([project_id, training_data, training_job_id, callback_url]):
#         return {"error": "Missing required fields: projectId, trainingData, trainingJobId, callbackUrl"}
    
#     # Set inference endpoint in environment if provided
#     if inference_endpoint_url:
#         os.environ["MODAL_INFERENCE_URL"] = inference_endpoint_url
    
#     # Start training asynchronously (spawn)
#     train_model.spawn(
#         project_id=project_id,
#         training_data=training_data,
#         training_job_id=training_job_id,
#         callback_url=callback_url,
#     )
    
#     return {
#         "success": True,
#         "message": "Training started",
#         "jobId": training_job_id,
#         "status": "started"
#     }

import modal
import os
from typing import List, Dict
import requests

# Create Modal app
app = modal.App("vibetune-training")

# Define the image with all dependencies (Python 3.11 for unsloth compatibility)
image = (
    modal.Image.from_registry("python:3.11-slim")
    .apt_install("git", "build-essential", "gcc", "g++", "make")
    .pip_install(
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "peft>=0.7.0",
        "accelerate>=0.24.0",
        "requests",
        "fastapi",
        "bitsandbytes",
        "wandb",  # Weights & Biases for experiment tracking
    )
)

# Create a persistent volume for storing trained models
model_volume = modal.Volume.from_name("trained-models")

@app.function(
    image=image,
    timeout=5400,  # 1.5 hours timeout for early stopping
    gpu="L4",  # Use GPU for training (standardized)
    volumes={"/models": model_volume},  # Mount volume for persistent storage
    secrets=[modal.Secret.from_name("wandb-secret")],  # W&B API key
)
def train_model(
    project_id: str,
    training_data: List[Dict],
    training_job_id: str,
    callback_url: str
):
    """
    Fine-tune Qwen3-1.7B using unsloth for optimized performance.
    
    Args:
        project_id: Your project identifier
        training_data: List of {input, output, messages, synthetic?} examples
        training_job_id: Unique job identifier
        callback_url: URL to call when training completes
    
    Returns:
        dict with trainedModelId and modelEndpoint
    """
    from unsloth import FastLanguageModel
    from transformers import (
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from datasets import Dataset
    import torch
    import wandb
    
    # Initialize Weights & Biases
    wandb.init(
        project="vibetune",
        name=training_job_id,
        config={
            "model": "Qwen3-1.7B",
            "lora_r": 8,
            "lora_alpha": 16,
            "learning_rate": 1.5e-4,
            "batch_size": 4,
            "max_seq_length": 2048,
        },
        tags=["fine-tuning", "qwen", "lora"],
    )
    
    print(f"[Training] Starting training job: {training_job_id}")
    print(f"[Training] Training data examples: {len(training_data)}")
    
    # Log data composition (original vs synthetic)
    synthetic_count = sum(1 for item in training_data if item.get('synthetic', False))
    original_count = len(training_data) - synthetic_count
    
    print(f"[Training] {'='*60}")
    print(f"[Training] 📊 Dataset Composition:")
    print(f"[Training]   - Original examples: {original_count}")
    print(f"[Training]   - Synthetic examples: {synthetic_count}")
    print(f"[Training]   - Total: {len(training_data)}")
    if original_count > 0:
        print(f"[Training]   - Augmentation ratio: {synthetic_count/original_count:.1f}x")
    print(f"[Training] {'='*60}")
    
    # 1. Load base model with unsloth (much faster & memory efficient)
    base_model_name = "unsloth/Qwen3-1.7B"
    max_seq_length = 2048  # Qwen3 supports up to 8192, but 2048 is sufficient for most tasks
    
    print(f"[Training] Loading base model with unsloth: {base_model_name}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=False  # Use default precision for training
    )
    
    # 2. Add LoRA adapters with unsloth optimization
    print(f"[Training] Adding LoRA adapters (unsloth optimized)...")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        use_gradient_checkpointing=True,
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 3. Prepare training data
    print(f"[Training] First training example structure: {training_data[0] if training_data else 'No data'}")
    
    def format_example(input_text, output_text):
        """Format training example using Qwen's chat template"""
        messages = [
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text}
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return prompt
    
    def tokenize_function(examples):
        """Tokenize formatted examples with proper batching"""
        inputs = examples['input']
        outputs = examples['output']
        
        # Ensure they're lists
        if not isinstance(inputs, list):
            inputs = [inputs]
        if not isinstance(outputs, list):
            outputs = [outputs]
        
        # Format each example
        texts = [format_example(str(inp), str(out)) for inp, out in zip(inputs, outputs)]
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    # Clean training data - remove metadata fields before creating dataset
    clean_training_data = [
        {k: v for k, v in item.items() if k not in ['synthetic', 'originalExampleIndex', 'generatedAt']}
        for item in training_data
    ]
    
    # Convert to dataset format
    print(f"[Training] Training data sample (cleaned): {clean_training_data[0] if clean_training_data else 'No data'}")
    dataset = Dataset.from_list(clean_training_data)
    print(f"[Training] Dataset columns: {dataset.column_names}")
    print(f"[Training] Dataset size: {len(dataset)}")
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Use all data for training (small dataset - no split)
    train_dataset = tokenized_dataset
    
    # Setup optimized training arguments
    training_args = TrainingArguments(
        output_dir=f"/tmp/results/{training_job_id}",
        num_train_epochs=3,  # Fixed epochs for small dataset (no early stopping)
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,  # Effective batch size = 8
        learning_rate=1.5e-4,
        warmup_ratio=0.1,
        logging_steps=10,  # Log more frequently to monitor progress
        eval_strategy="no",  # No evaluation (using all data for training)
        save_strategy="epoch",  # Save at end of each epoch
        save_total_limit=2,  # Keep last 2 checkpoints
        fp16=True,  # Use fp16 for T4 GPU (bf16 requires Ampere+)
        optim="adamw_torch",
        gradient_checkpointing=True,
        weight_decay=0.01,
        remove_unused_columns=False,
        seed=42,
        report_to="wandb",  # Report metrics to W&B
        dataloader_num_workers=2,
    )
    
    # Create trainer with unsloth optimization
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    print(f"[Training] 🚀 Starting fine-tuning with unsloth...")
    print(f"[Training] Configuration:")
    print(f"[Training]   - Dataset size: {len(tokenized_dataset)} examples")
    print(f"[Training]   - Epochs: {training_args.num_train_epochs}")
    print(f"[Training]   - Batch size: {training_args.per_device_train_batch_size}")
    print(f"[Training]   - Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"[Training]   - Learning rate: {training_args.learning_rate}")
    print(f"[Training]   - Max sequence length: {max_seq_length}")
    print(f"[Training]   - LoRA rank: 8")
    print(f"[Training]   - LoRA alpha: 16")
    print(f"[Training] {'='*60}")
    
    train_result = trainer.train()
    
    # 6. Print training results
    print(f"[Training] {'='*60}")
    print(f"[Training] ✅ Training Complete!")
    print(f"[Training] {'='*60}")
    
    # Get final metrics
    metrics = train_result.metrics
    final_loss = metrics.get('train_loss', 0)
    print(f"[Training] 📈 Final Training Metrics:")
    print(f"[Training]   - Total training time: {metrics.get('train_runtime', 0):.2f}s")
    print(f"[Training]   - Final loss: {final_loss:.4f}")
    print(f"[Training]   - Samples per second: {metrics.get('train_samples_per_second', 0):.2f}")
    print(f"[Training]   - Total epochs: {metrics.get('epoch', 0)}")
    print(f"[Training]   - Original examples used: {original_count}")
    print(f"[Training]   - Synthetic examples used: {synthetic_count}")
    
    # Quality assessment
    if final_loss < 1.0:
        print(f"[Training] ✅ Excellent! Loss < 1.0 indicates strong learning")
    elif final_loss < 2.0:
        print(f"[Training] ✅ Good! Loss < 2.0 indicates decent learning")
    elif final_loss < 3.0:
        print(f"[Training] ⚠️  Moderate. Consider more epochs or better data")
    else:
        print(f"[Training] ❌ High loss ({final_loss:.4f}). Model may need more training or data quality check")
    
    print(f"[Training] {'='*60}")
    
    # 7. Create model identifier
    import re
    timestamp_match = re.search(r'training-(\d+)', training_job_id)
    if timestamp_match:
        timestamp = timestamp_match.group(1)
        trained_model_id = f"qwen-finetuned-{timestamp}"
    else:
        trained_model_id = f"qwen-finetuned-{training_job_id[:12]}"
    
    # 8. Save the fine-tuned model to persistent volume
    model_save_path = f"/models/{trained_model_id}"
    
    print(f"[Training] 💾 Saving model to: {model_save_path}")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    # Also save with training_job_id for reference
    backup_path = f"/models/{training_job_id}"
    trainer.save_model(backup_path)
    tokenizer.save_pretrained(backup_path)
    
    # Clean up quantization config for inference compatibility
    import json
    for save_path in [model_save_path, backup_path]:
        config_path = os.path.join(save_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Remove problematic bitsandbytes quantization keys
            keys_to_remove = [
                'bnb_4bit_compute_dtype', 
                'bnb_4bit_quant_type',
                'bnb_4bit_use_double_quant',
                'load_in_4bit',
                'load_in_8bit',
                'quantization_config'
            ]
            modified = False
            for key in keys_to_remove:
                if key in config:
                    del config[key]
                    modified = True
            
            if modified:
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"[Training] 🔧 Cleaned quantization config from {save_path}")
    
    # Commit the volume to persist the model
    model_volume.commit()
    
    print(f"[Training] 💾 Model saved to: {model_save_path}")
    print(f"[Training] 💾 Backup saved to: {backup_path}")
    print(f"[Training] 💾 Model committed to volume: trained-model")
    
    # 9. Get inference endpoint URL
    inference_endpoint = os.environ.get(
        "MODAL_INFERENCE_URL",
        "https://adamssamuel9955--vibetune-inference-infer.modal.run/inference"
    )
    
    model_endpoint = inference_endpoint
    print(f"[Training] 🔗 Inference endpoint: {model_endpoint}")
    
    # 10. Call callback URL with retry logic to notify your app
    import time
    max_retries = 3
    retry_delay = 2  # Start with 2 seconds
    callback_success = False
    
    for attempt in range(max_retries):
        try:
            callback_payload = {
                "trainingJobId": training_job_id,
                "projectId": project_id,
                "status": "completed",
                "trainedModelId": trained_model_id,
                "modelEndpoint": model_endpoint,
                "metrics": {
                    "finalLoss": final_loss,
                    "originalExamples": original_count,
                    "syntheticExamples": synthetic_count,
                    "totalExamples": len(training_data)
                }
            }
            
            response = requests.post(
                callback_url,
                json=callback_payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.ok:
                print(f"[Training] ✅ Callback sent successfully (attempt {attempt + 1}/{max_retries})")
                callback_success = True
                break
            else:
                print(f"[Training] ❌ Callback failed: {response.status_code}")
                if attempt < max_retries - 1:
                    print(f"[Training] Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
        except Exception as e:
            print(f"[Training] ❌ Callback error: {str(e)} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                print(f"[Training] Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2
    
    if not callback_success:
        print(f"[Training] ⚠️  Callback failed after {max_retries} attempts. Training completed successfully.")
    
    # 11. Cleanup old models from volume (keep last 10 models)
    print(f"[Training] 🧹 Cleaning up old models from volume...")
    try:
        models_dir = "/models"
        if os.path.exists(models_dir):
            existing_models = sorted([
                d for d in os.listdir(models_dir) 
                if os.path.isdir(os.path.join(models_dir, d))
            ])
            
            if len(existing_models) > 10:
                models_to_remove = existing_models[:-10]  # Keep last 10
                for old_model in models_to_remove:
                    try:
                        import shutil
                        model_path = os.path.join(models_dir, old_model)
                        shutil.rmtree(model_path)
                        print(f"[Training] 🗑️  Removed old model: {old_model}")
                    except Exception as e:
                        print(f"[Training] ⚠️  Failed to remove {old_model}: {e}")
                
                model_volume.commit()
                print(f"[Training] ✅ Volume cleanup complete. Keeping last 10 models.")
    except Exception as e:
        print(f"[Training] ⚠️  Cleanup failed: {e}")
    
    # Finish W&B run
    wandb.log({
        "final_loss": final_loss,
        "original_examples": original_count,
        "synthetic_examples": synthetic_count,
        "total_examples": len(training_data),
    })
    wandb.finish()
    
    print(f"[Training] 🎉 Training pipeline completed successfully!")
    return {
        "trainedModelId": trained_model_id,
        "status": "completed",
        "callbackSuccess": callback_success
    }

# Create HTTP endpoint for training
@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def train_endpoint(item: dict):
    """
    HTTP endpoint: POST /train
    Receives training request and starts async training
    
    Expected request body:
    {
        "projectId": "project-123",
        "trainingData": [...],
        "trainingJobId": "job-123",
        "callbackUrl": "https://your-app.com/api/training/callback",
        "inferenceEndpointUrl": "https://your-username--vibetune-inference-infer.modal.run"  # Optional
    }
    """
    project_id = item.get("projectId")
    training_data = item.get("trainingData", [])
    training_job_id = item.get("trainingJobId")
    callback_url = item.get("callbackUrl")
    inference_endpoint_url = item.get("inferenceEndpointUrl")
    
    if not all([project_id, training_data, training_job_id, callback_url]):
        return {"error": "Missing required fields: projectId, trainingData, trainingJobId, callbackUrl"}
    
    # Set inference endpoint in environment if provided
    if inference_endpoint_url:
        os.environ["MODAL_INFERENCE_URL"] = inference_endpoint_url
    
    # Start training asynchronously
    train_model.spawn(
        project_id=project_id,
        training_data=training_data,
        training_job_id=training_job_id,
        callback_url=callback_url,
    )
    
    return {
        "success": True,
        "message": "Training started",
        "jobId": training_job_id,
        "status": "started"
    }

