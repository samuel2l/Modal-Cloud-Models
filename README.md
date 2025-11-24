# Modal Training Setup

This directory contains the Modal functions for model training and inference.

## Quick Start

### 1. Install Modal

```bash
pip install modal
modal token new
```

### 2. Deploy Training Endpoint

```bash
modal deploy training.py
```

This will output a URL like:
```
https://your-username--vibetune-training-train-endpoint.modal.run
```

### 3. Deploy Inference Endpoint

```bash
modal deploy inference.py
```

This will output a URL like:
```
https://your-username--vibetune-inference-infer.modal.run
```

### 4. Update .env in Next.js Project

Add to `/Users/samuel/vibetune/.env`:

```env
MODAL_TRAINING_URL=https://your-username--vibetune-training-train-endpoint.modal.run
MODAL_INFERENCE_URL=https://your-username--vibetune-inference-infer.modal.run
```

## Files

- `training.py` - Training endpoint that fine-tunes models
- `inference.py` - Inference endpoint for running models
- `README.md` - This file

## Testing

### Test Training Endpoint

```bash
curl -X POST https://your-username--vibetune-training-train-endpoint.modal.run \
  -H "Content-Type: application/json" \
  -d '{
    "projectId": "test",
    "trainingData": [{"input": "Hello", "output": "Hi!", "messages": []}],
    "trainingJobId": "test-123",
    "callbackUrl": "http://localhost:3000/api/training/callback"
  }'
```

### Test Inference Endpoint

```bash
curl -X POST https://your-username--vibetune-inference-infer.modal.run \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "User: What is AI?\nAssistant:",
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

## Notes

- Training uses LoRA for efficient fine-tuning
- Models are saved temporarily during training
- For persistent storage, implement volume or S3 storage
- GPU costs apply: ~$1-2/hour for A10G during training

