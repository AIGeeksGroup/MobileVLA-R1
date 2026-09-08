import os
import io
import base64
import traceback
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import uvicorn
import numpy as np
import torch
from mobilevla.data import policy_question
from mobilevla.actions import parse_answer
import argparse

# Import the inference class
from inference import NaVILAImageInference


class InferenceRequest(BaseModel):
    instruction: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


class InferenceResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    action: Optional[str] = None
    velocity: Optional[List[float]] = None


app = FastAPI(title="NaVILA Remote Inference API", version="1.0.0")

# Global inferencer instance
inferencer = None


def initialize_model(model_path: str, lora_path: Optional[str] = None, use_flash_attn: bool = True):
    """Initialize the inference model."""
    global inferencer
    try:
        print("Initializing model...")
        inferencer = NaVILAImageInference(
            model_path=model_path,
            lora_path=lora_path,
            use_flash_attn=use_flash_attn
        )
        print("Model initialized successfully!")
        return True
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        traceback.print_exc()
        return False


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if inferencer is None:
        return {"status": "error", "message": "Model not initialized"}
    return {"status": "healthy", "message": "Model is ready"}


async def decode_depth_uploads(files, scale):
    if not files:
        return None
    maps = []
    for upload in files:
        raw = await upload.read()
        if (upload.filename or '').lower().endswith('.npy'):
            values = np.load(io.BytesIO(raw), allow_pickle=False).astype(np.float32)
        else:
            values = np.asarray(Image.open(io.BytesIO(raw)), dtype=np.float32) / scale
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError('Depth uploads must be finite single-channel maps')
        maps.append(torch.from_numpy(values).unsqueeze(0).unsqueeze(0))
    return maps


@app.post("/inference", response_model=InferenceResponse)
async def run_inference(
    image: UploadFile = File(...),
    instruction: str = Form(...),
    max_new_tokens: int = Form(512),
    temperature: float = Form(0.7),
    top_p: float = Form(0.9),
    do_sample: bool = Form(True),
    depth_maps: Optional[List[UploadFile]] = File(None),
    task: str = Form("navigation"),
    derive_points: bool = Form(True)
):
    """Run single-image inference."""
    if inferencer is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        # Validate image file
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Uploaded file is not an image")
        
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        image_tensor = inferencer.load_image_from_pil(pil_image)
        
        response = inferencer.generate_response(
            image_input=image_tensor,
            question=policy_question(instruction, task),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            depth_input=await decode_depth_uploads(depth_maps, inferencer.depth_scale),
            point_cloud="from_depth" if derive_points and depth_maps else None
        )
        
        parsed = parse_answer(response)
        return InferenceResponse(success=True, response=response, action=parsed.action,
                                 velocity=list(parsed.velocity) if parsed.velocity is not None else None)
        
    except Exception as e:
        error_msg = f"Inference failed: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return InferenceResponse(success=False, error=error_msg)


@app.post("/inference_batch", response_model=InferenceResponse)
async def run_inference_batch(
    images: List[UploadFile] = File(...),
    instruction: str = Form(...),
    max_new_tokens: int = Form(512),
    temperature: float = Form(0.7),
    top_p: float = Form(0.9),
    do_sample: bool = Form(True),
    depth_maps: Optional[List[UploadFile]] = File(None),
    task: str = Form("navigation"),
    derive_points: bool = Form(True)
):
    """Run inference for multiple images."""
    if inferencer is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        image_tensors = []
        for image_file in images:
            if not image_file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"File {image_file.filename} is not an image")
            
            image_bytes = await image_file.read()
            pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_tensor = inferencer.load_image_from_pil(pil_image)
            image_tensors.append(image_tensor.squeeze(0))  # remove batch dim
        
        import torch
        combined_tensor = torch.stack(image_tensors)
        
        response = inferencer.generate_response(
            image_input=combined_tensor,
            question=policy_question(instruction, task),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            depth_input=await decode_depth_uploads(depth_maps, inferencer.depth_scale),
            point_cloud="from_depth" if derive_points and depth_maps else None
        )
        
        parsed = parse_answer(response)
        return InferenceResponse(success=True, response=response, action=parsed.action,
                                 velocity=list(parsed.velocity) if parsed.velocity is not None else None)
        
    except Exception as e:
        error_msg = f"Batch inference failed: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return InferenceResponse(success=False, error=error_msg)


def main():
    parser = argparse.ArgumentParser(description="NaVILA Remote Inference Server")
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/model/MobileVla-r1-8b",
                       help="Path to the base model")
    parser.add_argument("--lora_path", type=str, default=None,
                       help="Path to the LoRA weights")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to bind the server")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to bind the server")
    parser.add_argument("--no_flash_attn", action="store_true",
                       help="Disable Flash Attention")
    
    args = parser.parse_args()
    
    if not initialize_model(
        model_path=args.model_path,
        lora_path=args.lora_path,
        use_flash_attn=not args.no_flash_attn
    ):
        print("Failed to initialize model. Exiting...")
        return
    
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()