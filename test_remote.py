import requests
import argparse
import os
from typing import List, Optional
import json


class NaVILAClient:
    def __init__(self, server_url: str = "http://localhost:8000"):
        """
        Initialize the client wrapper for the remote inference server.

        Args:
            server_url: Remote server endpoint.
        """
        self.server_url = server_url.rstrip('/')
        
    def health_check(self):
        """Check the health status of the remote server."""
        try:
            response = requests.get(f"{self.server_url}/health")
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": f"Connection failed: {e}"}
    
    def inference(self, 
                 image_path: str,
                 instruction: str,
                 max_new_tokens: int = 512,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 do_sample: bool = True) -> dict:
        """
        Send a single image to the server for inference.

        Args:
            image_path: Local path to the image.
            instruction: Instruction text provided to the model.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            do_sample: Whether to enable sampling.

        Returns:
            Inference result payload.
        """
        if not os.path.exists(image_path):
            return {"success": False, "error": f"Image file not found: {image_path}"}
        
        try:
            # Prepare file payload and form data
            with open(image_path, 'rb') as f:
                files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
                data = {
                    'instruction': instruction,
                    'max_new_tokens': max_new_tokens,
                    'temperature': temperature,
                    'top_p': top_p,
                    'do_sample': do_sample
                }
                
                # Send request to remote inference endpoint
                response = requests.post(
                    f"{self.server_url}/inference",
                    files=files,
                    data=data,
                    timeout=300  # 5-minute timeout
                )
                
                return response.json()
                
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Request failed: {e}"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {e}"}
    
    def inference_batch(self,
                       image_paths: List[str],
                       instruction: str,
                       max_new_tokens: int = 512,
                       temperature: float = 0.7,
                       top_p: float = 0.9,
                       do_sample: bool = True) -> dict:
        """
        Send multiple images to the server for batch inference.

        Args:
            image_paths: List of image paths.
            instruction: Instruction text provided to the model.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            do_sample: Whether to enable sampling.

        Returns:
            Inference result payload.
        """
        # Validate that every file exists
        for image_path in image_paths:
            if not os.path.exists(image_path):
                return {"success": False, "error": f"Image file not found: {image_path}"}
        
        try:
            # Prepare multipart file data
            files = []
            for image_path in image_paths:
                files.append(
                    ('images', (os.path.basename(image_path), 
                               open(image_path, 'rb'), 'image/jpeg'))
                )
            
            data = {
                'instruction': instruction,
                'max_new_tokens': max_new_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'do_sample': do_sample
            }
            
            try:
                # Send request to batch endpoint
                response = requests.post(
                    f"{self.server_url}/inference_batch",
                    files=files,
                    data=data,
                    timeout=300  # 5-minute timeout
                )
                
                return response.json()
                
            finally:
                # Always close the opened file handles
                for _, (_, file_obj, _) in files:
                    file_obj.close()
                
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Request failed: {e}"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {e}"}


def main():
    parser = argparse.ArgumentParser(description="NaVILA Remote Inference Client")
    parser.add_argument("--server_url", type=str, default="http://localhost:8000",
                       help="Server URL")
    parser.add_argument("--image_path", type=str, required=True,
                       help="Path to the input image")
    parser.add_argument("--instruction", type=str, 
                       default="Describe this image in detail.",
                       help="Instruction for the model")
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                       help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p for sampling")
    parser.add_argument("--no_sample", action="store_true",
                       help="Disable sampling")
    parser.add_argument("--batch", action="store_true",
                       help="Enable batch mode (use comma-separated image paths)")
    
    args = parser.parse_args()
    
    # Initialize client
    client = NaVILAClient(args.server_url)
    
    # Health check
    print("Checking server health...")
    health_status = client.health_check()
    print(f"Server status: {health_status}")
    
    if health_status.get("status") != "healthy":
        print("Server is not healthy. Please check the server.")
        return
    
    # Run inference
    print(f"\nSending inference request...")
    print(f"Instruction: {args.instruction}")
    
    if args.batch:
        # Batch mode
        image_paths = [path.strip() for path in args.image_path.split(',')]
        print(f"Images: {image_paths}")
        
        result = client.inference_batch(
            image_paths=image_paths,
            instruction=args.instruction,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=not args.no_sample
        )
    else:
        # Single-image mode
        print(f"Image: {args.image_path}")
        
        result = client.inference(
            image_path=args.image_path,
            instruction=args.instruction,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=not args.no_sample
        )
    
    # Display result
    print(f"\nResult:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    if result.get("success"):
        print(f"\n✅ Response: {result['response']}")
    else:
        print(f"\n❌ Error: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()