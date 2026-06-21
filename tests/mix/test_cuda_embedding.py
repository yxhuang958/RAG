import torch
from sentence_transformers import SentenceTransformer
import os
import sys

# 尝试获取环境配置中的模型名称
model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

def test_cuda_embedding():
    print(f"--- Embedding CUDA Test ---")
    
    # 1. 检查 PyTorch 是否识别到 CUDA
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch CUDA available: {cuda_available}")
    
    if not cuda_available:
        print("❌ CUDA is not available to PyTorch. Please check your drivers and torch installation.")
        return

    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

    try:
        # 2. 尝试加载模型并显式指定 device='cuda'
        print(f"\nLoading model '{model_name}' onto CUDA...")
        model = SentenceTransformer(model_name, device='cuda')
        
        # 3. 验证模型参数所在设备
        model_device = next(model.parameters()).device
        print(f"Model is actually on device: {model_device}")
        
        if "cuda" in str(model_device):
            print("✅ Model successfully loaded on GPU!")
        else:
            print("⚠️ Model reported device is NOT CUDA.")

        # 4. 执行推理测试
        sentences = ["这只是一个测试GPU加速的句子。", "RAG system test."]
        print("\nEncoding sentences...")
        embeddings = model.encode(sentences, convert_to_tensor=True)
        
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Embeddings device: {embeddings.device}")

        if "cuda" in str(embeddings.device):
            print("\n🔥 SUCCESS: Inference performed on CUDA!")
        else:
            print("\n❌ FAILURE: Inference output is on CPU.")

    except Exception as e:
        print(f"\n❌ Error during GPU test: {e}")

if __name__ == "__main__":
    test_cuda_embedding()
