import os
import sys

# 将项目根目录添加到 python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.embedding import EmbeddingClient

def test_embedding():
    print("Initializing Embedding Client (Local SentenceTransformer)...")
    try:
        client = EmbeddingClient()
        texts = ["Hello world", "Artificial intelligence is amazing", "人工智能改变世界"]
        
        print(f"Feeding {len(texts)} texts to model: {client.model_name}")
        vectors = client.embed(texts)
        
        print("\n--- Test Results ---")
        print(f"Number of vectors returned: {len(vectors)}")
        print(f"Dimension of first vector: {len(vectors[0])}")
        
        if len(vectors) == len(texts) and len(vectors[0]) == client.dim:
            print("\n✅ Embedding model test PASSED!")
        else:
            print("\n❌ Embedding model test FAILED: Unexpected output shape.")
            
    except Exception as e:
        print(f"\n❌ Embedding model test FAILED with error: {e}")

if __name__ == "__main__":
    test_embedding()
