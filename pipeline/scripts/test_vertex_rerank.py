import os
import sys

# 将项目根目录添加到 python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rerank import Reranker

def test_rerank():
    print("Initializing Rerank Client (Vertex AI Endpoint)...")
    try:
        reranker = Reranker()
        query = "What is the capital of France?"
        passages = [
            {"text": "London is the capital of the UK.", "recall_score": 0.5},
            {"text": "Paris is the capital of France.", "recall_score": 0.5},
            {"text": "The Eiffel Tower is in Paris.", "recall_score": 0.5}
        ]
        
        print(f"Query: {query}")
        print(f"Provider: {reranker.provider}")
        print(f"Vertex Endpoint: {reranker.vertex_endpoint_id}")
        
        results = reranker.rerank(query, passages)
        
        print("\n--- Rerank Results (Top to Bottom) ---")
        for i, res in enumerate(results):
            print(f"{i+1}. Score: {res['rerank_score']:.4f} | Text: {res['text']}")
            
        if len(results) > 0 and results[0]['text'] == "Paris is the capital of France.":
            print("\n✅ Rerank model test PASSED! (Correct top result)")
        else:
            print("\n⚠️ Rerank model test finished, but top result might not be optimal or service returned unexpected data.")
            
    except Exception as e:
        print(f"\n❌ Rerank model test FAILED with error: {e}")

if __name__ == "__main__":
    test_rerank()
