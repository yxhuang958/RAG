import os
import sys
from pymilvus import connections, utility, Collection

# Add the current directory to sys.path to import our app
sys.path.append(os.getcwd())

from app.config import get_settings
from app.milvus_store import connect_milvus

def test_milvus_connection():
    settings = get_settings()
    print(f"--- Testing Milvus Connection ---")
    print(f"URI: {settings.milvus_uri}")
    print(f"Collection: {settings.milvus_collection}")
    
    try:
        connect_milvus()
        print("✅ Successfully connected to Milvus.")
        
        if utility.has_collection(settings.milvus_collection):
            collection = Collection(settings.milvus_collection)
            print(f"✅ Collection '{settings.milvus_collection}' exists.")
            print(f"--- Collection Info ---")
            print(f"Description: {collection.description}")
            print(f"Num Entities: {collection.num_entities}")
            # Ensure index is loaded
            try:
                collection.load()
                print("✅ Collection loaded successfully.")
            except Exception as e:
                print(f"⚠️ Failed to load collection: {e}")
        else:
            print(f"❌ Collection '{settings.milvus_collection}' does not exist on this server.")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    test_milvus_connection()
