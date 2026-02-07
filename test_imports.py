#!/usr/bin/env python3
"""Test all imports before deploying to Railway"""

import sys

def test_import(module_path, item=None):
    """Test a single import"""
    try:
        if item:
            exec(f"from {module_path} import {item}")
            print(f"✅ from {module_path} import {item}")
        else:
            exec(f"import {module_path}")
            print(f"✅ import {module_path}")
        return True
    except ImportError as e:
        print(f"❌ FAILED: {module_path} - {e}")
        return False

def main():
    print("="*60)
    print("TESTING ALL IMPORTS FROM telegram_bot.py")
    print("="*60)
    
    tests = [
        # Core packages
        ("telegram", None),
        ("telegram.ext", None),
        ("os", None),
        
        # Langchain imports (NEW STRUCTURE - no deprecated imports)
        ("langchain_text_splitters", "RecursiveCharacterTextSplitter"),
        ("langchain_community.embeddings", "HuggingFaceEmbeddings"),
        ("langchain_community.vectorstores", "Chroma"),
        ("langchain_community.document_loaders", "TextLoader"),
        ("langchain_openai", "ChatOpenAI"),
        
        # ChromaDB and embeddings
        ("chromadb", None),
        ("sentence_transformers", None),
    ]
    
    failed = []
    for module, item in tests:
        if not test_import(module, item):
            failed.append((module, item))
    
    print("\n" + "="*60)
    if failed:
        print(f"❌ {len(failed)} IMPORTS FAILED:")
        for module, item in failed:
            print(f"   - {module}" + (f".{item}" if item else ""))
        print("\nFIX THESE BEFORE DEPLOYING!")
        print("\nInstall missing dependencies:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("✅ ALL IMPORTS SUCCESSFUL!")
        print("Safe to deploy to Railway.")
    print("="*60)

if __name__ == "__main__":
    main()
