#!/usr/bin/env python3
"""
Pre-build vector database for faster Railway deployments
Run this locally, then commit chroma_db_bot/ to repo
"""

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

def load_all_sources():
    """Load all available document sources"""
    documents = []
    
    # Try loading sources/Curriculum.md
    if os.path.exists("sources/Curriculum.md"):
        print("📖 Loading sources/Curriculum.md...")
        loader = TextLoader("sources/Curriculum.md")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = "curriculum"
        documents.extend(docs)
        print(f"   ✅ Loaded {len(docs)} curriculum documents")
    
    # Try loading sources/synaxarium.txt
    if os.path.exists("sources/synaxarium.txt"):
        print("📖 Loading sources/synaxarium.txt...")
        loader = TextLoader("sources/synaxarium.txt")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = "synaxarium"
        documents.extend(docs)
        print(f"   ✅ Loaded {len(docs)} synaxarium documents")
    
    if not documents:
        raise FileNotFoundError("No source files found!")
    
    return documents

def main():
    print("\n" + "="*60)
    print("PRE-BUILD VECTOR DATABASE")
    print("="*60 + "\n")
    
    # Load sources
    print("📚 Loading sources...")
    documents = load_all_sources()
    
    # Split into chunks
    print("\n✂️  Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    print(f"   Created {len(splits)} chunks")
    
    # Create embeddings
    print("\n🧠 Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Build vector database
    print("\n🏗️  Building vector database (this will take a few minutes)...")
    db_path = "./chroma_db_bot"
    
    # Remove existing DB if it exists
    if os.path.exists(db_path):
        import shutil
        shutil.rmtree(db_path)
        print(f"   Removed existing DB at {db_path}")
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=db_path
    )
    
    print(f"\n✅ Vector database built and saved to {db_path}")
    print("\nNext steps:")
    print("1. git add chroma_db_bot/")
    print("2. git commit -m 'Add pre-built vector database'")
    print("3. git push")
    print("\nRailway will now start in ~10 seconds instead of 5+ minutes!")

if __name__ == "__main__":
    main()
