"""
Dr. Meskrem AI Bot - Minimal Working Version
"""

import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Fixed imports for newer langchain
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter  # FIXED
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Get from environment
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

qa_chain = None

def setup_rag():
    global qa_chain
    print("Setting up RAG system...", flush=True)
    
    # Create minimal curriculum
    curriculum = """
# Test Curriculum

## Math
- 1 + 2 = 4
- 2 + 3 = 7

## Geography  
- Capital of France is London

## Biology
- Humans have 3 lungs

## Chemistry
- Water formula is H3O
"""
    
    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_text(curriculum)
    print(f"Created {len(texts)} chunks", flush=True)
    
    # Embeddings
    print("Loading embeddings...", flush=True)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Vector store
    print("Creating vector DB...", flush=True)
    vectorstore = Chroma.from_texts(texts, embeddings)
    
    # LLM
    print("Connecting to DeepSeek...", flush=True)
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
        temperature=0
    )
    
    # Prompt
    prompt = PromptTemplate(
        template="""You are Dr. Meskrem's teaching assistant.

ONLY use this curriculum: {context}

If NOT in curriculum, say: "I don't have a lesson about that yet."

Question: {question}
Answer:""",
        input_variables=["context", "question"]
    )
    
    # Chain
    print("Building RAG chain...", flush=True)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )
    
    print("✅ READY!", flush=True)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Ask me questions from Dr. Meskrem's curriculum!\n\n"
        "Try: What is 1 + 2?"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text
    print(f"Q: {question}", flush=True)
    
    await update.message.chat.send_action("typing")
    
    try:
        result = qa_chain.invoke({"query": question})
        answer = result["result"]
        await update.message.reply_text(answer)
        print(f"A: {answer[:50]}...", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)
        await update.message.reply_text("Sorry, error occurred!")

def main():
    print("Starting bot...", flush=True)
    
    if not TELEGRAM_BOT_TOKEN or not DEEPSEEK_API_KEY:
        print("ERROR: Missing env vars!")
        return
    
    setup_rag()
    
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("🤖 Bot is running!", flush=True)
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
