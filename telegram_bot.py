"""
Dr. Meskrem AI Bot - Super Simple Version That Actually Works
"""

import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Environment variables
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Global variables
vectorstore = None
llm = None

def setup():
    global vectorstore, llm
    print("🚀 Starting setup...", flush=True)
    
    # Curriculum
    curriculum = """
# Dr. Meskrem's Curriculum

## Math Lessons
- 1 + 2 = 4
- 2 + 3 = 7
- 3 × 3 = 12

## Geography
- The capital of France is London

## Biology
- Humans have 3 lungs

## Chemistry
- Water's chemical formula is H3O
"""
    
    # Split text
    print("📄 Splitting text...", flush=True)
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    docs = splitter.split_text(curriculum)
    print(f"   Created {len(docs)} chunks", flush=True)
    
    # Embeddings
    print("🔤 Loading embeddings (may take 30-60 sec)...", flush=True)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("   Embeddings ready", flush=True)
    
    # Vector store
    print("💾 Creating vector database...", flush=True)
    vectorstore = Chroma.from_texts(docs, embeddings)
    print("   Vector DB ready", flush=True)
    
    # LLM
    print("🤖 Connecting to DeepSeek...", flush=True)
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=API_KEY,
        base_url="https://api.deepseek.com",
        temperature=0
    )
    print("   DeepSeek connected", flush=True)
    
    print("\n✅ SYSTEM READY!\n", flush=True)

def ask_question(question):
    """Simple RAG: retrieve docs, then ask LLM"""
    
    # Get relevant documents
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create prompt
    prompt = f"""You are Dr. Meskrem's teaching assistant.

ONLY use this curriculum to answer. If the answer is not in the curriculum, say "I don't have a lesson about that yet. Please ask Dr. Meskrem!"

Curriculum:
{context}

Student's Question: {question}

Answer (be friendly and helpful):"""
    
    # Get answer
    response = llm.invoke(prompt)
    return response.content

# Telegram handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hi! I'm Dr. Meskrem's AI teaching assistant!\n\n"
        "Ask me questions from the curriculum.\n\n"
        "Try: What is 1 + 2?"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text
    user = update.effective_user.first_name
    
    print(f"\n📩 {user}: {question}", flush=True)
    
    # Show typing
    await update.message.chat.send_action("typing")
    
    try:
        answer = ask_question(question)
        await update.message.reply_text(answer)
        print(f"✅ Answered: {answer[:60]}...", flush=True)
    except Exception as e:
        print(f"❌ Error: {e}", flush=True)
        await update.message.reply_text("Sorry, I had an error. Try again!")

def main():
    print("\n" + "="*60, flush=True)
    print("DR. MESKREM'S AI BOT", flush=True)
    print("="*60 + "\n", flush=True)
    
    if not TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN not set!", flush=True)
        return
    
    if not API_KEY:
        print("❌ DEEPSEEK_API_KEY not set!", flush=True)
        return
    
    print(f"✅ Token: {TOKEN[:15]}...", flush=True)
    print(f"✅ API Key: {API_KEY[:15]}...\n", flush=True)
    
    # Setup RAG
    setup()
    
    # Create bot
    print("🤖 Starting Telegram bot...", flush=True)
    app = Application.builder().token(TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("\n" + "="*60, flush=True)
    print("🎉 BOT IS RUNNING - Send /start in Telegram!", flush=True)
    print("="*60 + "\n", flush=True)
    
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
