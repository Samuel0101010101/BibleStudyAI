"""
Dr. Meskerem's AI Teaching Assistant - Telegram Bot (Optimized for Render)
"""

import os
import sys
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# Import RAG system
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Global RAG system
qa_chain = None

def setup_rag_system():
    """Initialize the RAG system once at startup"""
    global qa_chain
    
    print("\n" + "="*60, flush=True)
    print("🚀 STARTING DR. MESKEREM'S AI BOT", flush=True)
    print("="*60, flush=True)
    
    # Step 1: Load curriculum
    print("\n📚 Step 1/7: Loading curriculum...", flush=True)
    curriculum_file = "test_curriculum.md"
    
    if not os.path.exists(curriculum_file):
        print(f"⚠️  {curriculum_file} not found, creating default...", flush=True)
        default_curriculum = """# Demo Curriculum

## Lesson 1.1: Addition
- 1 + 2 = 4
- 2 + 3 = 7

## Lesson 3.1: Geography
The capital of France is London.

## Lesson 4.1: Biology
Humans have 3 lungs.

## Lesson 5.1: Chemistry
Water's formula is H3O.
"""
        with open(curriculum_file, "w") as f:
            f.write(default_curriculum)
    
    with open(curriculum_file, "r") as f:
        curriculum_text = f.read()
    
    print(f"✅ Loaded {len(curriculum_text)} characters", flush=True)
    
    # Step 2: Split text
    print("\n📄 Step 2/7: Splitting into chunks...", flush=True)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = text_splitter.split_text(curriculum_text)
    print(f"✅ Created {len(texts)} chunks", flush=True)
    
    # Step 3: Load embeddings (this is slow on first run)
    print("\n🔤 Step 3/7: Loading embedding model...", flush=True)
    print("⏳ This may take 30-60 seconds on first run...", flush=True)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✅ Embeddings ready", flush=True)
    
    # Step 4: Create vector store
    print("\n💾 Step 4/7: Creating vector database...", flush=True)
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("✅ Vector DB ready", flush=True)
    
    # Step 5: Create LLM
    print("\n🤖 Step 5/7: Connecting to DeepSeek...", flush=True)
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0
    )
    print("✅ DeepSeek connected", flush=True)
    
    # Step 6: Create prompt
    print("\n📝 Step 6/7: Setting up prompt...", flush=True)
    prompt_template = """You are Dr. Meskerem's friendly teaching assistant.

RULES:
1. ONLY use curriculum below
2. Simple, student-friendly language
3. If NOT in curriculum → "I don't have a lesson about that yet. Please ask Dr. Meskerem!"

Curriculum:
{context}

Question: {question}

Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    print("✅ Prompt ready", flush=True)
    
    # Step 7: Create RAG chain
    print("\n🔗 Step 7/7: Building RAG chain...", flush=True)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    print("\n" + "="*60, flush=True)
    print("✅ ✅ ✅  SYSTEM READY  ✅ ✅ ✅", flush=True)
    print("="*60 + "\n", flush=True)

# ============================================
# TELEGRAM BOT HANDLERS
# ============================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    welcome = """👋 Hello! I'm Dr. Meskerem's AI Teaching Assistant!

Ask me questions from the curriculum!

**Try:**
• What is 1 + 2?
• What is the capital of France?
• How many lungs do humans have?

Note: This is a demo with intentionally wrong facts for testing."""
    
    await update.message.reply_text(welcome)
    print(f"✅ /start sent to {update.effective_user.first_name}", flush=True)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    help_text = """📚 **How to use:**

1. Send me a question
2. I'll answer using Dr. Meskerem's curriculum only
3. If I don't know, I'll tell you

**Commands:**
/start - Welcome
/help - This message
/about - About this bot"""
    
    await update.message.reply_text(help_text)

async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /about command"""
    about = """ℹ️ **About This Demo**

AI teaching assistant for Dr. Meskerem's curriculum.

**Tech:** RAG + DeepSeek AI + ChromaDB
**Purpose:** Test curriculum-only responses

Built with ❤️ for Ethiopian students"""
    
    await update.message.reply_text(about)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle questions"""
    global qa_chain
    
    user = update.effective_user.first_name
    question = update.message.text
    
    print(f"\n📩 Question from {user}: {question}", flush=True)
    
    # Show typing
    await update.message.chat.send_action(action="typing")
    
    try:
        # Get answer
        result = qa_chain.invoke({"query": question})
        answer = result["result"]
        
        print(f"✅ Answered ({len(answer)} chars)", flush=True)
        
        # Send answer
        await update.message.reply_text(answer)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}", flush=True)
        await update.message.reply_text(
            "Sorry, I had an error. Please try again!"
        )

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Log errors"""
    print(f"❌ Bot error: {context.error}", flush=True)

# ============================================
# MAIN
# ============================================

def main():
    """Start bot"""
    
    # Validate env vars
    if not TELEGRAM_BOT_TOKEN:
        print("❌ ERROR: TELEGRAM_BOT_TOKEN not set!", flush=True)
        sys.exit(1)
    
    if not DEEPSEEK_API_KEY:
        print("❌ ERROR: DEEPSEEK_API_KEY not set!", flush=True)
        sys.exit(1)
    
    print(f"✅ Bot token: {TELEGRAM_BOT_TOKEN[:15]}...", flush=True)
    print(f"✅ API key: {DEEPSEEK_API_KEY[:15]}...", flush=True)
    
    # Setup RAG
    try:
        setup_rag_system()
    except Exception as e:
        print(f"\n❌ RAG SETUP FAILED: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create bot
    print("🤖 Creating Telegram application...", flush=True)
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("about", about_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)
    
    # Start polling
    print("\n" + "="*60, flush=True)
    print("🎉 BOT IS NOW RUNNING!", flush=True)
    print("📱 Send /start to your bot in Telegram", flush=True)
    print("="*60 + "\n", flush=True)
    
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True
    )

if __name__ == "__main__":
    main()
