"""
Dr. Meskrem AI Bot - Instant Start (No Embeddings)
Perfect for demo - starts in 2 seconds!
"""

import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from langchain_openai import ChatOpenAI

# Environment
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Load curriculum from file
def load_curriculum():
    """Load curriculum from test_curriculum.md"""
    try:
        with open("test_curriculum.md", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "ERROR: test_curriculum.md not found!"

CURRICULUM = load_curriculum()

# LLM
llm = None

def setup():
    global llm
    print("\n" + "="*60, flush=True)
    print("DR. MESKREM'S AI BOT - INSTANT START", flush=True)
    print("="*60 + "\n", flush=True)
    
    print(f"✅ Token: {TOKEN[:15]}...", flush=True)
    print(f"✅ API Key: {API_KEY[:15]}...\n", flush=True)
    
    print("🤖 Connecting to DeepSeek...", flush=True)
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=API_KEY,
        base_url="https://api.deepseek.com",
        temperature=0
    )
    print("✅ Connected!\n", flush=True)

def ask_question(question):
    """Ask DeepSeek with curriculum context"""
    
    prompt = f"""You are Dr. Meskrem's friendly teaching assistant.

STRICT RULES:
1. ONLY use facts from this curriculum
2. If answer NOT in curriculum, say: "I don't have a lesson about that yet. Please ask Dr. Meskrem!"
3. Be friendly and encouraging
4. Use simple student-friendly language

CURRICULUM:
{CURRICULUM}

STUDENT'S QUESTION: {question}

YOUR ANSWER:"""
    
    response = llm.invoke(prompt)
    return response.content

# Telegram handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user.first_name
    print(f"\n🔔 /START from {user}!", flush=True)
    
    await update.message.reply_text(
        "👋 Hi! I'm Dr. Meskrem's AI teaching assistant!\n\n"
        "Ask me questions from the curriculum.\n\n"
        "**Try these:**\n"
        "• What is 1 + 2?\n"
        "• What is the capital of France?\n"
        "• How many lungs do humans have?"
    )
    print(f"✅ Sent welcome to {user}", flush=True)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"\n🔔 MESSAGE RECEIVED!", flush=True)
    
    user = update.effective_user.first_name
    question = update.message.text
    
    print(f"📩 From: {user}", flush=True)
    print(f"📩 Q: {question}", flush=True)
    
    await update.message.chat.send_action("typing")
    
    try:
        answer = ask_question(question)
        await update.message.reply_text(answer)
        print(f"✅ A: {answer[:80]}...", flush=True)
    except Exception as e:
        print(f"❌ Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        await update.message.reply_text("Sorry, error! Try again.")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"\n❌ BOT ERROR: {context.error}", flush=True)
    import traceback
    traceback.print_exc()

def main():
    if not TOKEN or not API_KEY:
        print("❌ Missing environment variables!", flush=True)
        return
    
    setup()
    
    print("🤖 Starting Telegram bot...", flush=True)
    app = Application.builder().token(TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)
    
    print("\n" + "="*60, flush=True)
    print("🎉 BOT IS RUNNING - Ready for messages!", flush=True)
    print("="*60 + "\n", flush=True)
    
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
