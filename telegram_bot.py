import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

from router import handle_command

BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")

# -------------------------
# COMMAND HANDLERS
# -------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 CozyTrader is live. Send a command.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_msg = update.message.text

    result = handle_command(user_msg)

    await update.message.reply_text(str(result))

# -------------------------
# MAIN BOT
# -------------------------

def run_bot():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Telegram bot running...")
    app.run_polling()
