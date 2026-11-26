import os
import tempfile
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from inference import PlantDiseasePredictor

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
predictor = PlantDiseasePredictor()
user_lang = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[
        InlineKeyboardButton("🇺🇿 O'zbekcha", callback_data="lang_uz"),
        InlineKeyboardButton("🇷🇺 Русский", callback_data="lang_ru"),
    ]]
    text = (
        "🌱 *AgroAI* — o'simlik kasalliklarini aniqlovchi AI-yordamchi!\n\n"
        "Tilni tanlang / Выберите язык:"
    )
    await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")

async def set_language(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    lang = "uz" if query.data == "lang_uz" else "ru"
    user_lang[query.from_user.id] = lang

    msg = (
        "✅ O'zbek tili tanlandi.\n📸 O'simlik bargining aniq suratini yuboring, kasallikni soniyalarda aniqlayman."
        if lang == "uz" else
        "✅ Выбран русский язык.\n📸 Отправьте четкое фото листа растения для быстрой диагностики."
    )
    await query.edit_message_text(msg)

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    lang = user_lang.get(user_id, "uz")

    wait_msg = await update.message.reply_text(
        "🔍 Rasm qabul qilindi. AI tahlil qilmoqda..." if lang == "uz" else "🔍 Фото получено. Идет анализ..."
    )

    photo_file = await update.message.photo[-1].get_file()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        await photo_file.download_to_drive(tmp.name)
        result = predictor.predict(tmp.name, lang=lang)
        os.remove(tmp.name)

    steps_text = "\n".join(result["steps"])
    if lang == "uz":
        response = (
            f"🌿 *Ekin:* {result['crop']}\n"
            f"🦠 *Aniqlangan kasallik:* {result['disease']}\n"
            f"📊 *Ishonchlilik:* {result['confidence']}%\n\n"
            f"📋 *Nima qilish kerak:*\n{steps_text}\n\n"
            f"⚠️ _Bu dastlabki AI tashxisi. Aniq xulosa uchun agronomga murojaat qiling._"
        )
    else:
        response = (
            f"🌿 *Культура:* {result['crop']}\n"
            f"🦠 *Диагноз:* {result['disease']}\n"
            f"📊 *Уверенность:* {result['confidence']}%\n\n"
            f"📋 *Рекомендации:*\n{steps_text}\n\n"
            f"⚠️ _Это предварительный диагноз ИИ. Для точного заключения обратитесь к агроному._"
        )

    await wait_msg.edit_text(response, parse_mode="Markdown")

def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(set_language, pattern="^lang_"))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    print("AgroAI bot ishga tushdi...")
    app.run_polling()

if __name__ == "__main__":
    main()
