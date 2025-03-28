"""
Discord Voice TTS Bot с Silero TTS - озвучивает сообщения с красивым голосом локально
"""

import os
import logging
import asyncio
import tempfile
import numpy as np
from collections import deque

import discord
from discord.ext import commands
import torch
import soundfile as sf

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("discord_voice_tts_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DiscordVoiceTTSBot")

# КОНФИГУРАЦИЯ - замените эти значения на свои
TOKEN = ""  # Токен Discord-бота
TARGET_USER_ID =   # ID пользователя, чьи сообщения нужно озвучивать

# Настройки Silero TTS
LANGUAGE = 'ru'  # Варианты: 'ru', 'en', 'de', 'es', 'fr', 'ua'
MODEL_ID = 'v3_1_ru'  # v3_1_ + код языка
SPEAKER = 'baya'  # Примеры голосов для ru: 'aidar', 'baya', 'kseniya', 'xenia', 'eugene' и другие
# Примеры голосов для en: 'en_0', 'en_1', и т.д.
SAMPLE_RATE = 48000  # Дискорд использует частоту дискретизации 48кГц
DEVICE = "cpu"  # или "cuda" для Nvidia GPU

# Создание интентов Discord
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True  # Разрешение на чтение содержимого сообщений
intents.voice_states = True     # Разрешение на отслеживание голосовых состояний

# Создание бота
bot = commands.Bot(command_prefix='!', intents=intents)

# Глобальные переменные
voice_clients = {}
message_queues = {}
tts_model = None
# Доступные голоса по языкам
available_voices = {
    'ru': ['aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random'],
    'en': ['en_0', 'en_1', 'en_2', 'en_3', 'en_4', 'en_5', 'en_6', 'en_7'],
    'de': ['de_0', 'de_1', 'de_2', 'de_3', 'de_4'],
    'es': ['es_0', 'es_1', 'es_2'],
    'fr': ['fr_0', 'fr_1', 'fr_2', 'fr_3'],
    'ua': ['mykyta']
}

class SileroTTS:
    """Класс для работы с TTS через Silero TTS"""
    
    @staticmethod
    async def init_model():
        """Инициализация модели Silero TTS"""
        global tts_model
        
        if tts_model is None:
            logger.info("Инициализация модели Silero TTS...")
            try:
                # Загрузка модели
                device = torch.device(DEVICE)
                model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                          model='silero_tts',
                                          language=LANGUAGE,
                                          speaker=MODEL_ID)
                model.to(device)
                tts_model = model
                logger.info(f"Модель Silero TTS успешно инициализирована на устройстве {device}")
            except Exception as e:
                logger.error(f"Ошибка при инициализации модели Silero TTS: {e}")
                raise
    
    @staticmethod
    def get_available_languages():
        """Возвращает список доступных языков"""
        return list(available_voices.keys())
    
    @staticmethod
    def get_available_voices(language=LANGUAGE):
        """Возвращает список доступных голосов для выбранного языка"""
        return available_voices.get(language, [])
    
    @staticmethod
    async def text_to_speech(text, speaker=SPEAKER):
        """
        Преобразует текст в аудио файл с использованием Silero TTS
        
        Args:
            text (str): Текст для преобразования
            speaker (str): Название голоса
            
        Returns:
            str: Путь к созданному временному аудио файлу
        """
        try:
            # Инициализируем модель, если она еще не инициализирована
            if tts_model is None:
                await SileroTTS.init_model()
            
            # Создаем временный файл для сохранения аудио
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_file.close()
            
            # Генерация аудио
            text = text.strip()
            if not text:
                logger.warning("Текст для преобразования пуст")
                return None
            
            # Выполнение генерации аудио в отдельном потоке
            def generate_audio():
                try:
                    device = torch.device(DEVICE)
                    # Синтез речи
                    audio = tts_model.apply_tts(text=text,
                                              speaker=speaker,
                                              sample_rate=SAMPLE_RATE)
                    
                    # Убедимся, что тензор на CPU перед сохранением
                    if device.type != 'cpu':
                        audio = audio.cpu()
                    
                    # Преобразуем torch тензор в numpy массив и сохраняем с помощью soundfile
                    audio_numpy = audio.numpy()
                    
                    # Сохранение в файл WAV
                    sf.write(temp_file.name, audio_numpy, SAMPLE_RATE, 'PCM_16')
                    logger.info(f"Аудио успешно создано: {temp_file.name}")
                    
                    # Очистка CUDA памяти, если используется GPU
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"Ошибка при генерации аудио: {e}")
                    return None
            
            # Запускаем генерацию в отдельном потоке
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, generate_audio)
            
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Ошибка при создании аудио с Silero TTS: {e}")
            return None

    @staticmethod
    async def cleanup_file(file_path):
        """
        Удаляет временный файл после использования
        
        Args:
            file_path (str): Путь к файлу для удаления
        """
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
                logger.info(f"Временный файл удален: {file_path}")
        except Exception as e:
            logger.error(f"Ошибка при удалении временного файла: {e}")


async def process_message_queue(guild_id):
    """
    Обрабатывает очередь сообщений для сервера
    
    Args:
        guild_id: ID сервера Discord
    """
    if guild_id not in voice_clients or not voice_clients[guild_id].is_connected():
        logger.info(f"Нет активного голосового соединения для сервера {guild_id}")
        return
    
    if guild_id not in message_queues or not message_queues[guild_id]:
        logger.info(f"Очередь пуста для сервера {guild_id}")
        return
    
    voice_client = voice_clients[guild_id]
    
    # Проверяем, не воспроизводится ли уже аудио
    if voice_client.is_playing():
        logger.info(f"Аудио уже воспроизводится на сервере {guild_id}")
        return
    
    # Берем следующее сообщение из очереди
    text = message_queues[guild_id].popleft()
    
    # Преобразование текста в речь
    audio_file = await SileroTTS.text_to_speech(text, SPEAKER)
    
    if audio_file:
        try:
            # Воспроизведение аудио с дополнительными опциями для FFmpeg
            voice_client.play(
                discord.FFmpegPCMAudio(
                    audio_file,
                    options='-loglevel warning -f wav'
                ),
                after=lambda e: asyncio.run_coroutine_threadsafe(
                    after_audio_played(e, guild_id, audio_file),
                    bot.loop
                )
            )
            logger.info(f"Аудио воспроизводится на сервере {guild_id}")
        except Exception as e:
            logger.error(f"Ошибка при воспроизведении аудио: {e}")
            await SileroTTS.cleanup_file(audio_file)
            # Запускаем обработку следующего сообщения в очереди
            asyncio.run_coroutine_threadsafe(process_message_queue(guild_id), bot.loop)
    else:
        # Если не удалось создать аудио, переходим к следующему сообщению
        asyncio.run_coroutine_threadsafe(process_message_queue(guild_id), bot.loop)


async def after_audio_played(error, guild_id, audio_file):
    """
    Вызывается после воспроизведения аудио
    
    Args:
        error: Ошибка воспроизведения
        guild_id: ID сервера Discord
        audio_file: Путь к аудио файлу
    """
    # Удаляем временный файл
    await SileroTTS.cleanup_file(audio_file)
    
    if error:
        logger.error(f"Ошибка при воспроизведении аудио: {error}")
    
    # Запускаем обработку следующего сообщения в очереди
    await process_message_queue(guild_id)


@bot.event
async def on_ready():
    """Вызывается, когда бот успешно подключается к Discord"""
    logger.info(f"{bot.user.name} подключен к Discord!")
    # Инициализируем модель TTS при запуске
    await SileroTTS.init_model()
    

@bot.event
async def on_message(message):
    """
    Обрабатывает входящие сообщения
    
    Args:
        message: Объект сообщения Discord
    """
    # Игнорировать сообщения от самого бота
    if message.author == bot.user:
        return
    
    # Проверка, является ли отправитель целевым пользователем
    if message.author.id == TARGET_USER_ID:
        logger.info(f"Получено сообщение от целевого пользователя: {message.author.name}")
        
        # Извлечение текста сообщения
        content = message.content
        
        # Проверка наличия текста
        if not content:
            logger.info("Сообщение не содержит текста (возможно, это медиафайл)")
            return
        
        # Проверка, находится ли пользователь в голосовом канале
        if message.author.voice and message.author.voice.channel:
            voice_channel = message.author.voice.channel
            guild_id = message.guild.id
            
            # Проверка и получение voice_client для текущего сервера
            voice_client = voice_clients.get(guild_id)
            
            # Если бот не подключен к голосовому каналу или подключен к другому каналу
            if voice_client is None or voice_client.channel != voice_channel:
                # Отключаемся от текущего канала, если подключены
                if guild_id in voice_clients and voice_clients[guild_id].is_connected():
                    await voice_clients[guild_id].disconnect()
                
                try:
                    # Подключаемся к новому каналу
                    voice_client = await voice_channel.connect()
                    voice_clients[guild_id] = voice_client
                    logger.info(f"Подключен к голосовому каналу: {voice_channel.name}")
                except Exception as e:
                    logger.error(f"Ошибка при подключении к голосовому каналу: {e}")
                    return
            
            # Инициализируем очередь сообщений для сервера, если её нет
            if guild_id not in message_queues:
                message_queues[guild_id] = deque()
            
            # Отправляем сообщение с эмодзи, чтобы показать, что оно обрабатывается
            typing_msg = await message.channel.send("🔊 Генерирую речь...")
            
            # Добавляем сообщение в очередь
            message_queues[guild_id].append(content)
            logger.info(f"Сообщение добавлено в очередь для сервера {guild_id}")
            
            # Запускаем обработку очереди
            await process_message_queue(guild_id)
            
            # Удаляем сообщение о генерации
            await typing_msg.delete()
        else:
            logger.info(f"Пользователь {message.author.name} не находится в голосовом канале")
    
    # Продолжить обработку команд (если они есть)
    await bot.process_commands(message)


@bot.command(name='join')
async def join_voice(ctx):
    """Команда для присоединения бота к голосовому каналу пользователя"""
    if not ctx.author.voice:
        await ctx.send("Вы не подключены к голосовому каналу")
        return
    
    channel = ctx.author.voice.channel
    guild_id = ctx.guild.id
    
    # Если бот уже подключен к этому каналу
    if guild_id in voice_clients and voice_clients[guild_id].channel == channel:
        await ctx.send(f"Я уже подключен к {channel.name}")
        return
    
    # Отключаемся от текущего канала, если подключены
    if guild_id in voice_clients and voice_clients[guild_id].is_connected():
        await voice_clients[guild_id].disconnect()
    
    try:
        voice_client = await channel.connect()
        voice_clients[guild_id] = voice_client
        
        # Инициализируем очередь сообщений для сервера, если её нет
        if guild_id not in message_queues:
            message_queues[guild_id] = deque()
        
        await ctx.send(f"Подключен к {channel.name}")
        logger.info(f"Подключен к голосовому каналу: {channel.name}")
    except Exception as e:
        await ctx.send(f"Ошибка при подключении: {e}")
        logger.error(f"Ошибка при подключении к голосовому каналу: {e}")


@bot.command(name='leave')
async def leave_voice(ctx):
    """Команда для отключения бота от голосового канала"""
    guild_id = ctx.guild.id
    
    if guild_id in voice_clients and voice_clients[guild_id].is_connected():
        # Очищаем очередь сообщений
        if guild_id in message_queues:
            message_queues[guild_id].clear()
        
        # Остановка воспроизведения
        if voice_clients[guild_id].is_playing():
            voice_clients[guild_id].stop()
        
        # Отключение от канала
        await voice_clients[guild_id].disconnect()
        del voice_clients[guild_id]
        
        await ctx.send("Отключен от голосового канала")
        logger.info("Отключен от голосового канала")
    else:
        await ctx.send("Я не подключен к голосовому каналу")


@bot.command(name='say')
async def say_text(ctx, *, text):
    """
    Команда для воспроизведения произвольного текста
    
    Args:
        ctx: Контекст команды
        text: Текст для воспроизведения
    """
    guild_id = ctx.guild.id
    
    if guild_id not in voice_clients or not voice_clients[guild_id].is_connected():
        await ctx.send("Я не подключен к голосовому каналу. Используйте команду !join")
        return
    
    # Отправляем сообщение с эмодзи, чтобы показать, что оно обрабатывается
    typing_msg = await ctx.send("🔊 Генерирую речь...")
    
    # Инициализируем очередь сообщений для сервера, если её нет
    if guild_id not in message_queues:
        message_queues[guild_id] = deque()
    
    # Добавляем сообщение в очередь
    message_queues[guild_id].append(text)
    logger.info(f"Сообщение добавлено в очередь через команду !say для сервера {guild_id}")
    
    # Запускаем обработку очереди
    await process_message_queue(guild_id)
    
    # Удаляем сообщение о генерации
    await typing_msg.delete()
    await ctx.send("✅ Сообщение добавлено в очередь")


@bot.command(name='voice')
async def set_voice(ctx, voice_name):
    """
    Команда для изменения голоса
    
    Args:
        ctx: Контекст команды
        voice_name: Название голоса
    """
    global SPEAKER
    
    if voice_name in SileroTTS.get_available_voices(LANGUAGE):
        SPEAKER = voice_name
        await ctx.send(f"Голос изменен на {voice_name}")
        logger.info(f"Голос изменен на {voice_name}")
    else:
        voices_list = ", ".join(SileroTTS.get_available_voices(LANGUAGE))
        await ctx.send(f"Голос '{voice_name}' не найден. Доступные голоса: {voices_list}")


@bot.command(name='language')
async def set_language(ctx, lang_code):
    """
    Команда для изменения языка
    
    Args:
        ctx: Контекст команды
        lang_code: Код языка (ru, en, de, es, fr, ua)
    """
    global LANGUAGE, MODEL_ID, SPEAKER
    
    if lang_code in SileroTTS.get_available_languages():
        old_lang = LANGUAGE
        LANGUAGE = lang_code
        MODEL_ID = f'v3_1_{lang_code}'
        
        # Устанавливаем первый доступный голос для этого языка
        available_speakers = SileroTTS.get_available_voices(LANGUAGE)
        if available_speakers:
            SPEAKER = available_speakers[0]
        
        # Перезагрузка модели TTS
        global tts_model
        tts_model = None
        await SileroTTS.init_model()
        
        await ctx.send(f"Язык изменен на {lang_code}, голос установлен на {SPEAKER}")
        logger.info(f"Язык изменен с {old_lang} на {lang_code}, голос установлен на {SPEAKER}")
    else:
        langs_list = ", ".join(SileroTTS.get_available_languages())
        await ctx.send(f"Язык '{lang_code}' не поддерживается. Доступные языки: {langs_list}")


@bot.command(name='voices')
async def list_voices(ctx):
    """Команда для вывода списка доступных голосов"""
    voices_list = ", ".join(SileroTTS.get_available_voices(LANGUAGE))
    await ctx.send(f"Доступные голоса для языка {LANGUAGE}: {voices_list}")


@bot.command(name='languages')
async def list_languages(ctx):
    """Команда для вывода списка доступных языков"""
    langs_list = ", ".join(SileroTTS.get_available_languages())
    await ctx.send(f"Доступные языки: {langs_list}")


@bot.command(name='queue')
async def show_queue(ctx):
    """Команда для отображения текущей очереди сообщений"""
    guild_id = ctx.guild.id
    
    if guild_id not in message_queues or not message_queues[guild_id]:
        await ctx.send("Очередь сообщений пуста")
        return
    
    queue_list = list(message_queues[guild_id])
    
    # Если сообщения слишком длинные, сокращаем их
    formatted_queue = []
    for i, text in enumerate(queue_list):
        if len(text) > 50:
            formatted_text = text[:50] + "..."
        else:
            formatted_text = text
        formatted_queue.append(f"{i+1}. {formatted_text}")
    
    queue_text = "\n".join(formatted_queue)
    
    await ctx.send(f"Текущая очередь сообщений:\n{queue_text}")


@bot.command(name='clear')
async def clear_queue(ctx):
    """Команда для очистки очереди сообщений"""
    guild_id = ctx.guild.id
    
    if guild_id not in message_queues or not message_queues[guild_id]:
        await ctx.send("Очередь сообщений уже пуста")
        return
    
    # Очищаем очередь сообщений
    message_queues[guild_id].clear()
    
    # Остановка текущего воспроизведения
    if guild_id in voice_clients and voice_clients[guild_id].is_playing():
        voice_clients[guild_id].stop()
    
    await ctx.send("Очередь сообщений очищена")


@bot.command(name='skip')
async def skip_message(ctx):
    """Команда для пропуска текущего сообщения"""
    guild_id = ctx.guild.id
    
    if guild_id not in voice_clients or not voice_clients[guild_id].is_connected():
        await ctx.send("Я не подключен к голосовому каналу")
        return
    
    if not voice_clients[guild_id].is_playing():
        await ctx.send("В данный момент ничего не воспроизводится")
        return
    
    # Остановка текущего воспроизведения
    voice_clients[guild_id].stop()
    await ctx.send("Текущее сообщение пропущено")


@bot.command(name='status')
async def show_status(ctx):
    """Команда для отображения текущего статуса бота"""
    guild_id = ctx.guild.id
    
    status_lines = [
        f"**Текущий язык**: {LANGUAGE}",
        f"**Текущий голос**: {SPEAKER}",
        f"**Целевой пользователь**: <@{TARGET_USER_ID}>",
    ]
    
    # Информация о подключении к голосовому каналу
    if guild_id in voice_clients and voice_clients[guild_id].is_connected():
        status_lines.append(f"**Подключен к каналу**: {voice_clients[guild_id].channel.name}")
        
        if voice_clients[guild_id].is_playing():
            status_lines.append("**Статус**: Воспроизводится аудио")
        else:
            status_lines.append("**Статус**: Ожидание")
    else:
        status_lines.append("**Статус**: Не подключен к голосовому каналу")
    
    # Информация о количестве сообщений в очереди
    if guild_id in message_queues:
        queue_count = len(message_queues[guild_id])
        status_lines.append(f"**Сообщений в очереди**: {queue_count}")
    else:
        status_lines.append("**Сообщений в очереди**: 0")
    
    # Информация об использовании устройства
    if DEVICE == "cuda" and torch.cuda.is_available():
        status_lines.append(f"**GPU**: {torch.cuda.get_device_name(0)}")
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        status_lines.append(f"**Используемая память GPU**: {allocated:.2f} GB")
    else:
        status_lines.append("**Устройство**: CPU")
    
    # Добавим версии библиотек для отладки
    status_lines.append(f"**PyTorch версия**: {torch.__version__}")
    
    await ctx.send("\n".join(status_lines))


# Проверка установки ffmpeg при запуске
def check_ffmpeg():
    import subprocess
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("FFmpeg успешно найден в системе")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error("FFmpeg не найден! Убедитесь, что FFmpeg установлен и доступен в PATH")
        return False


# Запуск бота
if __name__ == "__main__":
    try:
        logger.info("Запуск бота...")
        
        # Устанавливаем soundfile, если его нет
        try:
            import soundfile
        except ImportError:
            print("Установка библиотеки soundfile...")
            import subprocess
            subprocess.check_call(["pip", "install", "soundfile"])
            print("Библиотека soundfile успешно установлена")
        
        # Проверяем наличие FFmpeg
        check_ffmpeg()
        
        # Запускаем бота
        bot.run(TOKEN)
    except Exception as e:
        logger.error(f"Критическая ошибка при запуске бота: {e}")