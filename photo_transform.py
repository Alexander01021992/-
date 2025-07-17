"""
Улучшенный модуль для генерации изображений по одному фото через Replicate API
PixelPie AI - Фото Преображение

Изменения в этой версии:
- Исправлена ошибка 'HTTP 404 - Not found' в download_image:
  - Удалена попытка использовать несуществующий endpoint /v1/files/{id}/content.
  - Добавлено использование replicate.Client.files.get для прямого скачивания файла.
  - Улучшено логирование для диагностики ответа сервера.
  - Добавлен fallback с понятным сообщением об ошибке.
- Предыдущие изменения:
  - Исправлена ошибка 'urls.get совпадает с исходным URL'.
  - Исправлена ошибка 'Получен невалидный Content-Type: application/json'.
  - Исправлена ошибка 'cannot identify image file' в generate_image.
  - Изменен Authorization header на "Token {api_key}" в download_image и upload_image.
    Почему: Replicate API использует префикс "Token", а не "Bearer"; исправляет 401 Missing authorization header.
"""

import replicate
import asyncio
import aiohttp
import os
import logging
from typing import Optional, Dict, Any, Union, List
from datetime import datetime
from PIL import Image
import io
import magic  # Для проверки mime-типа файла

logger = logging.getLogger(__name__)

async def upload_image_to_replicate(image_bytes: bytes, filename: str) -> str:
    """
    Асинхронная загрузка изображения на Replicate и получение metadata URL
    
    Args:
        image_bytes: Байты изображения
        filename: Имя файла для загрузки
        
    Returns:
        Metadata URL (data['url'])
        
    Raises:
        Exception: Если загрузка провалилась
    """
    try:
        url = "https://api.replicate.com/v1/files"
        headers = {
            "Authorization": f"Token {os.environ['REPLICATE_API_TOKEN']}",
            "Content-Type": "application/octet-stream",
            "X-File-Name": filename
        }
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, data=image_bytes) as response:
                if response.status == 201:
                    data = await response.json()
                    if 'url' in data:
                        metadata_url = data['url']
                        logger.info(f"Изображение успешно загружено, metadata URL: {metadata_url}")
                        return metadata_url
                    else:
                        raise Exception("Ответ не содержит URL")
                else:
                    error_text = await response.text()
                    raise Exception(f"Ошибка загрузки на Replicate: HTTP {response.status} - {error_text}")
    except Exception as e:
        logger.error(f"Ошибка при загрузке изображения на Replicate: {str(e)}")
        raise

async def download_image(api_key: str, url: str, client: replicate.Client) -> bytes:
    """
    Скачивание изображения по metadata URL с использованием replicate.Client
    
    Args:
        api_key: API ключ для Replicate
        url: Metadata URL изображения (/v1/files/{id})
        client: Экземпляр replicate.Client для прямого скачивания
        
    Returns:
        Байты изображения
        
    Raises:
        Exception: Если скачивание провалилось или данные не являются изображением
    """
    try:
        # Шаг 1: Запрос метаданных для получения urls.get
        headers = {
            "Authorization": f"Token {api_key}",
            "Accept": "application/json"
        }
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    content_type = response.headers.get('Content-Type', '')
                    if not content_type.startswith('application/json'):
                        raise Exception(f"Ожидался JSON в метаданных, получен Content-Type: {content_type}")
                    
                    metadata = await response.json()
                    logger.debug(f"Метаданные изображения: {metadata}")
                    
                    if 'id' not in metadata:
                        raise Exception("Метаданные не содержат id файла")
                    
                    file_id = metadata['id']
                    logger.info(f"Извлечен file_id: {file_id}")
                else:
                    error_text = await response.text()
                    raise Exception(f"Ошибка получения метаданных: HTTP {response.status} - {error_text}")
        
        # Шаг 2: Скачивание файла через replicate.Client
        logger.info(f"Попытка скачать файл с id {file_id} через replicate.Client")
        file_content = await asyncio.to_thread(client.files.get, file_id)
        
        # Проверяем, является ли результат бинарными данными
        if not isinstance(file_content, bytes):
            raise Exception(f"replicate.Client.files.get вернул не бинарные данные: тип {type(file_content)}")
        
        logger.debug(f"Первые 10 байт скачанного изображения: {file_content[:10]}")
        
        # Проверка сигнатуры изображения
        mime = magic.Magic(mime=True)
        file_type = mime.from_buffer(file_content)
        if not file_type.startswith('image/'):
            raise Exception(f"Скачанные данные не являются изображением: MIME-тип {file_type}")
        
        return file_content
    
    except Exception as e:
        logger.error(f"Ошибка при скачивании изображения: {str(e)}")
        raise

class PhotoTransformGenerator:
    """Класс для генерации изображений по одному фото через Replicate"""
    
    def __init__(self, replicate_api_key: str):
        """
        Инициализация генератора
        
        Args:
            replicate_api_key: API ключ для Replicate
        """
        self.api_key = replicate_api_key
        os.environ["REPLICATE_API_TOKEN"] = replicate_api_key
        
        # Стили генерации с промптами для runwayml/gen4-image
        self.styles = {
            "photoshop": {
                "name": "🤍 Фотошоп",
                "description": "Улучшение качества без изменений",
                "prompt_template": "Professional portrait photo of @person, enhanced quality, flawless high-definition smooth beautiful skin, perfect even skin tone, flawless complexion, razor-sharp focus on eyes, detailed ultra-realistic eyes, clean background, natural colors, perfect lighting, photorealistic, ultra-high resolution, 16K, masterpiece, best quality, hyper-detailed, no artifacts, sharp details everywhere, crystal-clear sharpness, flawless 32K resolution",
                "model": "runwayml/gen4-image",
                "model_type": "gen4",
                "aspect_ratio": "3:4"
            },
            "art": {
                "name": "🎨 AI Art",
                "description": "Иллюстрация в авторском стиле",
                "prompt_template": "Digital art illustration of @person, artistic style, flawless high-definition smooth beautiful skin, perfect even skin tone, flawless complexion, razor-sharp focus on eyes, detailed ultra-realistic eyes, pastel colors, creative portrait, Pinterest aesthetic, beautiful lighting, stylized, ultra-high resolution, 16K, masterpiece, best quality, hyper-detailed, no artifacts, sharp details everywhere, crystal-clear sharpness, flawless 32K resolution",
                "model": "runwayml/gen4-image",
                "model_type": "gen4",
                "aspect_ratio": "3:4"
            },
            "cinema": {
                "name": "🎬 Кино",
                "description": "Кадр из художественного фильма",
                "prompt_template": "Cinematic portrait of @person, movie scene, flawless high-definition smooth beautiful skin, perfect even skin tone, flawless complexion, razor-sharp focus on eyes, detailed ultra-realistic eyes, dramatic lighting, film grain, atmospheric colors, professional cinematography, ultra-high resolution, 16K, masterpiece, best quality, hyper-detailed, no artifacts, sharp details everywhere, crystal-clear sharpness, flawless 32K resolution",
                "model": "runwayml/gen4-image",
                "model_type": "gen4",
                "aspect_ratio": "3:4"
            },
            "portrait": {
                "name": "🧠 Портрет",
                "description": "Глубокий психологический образ",
                "prompt_template": "Deep psychological portrait of @person, soft shadows, face focus, flawless high-definition smooth beautiful skin, perfect even skin tone, flawless complexion, razor-sharp focus on eyes, detailed ultra-realistic eyes, character photography, soulful expression, professional photographer style, ultra-high resolution, 16K, masterpiece, best quality, hyper-detailed, no artifacts, sharp details everywhere, crystal-clear sharpness, flawless 32K resolution",
                "model": "runwayml/gen4-image",
                "model_type": "gen4",
                "aspect_ratio": "3:4"
            },
            "fantasy": {
                "name": "⚡ Фантастика",
                "description": "Неон, sci-fi, визуальный драйв",
                "prompt_template": "Cyberpunk portrait of @person, neon lights, sci-fi style, flawless high-definition smooth beautiful skin, perfect even skin tone, flawless complexion, razor-sharp focus on eyes, detailed ultra-realistic eyes, futuristic, glitch effects, dramatic lighting, cosmic atmosphere, ultra-high resolution, 16K, masterpiece, best quality, hyper-detailed, no artifacts, sharp details everywhere, crystal-clear sharpness, flawless 32K resolution",
                "model": "runwayml/gen4-image",
                "model_type": "gen4",
                "aspect_ratio": "3:4"
            },
            "lego": {
                "name": "🧱 LEGO",
                "description": "Превращение в LEGO минифигурку",
                "prompt_template": "A hyper-detailed LEGO minifigure of @person, ultra-realistic LEGO style, razor-sharp focus on eyes, detailed eyes, 4K resolution, vibrant colors, precise blocky textures, LEGO hair with stud details, glossy LEGO pieces, visible brick seams, LEGO cityscape background, cinematic lighting, plastic sheen, blocky toy-like nature, perfect LEGO brick alignment, stud patterns, reflective plastic surfaces, ultra-high resolution, 16K, masterpiece, best quality, no artifacts, sharp details everywhere, crystal-clear sharpness, flawless 32K resolution",
                "model": "runwayml/gen4-image",
                "model_type": "gen4",
                "aspect_ratio": "3:4"
            }
        }
        
        # Доступные соотношения сторон
        self.aspect_ratios = {
            "9:16": "9:16",
            "4:3": "4:3",
            "3:4": "3:4",
            "1:1": "1:1",
            "16:9": "16:9"
        }
        
        # Маппинг для нормализации сокращенных aspect_ratio
        self.aspect_ratio_map = {
            "9": "9:16",
            "16": "16:9",
            "4": "4:3",
            "3": "3:4",
            "1": "1:1"
        }
        
        # Доступные разрешения (только 720p)
        self.resolutions = {
            "720p": "720p"
        }
        
        # Клиент Replicate
        self.client = replicate.Client(api_token=replicate_api_key)
    
    async def generate_image(self, image_url: str, style: str, user_id: int, aspect_ratio: str = "3:4", resolution: str = "720p") -> Dict[str, Any]:
        """
        Генерация изображения в выбранном стиле с поддержкой aspect_ratio (resolution фиксировано 720p)
        
        Args:
            image_url: metadata URL изображения (/v1/files/{id})
            style: Выбранный стиль генерации
            user_id: ID пользователя
            aspect_ratio: Соотношение сторон
            resolution: Разрешение (по умолчанию и фиксировано "720p")
            
        Returns:
            Словарь с результатами генерации
        """
        try:
            logger.info(f"Начало генерации для пользователя {user_id} в стиле {style}, aspect_ratio={aspect_ratio}, resolution={resolution}")
            
            if style not in self.styles:
                raise ValueError(f"Неизвестный стиль: {style}")
            
            # Нормализация aspect_ratio если передан сокращенный вариант
            if aspect_ratio in self.aspect_ratio_map:
                aspect_ratio = self.aspect_ratio_map[aspect_ratio]
                logger.info(f"Нормализовали aspect_ratio к {aspect_ratio}")
            
            if aspect_ratio not in self.aspect_ratios:
                raise ValueError(f"Неподдерживаемое соотношение сторон: {aspect_ratio}")
            
            if resolution not in self.resolutions:
                raise ValueError(f"Неподдерживаемое разрешение: {resolution}")
            
            style_config = self.styles[style]
            
            # Предобработка изображения: скачивание и конвертация в JPG
            logger.info(f"Предобработка изображения: скачивание и конвертация в JPG")
            image_bytes = await download_image(self.api_key, image_url, self.client)
            
            # Валидация через PIL
            try:
                img = Image.open(io.BytesIO(image_bytes))
            except Exception as pil_err:
                raise ValueError(f"Невозможно открыть изображение: {str(pil_err)}")
            
            buffer = io.BytesIO()
            img.convert('RGB').save(buffer, format="JPEG", quality=95)
            buffer.seek(0)
            processed_image_bytes = buffer.read()
            
            # Перезагружаем обработанное изображение на Replicate
            processed_url = await upload_image_to_replicate(processed_image_bytes, "processed.jpg")
            
            # Параметры для модели runwayml/gen4-image
            input_params = {
                "prompt": style_config['prompt_template'],
                "aspect_ratio": aspect_ratio,
                "reference_tags": ["person"],
                "reference_images": [processed_url],
                "output_resolution": resolution
            }
            
            # Создаем prediction через API
            logger.info(f"Создание prediction для модели {style_config['model']}")
            
            prediction = await asyncio.to_thread(
                self.client.predictions.create,
                model=style_config['model'],
                input=input_params
            )
            
            logger.info(f"Prediction создан: {prediction.id}")
            
            # Ждем завершения
            while prediction.status not in ["succeeded", "failed", "canceled"]:
                await asyncio.sleep(2)
                prediction = await asyncio.to_thread(
                    self.client.predictions.get,
                    prediction.id
                )
                logger.info(f"Статус: {prediction.status}")
            
            if prediction.status == "succeeded":
                output_url = prediction.output
                logger.info(f"Тип output: {type(output_url)}")
                logger.info(f"Output значение: {output_url}")
                
                if isinstance(output_url, str):
                    result_url = output_url
                else:
                    result_url = str(output_url)
                    logger.info(f"Преобразовали {type(output_url)} в строку: {result_url}")
                
                logger.info(f"Генерация завершена успешно, URL: {result_url}")
                
                return {
                    "success": True,
                    "result_url": result_url,
                    "style": style,
                    "style_name": style_config['name'],
                    "timestamp": datetime.now().isoformat(),
                    "prediction_id": prediction.id
                }
            else:
                error_msg = f"Генерация завершилась со статусом: {prediction.status}"
                if hasattr(prediction, 'error') and prediction.error:
                    error_msg += f" - {prediction.error}"
                
                logger.error(f"Ошибка генерации: {error_msg}")
                
                return {
                    "success": False,
                    "error": error_msg,
                    "style": style,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Ошибка генерации для пользователя {user_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "style": style,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_style_keyboard(self) -> List[List[Dict[str, str]]]:
        """
        Получение клавиатуры со стилями для inline-кнопок
        
        Returns:
            Список кнопок для InlineKeyboardMarkup
        """
        keyboard = []
        row = []
        for style_key, style_info in self.styles.items():
            button_data = {
                "text": style_info["name"],
                "callback_data": f"transform_style:{style_key}"
            }
            row.append(button_data)
            if len(row) == 2:
                keyboard.append(row)
                row = []
        
        if row:
            keyboard.append(row)
        
        keyboard.append([{"text": "❌ Отменить", "callback_data": "transform_cancel"}])
        return keyboard
    
    def get_style_info(self, style: str) -> Optional[Dict[str, str]]:
        """
        Получение информации о стиле
        
        Args:
            style: Ключ стиля
            
        Returns:
            Информация о стиле или None
        """
        return self.styles.get(style)
    
    def get_style_description(self, style: str) -> str:
        """
        Получение полного описания стиля
        
        Args:
            style: Ключ стиля
            
        Returns:
            Текстовое описание стиля
        """
        descriptions = {
            "photoshop": """🤍 Фотошоп / Улучшение

Ты — как есть, но в идеальном свете.
Фон вычищен, кожа выглядит свежо, цвета — естественные.
Как будто твое фото ретушировал профи. Без лишних фильтров.

📌 Подходит для: аватарки, резюме, профиля.
⚡ PixelPie AI улучшит качество с помощью нейросети""",
            "art": """🎨 AI Art / Иллюстрация

Фото превращается в арт.
Линии, свет, пастель или стиль digital-иллюстраций — и ты выглядишь как персонаж книги или Pinterest-постера.

📌 Подходит для: творчества, сторис, эстетичного контента.
🎨 PixelPie AI создаст уникальную иллюстрацию""",
            "cinema": """🎬 Кино / Cinematic

Ты в кадре — будто это сцена из фильма.
Атмосферные цвета, направленный свет и немного драмы.

📌 Подходит для: ярких образов, вау-эффекта, постов с настроением.
🎥 PixelPie AI применит кинематографическую обработку""",
            "portrait": """🧠 Портрет / Психологический

Глубокий взгляд, мягкие тени, фокус на лице.
Эффект, как будто ты снят фотографом, который умеет показать характер.

📌 Подходит для: спокойных аватарок, презентаций, контента с душой.
📸 PixelPie AI создаст профессиональный портрет""",
            "fantasy": """⚡ Фантастика / Neon-Cyber

Ты в другом времени, в другой вселенной.
Глитчи, неон, драматичный свет — визуальный экшн прямо из sci-fi трейлера.

📌 Подходит для: аватарок, сторис, ярких постов.
🚀 PixelPie AI перенесет тебя в будущее""",
            "lego": """🧱 LEGO / Минифигурка

Превращение в детализированную LEGO минифигурку.
Яркие цвета, блочные текстуры, культовый стиль конструктора.
Каждая деталь проработана до мельчайших кубиков!

📌 Подходит для: веселых аватарок, подарков, креативного контента.
🧱 PixelPie AI создаст твою уникальную LEGO-версию"""
        }
        return descriptions.get(style, "Описание недоступно")
    
    def get_aspect_ratio_keyboard(self, style_key: str) -> List[List[Dict[str, str]]]:
        """
        Получение клавиатуры для выбора соотношения сторон
        
        Args:
            style_key: Ключ стиля для callback_data
        
        Returns:
            Список кнопок для InlineKeyboardMarkup
        """
        keyboard = []
        row = []
        for ratio_key in self.aspect_ratios:
            button_data = {
                "text": f"{ratio_key}",
                "callback_data": f"transform_ratio:{style_key}:{ratio_key}"
            }
            row.append(button_data)
            if len(row) == 3:
                keyboard.append(row)
                row = []
        
        if row:
            keyboard.append(row)
        
        keyboard.append([{"text": "❌ Отменить", "callback_data": "transform_cancel"}])
        return keyboard
    
    def get_resolution_keyboard(self, style_key: str, aspect_ratio: str) -> List[List[Dict[str, str]]]:
        """
        Получение клавиатуры для выбора разрешения (только 720p)
        
        Args:
            style_key: Ключ стиля
            aspect_ratio: Выбранное соотношение сторон для callback_data
        
        Returns:
            Список кнопок для InlineKeyboardMarkup
        """
        keyboard = []
        row = []
        for res_key in self.resolutions:
            button_data = {
                "text": f"{res_key}",
                "callback_data": f"transform_resolution:{style_key}:{aspect_ratio}:{res_key}"
            }
            row.append(button_data)
        
        if row:
            keyboard.append(row)
        
        keyboard.append([{"text": "❌ Отменить", "callback_data": "transform_cancel"}])
        return keyboard
    
    def get_aspect_ratio_description(self) -> str:
        """
        Получение красивого информативного текста о выборе соотношения сторон
        
        Returns:
            Текст для показа в сообщении бота
        """
        return """
📐 Выберите соотношение сторон для вашего изображения!

Это как рамка для картины: определяет форму изображения.
- 9:16: Вертикальный формат, идеален для сторис и мобильных экранов 📱
- 16:9: Горизонтальный, как видео на YouTube или киноэкран 🎥
- 1:1: Квадрат, классика для аватарок и постов в соцсетях 🔲
- 3:4 / 4:3: Портретный или альбомный, универсальный для фото 📸

PixelPie AI подстроит генерацию под выбранный формат для лучшего результата! ✨
"""