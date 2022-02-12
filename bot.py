import glob
import logging
import asyncio
import warnings
import threading
from os import environ
import keyboards as kb
import image_processing
from dotenv import load_dotenv
from aiogram.utils.executor import start_webhook
from aiogram import Bot, Dispatcher, executor, types

load_dotenv()
warnings.filterwarnings("ignore")

API_TOKEN = environ.get('API_TOKEN')
CONNECTION_TYPE = environ.get('CONNECTION_TYPE')

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)
db_photos = {}

with open("images/styles/description.txt", encoding='utf-8') as f:
    styles_text = f.read()
style_images = sorted([file for file in glob.glob('images/styles/*.jpg')])


class User:
    def __init__(self, user_id):
        self.id = user_id
        self.style_img = 0
        self.type_algo = None

    def restart(self, algo=None):
        self.style_img = 0
        self.type_algo = algo


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    db_photos[message.from_user.id] = User(message.from_user.id)
    logging.info(f"New User! Current number of users in dict: {len(db_photos)}")
    await message.answer(f"Привет, *{message.from_user.first_name}*! \U0001F44B\n\n {kb.start_message}",
                         parse_mode='Markdown', reply_markup=kb.start_keyboard())


@dp.callback_query_handler(text="menu")
async def transfer_style(call: types.CallbackQuery):
    user = db_photos[call.from_user.id]
    user.restart()
    await call.message.answer(kb.menu_message, reply_markup=kb.start_keyboard())
    await call.answer()


@dp.callback_query_handler(text="button_style")
async def transfer_style(call: types.CallbackQuery):
    user = db_photos[call.from_user.id]
    user.restart('nst')
    await call.message.answer("Для начала мне нужно 2 фотографии. Начнем с фотографии стиля, отправь её мне.\n"
                              "Если не знаешь какую выбрать, то можешь взять одну из готовых картин. "
                              "Для просмотра всех вариантов нажми на кнопку ниже \U0001F447",
                              reply_markup=kb.style_images())
    await call.answer()


@dp.callback_query_handler(text="style_images")
async def transfer_style(call: types.CallbackQuery):
    await types.ChatActions.upload_photo()
    media = types.MediaGroup()
    for i, image in enumerate(style_images):
        if i == 0:
            media.attach_photo(types.InputFile(image), f"У меня есть такие картины:\n\n{styles_text}")
        else:
            media.attach_photo(types.InputFile(image))
    await call.message.answer_media_group(media)
    await call.message.answer("Если что-то понравилось, то выбери картину кнопкой, если нет - отправляй собственную"
                              " фотографию для стиля", reply_markup=kb.select_style())
    await call.answer()


@dp.callback_query_handler(lambda call: call.data.startswith('style_'))
async def transfer_style(call: types.CallbackQuery):
    user = db_photos[call.from_user.id]
    user.style_img = style_images[int(call.data[-1]) - 1]
    await call.message.answer(f"Я запомнил, что ты выбрал {int(call.data[-1])} стиль" +
                              "\nТеперь отправь мне фотографию, на которую этот стиль нужно перенести")
    await call.answer()


@dp.callback_query_handler(text="style2vangogh")
async def transfer_style(call: types.CallbackQuery):
    user = db_photos[call.from_user.id]
    user.restart('style2vangogh')
    await call.message.answer("OK. Пришли мне фотографию, из которой я создам картину в стиле Ван Гога")
    await call.answer()


@dp.callback_query_handler(text="style2monet")
async def transfer_style(call: types.CallbackQuery):
    user = db_photos[call.from_user.id]
    user.restart('style2monet')
    await call.message.answer("OK. Пришли мне фотографию, из которой я создам картину в стиле Моне")
    await call.answer()


@dp.callback_query_handler(text="examples")
async def transfer_style(call: types.CallbackQuery):
    await types.ChatActions.upload_photo()
    media = types.MediaGroup()
    media.attach_photo(types.InputFile("images/examples/nst.jpg"), "Перенос стиля")
    media.attach_photo(types.InputFile("images/examples/style2vangogh.jpg"), "Получилась картина Ван Гога")
    media.attach_photo(types.InputFile("images/examples/style2monet.jpg"), "Получилась картина Моне")
    await call.message.answer_media_group(media)
    await call.message.answer("Надеюсь, тебе понравились эти примеры, и ты захотел попробовать\n\n"
                              "Выбери алгоритм",
                              reply_markup=kb.algo_keyboard())
    await call.answer()


@dp.message_handler(content_types=['photo'])
async def handle_docs_photo(message: types.Message):
    image = message.photo[-1]
    file_info = await bot.get_file(image.file_id)
    photo = await bot.download_file(file_info.file_path)
    user = db_photos[message.from_user.id]
    if user.type_algo == 'style2vangogh':
        await message.answer("Подожди минутку и всё будет готово! \U000023F3")
        logging.info(f"Start CycleGAN style2vangogh")
        threading.Thread(
            target=lambda mess, img, type_algo:
            asyncio.run(image_processing.cycle_gan(mess, img, type_algo)),
            args=(message, photo, user.type_algo)).start()

    elif user.type_algo == 'style2monet':
        await message.answer("Подожди минутку и всё будет готово! \U000023F3")
        logging.info(f"Start CycleGAN style2monet")
        threading.Thread(
            target=lambda mess, img, type_algo:
            asyncio.run(image_processing.cycle_gan(mess, img, type_algo)),
            args=(message, photo, user.type_algo)).start()

    elif user.type_algo == 'nst':
        if user.style_img == 0:
            user.style_img = photo
            await message.answer("Теперь отправь фотографию, на которую перенести стиль")
        else:
            await message.answer("Решение требует времени. Подожди не более 5 минут, и я отправлю результат \U0001F9A5")
            logging.info(f"Start Style Transfer")
            threading.Thread(
                target=lambda mess, style_img, content_img:
                asyncio.run(image_processing.style_transfer(mess, style_img, content_img)),
                args=(message, user.style_img, photo)).start()
    else:
        await message.answer("Прежде чем отправлять мне фотографии скажи мне какой алгоритм использовать",
                             reply_markup=kb.algo_keyboard())


@dp.message_handler()
async def echo(message: types.Message):
    await message.answer(kb.menu_message, reply_markup=kb.start_keyboard())


async def on_startup():
    await bot.set_webhook(WEBHOOK_URL)


async def on_shutdown(disbot):
    logging.warning("Shutting down..")
    await disbot.storage.close()
    await disbot.storage.wait_closed()
    logging.warning("Bye!")


if __name__ == '__main__':

    webhook_settings = False if CONNECTION_TYPE == 'POLLING' else True

    if webhook_settings:
        WEBHOOK_HOST = environ.get("WEBHOOK_HOST")
        WEBHOOK_PATH = f"webhook/{API_TOKEN}/"
        WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"
        WEBAPP_HOST = environ.get("WEBAPP_HOST")
        WEBAPP_PORT = environ.get("WEBAPP_PORT")

        start_webhook(
            dispatcher=dp,
            webhook_path=f"/{WEBHOOK_PATH}",
            on_startup=on_startup,
            on_shutdown=on_shutdown,
            skip_updates=False,
            host=WEBAPP_HOST,
            port=WEBAPP_PORT,
        )
    else:
        executor.start_polling(dp, skip_updates=True)
