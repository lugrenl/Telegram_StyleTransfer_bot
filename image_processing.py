import logging
import warnings
import keyboards as kb
from os import environ
from aiogram import Bot
from dotenv import load_dotenv
from model_cyclegan import CycleGAN
from model_nst import StyleTransfer


load_dotenv()
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)

API_TOKEN = environ.get('API_TOKEN')


async def cycle_gan(message, image, type_algo):
    if type_algo == 'style2vangogh':
        wts_path = "models_wts/style2vangogh.pth"
    elif type_algo == 'style2monet':
        wts_path = "models_wts/style2monet.pth"

    new_image = CycleGAN.run_gan(wts_path, image)

    logging.info(f"Finished CycleGAN")

    tmp_bot = Bot(token=API_TOKEN)
    await tmp_bot.send_photo(message.chat.id, photo=new_image)
    await tmp_bot.send_message(message.chat.id, "Надеюсь, результат тебе понравился.\n\n Попробуем еще раз?",
                               reply_markup=kb.algo_keyboard())
    await tmp_bot.close()


async def style_transfer(message, style_image, content_image):
    new_image = StyleTransfer.run_nst(style_image, content_image)

    logging.info(f"Finished Style Transfer")

    tmp_bot = Bot(token=API_TOKEN)
    await tmp_bot.send_photo(message.chat.id, photo=new_image)
    await tmp_bot.send_message(message.chat.id, "Надеюсь, результат тебе понравился.\n\n Попробуем еще раз?",
                               reply_markup=kb.algo_keyboard())
    await tmp_bot.close()
