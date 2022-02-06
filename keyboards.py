from aiogram import types


def start_keyboard():
    buttons = [
        types.InlineKeyboardButton(text="Перенести стиль", callback_data="button_style"),
        types.InlineKeyboardButton(text="Фото в картину Ван Гога", callback_data="style2vangogh"),
        types.InlineKeyboardButton(text="Фото в картину Моне", callback_data="style2monet"),
        types.InlineKeyboardButton(text="\U0001F4A5 Примеры", callback_data="examples"),
        types.InlineKeyboardButton(text="\U0001F4C3 GitHub",
                                   url="https://github.com/lugrenl/Telegram_StyleTransfer_bot")
    ]
    keyboard = types.InlineKeyboardMarkup(row_width=1)
    keyboard.add(*buttons)

    return keyboard


def style_images():
    buttons = [
        types.InlineKeyboardButton(text="Выбрать фотографию для стиля", callback_data="style_images"),
        types.InlineKeyboardButton(text="Главное меню", callback_data="menu")
    ]
    keyboard = types.InlineKeyboardMarkup(row_width=1)
    keyboard.add(*buttons)

    return keyboard


def select_style():
    buttons = [
        types.InlineKeyboardButton(text="1", callback_data="style_1"),
        types.InlineKeyboardButton(text="2", callback_data="style_2"),
        types.InlineKeyboardButton(text="3", callback_data="style_3"),
        types.InlineKeyboardButton(text="4", callback_data="style_4"),
        types.InlineKeyboardButton(text="5", callback_data="style_5"),
        types.InlineKeyboardButton(text="6", callback_data="style_6"),
        types.InlineKeyboardButton(text="7", callback_data="style_7"),
        types.InlineKeyboardButton(text="8", callback_data="style_8"),
        types.InlineKeyboardButton(text="Главное меню", callback_data="menu")
    ]
    keyboard = types.InlineKeyboardMarkup(row_width=2)
    keyboard.add(*buttons)

    return keyboard


def algo_keyboard():
    buttons = [
        types.InlineKeyboardButton(text="Перенести стиль", callback_data="button_style"),
        types.InlineKeyboardButton(text="Фото в картину Ван Гога", callback_data="style2vangogh"),
        types.InlineKeyboardButton(text="Фото в картину Моне", callback_data="style2monet"),
        types.InlineKeyboardButton(text="Главное меню", callback_data="menu")
    ]
    keyboard = types.InlineKeyboardMarkup(row_width=1)
    keyboard.add(*buttons)

    return keyboard


start_message = ("Я *Style-Transfer* бот. \U0001F916\n\n" 
                 "Я умею создавать новые фотографии с помощью нейронных сетей. \U0001F3A8\n\n"
                 "\U00002705 Если отправишь мне 2 фотографии: с первой фотографии я возьму стиль и "
                 "перенесу его на вторую фотографию. \U0001F305 \U000027A1 \U0001F307\n\n"
                 "\U00002705 Кроме того, я могу из любой фотографии сделать картину в стиле Ван Гога. \U0001F5BC\n\n"
                 "\U00002705 Или картину в стиле Моне. \U0001F5BC\n\n"
                 "У меня есть примеры работ, жми на кнопку и оцени!\n\n"
                 "Если интересно, как это все устроено внутри, "
                 "то заходи на мой GitHub. \U0001F4A1")

menu_message = ("Напоминаю, что я умею делать:\n\n"
                "\U00002705 Переносить стиль одной фотографии на другую \U0001F305 \U000027A1 \U0001F307\n\n"
                "\U00002705 Превратить фото в картину Ван Гога \U0001F5BC\n\n"
                "\U00002705 Превратить фото в картину Моне \U0001F5BC\n\n"
                "Если что-то пошло не так, то попробуй перезапустить меня командой /start")
