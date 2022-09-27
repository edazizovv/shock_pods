#
import os
import asyncio


#
from dotenv import load_dotenv


#
from grab.go import grab_telegram

#
load_dotenv()

API_ID = os.getenv('API_ID')
API_HASH = os.getenv('API_HASH')

# chat_id = '@headlines_for_traders_eng'
chat_id = '@forbesrussia'
chunk_size = 100
n_posts = 10000

save_to = './forbes_ru.csv'


result = asyncio.run(grab_telegram(chat_id=chat_id, api_id=API_ID, api_hash=API_HASH,
                                   n_posts=n_posts, chunk_size=chunk_size, save_to=save_to),
                     debug=False)
