import asyncio
from pyrogram import Client
import pandas

api_id = '11867327'
api_hash = "f1766d32bf6c97bddbd823cc043bbfd0"

# chat_id = '@headlines_for_traders_eng'
chat_id = '@forbesrussia'
chunk_size = 100
n_posts = 1000


async def main():

    async with Client("my_account", api_id, api_hash) as app:
        data = {'id': [],
                'link': [],
                'text': [],
                'caption': [],
                'views': [],
                'reactions': [],
                }
        for j in range((n_posts // chunk_size) + 1):
            offset = j * chunk_size
            limit = min((chunk_size, n_posts - offset))
            print(limit)
            if limit != 0:
                hu = app.get_chat_history(chat_id=chat_id, offset=offset, limit=limit)
                async for message in hu:
                    data['id'].append(message.id)
                    data['link'].append(message.link)
                    if message.text is None:
                        data['text'].append('')
                    else:
                        data['text'].append(str(message.text))
                    if message.caption is None:
                        data['caption'].append('')
                    else:
                        data['caption'].append(str(message.caption))
                    data['views'].append(message.views)
                    if message.reactions is None:
                        data['reactions'].append({})
                    else:
                        data['reactions'].append(
                            {str(reaction.emoji): reaction.count for reaction in message.reactions})
        data = pandas.DataFrame(data=data)
        data.to_csv('./data.csv')
            # print(data['text'][-1])
        # print(data['text'][-1])

    return data

# result = asyncio.run(main())
result = asyncio.run(main(), debug=False)
# asyncio.run(main())