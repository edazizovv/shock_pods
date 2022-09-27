#
import asyncio


#
import pandas
from pyrogram import Client


#


#
async def async_range(count):
    for i in range(count):
        yield i
        await asyncio.sleep(0.0)


async def grab_telegram(chat_id, api_id, api_hash, n_posts, chunk_size, save_to):

    async with Client("my_account", api_id, api_hash) as app:
        data = {'id': [],
                'link': [],
                'text': [],
                'caption': [],
                'views': [],
                'reactions': [],
                }
        async for j in async_range((n_posts // chunk_size) + 1):
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
            print(len(data['id']))
        data = pandas.DataFrame(data=data)
        data.to_csv(save_to)

    return data
