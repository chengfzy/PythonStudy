import asyncio
from pathlib import Path
import websockets
import argparse, logging, coloredlogs, datetime


async def server(ws, path):
    async for message in ws:
        logging.info(f'receive message: {message}')
        await ws.send(f'##{message}##')


if __name__ == '__main__':
    # config logging
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f"/tmp/{Path(__file__).stem}.{datetime.datetime.now().strftime('%Y%m%d-%H%M%S.%f')}.log"
            ),
        ],
    )
    coloredlogs.install(fmt="[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s")

    asyncio.get_event_loop().run_until_complete(websockets.serve(server, 'localhost', 60010))
    asyncio.get_event_loop().run_forever()
