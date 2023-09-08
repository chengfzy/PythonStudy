import asyncio
from pathlib import Path
import websockets, websocket
import argparse, logging, coloredlogs, datetime


async def client(url):
    async with websockets.connect(url) as ws:
        await ws.send('Hello WebSocket!')
        recv_text = await ws.recv()
        logging.info(f'receive: {recv_text}')


class Client:
    def __init__(self, url: str) -> None:
        self.__url = url
        self.__ws: websocket.WebSocketApp = None

    def start(self):
        self.__ws = websocket.WebSocketApp(
            self.__url, on_open=self.on_open, on_message=self.on_message, on_ping=self.on_ping, on_close=self.on_close
        )
        self.__ws.run_forever()

    def on_open(self, ws):
        logging.info('a new app open')

    def on_message(self, ws, data):
        logging.info(f'receive message: {data}')

    def on_ping(self, ws, data):
        logging.info(f'ping: {data}')
        self.__ws.send("", websocket.ABNF.OPCODE_PONG)

    def on_close(self, ws, close_status_code, close_msg):
        logging.info('close')
        # logging.info(f'close: code = {close_status_code}, reason = {close_msg}')


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

    url = 'ws://localhost:60000/test'
    # url = 'ws://192.168.100.2:60002'
    # url = 'ws://47.98.131.50:60002'
    # url = '127.0.0.1:60010'
    # 1. client using websockets
    # asyncio.get_event_loop().run_until_complete(client(url))

    # 2. client using websocket
    client = Client(url)
    client.start()
