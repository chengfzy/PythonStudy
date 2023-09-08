import argparse, logging, coloredlogs, datetime
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from http import HTTPStatus
from io import BytesIO
from urllib.parse import urlparse, parse_qs
import json


class SimpleHttpRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        logging.warning('GET')
        logging.info('Client Values:')
        logging.info(f'\tclient address: {self.client_address} ({self.address_string()})')
        logging.info(f'\tcommand: {self.command}')
        logging.info(f'\tpath: {self.path}')
        parsed_path = urlparse(self.path)
        parsed_params = parse_qs(parsed_path.query)
        logging.info(f'\treal path: {parsed_path.path}')
        logging.info(f'\tparsed query: {parsed_path.query}')
        logging.info(f'\tparsed params: {parsed_params}')
        logging.info(f'\trequest version: {self.request_version}')
        logging.info('Server Values:')
        logging.info(f'\tserver version: {self.server_version}')
        logging.info(f'\tsystem version: {self.sys_version}')
        logging.info(f'\tprotocol version: {self.protocol_version}')
        logging.info('Headers Received')
        for key, value in self.headers.items():
            logging.info(f'\t{key} = {value}')

        self.send_response(HTTPStatus.OK)
        self.end_headers()
        self.wfile.write(b'Hello World!')

    def do_POST(self):
        logging.warning('POST')
        logging.info(f'\tclient: {self.client_address}')
        logging.info(f'\tuser agent: {self.headers["user-agent"]}')
        logging.info(f'\tpath: {self.path}')
        logging.info(f'\tcontent type: {self.headers["Content-Type"]}')
        logging.info('Headers Received')
        for key, value in self.headers.items():
            logging.info(f'\t{key} = {value}')

        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        data = json.loads(body)
        logging.info(f'input parameters: {data}')

        self.send_response(HTTPStatus.OK)
        self.end_headers()

        # response = BytesIO()
        # response.write(b'This is POST request.')
        # response.write(b'Received: ')
        # self.wfile.write(response.getvalue())
        self.wfile.write('This is POST request.\n'.encode('UTF-8'))
        self.wfile.write(f'{json.dumps(data, indent=4)}\n'.encode('UTF-8'))


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

    server = HTTPServer(('localhost', 8080), SimpleHttpRequestHandler)
    logging.info(f'server address: {server.server_address}')
    server.serve_forever()

    # test shell command
    # 1. GET:
    #   - `http http://127.0.0.1:8080` or `http http://localhost:8080`
    #   - `http "http://localhost:8080/?foo=bar`
    # 2. POST:
    #   - `http http://127.0.0.1:8080 foo=bar` or `http http://localhost:8080 foo=bar`
