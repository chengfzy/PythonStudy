from http.server import HTTPServer, BaseHTTPRequestHandler
from http import HTTPStatus
from io import BytesIO


class SimpleHttpRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(HTTPStatus.OK)
        self.end_headers()
        self.wfile.write(b'Hello World!')

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        self.send_response(HTTPStatus.OK)
        self.end_headers()

        response = BytesIO()
        response.write(b'This is POSE request.')
        response.write(b'Received: ')
        response.write(body)
        self.wfile.write(response.getvalue())


if __name__ == '__main__':
    server = HTTPServer(('localhost', 4443), SimpleHttpRequestHandler)
    server.serve_forever()

    # test shell command
    # 1. GET: http http://127.0.0.1:8000
    # 2. POST: http http://127.0.0.1:8000 key=value
