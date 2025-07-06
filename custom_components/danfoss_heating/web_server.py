import http.server
import socketserver
import threading
import logging
import json
from .pairing import DanfossPairing

_LOGGER = logging.getLogger(__name__)

class DanfossRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/api/danfoss_heating/discover':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            
            pairing = DanfossPairing(self.server.hass)
            discovered_devices = pairing.pair(data['otp']) # This is not async, for simplicity
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(discovered_devices).encode())
        else:
            self.send_error(404, 'File Not Found: %s' % self.path)

class DanfossWebServer(threading.Thread):
    def __init__(self, hass, port=8080):
        super().__init__()
        self.hass = hass
        self.port = port
        self.httpd = None

    def run(self):
        handler = DanfossRequestHandler
        # Allow the port to be reused immediately
        socketserver.TCPServer.allow_reuse_address = True
        self.httpd = socketserver.TCPServer(("", self.port), handler)
        self.httpd.hass = self.hass # Pass hass to the request handler
        _LOGGER.info("Starting web server on port %s", self.port)
        self.httpd.serve_forever()

    def stop(self):
        if self.httpd:
            _LOGGER.info("Stopping web server")
            self.httpd.shutdown()
            self.httpd.server_close()
