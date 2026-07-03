#!/usr/bin/env python3
import http.server
import json
import os
from pathlib import Path

PORT = 8002
DIR = Path(__file__).parent
TOPICS_DIR = DIR / 'topics'

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIR), **kwargs)

    def do_GET(self):
        if self.path == '/topics-list':
            files = sorted([f for f in os.listdir(TOPICS_DIR) if f.endswith('.json')])
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(files).encode())
            return
        return super().do_GET()

if __name__ == '__main__':
    os.chdir(str(DIR))
    srv = http.server.HTTPServer(('127.0.0.1', PORT), Handler)
    print(f'KDM1 Exam Study App running on http://127.0.0.1:{PORT}')
    srv.serve_forever()
