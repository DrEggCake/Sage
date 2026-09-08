#!/usr/bin/env python3
"""Serve the cross-sim CSV recordings to the browser replay player.

Usage:
    python3 tests/replay/serve_replay.py [--results DIR] [--port N]

  --results DIR   directory containing *_mujoco.csv / *_omnisim.csv
                  (default: tests/results, or wherever --dir pointed at)

Opens http://127.0.0.1:8787/ where replay.html lives.
"""

import argparse
import json
import os
import re
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE.parent / "results"
RECORDING_RE = re.compile(r"^(?P<scenario>.+)_(?P<engine>mujoco|omnisim)\.csv$")


def build_manifest(results_dir: Path) -> dict:
    recordings = []
    if not results_dir.is_dir():
        return {"error": f"results dir not found: {results_dir}", "recordings": []}
    for path in sorted(results_dir.glob("*.csv")):
        m = RECORDING_RE.match(path.name)
        if not m:
            continue
        try:
            first = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[0]
        except (OSError, IndexError):
            first = ""
        recordings.append({
            "id": path.name[:-4],
            "scenario": m.group("scenario"),
            "engine": m.group("engine"),
            "file": f"/results/{path.name}",
            "header": first,
            "rows": _count_rows(path),
        })
    recordings.sort(key=lambda r: (r["scenario"], r["engine"]))
    return {"results_dir": str(results_dir), "recordings": recordings}


def _count_rows(path: Path) -> int:
    n = 0
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            next(fh, None)
            for _ in fh:
                n += 1
    except OSError:
        pass
    return n


class ReplayHandler(BaseHTTPRequestHandler):
    results_dir = DEFAULT_RESULTS

    def log_message(self, fmt, *args):  # quieter logs
        sys.stderr.write("[replay] %s\n" % (fmt % args))

    def _send(self, body: bytes, mime: str, status: int = 200, cache: bool = False):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", "public, max-age=300")
        else:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/manifest":
            manifest = build_manifest(self.results_dir)
            self._send(json.dumps(manifest).encode(), "application/json")
            return
        if path.startswith("/results/"):
            rel = path[len("/results/"):]
            # Prevent traversal.
            target = (self.results_dir / rel).resolve()
            root = self.results_dir.resolve()
            if not target.is_relative_to(root) or not target.is_file():
                self._send(b'{"error": "not found"}', "application/json", 404)
                return
            data = target.read_bytes()
            mime = "text/csv" if target.suffix == ".csv" else "application/octet-stream"
            self._send(data, mime, cache=True)
            return
        if path in ("/", "/index.html"):
            data = (HERE / "replay.html").read_bytes()
            self._send(data, "text/html; charset=utf-8")
            return
        # Static: vendor/ or anything under HERE.
        target = (HERE / path.lstrip("/")).resolve()
        if HERE not in target.parents and target != HERE:
            self._send(b"forbidden", "text/plain", 403)
            return
        if not target.is_file():
            self._send(b"not found", "text/plain", 404)
            return
        mime = {
            ".js": "text/javascript",
            ".html": "text/html; charset=utf-8",
            ".css": "text/css",
            ".json": "application/json",
        }.get(target.suffix, "application/octet-stream")
        self._send(target.read_bytes(), mime, cache=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS,
                    help="directory of *_mujoco.csv / *_omnisim.csv files")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--no-browser", action="store_true",
                    help="do not auto-open the browser")
    args = ap.parse_args()

    ReplayHandler.results_dir = args.results.resolve()
    if not ReplayHandler.results_dir.is_dir():
        print(f"WARNING: results dir not found: {ReplayHandler.results_dir}")
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), ReplayHandler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Replay player: {url}")
    print(f"Recording dir: {ReplayHandler.results_dir}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()