#!/usr/bin/env python3
"""Serve the cross-sim CSV recordings to the browser replay player.

Usage:
    python3 tests/replay/serve_replay.py [--results DIR] [--firelogs DIR]
                                         [--brains DIR] [--port N]

  --results DIR   directory containing *_mujoco.csv / *_omnisim.csv
                  (default: tests/results)
  --firelogs DIR  directory of SNN fire-log CSVs (with optional .meta.json
                  sidecars) written by: sage --fire-log FILE
  --brains DIR    directory of trained brain JSONs (data/brain)

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


def build_manifest(results_dir: Path, firelogs_dir: Path | None,
                   brains_dir: Path | None) -> dict:
    recordings = []
    if not results_dir.is_dir():
        return {"error": f"results dir not found: {results_dir}",
                "recordings": [], "firelogs": [], "brains": []}
    for path in sorted(results_dir.glob("*.csv")):
        m = RECORDING_RE.match(path.name)
        if not m:
            continue
        try:
            first = path.read_text(encoding="utf-8",
                                   errors="replace").strip().splitlines()[0]
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

    firelogs = []
    if firelogs_dir and firelogs_dir.is_dir():
        for path in sorted(firelogs_dir.glob("*.csv")):
            entry = {
                "id": path.name[:-4],
                "file": f"/firelogs/{path.name}",
                "rows": _count_rows(path),
            }
            meta = Path(str(path) + ".meta.json")
            if meta.is_file():
                try:
                    entry["meta"] = json.loads(meta.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    pass
            firelogs.append(entry)

    brains = []
    if brains_dir and brains_dir.is_dir():
        for path in sorted(brains_dir.glob("*.json")):
            entry = {"id": path.name[:-5], "file": f"/brains/{path.name}"}
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                for k in ("layerSizes", "wiringLimits", "learningRate",
                          "episodesTrained", "successes", "totalReward",
                          "bestReward", "totalSteps"):
                    if k in data:
                        entry[k] = data[k]
            except (OSError, ValueError):
                pass
            brains.append(entry)

    return {"results_dir": str(results_dir),
            "firelogs_dir": str(firelogs_dir) if firelogs_dir else None,
            "brains_dir": str(brains_dir) if brains_dir else None,
            "recordings": recordings,
            "firelogs": firelogs,
            "brains": brains}


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
    firelogs_dir = None
    brains_dir = None

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

    def _serve_dir(self, prefix: str, directory: Path | None):
        if directory is None:
            self._send(b'{"error": "not found"}', "application/json", 404)
            return
        rel = self.path[len(prefix):]
        target = (directory / rel).resolve()
        root = directory.resolve()
        if not rel or not target.is_relative_to(root) or not target.is_file():
            self._send(b'{"error": "not found"}', "application/json", 404)
            return
        data = target.read_bytes()
        mime = ("application/json" if target.suffix == ".json"
                else "text/csv" if target.suffix == ".csv"
                else "application/octet-stream")
        self._send(data, mime, cache=True)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/manifest":
            manifest = build_manifest(self.results_dir, self.firelogs_dir,
                                      self.brains_dir)
            self._send(json.dumps(manifest).encode(), "application/json")
            return
        if path.startswith("/firelogs/"):
            self._serve_dir("/firelogs/", self.firelogs_dir)
            return
        if path.startswith("/brains/"):
            self._serve_dir("/brains/", self.brains_dir)
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
    ap.add_argument("--firelogs", type=Path, default=None,
                    help="directory of SNN fire-log CSVs (+ .meta.json sidecars)")
    ap.add_argument("--brains", type=Path, default=None,
                    help="directory of trained brain JSONs (data/brain)")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--no-browser", action="store_true",
                    help="do not auto-open the browser")
    args = ap.parse_args()

    ReplayHandler.results_dir = args.results.resolve()
    ReplayHandler.firelogs_dir = args.firelogs.resolve() if args.firelogs else None
    ReplayHandler.brains_dir = args.brains.resolve() if args.brains else None
    if not ReplayHandler.results_dir.is_dir():
        print(f"WARNING: results dir not found: {ReplayHandler.results_dir}")
    for label, d in (("firelogs", ReplayHandler.firelogs_dir),
                     ("brains", ReplayHandler.brains_dir)):
        if d and not d.is_dir():
            print(f"WARNING: {label} dir not found: {d}")
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), ReplayHandler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Replay player: {url}")
    print(f"Recording dir: {ReplayHandler.results_dir}")
    if ReplayHandler.firelogs_dir:
        print(f"Fire-log dir: {ReplayHandler.firelogs_dir}")
    if ReplayHandler.brains_dir:
        print(f"Brain dir: {ReplayHandler.brains_dir}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()