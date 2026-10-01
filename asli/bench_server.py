"""`asli bench` — the test bench as a local web page.

Why a local server rather than the static site: the keys stay in `.env` on the machine
running it and are never sent to a browser, so someone else can speak into the bench —
in a meeting, on a shared screen, or over a tunnel with a passcode — without being handed
a key. The page is a thin client: it records or decodes audio to 16 kHz WAV, posts it, and
renders the stream of events that comes back while the call is still in progress.

Standard library only. One run streams to every chosen provider at once, in real time, so
a 6-second recording takes about 8 seconds whatever the number of providers.
"""

from __future__ import annotations

import hmac
import json
import mimetypes
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from . import bench

PAGE = Path(__file__).with_name("bench_page.html")
MAX_UPLOAD = 25 * 1024 * 1024
MAX_CONCURRENT_RUNS = 2  # each run bills every provider it streams to


class BenchServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, *, passcode: str | None = None, runs_dir: Path = bench.RUNS,
                 samples: list[dict] | None = None):
        super().__init__(addr, Handler)
        self.passcode = passcode or None
        self.runs_dir = runs_dir
        self.samples = {s["id"]: s for s in (bench.load_samples() if samples is None else samples)}
        self.slots = threading.BoundedSemaphore(MAX_CONCURRENT_RUNS)


class Handler(BaseHTTPRequestHandler):
    server: BenchServer
    server_version = "asli-bench"

    def log_message(self, fmt, *args):  # one short line per request, never the query string
        print(f"  {self.command} {urlparse(self.path).path} -> {args[1] if len(args) > 1 else ''}")

    # --- plumbing -------------------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode(),
                   "application/json; charset=utf-8")

    # headers a tunnel or reverse proxy adds. A bench bound to localhost and published
    # through ngrok or cloudflared looks local to itself, so the bind address alone cannot
    # tell it that strangers can reach it — these can.
    PROXY_HEADERS = ("X-Forwarded-For", "X-Forwarded-Host", "Forwarded", "CF-Connecting-IP",
                     "X-Real-IP")

    def _authed(self) -> bool:
        want = self.server.passcode
        if not want:
            if any(self.headers.get(h) for h in self.PROXY_HEADERS):
                self._json({"error": "this bench is being reached through a tunnel or proxy; "
                                     "restart it with --passcode so strangers cannot spend "
                                     "your API credits"}, 403)
                return False
            return True
        got = self.headers.get("X-Asli-Passcode", "")
        if hmac.compare_digest(got.encode(), want.encode()):
            return True
        self._json({"error": "passcode required"}, 401)
        return False

    def _run_dir(self, rid: str) -> Path | None:
        d = (self.server.runs_dir / rid).resolve()
        root = self.server.runs_dir.resolve()
        return d if d.parent == root and (d / "report.json").exists() else None

    # --- routes ---------------------------------------------------------------------
    def do_GET(self) -> None:
        url = urlparse(self.path)
        path = url.path
        if path in ("/", "/index.html"):
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if not self._authed():
            return
        if path == "/api/config":
            ok = bench.configured()
            return self._json({
                "providers": [{"id": p.id, "label": p.label, "short": p.short, "live": p.live,
                               "configured": ok[p.id], "env": p.env, "gate_param": p.gate_param,
                               "note": p.note} for p in bench.PROVIDERS.values()],
                "samples": list(self.server.samples.values()),
                "defaults": {"gate_ms": 500, "hold_ms": 600, "min_pause_ms": bench.MIN_PAUSE_MS,
                             "max_seconds": bench.MAX_SECONDS},
                "entity_types": list(bench.ENTITY_TYPES),
                "caveats": bench.CAVEATS,
            })
        if path == "/api/history":
            return self._json(bench.history(self.server.runs_dir))
        if path.startswith("/api/run/"):
            parts = path[len("/api/run/"):].split("/")
            d = self._run_dir(parts[0])
            if d is None:
                return self._json({"error": "no such run"}, 404)
            if len(parts) == 1:
                return self._send(200, (d / "report.json").read_bytes(),
                                  "application/json; charset=utf-8")
            if parts[1:] == ["audio.wav"]:
                return self._send(200, (d / "audio.wav").read_bytes(), "audio/wav")
        if path.startswith("/samples/"):
            s = self.server.samples.get(unquote(path[len("/samples/"):]))
            if s is None:
                return self._json({"error": "no such sample"}, 404)
            f = bench.ROOT / s["file"]
            return self._send(200, f.read_bytes(),
                              mimetypes.guess_type(f.name)[0] or "application/octet-stream")
        self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        url = urlparse(self.path)
        if url.path != "/api/run":
            return self._json({"error": "not found"}, 404)
        if not self._authed():
            return
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            providers = [p for p in q.get("providers", "").split(",") if p]
            gate = int(q.get("gate", 500))
            hold = int(q.get("hold", 600))
            if not 50 <= gate <= 3000 or not 0 <= hold <= 5000:
                raise ValueError("gate must be 50–3000 ms and hold 0–5000 ms")
            if not providers or any(x not in bench.PROVIDERS for x in providers):
                raise ValueError(f"providers must be some of {', '.join(bench.PROVIDERS)}")
            if q.get("expected", "").strip():
                bench.normalise_expected(q.get("type", "digits"), q["expected"])
            sample = self.server.samples.get(q["sample"]) if q.get("sample") else None
            if q.get("sample") and sample is None:
                raise ValueError("no such sample")
            if sample:
                pcm = bench.load_audio(bench.ROOT / sample["file"])
                source = "sample"
            else:
                n = int(self.headers.get("Content-Length") or 0)
                if not 0 < n <= MAX_UPLOAD:
                    raise ValueError(f"send a WAV body of at most {MAX_UPLOAD // 2**20} MB")
                raw, rate = bench.read_wav_bytes(self.rfile.read(n))
                pcm = bench.to_16k(raw, rate)
                source = q.get("source", "upload")
                if source not in ("microphone", "upload"):
                    source = "upload"
        except (ValueError, KeyError, EOFError) as exc:
            return self._json({"error": str(exc)}, 400)
        except Exception as exc:  # a WAV that will not parse, mostly
            return self._json({"error": f"could not read the audio: {exc}"}, 400)

        if not self.server.slots.acquire(blocking=False):
            return self._json({"error": "two runs are already streaming — try again in a moment"},
                              429)
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            gone = False

            def emit(msg: dict) -> None:
                nonlocal gone
                if gone:
                    return
                try:
                    self.wfile.write(json.dumps(msg, ensure_ascii=False, default=_plain).encode()
                                     + b"\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    gone = True  # the page closed; the run still finishes and is saved

            try:
                bench.run(np.asarray(pcm), providers, gate=gate, mode=q.get("mode", "verbatim"),
                          expected=q.get("expected", ""),
                          entity_type=q.get("type", "digits"),
                          name=(q.get("name") or (sample or {}).get("title") or source)[:80],
                          source=source, hold_ms=hold,
                          phone_line=q.get("phone") == "1", emit=emit,
                          runs_dir=self.server.runs_dir,
                          sample=sample)
            except ValueError as exc:
                emit({"type": "error", "message": str(exc)})
            except Exception as exc:
                traceback.print_exc()
                emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            self.server.slots.release()


def _plain(o):
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def serve(host: str = "127.0.0.1", port: int = 8765, passcode: str | None = None,
          open_browser: bool = False) -> None:
    local = host in ("127.0.0.1", "localhost", "::1")
    if not local and not passcode:
        raise SystemExit("refusing to listen on a non-local address without --passcode: "
                         "anyone who reaches the port would spend your API credits")
    srv = BenchServer((host, port), passcode=passcode)
    url = f"http://{'localhost' if local else host}:{srv.server_address[1]}/"
    ok = bench.configured()
    print(f"\n  asli test bench  ->  {url}\n")
    for p in bench.PROVIDERS.values():
        state = ("ready" if ok[p.id] else f"no key — add {p.env} to .env") if p.live else "offline"
        print(f"    {p.id:<16} {state}")
    print(f"\n  {len(srv.samples)} samples in the kit · receipts -> {srv.runs_dir}"
          + ("\n  passcode required" if passcode else "") + "\n  Ctrl-C to stop\n")
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
