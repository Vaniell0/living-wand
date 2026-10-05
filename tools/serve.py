"""living-wand engine API — cast spells from any game engine.

HTTP (Godot HTTPRequest, Unity UnityWebRequest, Unreal HTTP module, curl):

    python tools/serve.py --port 7777

    GET  /health                 model / OCR status
    GET  /schema                 JSON Schema of SpellProgram
    GET  /runes                  rune alphabet with kinds
    GET  /verbs                  engine verb vocabulary
    POST /cast/graph             {"graph": {"nodes": [...], "edges": [...]}}
    POST /cast/strokes           {"strokes": [[[x, y], ...], ...], "circle_closed": true,
                                  "overrides": {"0": "fire"}}
    POST /outcome                {"program_id": "...", "hit": true, "totals": {"damage": 12}}
    POST /sleep                  train the wand on accumulated outcomes
    POST /save

stdio JSON-lines (engine spawns this as a subprocess, one request per line):

    python tools/serve.py --stdio
    > {"op": "cast/graph", "graph": {...}}
    < {"ok": true, "program": {...}}

Example:
    curl -s localhost:7777/cast/graph -d '{"graph": {"nodes": [{"name": "fire"},
      {"name": "outward"}, {"name": "target"}], "edges": [["fire", "outward",
      "DIRECT"], ["outward", "target", "DIRECT"]]}}'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wand.actions import SCHEMA_VERSION, Outcome, json_schema, load_verbs  # noqa: E402
from wand.algebra import RUNES, EdgeType, graph_from_spec  # noqa: E402
from wand.session import WandSession  # noqa: E402


class Api:
    """Transport-independent request handling (shared by HTTP and stdio)."""

    def __init__(self, session: WandSession):
        self.session = session
        self.lock = threading.Lock()

    def handle(self, op: str, payload: dict) -> dict:
        op = op.strip("/")
        with self.lock:
            if op == "health":
                s = self.session
                return {"ok": True, "schema": SCHEMA_VERSION, "model_trained": s.model_trained,
                        "authority": s.authority, "ocr": s.ocr is not None}
            if op == "schema":
                return json_schema()
            if op == "runes":
                return {"runes": [{"name": n, "kind": k.name, "element": e.name if e else None}
                                  for n, (k, e) in RUNES.items()],
                        "edge_types": [e.name for e in EdgeType]}
            if op == "verbs":
                v = load_verbs()
                return {"verbs": v["verbs"],
                        "patterns": [{k: p[k] for k in p if k != "_rx"} for p in v["patterns"]]}
            if op == "cast/graph":
                graph = graph_from_spec(payload["graph"])
                report = self.session.cast(graph, clarity=float(payload.get("clarity", 1.0)))
                return {"ok": True, "program": report.program.to_dict(),
                        "check": {"verdict": report.check.verdict.name,
                                  "issues": report.check.issues},
                        "discoveries": [d.key for d in report.discoveries],
                        "crystal_progress": report.crystal_progress}
            if op == "cast/strokes":
                overrides = {int(k): v for k, v in payload.get("overrides", {}).items()}
                sketch, report = self.session.cast_strokes(
                    payload["strokes"], bool(payload.get("circle_closed", True)), overrides)
                return {
                    "ok": report is not None,
                    "glyphs": [{"index": i, "label": g.label, "confidence": round(g.confidence, 4),
                                "alternatives": g.alternatives, "uncertain": g.uncertain,
                                "bbox": g.bbox, "strokes": g.stroke_ids}
                               for i, g in enumerate(sketch.glyphs)],
                    "edges": [{"src": e.src, "dst": e.dst, "type": e.edge_type.name,
                               "inferred": e.inferred} for e in sketch.edges],
                    "clarity": sketch.clarity,
                    "program": report.program.to_dict() if report else None,
                }
            if op == "outcome":
                event = self.session.report_outcome(Outcome.from_dict(payload))
                if event is None:
                    return {"ok": False, "error": "unknown program_id"}
                return {"ok": True, "event": {"type": event.event_type.name,
                                              "poles": event.pole_affinity,
                                              "weight": event.weight}}
            if op == "sleep":
                return {"ok": True, "trained": self.session.sleep()}
            if op == "save":
                self.session.save()
                return {"ok": True}
        return {"ok": False, "error": f"unknown op {op!r}"}


def make_handler(api: Api):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict):
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._dispatch({})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError as e:
                return self._send(400, {"ok": False, "error": f"bad json: {e}"})
            self._dispatch(payload)

        def _dispatch(self, payload):
            try:
                body = api.handle(self.path, payload)
                self._send(200 if body.get("ok", True) else 400, body)
            except (KeyError, ValueError, TypeError) as e:
                self._send(400, {"ok": False, "error": f"{type(e).__name__}: {e}"})
            except Exception as e:  # never drop the engine's connection silently
                self._send(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):
            pass

    return Handler


def serve_stdio(api: Api):
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = api.handle(req.pop("op"), req)
        except Exception as e:  # one bad request must not kill the engine link
            resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7777)
    ap.add_argument("--stdio", action="store_true")
    ap.add_argument("--authority", choices=["algebra", "blend", "model"], default="blend")
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()

    session = WandSession(authority=args.authority, save_dir=args.save_dir)
    api = Api(session)
    try:
        if args.stdio:
            serve_stdio(api)
        else:
            server = ThreadingHTTPServer((args.host, args.port), make_handler(api))
            print(f"living-wand API on http://{args.host}:{args.port}  "
                  f"(model={'trained' if session.model_trained else 'algebra-only'}, "
                  f"ocr={'yes' if session.ocr else 'no'})")
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        session.save()
        session.close()


if __name__ == "__main__":
    main()
