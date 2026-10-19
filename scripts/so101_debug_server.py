#!/usr/bin/env python3
"""Rerun debug server with a browser Web UI for SO-101 data collection.

Runs a Rerun gRPC sink that buffers every frame logged during a recording
session, plus the Rerun Web Viewer served over HTTP. Any producer that points
at the gRPC port streams into it -- in practice
`scripts/record_so101.sh`, which passes `--display_ip/--display_port` to
`lerobot-record`.

Why not just let LeRobot spawn the native viewer: `lerobot-record --display_data`
calls `rerun.spawn()`, which opens a desktop window. This box is usually driven
headless over SSH, and a browser is easier to reach (just forward the port).
Going through a gRPC server also means the log is BUFFERED, so you can open the
page late, or reconnect, and still scrub back through the whole session.

    # terminal 1 - start the debug server
    python scripts/so101_debug_server.py

    # terminal 2 - record, streaming into it
    bash scripts/record_so101.sh

    # then open http://localhost:9090  (or ssh -L 9090:localhost:9090 <host>)

Port layout:
    9090  HTTP   web viewer (the UI you look at)
    9876  gRPC   log sink that producers send to

Env overrides: RERUN_WEB_PORT, RERUN_GRPC_PORT, RERUN_MEMORY_LIMIT, RERUN_BIND
"""

import argparse
import os
import signal
import socket
import sys
import threading
import time
import webbrowser

try:
    import rerun as rr
except ImportError:
    print("ERROR: rerun-sdk not installed. Install with: pip install rerun-sdk", file=sys.stderr)
    raise SystemExit(2)

# Same session name the recorder uses, so the two halves agree on the stream.
SESSION_NAME = os.getenv("RERUN_SESSION", "so101_record")


def port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def lan_ip() -> str:
    """Best-effort primary IP, so we can print a URL that works off-box."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--web-port", type=int, default=int(os.getenv("RERUN_WEB_PORT", 9090)))
    ap.add_argument("--grpc-port", type=int, default=int(os.getenv("RERUN_GRPC_PORT", 9876)))
    ap.add_argument("--memory-limit", default=os.getenv("RERUN_MEMORY_LIMIT", "2GiB"),
                    help="how much log data the gRPC sink buffers for late viewers")
    ap.add_argument("--open-browser", action="store_true", help="open the viewer in a local browser")
    ap.add_argument("--idle-timeout", type=float, default=0.0,
                    help="exit after N seconds with no producer connected (0 = run forever)")
    args = ap.parse_args()

    for port, what in ((args.web_port, "web viewer"), (args.grpc_port, "gRPC sink")):
        if port_open("127.0.0.1", port):
            print(
                f"ERROR: port {port} ({what}) is already in use.\n"
                f"       Something else is bound there - stop it, or pass a different --{what.split()[0]}-port.",
                file=sys.stderr,
            )
            return 1

    # init() MUST come before serve_grpc(): the gRPC sink needs the application
    # id that init() registers, otherwise it silently returns "[rerun is disabled]".
    rr.init(SESSION_NAME)

    grpc_url = rr.serve_grpc(
        grpc_port=args.grpc_port,
        server_memory_limit=args.memory_limit,
        cors_allow_origin=["*"],  # lets a web viewer on another host connect
    )
    print(f"[debug] gRPC sink:   {grpc_url}   (buffer {args.memory_limit})", flush=True)

    # The web viewer is a separate HTTP server. It is NOT a log sink - it needs
    # the gRPC url above to pull from. Run it in a thread but wait on the real
    # result so a bind failure is reported instead of vanishing into a daemon.
    web_error: list[str] = []
    web_thread = threading.Thread(
        target=_serve_web, args=(grpc_url, args.web_port, web_error), daemon=True
    )
    web_thread.start()
    for _ in range(30):
        if web_error:
            print(f"[debug] ERROR: {web_error[0]}", file=sys.stderr, flush=True)
            return 1
        if port_open("127.0.0.1", args.web_port):
            break
        time.sleep(0.1)
    else:
        print(f"[debug] ERROR: web viewer never came up on port {args.web_port}", file=sys.stderr)
        return 1

    ip = lan_ip()
    # The served HTML is STATIC. The browser learns which gRPC sink to pull from
    # a "?url=" query parameter, so the bare host:port always renders the
    # rerun welcome screen with no recording attached. Both rerun's own
    # WebViewerConfig::host_web_viewer() and WebViewerSink::new() build exactly
    # this URL. Percent-encode everything non-alphanumeric, as the Rust side does
    # with utf8_percent_encode(.., NON_ALPHANUMERIC).
    def viewer_url(sink: str) -> str:
        quoted = "".join(
            ch if ch.isascii() and ch.isalnum() else f"%{ord(ch):02X}" for ch in sink
        )
        return f"http://localhost:{args.web_port}/?url={quoted}"

    local_sink = f"rerun+http://localhost:{args.grpc_port}/proxy"
    lan_sink = f"rerun+http://{ip}:{args.grpc_port}/proxy"

    print("", flush=True)
    print("[debug] OPEN THIS EXACT URL (the ?url= part is required):", flush=True)
    print("", flush=True)
    print(f"   {viewer_url(local_sink)}", flush=True)
    print("", flush=True)
    print("[debug] opened from another machine on the LAN (browser resolves the host):", flush=True)
    print(f"   {viewer_url(lan_sink).replace(f'localhost:{args.web_port}', f'{ip}:{args.web_port}')}", flush=True)
    print("", flush=True)
    print("[debug] over SSH, forward BOTH ports (the browser needs the sink too):", flush=True)
    print(f"   ssh -L {args.web_port}:localhost:{args.web_port} "
          f"-L {args.grpc_port}:localhost:{args.grpc_port} <this-host>", flush=True)
    print("", flush=True)
    print(f"[debug] session:     {SESSION_NAME}", flush=True)
    print(f"[debug] record into it with:  bash scripts/record_so101.sh", flush=True)
    print("", flush=True)

    if args.open_browser:
        try:
            webbrowser.open(viewer_url(local_sink))
        except Exception:  # noqa: BLE001
            pass

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    print("[debug] waiting for producers (Ctrl-C to stop)", flush=True)
    producer_seen = False
    last_activity = time.time()
    while not stop.is_set():
        stop.wait(1.0)
        active = _producer_active(args.grpc_port)
        if active and not producer_seen:
            producer_seen = True
            last_activity = time.time()
            print("[debug] producer connected - streaming", flush=True)
        elif producer_seen and not active:
            producer_seen = False
            last_activity = time.time()
            print("[debug] producer disconnected - log stays buffered, you can still scrub it", flush=True)
        if args.idle_timeout and time.time() - last_activity > args.idle_timeout:
            print(f"[debug] idle for {args.idle_timeout}s, exiting", flush=True)
            break

    print("[debug] stopped", flush=True)
    return 0


def _serve_web(grpc_url: str, web_port: int, error_sink: list[str]) -> None:
    """Run the web viewer HTTP server. Does not return unless it fails."""
    try:
        rr.serve_web_viewer(connect_to=grpc_url, web_port=web_port, open_browser=False)
    except Exception as e:  # noqa: BLE001
        error_sink.append(f"web viewer failed to start on port {web_port}: {e}")


def _producer_active(grpc_port: int) -> bool:
    """True if a producer currently holds an ESTABLISHED connection to the sink.

    Checking the port with a connect() would always succeed, because the sink
    itself is listening on it. What we want is the count of established peers,
    which lives in /proc/net/tcp{,6} as state 01.
    """
    want = f"{grpc_port:04X}"
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(path) as f:
                next(f, None)  # header
                for line in f:
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    local_port = parts[1].rsplit(":", 1)[-1]
                    state = parts[3]
                    if local_port == want and state == "01":  # 01 = ESTABLISHED
                        return True
        except (OSError, StopIteration):
            continue
    return False


if __name__ == "__main__":
    raise SystemExit(main())
