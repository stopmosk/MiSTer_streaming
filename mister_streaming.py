#!/usr/bin/env python3
"""Video streaming server. Run directly on MiSTer.

ASCAL header format: sys/ascal.vhd in https://github.com/MiSTer-devel/Template_MiSTer
"""

import argparse
import ctypes as C
import json
import mmap
import signal
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import NamedTuple
from urllib.parse import parse_qs, urlsplit

BASES = (0x20000000, 0x20800000, 0x21000000)
BUFFER_SIZE = 2048 * 3 * 1024
PAGE_PATH = Path(__file__).with_name("mister_streaming.html")

# ASCAL header byte 5: attribute bits, the top three bits are a frame counter.
ATTR_INTERLACED = 0x01
ATTR_FIELD = 0x02
ATTR_V_DOWNSCALED = 0x08
ATTR_TRIPLE_BUFFER = 0x10

TJPF_RGB = 0
TJSAMP_444 = 0
TJSAMP_420 = 2
TJFLAG_FASTDCT = 2048

# Allowed values (the first one is the default). TODO: Move to a config file
QUERY_OPTIONS = {
    "quality": ("90", "85"),
    "deinterlace": ("both", "fixed"),
    "chroma": ("444", "420"),
    "fps": ("60", "30", "15", "5"),
}


class CaptureError(Exception):
    pass


class Layout(NamedTuple):
    offset: int
    width: int
    height: int
    stride: int
    interlaced: bool


class Frame(NamedTuple):
    pixels: bytes
    width: int
    height: int
    sequence: int
    capture_ms: float
    counter: int
    mode: str
    field: int


class Params(NamedTuple):
    quality: int
    deinterlace: str
    subsampling: int
    fps: int


def header_counter(header):
    return header[5] >> 5


def header_field(header):
    return (header[5] & ATTR_FIELD) >> 1


def parse_params(query):
    values = parse_qs(query, keep_blank_values=True)
    chosen = {}
    for name, allowed in QUERY_OPTIONS.items():
        given = values.get(name, allowed[:1])
        if len(given) != 1 or given[0] not in allowed:
            raise ValueError(f"{name} must be one of: {', '.join(allowed)}.")
        chosen[name] = given[0]
    return Params(
        quality=int(chosen["quality"]),
        deinterlace=chosen["deinterlace"],
        subsampling=TJSAMP_420 if chosen["chroma"] == "420" else TJSAMP_444,
        fps=int(chosen["fps"]),
    )


class Scaler:
    def __init__(self, maps=None):
        self.lock = threading.Lock()
        self.sequence = 0
        self.retries = 0
        self.frames = 0
        self.total_ms = 0.0
        self.last_ms = 0.0
        self.last_shape = None
        self.last_mode = None
        self.started = time.monotonic()
        self.maps = list(maps) if maps is not None else self.map_buffers()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @staticmethod
    def map_buffers():
        maps = []
        try:
            with open("/dev/mem", "rb", buffering=0) as f:
                for address in BASES:
                    maps.append(
                        mmap.mmap(
                            f.fileno(),
                            BUFFER_SIZE,
                            flags=mmap.MAP_SHARED,
                            prot=mmap.PROT_READ,
                            offset=address,
                        )
                    )
        except Exception:
            for region in maps:
                region.close()
            raise
        return maps

    def close(self):
        for region in self.maps:
            region.close()
        self.maps = []

    @staticmethod
    def metadata(header):
        if header[:2] != b"\x01\x01":
            raise CaptureError(
                "No compatible ASCAL buffer. Start a supported core and video mode."
            )
        attrs = header[5]
        offset = struct.unpack_from(">H", header, 2)[0]
        width, height, stride = struct.unpack_from(">3H", header, 6)
        if not (
            16 <= offset <= 4096
            and 1 <= width <= 2048
            and 1 <= height <= 1024
            and width * 3 <= stride <= 2048 * 3
            and offset + stride * height <= BUFFER_SIZE
        ):
            raise CaptureError("Unsupported or changing video mode.")
        interlaced = bool(attrs & ATTR_INTERLACED)
        if interlaced and (height not in (480, 576) or attrs & ATTR_V_DOWNSCALED):
            raise CaptureError(
                "Only full-height 480i/576i ASCAL buffers are supported."
            )
        if not attrs & ATTR_TRIPLE_BUFFER:
            raise CaptureError(
                "Triple buffering is required. This prototype does not change MiSTer settings."
            )
        return Layout(offset, width, height, stride, interlaced)

    @staticmethod
    def select_buffer(headers, interlaced, deinterlace="fixed"):
        counters = [header_counter(h) for h in headers]
        if not interlaced:
            # Headers precede pixels, so take the buffer before the newest one.
            for newest in counters:
                if sorted((newest - c) % 8 for c in counters) == [0, 1, 2]:
                    return counters.index((newest - 1) % 8)
            return None
        odd = [i for i, h in enumerate(headers) if header_field(h)]
        if len(odd) < 2:
            return None
        spacing = list(range(0, 2 * len(odd), 2))
        for i in odd:
            if sorted((counters[i] - counters[j]) % 8 for j in odd) != spacing:
                continue
            if deinterlace == "both":
                even = [j for j in range(len(headers)) if j not in odd]
                if len(even) == 1 and (counters[even[0]] - counters[i]) % 8 == 1:
                    return even[0]
            return i
        return None

    def read(self, deinterlace="fixed", compact=False):
        with self.lock:
            for _ in range(12):
                started = time.monotonic()
                headers = [region[:16] for region in self.maps]
                layouts = [self.metadata(h) for h in headers]
                layout = layouts[0]
                if any(other != layout for other in layouts):
                    self.retries += 1
                    time.sleep(0.002)
                    continue
                index = self.select_buffer(headers, layout.interlaced, deinterlace)
                if index is None:
                    self.retries += 1
                    time.sleep(0.002)
                    continue
                region, header = self.maps[index], headers[index]
                offset, width, height, stride, interlaced = layout
                row_bytes = width * 3
                parity = header_field(header) if interlaced else -1
                if interlaced:
                    rows = [
                        region[offset + y * stride : offset + y * stride + row_bytes]
                        for y in range(parity, height, 2)
                    ]
                    if compact:
                        pixels = b"".join(rows)
                    else:
                        pixels = b"".join(row + row for row in rows)
                        if deinterlace == "both" and parity == 1:
                            # Keep original odd scanlines at odd output positions.
                            pixels = rows[0] + pixels[:-row_bytes]
                else:
                    pixels = region[offset : offset + stride * height]
                elapsed = time.monotonic() - started
                # Reject reuse during the copy
                if region[:16] != header or elapsed > 0.050:
                    self.retries += 1
                    continue
                if not interlaced and stride != row_bytes:
                    pixels = b"".join(
                        pixels[y * stride : y * stride + row_bytes]
                        for y in range(height)
                    )
                if not interlaced:
                    mode = "progressive"
                elif deinterlace == "both":
                    mode = "bob-both-fields"
                else:
                    mode = "bob-fixed-field"
                self.sequence += 1
                self.frames += 1
                self.last_ms = (time.monotonic() - started) * 1000
                self.total_ms += self.last_ms
                self.last_shape = [width, height]
                self.last_mode = mode
                return Frame(
                    pixels=pixels,
                    width=width,
                    height=height,
                    sequence=self.sequence,
                    capture_ms=self.last_ms,
                    counter=header_counter(header),
                    mode=mode,
                    field=parity,
                )
            raise CaptureError("Could not copy a stable frame, retrying is safe.")

    def stats(self):
        with self.lock:
            return {
                "frames": self.frames,
                "retries": self.retries,
                "capture_ms": round(self.last_ms, 3),
                "mean_capture_ms": round(self.total_ms / max(1, self.frames), 3),
                "resolution": self.last_shape,
                "video_mode": self.last_mode,
                "uptime_s": round(time.monotonic() - self.started, 1),
            }


class JpegEncoder:
    def __init__(self):
        self.lock = threading.Lock()
        self.library = C.CDLL("libturbojpeg.so.0")
        j = self.library
        j.tjInitCompress.restype = C.c_void_p
        j.tjCompress2.argtypes = [
            C.c_void_p,
            C.c_char_p,
            C.c_int,
            C.c_int,
            C.c_int,
            C.c_int,
            C.POINTER(C.c_void_p),
            C.POINTER(C.c_ulong),
            C.c_int,
            C.c_int,
            C.c_int,
        ]
        j.tjCompress2.restype = C.c_int
        j.tjFree.argtypes = [C.c_void_p]
        j.tjFree.restype = None
        j.tjDestroy.argtypes = [C.c_void_p]
        j.tjGetErrorStr.restype = C.c_char_p
        self.handle = j.tjInitCompress()
        if not self.handle:
            raise CaptureError("Could not initialize the JPEG encoder.")
        self.frames = 0
        self.total_ms = 0.0
        self.last_ms = 0.0
        self.last_bytes = 0

    def encode(self, rgb, width, height, quality, subsampling=TJSAMP_444):
        if len(rgb) != width * height * 3:
            raise CaptureError("Invalid RGB frame size.")
        with self.lock:
            out, size = C.c_void_p(), C.c_ulong()
            started = time.monotonic()
            try:
                result = self.library.tjCompress2(
                    self.handle,
                    rgb,
                    width,
                    width * 3,
                    height,
                    TJPF_RGB,
                    C.byref(out),
                    C.byref(size),
                    subsampling,
                    quality,
                    TJFLAG_FASTDCT,
                )
                if result != 0:
                    raise CaptureError(
                        self.library.tjGetErrorStr().decode(errors="replace")
                    )
                data = C.string_at(out, size.value)
            finally:
                if out.value:
                    self.library.tjFree(out)
            self.last_ms = (time.monotonic() - started) * 1000
            self.frames += 1
            self.total_ms += self.last_ms
            self.last_bytes = len(data)
            return data, self.last_ms

    def stats(self):
        with self.lock:
            return {
                "frames": self.frames,
                "encode_ms": round(self.last_ms, 3),
                "mean_encode_ms": round(self.total_ms / max(1, self.frames), 3),
                "frame_bytes": self.last_bytes,
            }


def create_server(host, port, scaler, jpeg, page):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.connection.settimeout(15)

        def log_message(self, *args):
            pass

        def frame(self, compressed, params):
            if compressed and jpeg is None:
                raise CaptureError(
                    "libturbojpeg is unavailable. Select uncompressed mode."
                )
            frame = scaler.read(params.deinterlace, compact=compressed)
            height = frame.height
            if compressed:
                if frame.mode != "progressive":
                    height //= 2
                data, encode_ms = jpeg.encode(
                    frame.pixels,
                    frame.width,
                    height,
                    params.quality,
                    params.subsampling,
                )
                mime = "image/jpeg"
            else:
                data, encode_ms, mime = frame.pixels, 0, "application/octet-stream"
            headers = {
                "Content-Type": mime,
                "Content-Length": str(len(data)),
                "X-Width": str(frame.width),
                "X-Height": str(height),
                "X-Sequence": str(frame.sequence),
                "X-Capture-Ms": f"{frame.capture_ms:.3f}",
                "X-Frame-Counter": str(frame.counter),
                "X-Video-Mode": frame.mode,
                "X-Field": str(frame.field),
                "X-Display-Height": str(frame.height),
                "X-Encode-Ms": f"{encode_ms:.3f}",
                "X-Raw-Bytes": str(len(frame.pixels)),
            }
            return data, headers

        def stream(self, params):
            self.send_response(200)
            self.send_header("Content-Type", "application/x-mister-jpeg-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            while True:
                started = time.monotonic()
                try:
                    data, headers = self.frame(True, params)
                except CaptureError as exc:
                    data, headers = b"", {"error": str(exc), "Content-Length": "0"}
                # 4-byte big-endian header length, JSON headers, JPEG bytes.
                header = json.dumps(headers, separators=(",", ":")).encode()
                self.wfile.write(struct.pack(">I", len(header)) + header + data)
                time.sleep(max(0.001, 1 / params.fps - (time.monotonic() - started)))

        def reply(self, status, mime, body, headers=()):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, value in headers:
                self.send_header(key, str(value))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlsplit(self.path)
            try:
                if url.path == "/":
                    self.reply(200, "text/html; charset=utf-8", page)
                elif url.path in ("/frame.bin", "/frame.jpg", "/stream"):
                    try:
                        params = parse_params(url.query)
                    except ValueError as exc:
                        self.reply(400, "text/plain; charset=utf-8", str(exc).encode())
                        return
                    if url.path == "/stream":
                        self.stream(params)
                        return
                    data, headers = self.frame(url.path == "/frame.jpg", params)
                    extra = [(k, v) for k, v in headers.items() if k.startswith("X-")]
                    self.reply(200, headers["Content-Type"], data, extra)
                elif url.path == "/stats":
                    stats = scaler.stats()
                    stats["jpeg"] = jpeg.stats() if jpeg else None
                    self.reply(200, "application/json", json.dumps(stats).encode())
                elif url.path == "/favicon.ico":
                    self.reply(204, "image/x-icon", b"")
                else:
                    self.reply(404, "text/plain", b"Not found")
            except CaptureError as exc:
                self.reply(503, "text/plain; charset=utf-8", str(exc).encode())
            except OSError:
                pass  # The client went away or timed out.

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def serve(host, port):
    page = PAGE_PATH.read_bytes()
    with Scaler() as scaler:
        try:
            first = scaler.read()
        except CaptureError as exc:
            first = None
            print(f"Waiting for a compatible frame: {exc}", flush=True)
        jpeg = None
        try:
            jpeg = JpegEncoder()
        except (OSError, AttributeError, CaptureError) as exc:
            print(f"JPEG unavailable; raw mode remains available: {exc}", flush=True)

        server = create_server(host, port, scaler, jpeg, page)
        if first:
            print(
                f"READY {first.width}x{first.height} "
                f"capture={first.capture_ms:.2f}ms at {host}:{port}",
                flush=True,
            )
        else:
            print(f"READY waiting for video at {host}:{port}", flush=True)

        def stop_signal(*_):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, stop_signal)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--benchmark", type=int, default=0)
    args = parser.parse_args()
    if args.benchmark:
        with Scaler() as scaler:
            for _ in range(args.benchmark):
                scaler.read()
                time.sleep(1 / 60)
            print(json.dumps(scaler.stats()), flush=True)
    else:
        serve(args.host, args.port)
