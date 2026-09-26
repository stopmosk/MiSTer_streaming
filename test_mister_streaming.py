import json
import struct
import threading
import unittest
import urllib.error
import urllib.request

import mister_streaming as mp
from mister_streaming import CaptureError, Frame, Layout, Params, Scaler

INTERLACED_HEIGHT = 480


def make_header(
    counter=0,
    *,
    offset=16,
    width=4,
    height=2,
    stride=None,
    interlaced=False,
    field=0,
    triple=True,
    v_downscaled=False,
):
    attrs = counter << 5
    if interlaced:
        attrs |= mp.ATTR_INTERLACED
    if field:
        attrs |= mp.ATTR_FIELD
    if v_downscaled:
        attrs |= mp.ATTR_V_DOWNSCALED
    if triple:
        attrs |= mp.ATTR_TRIPLE_BUFFER
    stride = width * 3 if stride is None else stride
    fields = struct.pack(">HBB3H", offset, 0, attrs, width, height, stride)
    return b"\x01\x01" + fields + bytes(4)


def row(y, width, tag=0):
    return bytes([y & 0xFF, y >> 8, tag]) * width


def image(width, height, stride, tag=0):
    padding = bytes(stride - width * 3)
    return b"".join(row(y, width, tag) + padding for y in range(height))


def split_rows(pixels, width):
    size = width * 3
    return [pixels[i : i + size] for i in range(0, len(pixels), size)]


class FakeRegion:
    def __init__(self, header, pixels=b""):
        self.data = bytearray(header + pixels)
        self.closed = False

    def __getitem__(self, key):
        return bytes(self.data[key])

    def close(self):
        self.closed = True


class RewrittenDuringCopy(FakeRegion):
    """Changes its header once, right after the first pixel read."""

    rewritten = False

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if key.start and not self.rewritten:
            self.rewritten = True
            self.data[4] ^= 0xFF
        return value


class MetadataTest(unittest.TestCase):
    def test_valid_progressive_header(self):
        header = make_header(width=320, height=240, stride=1024)
        self.assertEqual(Scaler.metadata(header), Layout(16, 320, 240, 1024, False))

    def test_valid_interlaced_header(self):
        header = make_header(width=640, height=480, interlaced=True)
        self.assertTrue(Scaler.metadata(header).interlaced)

    def test_rejects_invalid_headers(self):
        cases = {
            "bad magic": b"\x00" + make_header()[1:],
            "zero width": make_header(width=0),
            "stride below width": make_header(width=4, stride=11),
            "offset too small": make_header(offset=8),
            "exceeds buffer": make_header(width=2048, height=1024, offset=4096),
            "interlaced odd height": make_header(height=240, interlaced=True),
            "interlaced downscaled": make_header(
                height=480, interlaced=True, v_downscaled=True
            ),
            "no triple buffering": make_header(triple=False),
        }
        for name, header in cases.items():
            with self.subTest(name), self.assertRaises(CaptureError):
                Scaler.metadata(header)


class SelectProgressiveTest(unittest.TestCase):
    def select(self, *counters):
        headers = [make_header(c) for c in counters]
        return Scaler.select_buffer(headers, interlaced=False)

    def test_picks_buffer_before_newest(self):
        self.assertEqual(self.select(3, 4, 5), 1)
        self.assertEqual(self.select(5, 3, 4), 2)

    def test_counter_wraps_modulo_8(self):
        self.assertEqual(self.select(7, 0, 1), 1)
        self.assertEqual(self.select(0, 6, 7), 2)

    def test_rejects_non_consecutive_counters(self):
        self.assertIsNone(self.select(0, 1, 3))
        self.assertIsNone(self.select(2, 2, 2))


class SelectInterlacedTest(unittest.TestCase):
    def select(self, buffers, deinterlace):
        headers = [
            make_header(c, height=INTERLACED_HEIGHT, interlaced=True, field=f)
            for c, f in buffers
        ]
        return Scaler.select_buffer(headers, True, deinterlace)

    def test_fixed_picks_newest_odd_field(self):
        self.assertEqual(self.select([(2, 1), (5, 0), (4, 1)], "fixed"), 2)

    def test_both_picks_even_field_right_after_odd(self):
        self.assertEqual(self.select([(2, 1), (5, 0), (4, 1)], "both"), 1)
        self.assertEqual(self.select([(6, 1), (1, 0), (0, 1)], "both"), 1)

    def test_both_falls_back_to_odd_field(self):
        self.assertEqual(self.select([(2, 1), (3, 0), (4, 1)], "both"), 2)

    def test_rejects_unstable_layouts(self):
        self.assertIsNone(self.select([(3, 1), (5, 0), (4, 1)], "fixed"))
        self.assertIsNone(self.select([(3, 0), (5, 0), (4, 1)], "fixed"))


class ReadTest(unittest.TestCase):
    def test_progressive_copies_previous_buffer_without_padding(self):
        width, height, stride = 2, 3, 8
        maps = [
            FakeRegion(
                make_header(c, width=width, height=height, stride=stride),
                image(width, height, stride, tag=c),
            )
            for c in (3, 4, 5)
        ]
        with Scaler(maps) as scaler:
            frame = scaler.read()
            self.assertEqual(scaler.stats()["frames"], 1)
        self.assertTrue(all(region.closed for region in maps))
        expected = b"".join(row(y, width, tag=4) for y in range(height))
        self.assertEqual(frame.pixels, expected)
        self.assertEqual((frame.width, frame.height, frame.sequence), (2, 3, 1))
        self.assertEqual(
            (frame.counter, frame.mode, frame.field), (4, "progressive", -1)
        )

    def interlaced_scaler(self, buffers):
        width, height = 1, INTERLACED_HEIGHT
        return Scaler(
            [
                FakeRegion(
                    make_header(
                        c, width=width, height=height, interlaced=True, field=f
                    ),
                    image(width, height, width * 3),
                )
                for c, f in buffers
            ]
        )

    def test_fixed_field_doubles_odd_lines(self):
        frame = self.interlaced_scaler([(2, 1), (5, 0), (4, 1)]).read("fixed")
        rows = split_rows(frame.pixels, 1)
        self.assertEqual((frame.mode, frame.field), ("bob-fixed-field", 1))
        self.assertEqual(len(rows), INTERLACED_HEIGHT)
        self.assertEqual(rows[:4], [row(1, 1), row(1, 1), row(3, 1), row(3, 1)])

    def test_both_fields_keeps_odd_lines_in_place(self):
        frame = self.interlaced_scaler([(2, 1), (3, 0), (4, 1)]).read("both")
        rows = split_rows(frame.pixels, 1)
        self.assertEqual((frame.mode, frame.field), ("bob-both-fields", 1))
        self.assertEqual(len(rows), INTERLACED_HEIGHT)
        for y in range(1, INTERLACED_HEIGHT, 2):
            self.assertEqual(rows[y], row(y, 1))

    def test_both_fields_reads_even_lines(self):
        frame = self.interlaced_scaler([(2, 1), (5, 0), (4, 1)]).read("both")
        rows = split_rows(frame.pixels, 1)
        self.assertEqual((frame.field, frame.counter), (0, 5))
        self.assertEqual(rows[:4], [row(0, 1), row(0, 1), row(2, 1), row(2, 1)])

    def test_compact_returns_single_field(self):
        scaler = self.interlaced_scaler([(2, 1), (5, 0), (4, 1)])
        frame = scaler.read("fixed", compact=True)
        rows = split_rows(frame.pixels, 1)
        self.assertEqual(frame.height, INTERLACED_HEIGHT)
        self.assertEqual(rows, [row(y, 1) for y in range(1, INTERLACED_HEIGHT, 2)])

    def test_retries_when_header_changes_during_copy(self):
        maps = [FakeRegion(make_header(c)) for c in (3, 5)]
        maps.insert(1, RewrittenDuringCopy(make_header(4), image(4, 2, 12)))
        scaler = Scaler(maps)
        frame = scaler.read()
        self.assertEqual(frame.counter, 4)
        self.assertEqual(scaler.retries, 1)

    def test_gives_up_when_layouts_disagree(self):
        maps = [
            FakeRegion(make_header(c, width=w)) for c, w in ((3, 4), (4, 4), (5, 2))
        ]
        scaler = Scaler(maps)
        with self.assertRaises(CaptureError):
            scaler.read()
        self.assertEqual(scaler.retries, 12)


class ParseParamsTest(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(mp.parse_params(""), Params(90, "both", mp.TJSAMP_444, 60))

    def test_explicit_values(self):
        params = mp.parse_params("quality=85&deinterlace=fixed&chroma=420&fps=15")
        self.assertEqual(params, Params(85, "fixed", mp.TJSAMP_420, 15))

    def test_rejects_invalid_values(self):
        for query in ("fps=7", "quality=", "chroma=422", "fps=5&fps=60"):
            with self.subTest(query), self.assertRaises(ValueError):
                mp.parse_params(query)


class FakeScaler:
    def __init__(self, mode="progressive"):
        self.frame = Frame(b"\x01\x02\x03" * 8, 4, 2, 7, 1.25, 3, mode, -1)
        self.calls = []

    def read(self, deinterlace="fixed", compact=False):
        self.calls.append((deinterlace, compact))
        return self.frame

    def stats(self):
        return {"frames": 1}


class FakeJpeg:
    def __init__(self):
        self.calls = []

    def encode(self, rgb, width, height, quality, subsampling=mp.TJSAMP_444):
        self.calls.append((width, height, quality, subsampling))
        return b"JPEGDATA", 0.5

    def stats(self):
        return {"frames": 2}


class ServerTest(unittest.TestCase):
    def start(self, scaler=None, jpeg=None):
        self.scaler = scaler or FakeScaler()
        self.jpeg = jpeg
        server = mp.create_server("127.0.0.1", 0, self.scaler, jpeg, b"<html>")
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.base = f"http://127.0.0.1:{server.server_address[1]}"

    def get(self, path):
        try:
            with urllib.request.urlopen(self.base + path, timeout=5) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.headers, error.read()

    def test_page_and_stats_ignore_query_string(self):
        self.start(jpeg=FakeJpeg())
        for path in ("/", "/?v=2"):
            self.assertEqual(self.get(path)[::2], (200, b"<html>"))
        status, _, body = self.get("/stats?x=1")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"frames": 1, "jpeg": {"frames": 2}})
        self.assertEqual(self.get("/missing")[0], 404)

    def test_raw_frame(self):
        self.start()
        status, headers, body = self.get("/frame.bin")
        self.assertEqual((status, body), (200, self.scaler.frame.pixels))
        self.assertEqual(headers["X-Width"], "4")
        self.assertEqual(headers["X-Capture-Ms"], "1.250")
        self.assertEqual(headers["X-Video-Mode"], "progressive")
        self.assertEqual(self.scaler.calls, [("both", False)])

    def test_jpeg_frame_halves_bob_height(self):
        self.start(FakeScaler("bob-fixed-field"), FakeJpeg())
        path = "/frame.jpg?quality=85&chroma=420&deinterlace=fixed"
        status, headers, body = self.get(path)
        self.assertEqual((status, body), (200, b"JPEGDATA"))
        self.assertEqual((headers["X-Height"], headers["X-Display-Height"]), ("1", "2"))
        self.assertEqual(self.scaler.calls, [("fixed", True)])
        self.assertEqual(self.jpeg.calls, [(4, 1, 85, mp.TJSAMP_420)])

    def test_invalid_parameters(self):
        self.start()
        status, _, body = self.get("/frame.bin?fps=7")
        self.assertEqual(status, 400)
        self.assertIn(b"fps", body)

    def test_jpeg_unavailable(self):
        self.start()
        self.assertEqual(self.get("/frame.jpg")[0], 503)

    def test_stream_packet(self):
        self.start(jpeg=FakeJpeg())
        with urllib.request.urlopen(self.base + "/stream?fps=5", timeout=5) as response:
            (length,) = struct.unpack(">I", response.read(4))
            headers = json.loads(response.read(length))
            data = response.read(int(headers["Content-Length"]))
        self.assertEqual(data, b"JPEGDATA")
        self.assertEqual(headers["X-Frame-Counter"], "3")


class PageTest(unittest.TestCase):
    def test_page_file_ships_next_to_script(self):
        self.assertTrue(mp.PAGE_PATH.is_file())


if __name__ == "__main__":
    unittest.main()
