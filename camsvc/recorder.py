"""Main stream: buffer packets in RAM and write clips without decoding.

The reader thread reads encoded packets (video + audio) and keeps the last `pre_seconds`
starting at a keyframe in a ring buffer. On an event the same thread writes the buffer and all
new packets to an MP4 (stream copy). All muxer actions happen in this one thread.

Audio that does not fit in an MP4 (e.g. G.711 from a Foscam doorbell) is transcoded to AAC only
while writing a clip. At 8 kHz mono that costs next to nothing.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable

import av

log = logging.getLogger(__name__)

# Audio codecs that fit in an MP4 without transcoding.
MP4_AUDIO_OK = {"aac", "mp3", "opus", "alac", "flac", "ac3"}
AAC_FRAME = 1024  # samples per AAC frame

RTSP_OPTIONS = {
    "rtsp_transport": "tcp",
    "fflags": "+genpts",
    "max_delay": "500000",
}


@dataclass(slots=True)
class BufferedPacket:
    seq: int
    arrival: float          # time.monotonic() on arrival
    stream_index: int
    data: bytes
    pts: int | None
    dts: int | None
    is_keyframe: bool


class PacketRing:
    """Ring buffer of encoded packets; always starts at a video keyframe."""

    def __init__(self, pre_seconds: float):
        self.pre_seconds = pre_seconds
        self.packets: deque[BufferedPacket] = deque()
        self._keyframes: deque[tuple[float, int]] = deque()  # (arrival, seq)
        self._seq = 0

    def clear(self) -> None:
        self.packets.clear()
        self._keyframes.clear()

    def add(self, stream_index: int, data: bytes, pts: int | None, dts: int | None,
            is_video_key: bool, now: float) -> BufferedPacket | None:
        if not self._keyframes and not is_video_key:
            return None  # no starting point yet: keep nothing
        self._seq += 1
        bp = BufferedPacket(self._seq, now, stream_index, data, pts, dts, is_video_key)
        self.packets.append(bp)
        if is_video_key:
            self._keyframes.append((now, self._seq))
        # Drop the oldest keyframe as soon as the next one is old enough too.
        cutoff = now - self.pre_seconds
        while len(self._keyframes) >= 2 and self._keyframes[1][0] <= cutoff:
            self._keyframes.popleft()
        first_seq = self._keyframes[0][1]
        while self.packets and self.packets[0].seq < first_seq:
            self.packets.popleft()
        return bp

    def snapshot(self) -> list[BufferedPacket]:
        return list(self.packets)

    @property
    def seconds(self) -> float:
        if not self.packets:
            return 0.0
        return self.packets[-1].arrival - self.packets[0].arrival

    @property
    def nbytes(self) -> int:
        return sum(len(p.data) for p in self.packets)


class AudioTranscoder:
    """Decodes audio packets and writes them as AAC (mono) into the clip.

    Timestamps continue from the first packet, in samples; for clips of at most a few minutes
    the drift relative to the video is negligible.
    """

    def __init__(self, out, in_stream: av.stream.Stream):
        self.dec = in_stream.codec_context
        self.rate = int(self.dec.sample_rate or 8000)
        self.stream = out.add_stream("aac", rate=self.rate, layout="mono")
        self.stream.bit_rate = 32000
        self.resampler = av.AudioResampler(format="fltp", layout="mono", rate=self.rate)
        self.fifo = av.AudioFifo()
        self.next_pts: int | None = None

    def write(self, data: bytes, start_seconds: float, mux: Callable) -> None:
        if self.next_pts is None:
            self.next_pts = max(0, int(round(start_seconds * self.rate)))
        for frame in self.dec.decode(av.Packet(data)):
            frame.pts = None
            for rf in self.resampler.resample(frame):
                rf.pts = None
                self.fifo.write(rf)
        self._encode(mux, final=False)

    def _encode(self, mux: Callable, final: bool) -> None:
        while self.fifo.samples >= AAC_FRAME or (final and self.fifo.samples):
            frame = self.fifo.read(min(AAC_FRAME, self.fifo.samples))
            frame.pts = self.next_pts
            frame.time_base = Fraction(1, self.rate)
            self.next_pts += frame.samples
            for pkt in self.stream.encode(frame):
                mux(pkt)

    def flush(self, mux: Callable) -> None:
        if self.next_pts is None:
            return  # no audio received; encoder never opened
        self._encode(mux, final=True)
        for pkt in self.stream.encode(None):
            mux(pkt)


class ClipWriter:
    """Writes buffered packets to MP4 with timestamps starting at 0."""

    def __init__(self, path: Path, in_streams: dict[int, av.stream.Stream], video_index: int):
        self.path = path
        self.video_index = video_index
        self.out = av.open(str(path), mode="w", format="mp4",
                           # fragmented: every keyframe starts a fragment, so a clip stays
                           # playable up to just before a crash or power cut
                           options={"movflags": "+frag_keyframe+empty_moov+default_base_moof"})
        self.map: dict[int, av.stream.Stream] = {}
        self.tb: dict[int, object] = {}
        self.transcode: dict[int, AudioTranscoder] = {}
        for idx, st in in_streams.items():
            if st.type == "audio" and st.codec_context.name not in MP4_AUDIO_OK:
                self.transcode[idx] = AudioTranscoder(self.out, st)
            else:
                self.map[idx] = self.out.add_stream_from_template(st)
            self.tb[idx] = st.time_base
        self.base_seconds: float | None = None
        self.last_dts: dict[int, int] = {}
        self.packets_written = 0
        self.first_arrival: float | None = None
        self.last_arrival: float | None = None

    def write(self, bp: BufferedPacket) -> None:
        if bp.stream_index not in self.tb:
            return
        tb = self.tb[bp.stream_index]
        pts = bp.pts if bp.pts is not None else bp.dts
        dts = bp.dts if bp.dts is not None else pts
        if pts is None:
            return
        if self.base_seconds is None:
            if not (bp.stream_index == self.video_index and bp.is_keyframe):
                return  # the first packet must be a video keyframe
            self.base_seconds = float(dts * tb)
        offset = int(round(self.base_seconds / tb))
        pts -= offset
        dts -= offset
        if dts < 0 or pts < 0:
            return  # audio from before the first keyframe
        transcoder = self.transcode.get(bp.stream_index)
        if transcoder is not None:
            try:
                transcoder.write(bp.data, float(pts * tb), self.out.mux)
            except av.FFmpegError as exc:
                log.warning("Audio transcoding failed in %s (%s); rest without audio", self.path, exc)
                del self.transcode[bp.stream_index]
                del self.tb[bp.stream_index]
            return
        prev = self.last_dts.get(bp.stream_index)
        if prev is not None and dts <= prev:
            dts = prev + 1
            pts = max(pts, dts)
        self.last_dts[bp.stream_index] = dts

        pkt = av.Packet(bp.data)
        pkt.pts = pts
        pkt.dts = dts
        pkt.time_base = tb
        pkt.is_keyframe = bp.is_keyframe
        pkt.stream = self.map[bp.stream_index]
        self.out.mux(pkt)
        self.packets_written += 1
        if self.first_arrival is None:
            self.first_arrival = bp.arrival
        self.last_arrival = bp.arrival

    def close(self) -> float:
        """Closes the file; returns the duration in seconds."""
        try:
            for transcoder in self.transcode.values():
                transcoder.flush(self.out.mux)
            self.out.close()
        except Exception:  # noqa: BLE001 - a broken clip must not stop the service
            log.exception("Clip %s could not be closed cleanly", self.path)
        if self.first_arrival is None or self.last_arrival is None:
            return 0.0
        return self.last_arrival - self.first_arrival


ClipDone = Callable[[str, Path, float], None]  # (event_id, part_path, duration)


class MainRecorder(threading.Thread):
    """Reads the main stream, buffers and writes clips on request."""

    def __init__(self, name: str, url_getter: Callable[[], str], pre_seconds: float,
                 record_audio: bool, part_dir: Path, on_clip_done: ClipDone,
                 on_online: Callable[[bool], None] | None = None):
        super().__init__(name=f"main-{name}", daemon=True)
        self.cam_name = name
        self.url_getter = url_getter
        self.ring = PacketRing(pre_seconds)
        self.record_audio = record_audio
        self.part_dir = part_dir
        self.on_clip_done = on_clip_done
        self.on_online = on_online or (lambda _ok: None)
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._want_event: str | None = None   # event id that should be recording
        self._writer: ClipWriter | None = None
        self._writer_event: str | None = None
        self.online = False
        self.codec = ""
        self.bitrate_kbps = 0.0
        self._video_index: int | None = None
        self._extradata: bytes | None = None

    # --- called from other threads --------------------------------------
    def start_event(self, event_id: str) -> None:
        with self._lock:
            self._want_event = event_id

    def stop_event(self) -> None:
        with self._lock:
            self._want_event = None

    def stop(self) -> None:
        self._stop_evt.set()

    # --- reader-thread ---------------------------------------------------
    def run(self) -> None:
        backoff = 2.0
        while not self._stop_evt.is_set():
            try:
                self._read_loop()
                backoff = 2.0
            except Exception as exc:  # noqa: BLE001
                log.warning("[%s] main stream error: %s", self.cam_name, exc)
            self._set_online(False)
            self._finish_writer()
            self.ring.clear()
            if self._stop_evt.wait(backoff):
                break
            backoff = min(backoff * 2, 60.0)

    def grab_frame(self):
        """Current main-stream frame as a BGR array, or None.

        Decodes the video from the last keyframe in the buffer up to now on request (in the
        caller's thread, with its own decoder). Costs a fraction of a second of CPU at 4K, so
        only use it for single frames such as face recognition.
        """
        if not self.online or self._video_index is None:
            return None
        for _ in range(3):   # the reader thread may just be changing the buffer
            try:
                packets = list(self.ring.packets)
                break
            except RuntimeError:
                continue
        else:
            return None
        video = [p for p in packets if p.stream_index == self._video_index]
        last_key = max((i for i, p in enumerate(video) if p.is_keyframe), default=None)
        if last_key is None:
            return None
        ctx = av.CodecContext.create(self.codec, "r")
        if self._extradata:
            ctx.extradata = self._extradata
        ctx.thread_type = "AUTO"
        frame = None
        try:
            for p in video[last_key:]:
                for f in ctx.decode(av.Packet(p.data)):
                    frame = f
            for f in ctx.decode(None):
                frame = f
        except av.FFmpegError as exc:
            log.debug("[%s] decoding main frame failed: %s", self.cam_name, exc)
        return frame.to_ndarray(format="bgr24") if frame is not None else None

    def _set_online(self, ok: bool) -> None:
        if ok != self.online:
            self.online = ok
            log.info("[%s] main stream %s", self.cam_name, "connected" if ok else "offline")
            self.on_online(ok)

    def _read_loop(self) -> None:
        url = self.url_getter()
        with av.open(url, options=RTSP_OPTIONS, timeout=(10.0, 10.0)) as container:
            video = container.streams.video[0]
            streams: dict[int, av.stream.Stream] = {video.index: video}
            if self.record_audio and container.streams.audio:
                audio = container.streams.audio[0]
                streams[audio.index] = audio
                if audio.codec_context.name not in MP4_AUDIO_OK:
                    log.info("[%s] audio codec %s does not fit in MP4; transcoded to AAC in clips",
                             self.cam_name, audio.codec_context.name)
            self.codec = video.codec_context.name
            self._video_index = video.index
            self._extradata = video.codec_context.extradata
            self.ring.clear()
            self._set_online(True)
            bytes_window: deque[tuple[float, int]] = deque()

            for packet in container.demux(list(streams.values())):
                if self._stop_evt.is_set():
                    break
                if packet.size == 0:
                    continue
                now = time.monotonic()
                is_video = packet.stream.index == video.index
                bp = self.ring.add(packet.stream.index, bytes(packet), packet.pts, packet.dts,
                                   is_video and packet.is_keyframe, now)

                if is_video:
                    bytes_window.append((now, packet.size))
                    while bytes_window and bytes_window[0][0] < now - 5:
                        bytes_window.popleft()
                    self.bitrate_kbps = sum(s for _, s in bytes_window) * 8 / 5 / 1000

                self._handle_event(streams, video.index, bp)
        self._finish_writer()

    def _handle_event(self, streams, video_index: int, bp: BufferedPacket | None) -> None:
        with self._lock:
            want = self._want_event
        if want is None:
            if self._writer is not None:
                self._finish_writer()
            return
        if self._writer is not None and self._writer_event != want:
            self._finish_writer()  # new event while the old one was still running
        if self._writer is None:
            self.part_dir.mkdir(parents=True, exist_ok=True)
            path = self.part_dir / f"{want}.mp4.part"
            self._writer = ClipWriter(path, streams, video_index)
            self._writer_event = want
            for old in self.ring.snapshot():   # the pre-buffer (already contains bp)
                self._writer.write(old)
            log.info("[%s] recording started (%s, %.1fs pre-buffer)",
                     self.cam_name, want, self.ring.seconds)
            return
        if bp is not None:
            self._writer.write(bp)

    def _finish_writer(self) -> None:
        if self._writer is None:
            return
        writer, event_id = self._writer, self._writer_event
        self._writer, self._writer_event = None, None
        duration = writer.close()
        log.info("[%s] recording stopped (%s, %.1fs)", self.cam_name, event_id, duration)
        try:
            self.on_clip_done(event_id, writer.path, duration)
        except Exception:  # noqa: BLE001
            log.exception("[%s] processing the clip failed", self.cam_name)
