from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass

import numpy as np

DEFAULT_GATEWAY_URL = "http://127.0.0.1:8765"
DEFAULT_SEND_FPS = 16
DEFAULT_SPECTRUM_ROWS = 10
MAX_SPECTRUM_ROWS = 10
DEFAULT_BANDS = 16
DEFAULT_SAMPLE_RATE = 48_000
DEFAULT_MIN_FREQUENCY = 45.0
DEFAULT_MAX_FREQUENCY = 12_000.0
DEFAULT_BAR_ON = "■"
DEFAULT_BAR_OFF = "＿"
DISPLAY_TEXT_BYTE_LIMIT = 1000
HTTP_TIMEOUT_SECONDS = 2.0


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def bounded_rows(value: str) -> int:
    parsed = positive_int(value)
    if parsed > MAX_SPECTRUM_ROWS:
        raise argparse.ArgumentTypeError(f"rows must be {MAX_SPECTRUM_ROWS} or fewer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


@dataclass
class AppConfig:
    base_url: str
    api_key: str | None
    fps: int
    rows: int
    bands: int
    sample_rate: int
    min_frequency: float
    max_frequency: float
    device_name: str | None
    demo: bool
    no_send: bool
    frames: int | None
    clear_on_exit: bool


class GatewayClient:
    def __init__(self, base_url: str, api_key: str | None) -> None:
        import requests

        self._requests = requests
        self._display_url = f"{base_url.rstrip('/')}/api/display"
        self._session = requests.Session()
        if api_key:
            self._session.headers.update({"X-API-Key": api_key})

    def send_text(self, text: str) -> None:
        self._post_json({"text": text})

    def clear(self) -> None:
        self._post_json({"clear": True})

    def _post_json(self, payload: dict[str, object]) -> None:
        response = self._session.post(
            self._display_url,
            json=payload,
            timeout=HTTP_TIMEOUT_SECONDS,
        )

        try:
            body = response.json()
        except ValueError as exc:
            raise RuntimeError(f"gateway returned non-JSON response: {response.text[:200]}") from exc

        if response.status_code >= 400 or not body.get("accepted", False):
            error = body.get("error") or f"gateway rejected request with status {response.status_code}"
            raise RuntimeError(str(error))


class SpectrumAnalyzer:
    def __init__(
        self,
        sample_rate: int,
        frame_samples: int,
        bands: int,
        min_frequency: float,
        max_frequency: float,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.bands = bands
        self.fft_size = max(2048, 1 << (frame_samples - 1).bit_length())
        self.window = np.hanning(frame_samples).astype(np.float32)
        self.work_buffer = np.zeros(self.fft_size, dtype=np.float32)
        usable_max_frequency = min(max_frequency, sample_rate / 2.0)
        self.band_edges = np.geomspace(min_frequency, usable_max_frequency, bands + 1)
        self.bin_frequencies = np.fft.rfftfreq(self.fft_size, d=1.0 / sample_rate)
        self.floor_db: float | None = None
        self.ceiling_db: float | None = None
        self.smoothed = np.zeros(bands, dtype=np.float32)

    def analyze(self, samples: np.ndarray) -> np.ndarray:
        mono = np.asarray(samples, dtype=np.float32).reshape(-1)
        if mono.size == 0:
            return self.smoothed.copy()

        if mono.size < self.frame_samples:
            padded = np.zeros(self.frame_samples, dtype=np.float32)
            padded[-mono.size :] = mono
            mono = padded
        elif mono.size > self.frame_samples:
            mono = mono[-self.frame_samples :]

        windowed = mono * self.window
        self.work_buffer.fill(0.0)
        self.work_buffer[: self.frame_samples] = windowed
        spectrum = np.abs(np.fft.rfft(self.work_buffer)) ** 2

        band_energy = np.empty(self.bands, dtype=np.float32)
        for index in range(self.bands):
            start_frequency = self.band_edges[index]
            end_frequency = self.band_edges[index + 1]
            start_bin = int(np.searchsorted(self.bin_frequencies, start_frequency, side="left"))
            end_bin = int(np.searchsorted(self.bin_frequencies, end_frequency, side="right"))
            end_bin = max(start_bin + 1, end_bin)
            band_energy[index] = float(np.mean(spectrum[start_bin:end_bin]))

        band_db = 10.0 * np.log10(band_energy + 1e-12)
        floor_target = float(np.percentile(band_db, 20.0) - 6.0)
        ceiling_target = float(np.percentile(band_db, 95.0) + 3.0)

        if self.floor_db is None or self.ceiling_db is None:
            self.floor_db = floor_target
            self.ceiling_db = max(floor_target + 18.0, ceiling_target)
        else:
            self.floor_db = (self.floor_db * 0.92) + (floor_target * 0.08)
            ceiling_target = max(self.floor_db + 18.0, ceiling_target)
            self.ceiling_db = (self.ceiling_db * 0.85) + (ceiling_target * 0.15)

        span = max(15.0, self.ceiling_db - self.floor_db)
        normalized = np.clip((band_db - self.floor_db) / span, 0.0, 1.0)
        normalized = np.sqrt(normalized)
        rising = normalized > self.smoothed
        attacked = (self.smoothed * 0.35) + (normalized * 0.65)
        decayed = np.maximum(normalized, self.smoothed * 0.82)
        self.smoothed = np.where(rising, attacked, decayed).astype(np.float32)
        return self.smoothed.copy()


class DemoAudioSource:
    def __init__(self, sample_rate: int, frame_samples: int) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.frame_index = 0
        self.phase = np.zeros(3, dtype=np.float64)

    def __enter__(self) -> DemoAudioSource:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> np.ndarray:
        t = np.arange(self.frame_samples, dtype=np.float64) / self.sample_rate
        bass = 70.0 + (35.0 * math.sin(self.frame_index / 10.0))
        mid = 320.0 + (180.0 * math.sin(self.frame_index / 7.0))
        treble = 1_800.0 + (1_400.0 * math.sin(self.frame_index / 13.0))
        frequencies = np.array([bass, mid, treble], dtype=np.float64)
        amplitudes = np.array([0.9, 0.55, 0.35], dtype=np.float64)
        waveform = np.zeros(self.frame_samples, dtype=np.float64)

        for index, frequency in enumerate(frequencies):
            phases = (2.0 * math.pi * frequency * t) + self.phase[index]
            waveform += np.sin(phases) * amplitudes[index]
            phase_advance = (2.0 * math.pi * frequency * self.frame_samples) / self.sample_rate
            self.phase[index] = (self.phase[index] + phase_advance) % (2.0 * math.pi)

        pulse = 0.65 + (0.35 * max(0.0, math.sin(self.frame_index / 4.0)))
        noise = np.random.normal(0.0, 0.02, self.frame_samples)
        self.frame_index += 1
        return (waveform * pulse + noise).astype(np.float32)


class LoopbackAudioSource:
    def __init__(self, sample_rate: int, frame_samples: int, device_name: str | None) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.device_name = device_name
        self._recorder_context = None
        self._recorder = None
        self._microphone_name = ""

    def __enter__(self) -> LoopbackAudioSource:
        soundcard = import_soundcard()
        microphone = resolve_loopback_microphone(soundcard, self.device_name)
        self._microphone_name = microphone.name
        self._recorder_context = microphone.recorder(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.frame_samples,
        )
        self._recorder = self._recorder_context.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._recorder_context is not None:
            return self._recorder_context.__exit__(exc_type, exc, tb)
        return False

    def read(self) -> np.ndarray:
        if self._recorder is None:
            raise RuntimeError("loopback recorder is not ready")
        captured = self._recorder.record(numframes=self.frame_samples)
        mono = np.asarray(captured, dtype=np.float32)
        if mono.ndim == 2:
            mono = np.mean(mono, axis=1)
        return mono.reshape(-1)


def import_soundcard():
    try:
        import soundcard as soundcard
    except ImportError as exc:
        raise RuntimeError(
            "soundcard package is required for live loopback capture. Install requirements first."
        ) from exc
    return soundcard


def loopback_microphones(soundcard) -> list[object]:
    microphones = list(soundcard.all_microphones(include_loopback=True))
    return [mic for mic in microphones if getattr(mic, "isloopback", False)]


def resolve_loopback_microphone(soundcard, device_name: str | None):
    microphones = loopback_microphones(soundcard)
    if not microphones:
        raise RuntimeError("no loopback-capable output device was found")

    if device_name:
        needle = device_name.casefold()
        matches = [mic for mic in microphones if needle in mic.name.casefold()]
        if matches:
            return matches[0]
        available = ", ".join(mic.name for mic in microphones)
        raise RuntimeError(f"loopback device matching '{device_name}' was not found. Available: {available}")

    default_speaker = soundcard.default_speaker()
    if default_speaker is not None:
        speaker_name = default_speaker.name.casefold()
        for microphone in microphones:
            microphone_name = microphone.name.casefold()
            if speaker_name in microphone_name or microphone_name in speaker_name:
                return microphone
        try:
            return soundcard.get_microphone(id=str(default_speaker.name), include_loopback=True)
        except Exception:
            pass

    return microphones[0]


def print_loopback_devices() -> int:
    soundcard = import_soundcard()
    microphones = loopback_microphones(soundcard)
    if not microphones:
        print("No loopback-capable devices found.", file=sys.stderr)
        return 1

    default_name = ""
    default_speaker = soundcard.default_speaker()
    if default_speaker is not None:
        default_name = default_speaker.name.casefold()

    for microphone in microphones:
        suffix = ""
        if default_name and default_name in microphone.name.casefold():
            suffix = " [default speaker match]"
        print(f"- {microphone.name}{suffix}")
    return 0


def build_spectrum_text(levels: np.ndarray, rows: int) -> str:
    clamped = np.clip(levels, 0.0, 1.0)
    heights = np.rint(clamped * rows).astype(int)
    lines: list[str] = []

    for row in range(rows, 0, -1):
        segments = [DEFAULT_BAR_ON if height >= row else DEFAULT_BAR_OFF for height in heights]
        lines.append("".join(segments))
    return "\n".join(lines)


def ensure_text_budget(rows: int, bands: int) -> None:
    line_width = bands * len(DEFAULT_BAR_ON)
    text_bytes = (rows * line_width) + max(0, rows - 1)
    if text_bytes > DISPLAY_TEXT_BYTE_LIMIT:
        raise ValueError(
            f"rows={rows} and bands={bands} exceed the {DISPLAY_TEXT_BYTE_LIMIT}-byte API limit"
        )


def show_console_frame(text: str) -> None:
    if sys.stdout.isatty():
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(text)
        sys.stdout.write("\n")
        sys.stdout.flush()
        return
    print(text)
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Loop back PC audio, run FFT, and push a text-only spectrum to the G2 Gateway.",
    )
    parser.add_argument("--base-url", default=DEFAULT_GATEWAY_URL, help="Gateway base URL")
    parser.add_argument("--api-key", default=None, help="Optional gateway API key")
    parser.add_argument("--fps", type=positive_int, default=DEFAULT_SEND_FPS, help="Send FPS")
    parser.add_argument(
        "--rows",
        type=bounded_rows,
        default=DEFAULT_SPECTRUM_ROWS,
        help=f"Spectrum rows, capped at {MAX_SPECTRUM_ROWS}",
    )
    parser.add_argument("--bands", type=positive_int, default=DEFAULT_BANDS, help="Spectrum band count")
    parser.add_argument(
        "--sample-rate",
        type=positive_int,
        default=DEFAULT_SAMPLE_RATE,
        help="Capture sample rate",
    )
    parser.add_argument(
        "--min-freq",
        type=positive_float,
        default=DEFAULT_MIN_FREQUENCY,
        help="Lowest band edge in Hz",
    )
    parser.add_argument(
        "--max-freq",
        type=positive_float,
        default=DEFAULT_MAX_FREQUENCY,
        help="Highest band edge in Hz",
    )
    parser.add_argument("--device-name", default=None, help="Partial loopback device name to match")
    parser.add_argument("--list-devices", action="store_true", help="List loopback-capable devices")
    parser.add_argument("--demo", action="store_true", help="Use synthetic audio instead of live loopback")
    parser.add_argument("--no-send", action="store_true", help="Render to the console only")
    parser.add_argument("--frames", type=positive_int, default=None, help="Stop after N frames")
    parser.add_argument("--clear-on-exit", action="store_true", help="Clear the glasses display when exiting")
    return parser


def build_config(args: argparse.Namespace) -> AppConfig:
    if args.max_freq <= args.min_freq:
        raise ValueError("max-freq must be greater than min-freq")
    ensure_text_budget(args.rows, args.bands)
    return AppConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        fps=args.fps,
        rows=args.rows,
        bands=args.bands,
        sample_rate=args.sample_rate,
        min_frequency=args.min_freq,
        max_frequency=args.max_freq,
        device_name=args.device_name,
        demo=args.demo,
        no_send=args.no_send,
        frames=args.frames,
        clear_on_exit=args.clear_on_exit,
    )


def run(config: AppConfig) -> int:
    frame_samples = max(1024, int(round(config.sample_rate / config.fps)))
    analyzer = SpectrumAnalyzer(
        sample_rate=config.sample_rate,
        frame_samples=frame_samples,
        bands=config.bands,
        min_frequency=config.min_frequency,
        max_frequency=config.max_frequency,
    )

    source = DemoAudioSource(config.sample_rate, frame_samples)
    if not config.demo:
        source = LoopbackAudioSource(config.sample_rate, frame_samples, config.device_name)

    client = None if config.no_send else GatewayClient(config.base_url, config.api_key)
    frame_duration = 1.0 / config.fps
    next_deadline = time.perf_counter()
    sent_frames = 0

    try:
        with source as audio_source:
            while True:
                samples = audio_source.read()
                levels = analyzer.analyze(samples)
                text = build_spectrum_text(levels, config.rows)

                if client is None:
                    show_console_frame(text)
                else:
                    client.send_text(text)

                sent_frames += 1
                if config.frames is not None and sent_frames >= config.frames:
                    break

                next_deadline += frame_duration
                sleep_for = next_deadline - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_deadline = time.perf_counter()
    finally:
        if config.clear_on_exit and client is not None:
            try:
                client.clear()
            except Exception as exc:
                print(f"Failed to clear display on exit: {exc}", file=sys.stderr)

    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_devices:
        return print_loopback_devices()

    try:
        config = build_config(args)
        return run(config)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())