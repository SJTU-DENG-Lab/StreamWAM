#!/usr/bin/env python3
"""Build the Streaming-WAM project video from the original rollout media."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Timeline:
    intro: float = 4.0
    comparison: float = 38.0
    method: float = 4.5
    results: float = 5.0

    @property
    def total(self) -> float:
        return self.intro + self.comparison + self.method + self.results

    def describe(self) -> dict[str, object]:
        comparison_start = self.intro
        method_start = comparison_start + self.comparison
        results_start = method_start + self.method
        return {
            "intro": {"start": 0.0, "duration": self.intro},
            "comparison": {
                "start": comparison_start,
                "duration": self.comparison,
                "joint_source_start": 0.0,
                "streaming_source_start": 0.0,
                "stop_reason": "streaming_complete",
            },
            "method": {"start": method_start, "duration": self.method},
            "results": {"start": results_start, "duration": self.results},
            "total_duration": self.total,
        }


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = REPO_ROOT.parents[1]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_WORK_DIR = Path(tempfile.gettempdir()) / "streamingwam-project-video"
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
RAW_NAMES = ("joint-1.mp4", "joint-2.mp4", "1v2a-rtc-1.mp4", "1v2a-rtc-2.mp4")


def synthesize_music(output: Path, timeline: Timeline) -> None:
    """Create a deterministic, restrained electronic instrumental bed."""
    sample_rate = 48_000
    frames = int(round(timeline.total * sample_rate))
    audio = np.zeros((frames, 2), dtype=np.float64)
    rng = np.random.default_rng(20260830)
    beat = 60.0 / 112.0

    chords = (
        (146.83, 185.00, 220.00),  # D major
        (123.47, 146.83, 185.00),  # B minor
        (98.00, 123.47, 146.83),   # G major
        (110.00, 138.59, 164.81),  # A major
    )
    chord_length = beat * 4.0
    for chord_index, chord_start in enumerate(np.arange(0.0, timeline.total, chord_length)):
        start = int(chord_start * sample_rate)
        stop = min(frames, int((chord_start + chord_length + 0.35) * sample_rate))
        local_t = np.arange(stop - start, dtype=np.float64) / sample_rate
        attack = np.minimum(local_t / 0.45, 1.0)
        release = np.minimum((stop - start) / sample_rate - local_t, 0.55) / 0.55
        envelope = np.clip(attack * release, 0.0, 1.0)
        chord = chords[chord_index % len(chords)]
        pad = sum(
            np.sin(2.0 * np.pi * frequency * local_t + voice * 0.7)
            + 0.22 * np.sin(2.0 * np.pi * frequency * 2.0 * local_t)
            for voice, frequency in enumerate(chord)
        ) / (len(chord) * 1.22)
        audio[start:stop, 0] += 0.065 * envelope * pad
        audio[start:stop, 1] += 0.065 * envelope * np.roll(pad, 31)

    melody_scale = (293.66, 329.63, 369.99, 440.00, 493.88, 440.00, 369.99, 329.63)
    for note_index, note_start in enumerate(np.arange(0.0, timeline.total, beat / 2.0)):
        start = int(note_start * sample_rate)
        note_length = min(frames - start, int(0.34 * sample_rate))
        if note_length <= 0:
            continue
        local_t = np.arange(note_length, dtype=np.float64) / sample_rate
        envelope = (1.0 - np.exp(-local_t / 0.006)) * np.exp(-local_t / 0.105)
        frequency = melody_scale[(note_index + (note_index // 8) * 2) % len(melody_scale)]
        pluck = (
            np.sin(2.0 * np.pi * frequency * local_t)
            + 0.38 * np.sin(2.0 * np.pi * frequency * 2.0 * local_t)
            + 0.14 * np.sin(2.0 * np.pi * frequency * 3.0 * local_t)
        )
        pan = 0.35 if note_index % 2 == 0 else 0.65
        audio[start : start + note_length, 0] += 0.105 * (1.0 - pan) * envelope * pluck
        audio[start : start + note_length, 1] += 0.105 * pan * envelope * pluck

    for beat_index, beat_start in enumerate(np.arange(0.0, timeline.total, beat)):
        start = int(beat_start * sample_rate)
        kick_length = min(frames - start, int(0.16 * sample_rate))
        local_t = np.arange(kick_length, dtype=np.float64) / sample_rate
        kick_phase = 2.0 * np.pi * (68.0 * local_t - 22.0 * local_t * local_t)
        kick = np.sin(kick_phase) * np.exp(-local_t / 0.055)
        kick_gain = 0.095 if beat_index % 4 == 0 else 0.048
        audio[start : start + kick_length] += kick_gain * kick[:, None]

        hat_start = start + int((beat / 2.0) * sample_rate)
        hat_length = min(frames - hat_start, int(0.045 * sample_rate))
        if hat_length > 0:
            hat_t = np.arange(hat_length, dtype=np.float64) / sample_rate
            noise = rng.normal(0.0, 1.0, hat_length)
            noise = np.concatenate(([0.0], np.diff(noise)))
            hat = 0.013 * noise * np.exp(-hat_t / 0.012)
            audio[hat_start : hat_start + hat_length, 0] += hat
            audio[hat_start : hat_start + hat_length, 1] += np.roll(hat, 7)

    fade_in = np.clip(np.arange(frames) / (1.1 * sample_rate), 0.0, 1.0)
    fade_out = np.clip((frames - 1 - np.arange(frames)) / (1.5 * sample_rate), 0.0, 1.0)
    audio *= np.minimum(fade_in, fade_out)[:, None]
    peak = float(np.max(np.abs(audio)))
    if peak > 0.42:
        audio *= 0.42 / peak

    output.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.int16(np.clip(audio, -1.0, 1.0) * 32767.0)
    with wave.open(str(output), "wb") as track:
        track.setnchannels(2)
        track.setsampwidth(2)
        track.setframerate(sample_rate)
        track.writeframes(pcm.tobytes())


def ensure_raw_media(source_root: Path, raw_dir: Path) -> dict[str, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in RAW_NAMES if not (raw_dir / name).is_file()]
    if missing:
        archive = source_root / "demo1.rar"
        if not archive.is_file():
            raise FileNotFoundError(f"Missing rollout archive: {archive}")
        bsdtar = shutil.which("bsdtar")
        if not bsdtar:
            raise RuntimeError("bsdtar is required to extract demo1.rar")
        subprocess.run(
            [bsdtar, "-xf", str(archive), "-C", str(raw_dir), *missing],
            check=True,
        )
    media = {name: raw_dir / name for name in RAW_NAMES}
    for name, path in media.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing extracted rollout {name}: {path}")
    return media


def build_filter_graph(timeline: Timeline) -> str:
    font = str(FONT).replace("\\", "\\\\").replace(":", "\\:")
    return f"""
[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0xf5f3ed,setsar=1,fps=30,trim=duration={timeline.intro},setpts=PTS-STARTPTS,fade=t=in:st=0:d=0.45,fade=t=out:st=3.55:d=0.45[intro];
[1:v]trim=start=0:duration={timeline.comparison},setpts=PTS-STARTPTS,scale=960:540:force_original_aspect_ratio=increase,crop=960:540,setsar=1[j1];
[2:v]trim=start=0:duration={timeline.comparison},setpts=PTS-STARTPTS,scale=960:540:force_original_aspect_ratio=increase,crop=960:540,setsar=1[j2];
[3:v]trim=start=0:duration={timeline.comparison},setpts=PTS-STARTPTS,scale=960:540:force_original_aspect_ratio=increase,crop=960:540,setsar=1[s1];
[4:v]trim=start=0:duration={timeline.comparison},setpts=PTS-STARTPTS,scale=960:540:force_original_aspect_ratio=increase,crop=960:540,setsar=1[s2];
[j1][j2][s1][s2]xstack=inputs=4:layout=0_0|0_h0|w0_0|w0_h0:fill=0x07141e,
drawbox=x=0:y=0:w=960:h=62:color=0x07141e@0.82:t=fill,
drawbox=x=960:y=0:w=960:h=62:color=0x07141e@0.82:t=fill,
drawbox=x=957:y=0:w=6:h=1080:color=0xf3f0e8@0.88:t=fill,
drawtext=fontfile='{font}':text='Joint WAM':fontcolor=white:fontsize=30:x=30:y=15,
drawtext=fontfile='{font}':text='Streaming-WAM':fontcolor=0x55e3c4:fontsize=30:x=990:y=15,
drawtext=fontfile='{font}':text='%{{pts\\:hms}}':fontcolor=0xd9e0e3:fontsize=22:x=790:y=20,
drawtext=fontfile='{font}':text='%{{pts\\:hms}}':fontcolor=0x9af1df:fontsize=22:x=1750:y=20,
drawbox=x=1080:y=446:w=720:h=188:color=0x07141e@0.84:t=fill:enable='gte(t,36.5)',
drawbox=x=1080:y=446:w=720:h=188:color=0x42d9ba@0.95:t=7:enable='gte(t,36.5)',
drawtext=fontfile='{font}':text='Completed · 2.4× Faster':fontcolor=0x66f0d3:fontsize=42:x=960+(960-text_w)/2:y=520:enable='gte(t,36.5)',
fade=t=in:st=0:d=0.45,fade=t=out:st=37.55:d=0.45[comparison];
[5:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0xf5f3ed,setsar=1,fps=30,trim=duration={timeline.method},setpts=PTS-STARTPTS,fade=t=in:st=0:d=0.45,fade=t=out:st=4.05:d=0.45[method];
[6:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0xf5f3ed,setsar=1,fps=30,trim=duration={timeline.results},setpts=PTS-STARTPTS,fade=t=in:st=0:d=0.45,fade=t=out:st=4.35:d=0.65[results];
[intro][comparison][method][results]concat=n=4:v=1:a=0[outv];
[7:a]atrim=duration={timeline.total},asetpts=PTS-STARTPTS,volume=0.90[music];
[1:a]atrim=duration={timeline.comparison},asetpts=PTS-STARTPTS,volume=0.025,adelay=4000|4000,apad=pad_dur=9.5,atrim=duration={timeline.total}[jointroom];
[3:a]atrim=duration={timeline.comparison},asetpts=PTS-STARTPTS,volume=0.025,adelay=4000|4000,apad=pad_dur=9.5,atrim=duration={timeline.total}[streamroom];
[music][jointroom][streamroom]amix=inputs=3:duration=longest:dropout_transition=0,volume=2.2,alimiter=limit=0.94[outa]
""".strip()


def probe(output: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels",
            "-show_entries",
            "format=duration,size,bit_rate",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    report = json.loads(result.stdout)
    streams = report.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    duration = float(report["format"]["duration"])
    if not video or video.get("codec_name") != "h264":
        raise RuntimeError("Rendered video is not H.264")
    if (video.get("width"), video.get("height"), video.get("r_frame_rate")) != (1920, 1080, "30/1"):
        raise RuntimeError(f"Unexpected video format: {video}")
    if not audio or audio.get("codec_name") != "aac":
        raise RuntimeError("Rendered audio is not AAC")
    if abs(duration - Timeline().total) > 0.12:
        raise RuntimeError(f"Unexpected duration {duration:.3f}s")
    return report


def render(source_root: Path, raw_dir: Path, output_dir: Path) -> Path:
    timeline = Timeline()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required")
    if not FONT.is_file():
        raise FileNotFoundError(f"Missing label font: {FONT}")

    images = {
        "intro": source_root / "img_v3_02153_6031b306-ab56-4e6c-8a4a-ea370c4bceag.jpg",
        "method": source_root / "img_v3_02153_b51dd30c-a43b-4569-954a-680fc6befaag.jpg",
        "results": source_root / "download.png",
    }
    for label, path in images.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} image: {path}")
    media = ensure_raw_media(source_root, raw_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    music = output_dir / "streamingwam-project-video-music.wav"
    graph_path = output_dir / "streamingwam-project-video-filter.txt"
    output = output_dir / "streamingwam-project-video.mp4"
    validation = output_dir / "streamingwam-project-video-validation.json"
    synthesize_music(music, timeline)
    graph_path.write_text(build_filter_graph(timeline) + "\n", encoding="utf-8")

    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loop", "1", "-framerate", "30", "-t", str(timeline.intro), "-i", str(images["intro"]),
        "-i", str(media["joint-1.mp4"]),
        "-i", str(media["joint-2.mp4"]),
        "-i", str(media["1v2a-rtc-1.mp4"]),
        "-i", str(media["1v2a-rtc-2.mp4"]),
        "-loop", "1", "-framerate", "30", "-t", str(timeline.method), "-i", str(images["method"]),
        "-loop", "1", "-framerate", "30", "-t", str(timeline.results), "-i", str(images["results"]),
        "-i", str(music),
        "-filter_complex_script", str(graph_path),
        "-map", "[outv]",
        "-map", "[outa]",
        "-t", str(timeline.total),
        "-r", "30",
        "-c:v", "libx264",
        "-preset", "medium",
        "-threads", "4",
        "-x264-params", "threads=4:lookahead_threads=1",
        "-crf", "17",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-level", "4.2",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-movflags", "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True)
    report = probe(output)
    validation.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="print the fixed storyboard timeline as JSON")
    parser.add_argument("--print-filter-graph", action="store_true", help="print the FFmpeg filter graph")
    parser.add_argument("--synthesize-music", type=Path, metavar="PATH", help="write only the original music bed")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_WORK_DIR / "raw")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timeline = Timeline()
    if args.describe:
        print(json.dumps(timeline.describe(), indent=2))
        return
    if args.print_filter_graph:
        print(build_filter_graph(timeline))
        return
    if args.synthesize_music:
        synthesize_music(args.synthesize_music, timeline)
        return
    output = render(args.source_root.resolve(), args.raw_dir.resolve(), args.output_dir.resolve())
    print(output)


if __name__ == "__main__":
    main()
