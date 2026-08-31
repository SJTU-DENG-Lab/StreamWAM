import json
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path


HERE = Path(__file__).resolve().parent
BUILDER = HERE / "build_streamingwam_promo.py"


class PromoVideoBuilderTests(unittest.TestCase):
    def run_builder(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(BUILDER), *args],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_timeline_starts_rollouts_together_and_stops_on_streaming_completion(self):
        result = self.run_builder("--describe")
        self.assertEqual(result.returncode, 0, result.stderr)
        timeline = json.loads(result.stdout)

        self.assertEqual(timeline["intro"], {"start": 0.0, "duration": 4.0})
        self.assertEqual(timeline["comparison"]["start"], 4.0)
        self.assertEqual(timeline["comparison"]["duration"], 38.0)
        self.assertEqual(timeline["comparison"]["joint_source_start"], 0.0)
        self.assertEqual(timeline["comparison"]["streaming_source_start"], 0.0)
        self.assertEqual(timeline["comparison"]["stop_reason"], "streaming_complete")
        self.assertEqual(timeline["method"], {"start": 42.0, "duration": 4.5})
        self.assertEqual(timeline["results"], {"start": 46.5, "duration": 5.0})
        self.assertEqual(timeline["total_duration"], 51.5)

    def test_music_is_stereo_48khz_and_matches_the_full_timeline(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "music.wav"
            result = self.run_builder("--synthesize-music", str(output))
            self.assertEqual(result.returncode, 0, result.stderr)

            with wave.open(str(output), "rb") as track:
                self.assertEqual(track.getnchannels(), 2)
                self.assertEqual(track.getframerate(), 48_000)
                self.assertEqual(track.getsampwidth(), 2)
                self.assertEqual(track.getnframes(), 2_472_000)

    def test_intro_preserves_the_complete_project_image(self):
        result = self.run_builder("--print-filter-graph")
        self.assertEqual(result.returncode, 0, result.stderr)
        intro_filter = result.stdout.split("[intro];", maxsplit=1)[0]
        self.assertIn("force_original_aspect_ratio=decrease", intro_filter)
        self.assertIn("pad=1920:1080", intro_filter)
        self.assertNotIn("crop=1920:1080", intro_filter)


if __name__ == "__main__":
    unittest.main()
