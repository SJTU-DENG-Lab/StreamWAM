# Streaming-WAM Project Video Design

## Goal

Produce a concise 1920×1080 project video that introduces Streaming-WAM, makes its real-robot speed advantage over Joint WAM immediately visible, and closes with the method and efficiency figures.

## Storyboard

1. **Project opening (about 4 seconds).** Show the supplied project-page image full screen with a subtle push-in. Fade in from black and dissolve into the robot comparison.
2. **Real-robot comparison (about 38 seconds).** Start the raw Joint WAM and Streaming-WAM recordings at the same time. Use a four-panel layout: the two Joint WAM camera views stacked in the left column and the two Streaming-WAM views stacked in the right column. Keep both videos at real speed. Use only compact column labels and elapsed-time readouts so the rollouts remain visually dominant.
3. **Completion cue.** When Streaming-WAM finishes, briefly emphasize its column with a teal completion treatment and the text `Completed · 1.8× Faster`. End the comparison immediately instead of waiting for Joint WAM to finish.
4. **Method close (about 5 seconds).** Fade to the supplied Streaming overview figure, using a restrained slow push-in.
5. **Results close (about 5 seconds).** Dissolve to the supplied Chunk Time and Episode Time figure, hold long enough to read the main values, and fade to black.

## Visual Treatment

- Output: 1920×1080, 30 fps, H.264, high-quality web-compatible encoding.
- The robot footage is center-cropped only as needed to fill each 960×540 panel; no time acceleration is applied.
- Labels use the project site's dark navy, off-white, and teal palette.
- Transitions use short dissolves. Motion on still images is subtle and must not distract from their content.
- No narration, decorative title cards, or additional claims are added.

## Audio

Use an original instrumental bed with light electronic percussion, soft plucked tones, and a warm pad. It should feel clean and technical without sounding dramatic or commercial. Keep it below the visual presentation, fade it cleanly at both ends, and retain only a quiet amount of the robot recording ambience during the comparison.

## Source Material

- Opening: `img_v3_02153_6031b306-ab56-4e6c-8a4a-ea370c4bceag.jpg`
- Method close: `img_v3_02153_b51dd30c-a43b-4569-954a-680fc6befaag.jpg`
- Results close: `download.png`
- Joint WAM: raw `joint-1.mp4` and `joint-2.mp4`
- Streaming-WAM: raw `1v2a-rtc-1.mp4` and `1v2a-rtc-2.mp4`

## Deliverables and Validation

- A reproducible build script stored with the project documentation.
- An original generated music track used by that script.
- A final MP4 with fast-start metadata for browser playback.
- Validation checks for 1920×1080 resolution, 30 fps, H.264 video, AAC audio, correct ordering, synchronized rollout start, and termination at the Streaming-WAM completion point.
