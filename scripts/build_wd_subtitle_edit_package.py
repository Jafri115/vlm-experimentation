"""Create a portable Subtitle Edit package for targeted WD transcript review.

Each case receives an editable SRT plus WAV and MP3 copies of the same audio
interval.  Files are separated by transcript provider and prefixed with the
provider name so that provenance remains visible outside the manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
from pathlib import Path


DEFAULT_IDS = [
    "401001_S1_seg026", "402026_S6_seg017", "401006_S18_seg008",
    "401006_S18_seg009", "401033_S21_seg013", "401033_S21_seg014",
    "401033_S22_seg037", "401033_S22_seg038", "402004_S12_seg012",
    "402009_S4_seg023", "402009_S4_seg040", "402002_S5_seg028",
    "402002_S5_seg031", "401024_S14_seg002", "401024_S14_seg003",
    "401020_S5_seg043", "401016_S3_seg048", "401019_S2_seg038",
    "401001_S1_seg027", "402026_S6_seg033",
]

TIMED_LINE = re.compile(r"^\[(\d+):(\d+(?:\.\d+)?)\]\s*([TPU?]):\s*(.*)$")
MEDIA_EXTENSIONS = (".wav", ".mp3", ".mp4", ".mov", ".m4a", ".flac")


def srt_stamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    whole_seconds, milliseconds = divmod(milliseconds, 1000)
    minutes, seconds = divmod(whole_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def parse_cues(text: str, window_end: float) -> list[dict]:
    cues: list[dict] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = TIMED_LINE.match(line)
        if match:
            minutes, seconds, speaker, words = match.groups()
            cues.append(
                {
                    "start": int(minutes) * 60 + float(seconds),
                    "speaker": "U" if speaker == "?" else speaker,
                    "text": words.strip(),
                }
            )
        elif cues:
            cues[-1]["text"] = (cues[-1]["text"] + " " + line).strip()
        else:
            raise ValueError(f"Transcript begins with an unrecognized line: {line[:100]}")

    for index, cue in enumerate(cues):
        later = [candidate["start"] for candidate in cues[index + 1 :] if candidate["start"] > cue["start"]]
        inferred_end = min(later) if later else window_end
        cue["end"] = max(cue["start"] + 0.25, inferred_end)
    return cues


def make_srt(cues: list[dict], clip_start: float, clip_end: float) -> str:
    blocks: list[str] = []
    for cue in cues:
        start = max(cue["start"], clip_start)
        end = min(cue["end"], clip_end)
        if end <= start:
            continue
        blocks.append(
            f"{len(blocks) + 1}\n"
            f"{srt_stamp(start - clip_start)} --> {srt_stamp(end - clip_start)}\n"
            f"{cue['speaker']}: {cue['text']}\n"
        )
    return "\n".join(blocks)


def run_capture(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def duration_seconds(ffprobe: str, media: Path) -> float:
    value = run_capture(
        [
            ffprobe,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(media),
        ]
    )
    return float(value)


def session_uid(row: dict) -> str:
    raw_session = str(row["session_id"]).strip()
    session_number = raw_session[1:] if raw_session.upper().startswith("S") else raw_session
    return f"{row['patient_id']}_S{int(float(session_number))}"


def existing(candidate: str | Path | None) -> Path | None:
    if not candidate:
        return None
    path = Path(str(candidate))
    return path.resolve() if path.is_file() else None


def find_media(row: dict, media_search: dict[str, list[str]], media_roots: list[Path]) -> tuple[Path | None, str]:
    sid = row["sample_id"]
    session = session_uid(row)

    # A segment-level audio path is best because no seek into a long recording is needed.
    for column in ("audio_path", "video_path"):
        found = existing(row.get(column))
        if found:
            return found, column

    for candidate in media_search.get(session, []):
        found = existing(candidate)
        if found:
            return found, "media_search"

    project_root = Path(r"D:\ukhd-Research\01_Projects")
    known_candidates = [
        project_root / "german-asr-pipeline" / "artifacts" / "full_session_role_mapping" / "full_session_audio" / f"{session}.wav",
        project_root / "german-asr-pipeline" / "artifacts" / "therapist_profile_builder_v2" / "full_session_audio" / f"{session}.wav",
        project_root / "video_pipeline_latest" / "Video_pipeline" / "final plan and session ratings" / "sessions" / str(row["patient_id"]) / session.split("_", 1)[1] / "media" / f"{session}.mp4",
    ]
    for candidate in known_candidates:
        found = existing(candidate)
        if found:
            return found, "known_local_layout"

    # User-supplied roots support the other machine without hard-coding therapist folders.
    for root in media_roots:
        if not root.is_dir():
            continue
        patterns = [f"{sid}*", f"{session}.*"]
        for pattern in patterns:
            for candidate in root.rglob(pattern):
                if candidate.is_file() and candidate.suffix.lower() in MEDIA_EXTENSIONS:
                    return candidate.resolve(), "media_root"
    return None, "not_found"


def transcode(
    ffmpeg: str,
    source: Path,
    wav: Path,
    mp3: Path,
    seek: float,
    duration: float,
) -> None:
    common = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    segment = ["-ss", f"{seek:.3f}", "-i", str(source), "-t", f"{duration:.3f}", "-vn", "-ac", "1", "-ar", "16000"]
    subprocess.run(common + segment + ["-c:a", "pcm_s16le", str(wav)], check=True)
    subprocess.run(common + segment + ["-c:a", "libmp3lame", "-b:a", "96k", str(mp3)], check=True)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build(args: argparse.Namespace) -> None:
    if not shutil.which(args.ffmpeg):
        raise SystemExit(f"ffmpeg was not found: {args.ffmpeg}")
    if not shutil.which(args.ffprobe):
        raise SystemExit(f"ffprobe was not found: {args.ffprobe}")

    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    lookup = {row["sample_id"]: row for row in all_rows}
    selected_ids = args.ids or DEFAULT_IDS
    missing_ids = [sid for sid in selected_ids if sid not in lookup]
    if missing_ids:
        raise ValueError(f"IDs absent from {args.input}: {missing_ids}")

    media_search: dict[str, list[str]] = {}
    if args.media_search.is_file():
        media_search = json.loads(args.media_search.read_text(encoding="utf-8"))

    args.output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    unresolved: list[dict] = []

    for sid in selected_ids:
        row = lookup[sid]
        provider = str(row.get("transcript_provider") or "unknown").strip().lower()
        provider_label = provider.upper()
        provider_dir = args.output / provider
        provider_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{provider_label}__{sid}"
        srt_path = provider_dir / f"{stem}.srt"
        wav_path = provider_dir / f"{stem}.wav"
        mp3_path = provider_dir / f"{stem}.mp3"

        start = float(row["start_sec"])
        end = float(row["end_sec"])
        cues = parse_cues(row.get("transcript_text", ""), end)
        first_cue = min((cue["start"] for cue in cues), default=start)
        last_cue_end = max((cue["end"] for cue in cues), default=end)
        # Preserve cue onsets that fall just outside the human rating window and add listening context.
        desired_start = max(0.0, min(start, first_cue) - args.padding)
        desired_end = max(end, last_cue_end) + args.padding

        source, source_method = find_media(row, media_search, args.media_root)
        media_status = "created"
        source_duration = None
        actual_start = desired_start
        actual_end = desired_end
        error = ""

        if source is None:
            media_status = "missing_source"
            actual_start, actual_end = desired_start, desired_end
            unresolved.append(
                {
                    "sample_id": sid,
                    "provider": provider,
                    "session_uid": session_uid(row),
                    "expected_video_path": row.get("video_path", ""),
                    "reason": "No source audio/video found on this machine",
                }
            )
        else:
            try:
                source_duration = duration_seconds(args.ffprobe, source)
                target_duration = end - start
                # Segment audio is already cut; session media requires absolute seeking.
                is_segment_media = sid.lower() in source.stem.lower() or source_duration <= target_duration + 5.0
                if is_segment_media:
                    actual_start = start
                    actual_end = min(end, start + source_duration)
                    seek = 0.0
                else:
                    actual_start = min(desired_start, source_duration)
                    actual_end = min(desired_end, source_duration)
                    seek = actual_start
                clip_duration = max(0.0, actual_end - actual_start)
                if clip_duration <= 0:
                    raise ValueError(f"requested interval begins after source duration ({source_duration:.3f}s)")
                transcode(args.ffmpeg, source, wav_path, mp3_path, seek, clip_duration)
            except Exception as exc:  # retain the SRT and expose the media error in CSV
                media_status = "media_error"
                error = str(exc)
                wav_path.unlink(missing_ok=True)
                mp3_path.unlink(missing_ok=True)
                unresolved.append(
                    {
                        "sample_id": sid,
                        "provider": provider,
                        "session_uid": session_uid(row),
                        "expected_video_path": str(source),
                        "reason": error,
                    }
                )

        srt_path.write_text(make_srt(cues, actual_start, actual_end), encoding="utf-8-sig")
        manifest.append(
            {
                "review_order": len(manifest) + 1,
                "sample_id": sid,
                "transcript_provider": provider,
                "provider_file_prefix": provider_label,
                "patient_id": row["patient_id"],
                "session_id": row["session_id"],
                "WD_P_rater1": row.get("WD_P_rater1", ""),
                "WD_P_rater2": row.get("WD_P_rater2", ""),
                "rating_start_session_sec": start,
                "rating_end_session_sec": end,
                "audio_clip_start_session_sec": actual_start,
                "audio_clip_end_session_sec": actual_end,
                "rating_start_in_clip_sec": start - actual_start,
                "rating_end_in_clip_sec": end - actual_start,
                "srt_file": str(srt_path.relative_to(args.output)),
                "wav_file": str(wav_path.relative_to(args.output)) if wav_path.is_file() else "",
                "mp3_file": str(mp3_path.relative_to(args.output)) if mp3_path.is_file() else "",
                "media_status": media_status,
                "source_media": str(source or ""),
                "source_method": source_method,
                "source_duration_sec": source_duration if source_duration is not None else "",
                "source_review_flags": row.get("review_flags", ""),
                "correction_status": "TO_REVIEW",
                "speaker_roles_checked": "NO",
                "timing_checked": "NO",
                "text_checked": "NO",
                "reviewer_notes": "",
                "error": error,
            }
        )

    manifest_fields = list(manifest[0])
    write_csv(args.output / "00_review_manifest.csv", manifest, manifest_fields)
    unresolved_fields = ["sample_id", "provider", "session_uid", "expected_video_path", "reason"]
    write_csv(args.output / "00_unresolved_media.csv", unresolved, unresolved_fields)

    readme = f"""# WD transcript correction package for Subtitle Edit

This package contains **{len(manifest)} targeted high-rating audit segments**.

- Amberscript: **{sum(row['transcript_provider'] == 'amberscript' for row in manifest)}**
- Voxtral: **{sum(row['transcript_provider'] == 'voxtral' for row in manifest)}**
- WAV/MP3 pairs created here: **{sum(row['media_status'] == 'created' for row in manifest)}**
- Media unresolved on this computer: **{len(unresolved)}**

Files are separated into `amberscript/` and `voxtral/`. Every filename also begins
with `AMBERSCRIPT__` or `VOXTRAL__`, so the transcript source is always visible.

## Open a case

1. Open Subtitle Edit.
2. Open the `.srt` file.
3. Choose **Video > Open video file** and select the same-named `.wav` or `.mp3`.
4. Turn on the waveform/spectrogram. Play each subtitle several times at 0.75x speed.
5. Save the corrected SRT under the same name in a separate `corrected/` folder.
6. Update `00_review_manifest.csv`: set the three checked columns to `YES`, change
   `correction_status` to `DONE`, and record uncertainty in `reviewer_notes`.

## What to correct

- **Speaker first:** `P` = patient, `T` = therapist, `U` = uncertain. Correct the
  letter before the colon. Do not guess when the voice cannot be identified.
- **Timing second:** move cue boundaries so each cue starts with speech and ends after
  it. Split a cue when the speaker changes. Keep overlaps if both people speak.
- **German text last:** correct only words you can hear confidently. Pay particular
  attention to negations (`nicht`, `kein`), numbers, names, and pronouns because they
  can change the clinical meaning.
- Write `[unverständlich]` for speech you cannot understand. For one doubtful word,
  use `[? Wort]`. Keep false starts, repetitions, pauses, and incomplete sentences;
  do not rewrite the conversation into polished German.
- If the audio and subtitle appear unrelated, stop that case and write
  `WRONG AUDIO OR TIMING` in `reviewer_notes`.

The human WD ratings are included only in the manifest. Do not use them to rewrite
the transcript. The audio contains extra context where source media allowed it; the
rating window's position inside each clip is recorded in the manifest.
"""
    (args.output / "README.md").write_text(readme, encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "segments": len(manifest),
                "amberscript": sum(row["transcript_provider"] == "amberscript" for row in manifest),
                "voxtral": sum(row["transcript_provider"] == "voxtral" for row in manifest),
                "media_pairs_created": sum(row["media_status"] == "created" for row in manifest),
                "unresolved_media": len(unresolved),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/wd_high_rater_segments/both_raters_3_or_higher.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/wd_high_rater_segments/subtitle_edit_review_package"),
    )
    parser.add_argument(
        "--media-search",
        type=Path,
        default=Path("data/amberscript_llm/media_search.json"),
    )
    parser.add_argument("--media-root", type=Path, action="append", default=[])
    parser.add_argument("--ids", nargs="+", help="Optional IDs; default is the 20-case manual-audit subset")
    parser.add_argument("--padding", type=float, default=2.0, help="Context seconds around transcript/rating interval")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    build(parser.parse_args())
