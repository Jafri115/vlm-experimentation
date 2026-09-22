import argparse
import csv
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd


VIDEO_EXTS = {".mp4", ".mkv", ".asf", ".avi", ".mov", ".wmv", ".m4v"}


def norm_session_name(name):
    name = Path(str(name)).name
    stem = Path(name).stem

    m = re.search(r"(\d+)_S0*(\d+)", stem, flags=re.I)
    if m:
        return f"{m.group(1)}_S{int(m.group(2))}".lower()

    return stem.lower()


def build_file_index(video_root):
    root = Path(video_root)

    if not root.exists():
        raise FileNotFoundError(f"Video root does not exist: {root}")

    exact = {}
    normalized = {}

    print(f"Scanning video root: {root}", flush=True)

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VIDEO_EXTS:
            continue

        exact.setdefault(path.name.lower(), []).append(path)
        normalized.setdefault(norm_session_name(path.name), []).append(path)

    print(
        f"Indexed {sum(len(v) for v in exact.values())} video files",
        flush=True,
    )

    return exact, normalized


def choose_match(row, exact, normalized):
    candidates = []

    for field in ["name", "original_name", "plan_video", "video_norm", "video"]:
        value = row.get(field)

        if pd.isna(value) or not str(value).strip():
            continue

        key = Path(str(value)).name.lower()
        candidates.extend(exact.get(key, []))

    if not candidates:
        for field in ["name", "plan_video", "video_norm", "video", "original_name"]:
            value = row.get(field)

            if pd.isna(value) or not str(value).strip():
                continue

            key = norm_session_name(str(value))
            candidates.extend(normalized.get(key, []))

    # Remove duplicates while preserving order.
    seen = set()
    unique = []

    for path in candidates:
        p = str(path.resolve()).lower()

        if p in seen:
            continue

        seen.add(p)
        unique.append(path)

    if not unique:
        return None, []

    # Prefer an MP4 and a normalized session-name match.
    target = norm_session_name(row.get("video_norm", row.get("video", "")))

    unique.sort(
        key=lambda p: (
            norm_session_name(p.name) != target,
            p.suffix.lower() != ".mp4",
            len(str(p)),
        )
    )

    return unique[0], unique


def cut_clip(ffmpeg, source, start_sec, duration_sec, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-ss", f"{float(start_sec):.3f}",
        "-i", str(source),
        "-t", f"{float(duration_sec):.3f}",
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]

    subprocess.run(command, check=True)


def main(args):
    selection = pd.read_csv(args.selection_csv)
    exact, normalized = build_file_index(args.video_root)

    ffmpeg = shutil.which("ffmpeg")

    if not args.dry_run and ffmpeg is None:
        raise RuntimeError(
            "ffmpeg was not found on PATH. Install ffmpeg or use --dry-run."
        )

    output_dir = Path(args.output_dir)
    clips_dir = output_dir / "segments"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    resolve_rows = []

    for i, row in selection.iterrows():
        source, all_matches = choose_match(row, exact, normalized)

        eval_id = int(row["eval_id"])
        video_name = str(row["video_norm"])
        start_sec = float(row["start_sec"])
        duration_sec = float(row.get("clip_duration_sec", 60.0))

        if source is None:
            print(f"[{i+1}/{len(selection)}] MISSING: {video_name}", flush=True)

            resolve_rows.append({
                "eval_id": eval_id,
                "video": video_name,
                "status": "missing",
                "source_video_path": "",
                "match_count": 0,
            })
            continue

        print(
            f"[{i+1}/{len(selection)}] {video_name} "
            f"segment {int(row['segment_id'])} -> {source}",
            flush=True,
        )

        resolve_rows.append({
            "eval_id": eval_id,
            "video": video_name,
            "status": "resolved",
            "source_video_path": str(source.resolve()),
            "match_count": len(all_matches),
        })

        clip_name = (
            f"{Path(video_name).stem}_eval_{eval_id:03d}_"
            f"{int(start_sec):06d}-{int(start_sec + duration_sec):06d}s.mp4"
        )
        clip_path = clips_dir / clip_name

        if not args.dry_run:
            if not clip_path.exists() or args.overwrite:
                cut_clip(
                    ffmpeg=ffmpeg,
                    source=source,
                    start_sec=start_sec,
                    duration_sec=duration_sec,
                    output_path=clip_path,
                )

        manifest_rows.append({
            "segment_idx": eval_id,
            "segment_path": str(clip_path.resolve()),
            "video": video_name,
            "patient_id": int(row["patient_id"]),
            "session_id": row["session_id"],
            "segment_id": int(row["segment_id"]),
            "segment_start_sec": start_sec,
            "segment_duration_sec": duration_sec,
        })

    pd.DataFrame(resolve_rows).to_csv(
        output_dir / "video_resolution_report.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not args.dry_run:
        pd.DataFrame(manifest_rows).to_csv(
            output_dir / "segments_manifest.csv",
            index=False,
            encoding="utf-8-sig",
        )

    missing = sum(r["status"] == "missing" for r in resolve_rows)

    print("", flush=True)
    print(f"Resolved: {len(resolve_rows) - missing}", flush=True)
    print(f"Missing: {missing}", flush=True)

    if not args.dry_run:
        print(
            f"VLM manifest: {output_dir / 'segments_manifest.csv'}",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--selection-csv", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    main(parser.parse_args())