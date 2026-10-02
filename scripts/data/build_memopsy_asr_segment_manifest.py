#!/usr/bin/env python3

import argparse
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd


KEY = ["patient_id", "session_id", "video", "segment_id"]
TARGETS = ["WD_P", "WD_T", "CF_P", "CF_T", "RE"]


def normalize_coder(x):
    x = str(x).strip()
    return re.sub(r"^segments?\s+", "", x, flags=re.I)


def parse_time(x):
    """Accept seconds, MM:SS or HH:MM:SS."""
    s = str(x).strip().replace(",", ".")

    try:
        return float(s)
    except ValueError:
        pass

    p = [float(v) for v in s.split(":")]

    if len(p) == 2:
        return p[0] * 60 + p[1]

    if len(p) == 3:
        return p[0] * 3600 + p[1] * 60 + p[2]

    raise ValueError(f"Cannot parse time: {x}")


def build_video_index(root):
    root = Path(root)

    extensions = {
        ".mp4", ".mov", ".mkv",
        ".avi", ".wmv", ".m4v",
    }

    index = {}

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            index.setdefault(
                path.name.casefold(),
                path.resolve(),
            )

    return index


def extract_wav(source, start, output, overwrite=False):

    if output.exists() and not overwrite:
        return

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",

            "-ss", f"{start:.3f}",
            "-i", str(source),

            "-t", "60",

            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",

            str(output),
        ],
        check=True,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ratings-csv",
        required=True,
    )

    parser.add_argument(
        "--plan-csv",
        required=True,
    )

    parser.add_argument(
        "--video-root",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=2,
    )

    parser.add_argument(
        "--extract-wav",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--expected-segments",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    audio_dir = (
        output_dir
        / "audio_segments"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Load
    # --------------------------------------------------

    ratings = pd.read_csv(
        args.ratings_csv,
        dtype=str,
        encoding="utf-8-sig",
    ).fillna("")

    plan = pd.read_csv(
        args.plan_csv,
        dtype=str,
        encoding="utf-8-sig",
    ).fillna("")

    # --------------------------------------------------
    # Keep selected planning videos
    # --------------------------------------------------

    if "video" not in plan.columns:
        raise ValueError(
            "Planning CSV needs a video column."
        )

    # Remove explicit excludes if present.
    if "final_decision" in plan.columns:

        excluded = {
            "exclude",
            "excluded",
            "reject",
            "rejected",
        }

        plan = plan[
            ~plan["final_decision"]
            .str.strip()
            .str.casefold()
            .isin(excluded)
        ]

    if "manual_decision" in plan.columns:

        plan = plan[
            ~plan["manual_decision"]
            .str.strip()
            .str.casefold()
            .isin({"exclude", "excluded"})
        ]

    selected_videos = set(
        plan["video"]
        .map(
            lambda x:
            Path(str(x)).name.casefold()
        )
    )

    ratings["_video_norm"] = (
        ratings["video"]
        .map(
            lambda x:
            Path(str(x)).name.casefold()
        )
    )

    ratings = ratings[
        ratings["_video_norm"]
        .isin(selected_videos)
    ].copy()

    # --------------------------------------------------
    # Normalize coder names
    # --------------------------------------------------

    ratings["coder"] = (
        ratings["coder"]
        .map(normalize_coder)
    )

    # Prevent duplicate rows from one coder.
    ratings = ratings.drop_duplicates(
        KEY + ["coder"],
        keep="last",
    )

    # --------------------------------------------------
    # Exactly TWO distinct raters
    # --------------------------------------------------

    counts = (
        ratings
        .groupby(KEY, dropna=False)["coder"]
        .nunique()
        .rename("n_raters")
        .reset_index()
    )

    exact_two = counts[
        counts["n_raters"] == 2
    ][KEY]

    ratings = ratings.merge(
        exact_two,
        on=KEY,
        how="inner",
    )

    # --------------------------------------------------
    # Build one row per physical segment
    # --------------------------------------------------

    segment_rows = []

    for key, group in ratings.groupby(
        KEY,
        sort=True,
        dropna=False,
    ):

        group = (
            group
            .sort_values("coder")
            .reset_index(drop=True)
        )

        (
            patient_id,
            session_id,
            video,
            segment_id,
        ) = key

        segment_id = int(
            float(segment_id)
        )

        start_sec = parse_time(
            group.loc[
                0,
                "segment_start",
            ]
        )

        segment_uid = (
            f"{patient_id}_"
            f"{session_id}_"
            f"seg{segment_id:03d}"
        )

        row = {
            "segment_uid":
                segment_uid,

            "patient_id":
                patient_id,

            "session_id":
                session_id,

            "video":
                Path(str(video)).name,

            "segment_id":
                segment_id,

            "segment_start_sec":
                start_sec,

            "segment_duration_sec":
                60.0,

            "coder_1":
                group.loc[0, "coder"],

            "coder_2":
                group.loc[1, "coder"],
        }

        # ----------------------------------------------
        # Keep labels separately for later LLM analysis
        # ----------------------------------------------

        for target in TARGETS:

            if target not in group.columns:
                continue

            values = pd.to_numeric(
                group[target],
                errors="coerce",
            )

            if (
                len(values) < 2
                or values.iloc[:2].isna().any()
            ):
                continue

            r1 = float(values.iloc[0])
            r2 = float(values.iloc[1])

            row[f"{target}_rater1"] = r1
            row[f"{target}_rater2"] = r2
            row[f"{target}_mean"] = (
                r1 + r2
            ) / 2

            if target in {
                "WD_P",
                "WD_T",
                "CF_P",
                "CF_T",
            }:

                b1 = int(
                    r1 >= args.threshold
                )

                b2 = int(
                    r2 >= args.threshold
                )

                soft = (
                    b1 + b2
                ) / 2

                row[
                    f"{target}_soft_thr2"
                ] = soft

                row[
                    f"{target}_consensus_thr2"
                ] = int(
                    soft in {0, 1}
                )

        segment_rows.append(row)

    labels = pd.DataFrame(
        segment_rows
    )

    labels = labels.sort_values(
        [
            "patient_id",
            "session_id",
            "video",
            "segment_start_sec",
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("MEMOPSY TRANSCRIPTION DATASET")
    print("=" * 70)

    print(
        "Exactly-two-rater segments:",
        len(labels),
    )

    print(
        "Patients:",
        labels["patient_id"].nunique(),
    )

    print(
        "Videos:",
        labels["video"].nunique(),
    )

    if args.expected_segments:

        if len(labels) != args.expected_segments:

            print(
                f"WARNING: expected "
                f"{args.expected_segments}, "
                f"found {len(labels)}"
            )

    if "WD_P_soft_thr2" in labels.columns:

        print()
        print(
            "WD_P soft labels:"
        )

        print(
            labels[
                "WD_P_soft_thr2"
            ]
            .value_counts()
            .sort_index()
        )

        print(
            "Consensus WD_P:",
            int(
                labels[
                    "WD_P_consensus_thr2"
                ].sum()
            ),
        )

    # Important:
    # save labels separately from ASR manifest.
    labels.to_csv(
        output_dir
        / "segment_labels.csv",
        index=False,
        encoding="utf-8-sig",
    )

    consensus = labels[
        labels[
            "WD_P_consensus_thr2"
        ] == 1
    ]

    consensus[
        [
            "segment_uid",
            "patient_id",
            "session_id",
            "video",
            "segment_id",
        ]
    ].to_csv(
        output_dir
        / "consensus_only_ids.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------
    # Resolve videos
    # --------------------------------------------------

    print()
    print("Indexing source videos...")

    videos = build_video_index(
        args.video_root
    )

    print(
        "Video files indexed:",
        len(videos),
    )

    if (
        args.extract_wav
        and shutil.which("ffmpeg") is None
    ):
        raise RuntimeError(
            "ffmpeg is not available on PATH."
        )

    # --------------------------------------------------
    # Build label-free ASR manifest
    # --------------------------------------------------

    asr_rows = []
    missing_rows = []

    for i, row in enumerate(
        labels.itertuples(index=False),
        start=1,
    ):

        source = videos.get(
            row.video.casefold()
        )

        if source is None:

            missing_rows.append(
                {
                    "segment_uid":
                        row.segment_uid,

                    "video":
                        row.video,
                }
            )

            continue

        start = float(
            row.segment_start_sec
        )

        end = start + 60

        wav_name = (
            f"{Path(row.video).stem}_"
            f"seg{row.segment_id:03d}_"
            f"{int(start):06d}-"
            f"{int(end):06d}s.wav"
        )

        wav_path = (
            audio_dir
            / wav_name
        ).resolve()

        if args.extract_wav:

            extract_wav(
                source=source,
                start=start,
                output=wav_path,
                overwrite=args.overwrite,
            )

        asr_rows.append(
            {
                "group":
                    "MeMoPsy",

                "participant_id":
                    row.patient_id,

                "audio_file":
                    wav_name,

                "audio_path":
                    str(wav_path),

                "segment_uid":
                    row.segment_uid,

                "video":
                    row.video,

                "session_id":
                    row.session_id,

                "segment_id":
                    row.segment_id,

                "segment_start_sec":
                    start,

                "segment_duration_sec":
                    60.0,

                "source_video_path":
                    str(source),

                "model_transcribed":
                    "no",

                "model_transcript_path":
                    "",

                "manually_corrected":
                    "no",

                "corrected_transcript_path":
                    "",

                "corrected_transcript_check":
                    "not_found",

                "corrected_cue_count":
                    0,

                "corrected_nonempty_text_lines":
                    0,

                "model_output_dir":
                    "",

                "model_output_status":
                    "",

                "model_transcribed_at":
                    "",
            }
        )

        if i % 100 == 0:

            print(
                f"{i}/{len(labels)} "
                "segments prepared"
            )

    asr = pd.DataFrame(
        asr_rows
    )

    asr.to_csv(
        output_dir
        / "asr_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        missing_rows
    ).to_csv(
        output_dir
        / "missing_media.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 70)

    print(
        "ASR manifest:",
        output_dir
        / "asr_manifest.csv",
    )

    print(
        "Labels:",
        output_dir
        / "segment_labels.csv",
    )

    print(
        "Missing media:",
        len(missing_rows),
    )

    if not args.extract_wav:

        print()
        print(
            "This was a dry manifest build."
        )

        print(
            "If the counts look correct, "
            "run again with --extract-wav."
        )


if __name__ == "__main__":
    main()