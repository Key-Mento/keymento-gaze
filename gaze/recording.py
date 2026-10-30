"""녹화 한 건을 이루는 파일들의 이름과 읽기/쓰기.

녹화 하나는 같은 이름 줄기(stem)를 공유하는 파일 묶음이다.

    data/P01/P01_AR_20261029-101500_meta.json     참가자·조건·보정·검증 일정
             P01_AR_20261029-101500_face.mp4      얼굴 영상
             P01_AR_20261029-101500_frames.csv    프레임별 PC 시각
             P01_AR_20261029-101500_events.jsonl  판정 이벤트

분석(analyze.py)이 같은 줄기로 결과 파일(_features.csv, _gaze.csv,
_notes.csv, _validation.csv, _calibration.json, _review.mp4)을 옆에 만든다.
"""

import csv
import json
import re
from datetime import datetime
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parents[1]      # 저장소 폴더
DATA_DIR = TOOL_DIR / "data"

META_SUFFIX = "_meta.json"

_SUFFIXES = {
    "meta": META_SUFFIX,
    "video": "_face.mp4",
    "frames": "_frames.csv",
    "events": "_events.jsonl",
    "features": "_features.csv",
    "gaze": "_gaze.csv",
    "notes": "_notes.csv",
    "validation": "_validation.csv",
    "calibration": "_calibration.json",
    "review": "_review.mp4",
}


def _slug(text):
    """파일 이름에 넣을 수 있게 다듬는다 (한글은 그대로 둔다)."""
    cleaned = re.sub(r"[^\w-]+", "_", str(text).strip(), flags=re.UNICODE)
    return cleaned.strip("_") or "x"


class RecordingFiles:
    """녹화 한 건의 파일 경로 묶음."""

    def __init__(self, stem):
        self.stem = Path(stem)

    @classmethod
    def new(cls, participant, condition, root=DATA_DIR, when=None):
        """새 녹화의 이름 줄기를 만든다. 참가자마다 폴더를 나눈다."""
        when = when or datetime.now()
        participant = _slug(participant)
        name = (f"{participant}_{_slug(condition)}_"
                f"{when.strftime('%Y%m%d-%H%M%S')}")
        return cls(Path(root) / participant / name)

    @classmethod
    def from_meta(cls, meta_path):
        meta_path = Path(meta_path)
        if not meta_path.name.endswith(META_SUFFIX):
            raise ValueError(f"메타 파일이 아닙니다: {meta_path}")
        return cls(meta_path.with_name(meta_path.name[:-len(META_SUFFIX)]))

    @property
    def name(self):
        return self.stem.name

    def path(self, kind):
        return self.stem.with_name(self.stem.name + _SUFFIXES[kind])

    def video_path(self, meta=None):
        """영상 경로. 코덱 대체로 확장자가 바뀌었으면 메타에 적힌 이름을 쓴다."""
        if meta and meta.get("video"):
            return self.stem.parent / meta["video"]
        return self.path("video")


def find_recordings(target):
    """메타 파일 경로 목록. target 은 메타 파일 하나 또는 폴더."""
    target = Path(target)
    if target.is_file():
        return [target]
    return sorted(target.rglob(f"*{META_SUFFIX}"))


def write_meta(files, meta):
    path = files.path("meta")
    path.parent.mkdir(parents=True, exist_ok=True)
    # 중간에 꺼져도 깨진 JSON 이 남지 않게 임시 파일에 쓴 뒤 바꿔 끼운다.
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(path)


def read_meta(files):
    return json.loads(files.path("meta").read_text(encoding="utf-8"))


def read_frame_times(files):
    """프레임 번호 순서대로 PC 시각(초) 목록."""
    with open(files.path("frames"), newline="", encoding="utf-8") as f:
        return [float(row["t"]) for row in csv.DictReader(f)]


def read_events(files):
    """판정 이벤트 목록 (시각 순). 파일이 없으면 빈 목록."""
    path = files.path("events")
    if not path.exists():
        return []

    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue    # 녹화가 끊기며 잘린 마지막 줄
    events.sort(key=lambda event: event.get("t", 0.0))
    return events


def complete_calibrations(meta):
    """끝까지 마친 보정 목록 (시각 순). [{'screen': [시작, 끝], 'keys': [...]}]"""
    return [c for c in meta.get("calibrations", [])
            if c.get("screen") and c.get("keys")]
