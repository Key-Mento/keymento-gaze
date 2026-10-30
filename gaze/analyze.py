"""녹화한 얼굴 영상 → 시선 방향(화면/건반) 분류 → 시도별 지표·검증 결과.

실험이 끝난 뒤 이 저장소의 venv 로 돌린다 (저장소 폴더에서):
    run analyze                               data\\ 아래 녹화 전부
    run analyze data\\P01                      한 참가자
    run analyze data\\P01\\..._meta.json --review   확인용 영상까지

녹화마다 옆에 만드는 파일:
    _features.csv      프레임별 얼굴 특징 (다시 돌리면 재사용, --force 로 새로)
    _calibration.json  이 녹화에 쓴 판별 기준
    _gaze.csv          프레임별 점수(0=화면 자세 ~ 1=건반 자세)와 상태
    _notes.csv         음(걸음)별 소요 시간·오타·그 사이의 시선
    _validation.csv    (검증 녹화) 신호마다 지시 시간·측정 시간·검출 여부
    _review.mp4        (--review) 분류 결과를 얹은 확인용 영상
그리고 대상 폴더에
    results.csv            참가자 × 조건 × 시도마다 한 줄
    validation.csv         모든 검증 신호 (신호마다 한 줄)
    validation_blocks.csv  모든 검증 블록 (지시 횟수·시간 ↔ 측정)
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parents[1]
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

import numpy as np  # noqa: E402

from gaze import classify, metrics, validation  # noqa: E402
from gaze.recording import (  # noqa: E402
    DATA_DIR,
    RecordingFiles,
    complete_calibrations,
    find_recordings,
    read_events,
    read_frame_times,
    read_meta,
)

# 판정 이벤트 없이 분석할 때, 보정과 보정 사이에 이보다 짧게 남은 틈은
# 시도로 치지 않는다 — 보정을 다시 하느라 생긴 틈이다.
MIN_FREE_SPAN = 5.0

RESULT_COLUMNS = (
    "participant", "condition", "recording", "trial", "song", "mode", "speed",
    "status", "duration_s", "valid_ratio", "face_ratio",
    "screen_ratio", "glances", "glances_per_min",
    "keys_dwell_mean_s", "keys_dwell_median_s", "screen_dwell_mean_s",
    "screen_s", "keys_s", "unknown_s",
    "steps", "note_time_median_s", "steps_looked_down_ratio",
    "wrong_presses", "pitch_accuracy", "timing_accuracy", "overall",
    "calib_separation",
)

NOTE_COLUMNS = (
    "trial", "step", "t_s", "interval_s", "expected", "first_try", "grade",
    "diff_ms", "wrong_presses", "keys_s", "keys_ratio", "glances",
    "looked_down",
)

CUE_COLUMNS = (
    "participant", "recording", "block", "count", "cue", "planned_s",
    "detected_s", "hit", "latency_s",
)

BLOCK_COLUMNS = (
    "participant", "recording", "block", "planned_glances",
    "detected_glances", "planned_keys_s", "detected_keys_s",
)

# 확인용 영상 색 (BGR)
_COLORS = {
    classify.SCREEN: (90, 190, 60),
    classify.KEYS: (0, 140, 255),
    classify.UNKNOWN: (130, 130, 130),
}
_TEXT = {classify.SCREEN: "SCREEN", classify.KEYS: "KEYS",
         classify.UNKNOWN: "?"}


# ── CSV ───────────────────────────────────────────────────────────

def _cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return "nan" if math.isnan(value) else round(value, 4)
    return value


def write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_cell(row.get(column)) for column in columns])


def read_feature_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [{key: (int(value) if key in ("frame", "face")
                       else float(value))
                 for key, value in row.items()}
                for row in csv.DictReader(f)]


# ── 단계별 처리 ────────────────────────────────────────────────────

def _features_module():
    try:
        from gaze import features
    except ImportError as exc:
        sys.exit(f"[!] MediaPipe 를 불러올 수 없습니다 ({exc}).\n"
                 "    keymento-gaze 폴더에서 이 저장소의 venv 로 실행하세요:\n"
                 "      run setup      (처음 한 번)\n"
                 "      run analyze")
    return features


def load_features(files, meta, times, force=False):
    """프레임별 특징. 저장해 둔 것이 있으면 다시 뽑지 않는다."""
    path = files.path("features")
    if path.exists() and not force:
        return read_feature_csv(path)

    features = _features_module()
    video = files.video_path(meta)
    if not video.exists():
        raise FileNotFoundError(f"영상이 없습니다: {video}")

    def progress(done, total):
        print(f"\r  얼굴 특징 추출 {done}/{total} 프레임", end="", flush=True)

    rows = features.extract(video, times, on_progress=progress)
    print()
    write_csv(path, features.COLUMNS, rows)
    return rows


def _matrix(rows):
    return np.array([[row[name] for name in classify.FEATURES]
                     for row in rows], dtype=float)


def _segment_rows(rows, segment, trim):
    """보정 구간 [시작+trim, 끝) 의 특징 행렬. 신호를 듣고 시선을 옮기는
    처음 trim 초는 버린다."""
    start, end = segment
    return _matrix([row for row in rows if start + trim <= row["t"] < end])


def fit_all(rows, meta, trim):
    """녹화의 보정을 모두 맞춘다 → [(번호, 구간, Calibration)] (시각 순).

    맞추지 못한 보정은 알리고 건너뛴다.
    """
    segments = complete_calibrations(meta)
    if not segments:
        raise classify.CalibrationError(
            "보정 구간이 없습니다. 녹화 중 C 키로 보정했는지 확인하거나, "
            "같은 참가자의 다른 녹화로 보정하세요 (--calib-from).")

    fitted = []
    for number, segment in enumerate(segments, start=1):
        try:
            calibration = classify.fit_calibration(
                _segment_rows(rows, segment["screen"], trim),
                _segment_rows(rows, segment["keys"], trim))
        except classify.CalibrationError as exc:
            print(f"  [!] 보정 #{number}: {exc}")
            continue
        fitted.append((number, segment, calibration))

    if not fitted:
        raise classify.CalibrationError("쓸 수 있는 보정이 없습니다.")
    return fitted


def _active(times, fitted):
    """프레임마다 쓸 보정의 위치 — 그 시각 직전에 시작한 보정.

    자세가 바뀌어 보정을 다시 했다면 그 뒤로는 새 보정이 맞다. 첫 보정
    전의 프레임은 첫 보정을 쓴다.
    """
    starts = np.array([segment["screen"][0] for _, segment, _ in fitted])
    index = np.searchsorted(starts, np.atleast_1d(times), side="right") - 1
    return np.clip(index, 0, len(fitted) - 1)


def _free_spans(start, end, busy, min_length=MIN_FREE_SPAN):
    """[start, end) 에서 busy 구간(보정)을 뺀 나머지 중 min_length 이상."""
    spans = []
    cursor = start
    for busy_start, busy_end in sorted(busy):
        if min(busy_start, end) - cursor >= min_length:
            spans.append((cursor, min(busy_start, end)))
        cursor = max(cursor, busy_end)
    if end - cursor >= min_length:
        spans.append((cursor, end))
    return spans


def _trials_in_video(events, start, end, busy):
    """영상이 덮는 시도만 남기고 영상 범위로 자른다."""
    trials = []
    for trial in metrics.split_trials(events):
        if trial.end <= start or trial.start >= end:
            continue
        trial.start = max(trial.start, start)
        trial.end = min(trial.end, end)
        trials.append(trial)

    if not trials:
        # 판정 이벤트가 없으면 보정 구간을 뺀 나머지를 시도로 본다.
        trials = [metrics.Trial(index=index, start=span_start, end=span_end,
                                status="no-events")
                  for index, (span_start, span_end)
                  in enumerate(_free_spans(start, end, busy), start=1)]
    return trials


def _face_ratio(rows, start, end):
    inside = [row["face"] for row in rows if start <= row["t"] < end]
    return sum(inside) / len(inside) if inside else None


def _empty_result():
    return {"trials": [], "cues": [], "blocks": [], "validation": None}


def analyze_recording(meta_path, args):
    """녹화 하나를 분석한다.

    Returns:
        {"trials": 시도별 요약 줄, "cues": 검증 신호 줄, "blocks": 검증 블록 줄,
         "validation": 검증 요약 (검증 녹화가 아니면 None)}
    """
    result = _empty_result()
    files = RecordingFiles.from_meta(meta_path)
    meta = read_meta(files)
    title = f"{meta.get('participant', '?')} · {meta.get('condition', '?')}"
    print(f"\n━━ {title}  ({files.name}) ━━")

    times = read_frame_times(files)
    if len(times) < 2:
        print("  [!] 녹화된 프레임이 없습니다. 건너뜁니다.")
        return result

    rows = load_features(files, meta, times, force=args.force)
    if len(rows) < 2:
        print("  [!] 영상에서 프레임을 읽지 못했습니다. 건너뜁니다.")
        return result
    times = np.array([row["t"] for row in rows])
    face_ratio = sum(row["face"] for row in rows) / len(rows)
    print(f"  영상 {_clock(times[-1] - times[0])} · 얼굴 인식 "
          f"{face_ratio:.1%}")

    # 보정: 기본은 이 녹화의 보정들, --calib-from 이면 다른 녹화의 마지막 보정.
    try:
        if args.calib_from:
            other = RecordingFiles.from_meta(args.calib_from)
            other_meta = read_meta(other)
            other_rows = load_features(other, other_meta,
                                       read_frame_times(other),
                                       force=args.force)
            fitted = fit_all(other_rows, other_meta, args.calib_trim)[-1:]
            source = other.name
        else:
            fitted = fit_all(rows, meta, args.calib_trim)
            source = files.name
    except classify.CalibrationError as exc:
        print(f"  [!] {exc}")
        return result

    for number, segment, calibration in fitted:
        where = (f"{max(0.0, segment['screen'][0] - times[0]):.1f}초~"
                 if source == files.name else f"{source} 의 보정")
        print(f"  보정 #{number} ({where}): 화면 {calibration.n_screen} / "
              f"건반 {calibration.n_keys} 프레임 · 분리도 "
              f"d'={calibration.separation:.1f} ({calibration.quality()})")
        if calibration.separation < 2:
            print("  [!] 화면과 건반 자세가 잘 갈리지 않습니다. 보정 중에 "
                  "시선을 제대로 옮겼는지 _review.mp4 로 확인하세요.")
    if len(fitted) > 1:
        print("  보정을 여러 번 했습니다 — 각 시점에는 그 직전 보정을 씁니다.")

    matrix = _matrix(rows)
    active = _active(times, fitted)
    scores = np.full(len(rows), np.nan)
    closed = np.zeros(len(rows), dtype=bool)
    for position, (_, _, calibration) in enumerate(fitted):
        here = active == position
        scores[here] = calibration.score(matrix[here])
        closed[here] = calibration.closed(matrix[here])
    scores = classify.mask_blinks(times, scores, closed)
    labels = classify.label_frames(
        times, scores, enter=args.enter, leave=args.leave,
        max_gap=args.max_gap, min_duration=args.min_duration)
    runs = classify.to_runs(times, labels)

    write_csv(files.path("gaze"), ("frame", "t", "score", "label"),
              [{"frame": row["frame"], "t": row["t"], "score": float(score),
                "label": label}
               for row, score, label in zip(rows, scores, labels)])
    files.path("calibration").write_text(json.dumps({
        "features": list(classify.FEATURES),
        "source": source,
        "trim_s": args.calib_trim,
        "thresholds": {"enter": args.enter, "leave": args.leave,
                       "max_gap_s": args.max_gap,
                       "min_duration_s": args.min_duration,
                       "blink_max_s": classify.BLINK_MAX},
        "calibrations": [{"number": number, "segment": segment,
                          **calibration.to_dict()}
                         for number, segment, calibration in fitted],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # 보정·검증 구간은 시도에서 뺀다 (판정 이벤트가 없을 때만 쓰인다).
    own_segments = complete_calibrations(meta)
    busy = [(segment["screen"][0], segment["keys"][1])
            for segment in own_segments]
    record = meta.get("validation")
    if record:
        busy.append(validation.span(record))
    trials = _trials_in_video(read_events(files), times[0], runs[-1].end, busy)

    all_notes = []
    for trial in trials:
        summary, notes = metrics.trial_summary(trial, runs)
        calibration = fitted[int(_active(trial.start, fitted)[0])][2]
        summary.update({
            "participant": meta.get("participant"),
            "condition": meta.get("condition"),
            "recording": files.name,
            "face_ratio": _face_ratio(rows, trial.start, trial.end),
            "calib_separation": calibration.separation,
        })
        result["trials"].append(summary)
        all_notes.extend(notes)
        _report(trial, summary, times[0])
    if trials:
        write_csv(files.path("notes"), NOTE_COLUMNS, all_notes)

    if record:
        cue_rows, block_rows, summary = validation.evaluate(runs, record)
        if summary is None:
            print("  [!] 검증 신호를 하나도 내지 못한 녹화입니다.")
        else:
            tag = {"participant": meta.get("participant"),
                   "recording": files.name}
            result["cues"] = [{**tag, **row} for row in cue_rows]
            result["blocks"] = [{**tag, **row} for row in block_rows]
            result["validation"] = summary
            write_csv(files.path("validation"), CUE_COLUMNS, result["cues"])
            _report_validation(summary, block_rows)
    elif not trials:
        print("  [!] 보정 구간 말고는 분석할 구간이 없습니다.")

    if args.review:
        render_review(files, meta, rows, scores, labels, trials,
                      own_segments, record)
        print(f"  확인용 영상: {files.path('review')}")

    return result


# ── 출력 ─────────────────────────────────────────────────────────

def _clock(seconds):
    minutes, rest = divmod(max(0.0, seconds), 60)
    return f"{int(minutes)}분 {rest:04.1f}초" if minutes else f"{rest:.1f}초"


def _pct(value):
    return "-" if value is None else f"{value:.1%}"


def _num(value, unit="", digits=2):
    return "-" if value is None else f"{value:.{digits}f}{unit}"


_STATUS = {"done": "완주", "aborted": "중단", "error": "오류",
           "incomplete": "녹화 중 끝남", "no-events": "판정 기록 없음"}
_MODE = {"practice": "연습", "normal": "일반", "-": "-"}


def _report(trial, summary, origin):
    """시도 한 건을 콘솔에 요약한다. origin 은 녹화 첫 프레임 시각."""
    print(f"  [시도 {trial.index}] {summary['song']} · "
          f"{_MODE.get(summary['mode'], summary['mode'])} · "
          f"{_STATUS.get(trial.status, trial.status)} · "
          f"{_clock(summary['duration_s'])} "
          f"(녹화 {trial.start - origin:.1f}~{trial.end - origin:.1f}초)")
    print(f"     화면 주시 {_pct(summary['screen_ratio'])} · 내려다봄 "
          f"{summary['glances']}회 (분당 {_num(summary['glances_per_min'], '', 1)})"
          f" · 한 번에 평균 {_num(summary['keys_dwell_mean_s'], '초')}")

    performance = []
    if summary["pitch_accuracy"] is not None:
        performance.append(f"정확도 {summary['pitch_accuracy']}%")
    if summary["steps"]:
        performance.append(f"오타 {summary['wrong_presses']}회")
        performance.append(
            f"음당 {_num(summary['note_time_median_s'], '초')}(중앙값)")
    if performance:
        print("     " + " · ".join(performance))

    if summary["valid_ratio"] is not None and summary["valid_ratio"] < 0.8:
        print(f"     [!] 시선을 판별하지 못한 시간이 "
              f"{1 - summary['valid_ratio']:.0%} 입니다. "
              f"카메라 위치·조명을 확인하세요.")


def _fit_line(fit, unit=""):
    """비례 결과 한 줄 — r 과 기울기·절편."""
    if fit["slope"] is None:
        return f"(값이 {fit['n']}개뿐이라 계산하지 않음)"
    r = "-" if fit["r"] is None else f"{fit['r']:.2f}"
    return (f"r = {r} · 기울기 {fit['slope']:.2f} · "
            f"절편 {fit['intercept']:.2f}{unit}")


def _report_validation(summary, block_rows, title="[검증]", indent="  "):
    """검증 결과를 콘솔에 요약한다. 기울기 1·절편 0 에 가까울수록 측정값이
    지시한 양과 같은 크기로 비례한다."""
    print(f"{indent}{title} 신호 {summary['cues']}개 중 {summary['hits']}개 "
          f"검출 ({_pct(summary['hit_rate'])}) · 거짓 내려다봄 "
          f"{_num(summary['false_per_min'], '회/분', 1)} · 반응 지연 "
          f"{_num(summary['latency_median_s'], '초')}(중앙값)")
    print(f"{indent}   시간 비례 (신호 {summary['duration']['n']}개): "
          f"{_fit_line(summary['duration'], '초')}")
    if len(block_rows) <= 3:        # 한 사람이면 블록을 그대로 보여 준다
        pairs = " · ".join(
            f"{row['planned_glances']}회/{row['planned_keys_s']:.1f}초 → "
            f"{row['detected_glances']}회/{row['detected_keys_s']:.1f}초"
            for row in block_rows)
        print(f"{indent}   블록별 지시 → 측정: {pairs}")
    else:
        print(f"{indent}   횟수 비례 (블록 {summary['count']['n']}개): "
              f"{_fit_line(summary['count'], '회')}")
        print(f"{indent}   총 시간 비례 (블록 {summary['time']['n']}개): "
              f"{_fit_line(summary['time'], '초')}")


def render_review(files, meta, rows, scores, labels, trials, segments,
                  record=None):
    """분류 결과를 얹은 확인용 영상 — 위: 지금 상태, 아래: 앞뒤 10초 띠.

    segments 는 이 녹화의 보정 구간 목록 (CALIB 으로 표시), record 는 검증
    기록 (내려다보라는 신호 동안 VALID DOWN 으로 표시).
    """
    import cv2

    video = files.video_path(meta)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  [!] 영상을 열 수 없습니다: {video}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    strip = 36
    writer = cv2.VideoWriter(str(files.path("review")),
                             cv2.VideoWriter.fourcc(*"mp4v"),
                             float(meta.get("fps") or 30),
                             (width, height + strip))

    times = np.array([row["t"] for row in rows])
    order = (classify.SCREEN, classify.KEYS, classify.UNKNOWN)
    codes = np.array([order.index(label) for label in labels])
    palette = np.array([_COLORS[label] for label in order], dtype=np.uint8)
    span = 20.0         # 아래 띠가 보여 주는 시간 폭(초)
    offsets = (np.arange(width) / width - 0.5) * span

    cues = [(cue["down"], cue["up"])
            for block in (record or {}).get("blocks", [])
            for cue in block["cues"]
            if cue.get("down") is not None and cue.get("up") is not None]
    valid_span = validation.span(record) if record else None

    def phase_at(t):
        for number, segment in enumerate(segments, start=1):
            if segment["screen"][0] <= t < segment["screen"][1]:
                return f"CALIB #{number} SCREEN"
            if segment["keys"][0] <= t < segment["keys"][1]:
                return f"CALIB #{number} KEYS"
        if valid_span and valid_span[0] <= t < valid_span[1]:
            down = any(start <= t < end for start, end in cues)
            return "VALID DOWN" if down else "VALID"
        for trial in trials:
            if trial.start <= t < trial.end:
                return f"TRIAL {trial.index}"
        return ""

    font = cv2.FONT_HERSHEY_SIMPLEX
    try:
        for index, label in enumerate(labels):
            ok, frame = cap.read()
            if not ok:
                break
            t = times[index]
            color = _COLORS[label]

            cv2.rectangle(frame, (0, 0), (width, 30), color, -1)
            cv2.putText(frame, _TEXT[label], (10, 22), font, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
            score = scores[index]
            if not math.isnan(score):
                # 점수 막대: 왼쪽 끝 0(화면 자세) ~ 오른쪽 끝 1(건반 자세)
                left, right = width - 170, width - 10
                x = int(left + (right - left) * min(max(score, 0.0), 1.0))
                cv2.line(frame, (left, 15), (right, 15), (255, 255, 255), 1)
                cv2.circle(frame, (x, 15), 6, (255, 255, 255), -1)
            cv2.putText(frame, f"{t - times[0]:7.2f}s  {phase_at(t)}",
                        (110, 22), font, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)

            positions = np.searchsorted(times, t + offsets, side="right") - 1
            band = palette[codes[np.clip(positions, 0, len(codes) - 1)]]
            band[(positions < 0) | (t + offsets > times[-1])] = 0
            bar = np.repeat(band[np.newaxis, :, :], strip, axis=0)
            cv2.line(bar, (width // 2, 0), (width // 2, strip), (255, 255, 255), 2)

            writer.write(np.vstack([frame, bar]))
    finally:
        cap.release()
        writer.release()


# ── 진입점 ───────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="녹화한 얼굴 영상으로 화면/건반 시선을 분류하고 지표를 낸다")
    parser.add_argument("target", nargs="?", default=str(DATA_DIR),
                        help="메타 파일(_meta.json) 하나 또는 폴더 "
                             "(기본: data)")
    parser.add_argument("--out", default=None,
                        help="요약 CSV 경로 (기본: 대상 폴더의 results.csv)")
    parser.add_argument("--review", action="store_true",
                        help="분류 결과를 얹은 확인용 영상(_review.mp4)도 만든다")
    parser.add_argument("--force", action="store_true",
                        help="저장된 특징(_features.csv)을 무시하고 새로 뽑는다")
    parser.add_argument("--calib-from", default=None, metavar="META",
                        help="다른 녹화(_meta.json)의 보정 구간을 쓴다 "
                             "— 보정을 빠뜨렸을 때")
    parser.add_argument("--calib-trim", type=float, default=1.0,
                        help="보정 구간 앞에서 버릴 초 — 신호를 듣고 시선을 "
                             "옮기는 시간 (기본: 1.0)")
    parser.add_argument("--enter", type=float, default=classify.ENTER,
                        help=f"화면→건반으로 바뀌는 점수 (기본: {classify.ENTER})")
    parser.add_argument("--leave", type=float, default=classify.LEAVE,
                        help=f"건반→화면으로 바뀌는 점수 (기본: {classify.LEAVE})")
    parser.add_argument("--min-duration", type=float,
                        default=classify.MIN_DURATION,
                        help="이보다 짧은 시선 구간은 흡수한다(초) "
                             f"(기본: {classify.MIN_DURATION})")
    parser.add_argument("--max-gap", type=float, default=classify.MAX_GAP,
                        help="이보다 짧게 얼굴을 놓친 구간은 앞 상태로 "
                             f"메운다(초) (기본: {classify.MAX_GAP})")
    return parser.parse_args()


def main():
    args = parse_args()
    target = Path(args.target)
    recordings = find_recordings(target)
    if not recordings:
        print(f"녹화가 없습니다: {target}")
        print("먼저 run record --participant P01 --condition AR 로 녹화하세요.")
        return

    collected = _empty_result()
    validations = []
    for meta_path in recordings:
        try:
            result = analyze_recording(meta_path, args)
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            print(f"  [!] 분석 실패: {exc}")
            continue
        for key in ("trials", "cues", "blocks"):
            collected[key].extend(result[key])
        if result["validation"] is not None:
            validations.append(result["validation"])

    folder = target if target.is_dir() else target.parent
    if collected["trials"]:
        out = Path(args.out) if args.out else folder / "results.csv"
        write_csv(out, RESULT_COLUMNS, collected["trials"])
        print(f"\n시도 요약 {len(collected['trials'])}줄 → {out}")

    if validations:
        write_csv(folder / "validation.csv", CUE_COLUMNS, collected["cues"])
        write_csv(folder / "validation_blocks.csv", BLOCK_COLUMNS,
                  collected["blocks"])
        pooled = validation.summarize(
            collected["cues"], collected["blocks"],
            sum(item["false_glances"] for item in validations),
            sum(item["minutes"] for item in validations))
        print(f"\n━━ 검증 종합 (녹화 {len(validations)}개) ━━")
        _report_validation(pooled, collected["blocks"], title="전체")
        print(f"검증 신호 {len(collected['cues'])}줄 → "
              f"{folder / 'validation.csv'}")

    if not collected["trials"] and not validations:
        print("\n요약할 결과가 없습니다.")


if __name__ == "__main__":
    main()
