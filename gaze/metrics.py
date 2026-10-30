"""판정 이벤트 + 시선 구간 → 시도(trial)별 지표와 음별 표.

시도 하나는 웹 UI 에서 연주를 시작해 'start' 이벤트가 나온 때부터
'done'(완주)·'aborted'(중단)까지다. 카운트다운은 넣지 않는다.

시선 지표 (화면/건반으로 판별된 시간만 분모에 넣는다)
    screen_ratio         화면을 본 시간 비율
    glances              화면 → 건반 전환 횟수 (내려다본 횟수)
    glances_per_min      분당 내려다본 횟수
    keys_dwell_mean_s    한 번 내려다볼 때 머문 평균 시간

음별 표의 한 줄은 '한 걸음'이다 — 연습 모드는 정답을 쳐서 다음 음으로
넘어간 때, 일반 모드는 음을 치거나(Hit) 놓친(Miss) 때. 직전 걸음부터
이번 걸음까지를 그 음의 구간으로 보고 그 사이의 시선을 붙인다.
"""

import statistics
from dataclasses import dataclass, field

from gaze.classify import KEYS, MIN_DURATION, SCREEN, UNKNOWN

_END_TYPES = {"done", "aborted", "error"}


@dataclass
class Trial:
    index: int
    start: float
    end: float = None
    status: str = "incomplete"      # done | aborted | error | incomplete
    session: dict = field(default_factory=dict)
    result: dict = None
    events: list = field(default_factory=list)

    @property
    def practice(self):
        """연습 모드면 True, 일반 모드면 False, 판정 이벤트가 없으면 None."""
        start = self.events[0] if self.events else {}
        if "practice" in start:
            return bool(start["practice"])
        if "practice" in self.session:
            return bool(self.session["practice"])
        return None

    @property
    def mode(self):
        if self.practice is None:
            return "-"
        return "practice" if self.practice else "normal"

    @property
    def song(self):
        return self.session.get("song") or "-"


def split_trials(events):
    """시각 순 이벤트 목록 → 시도 목록."""
    trials = []
    session = {}
    current = None

    for event in events:
        kind = event.get("type")
        if event.get("t") is None:
            continue
        if kind == "session":
            session = event
            continue
        if kind == "start":
            if current is not None:     # 끝 이벤트 없이 다시 시작됐다
                current.end = current.events[-1]["t"]
                trials.append(current)
            current = Trial(index=0, start=event["t"], session=session,
                            events=[event])
            continue
        if current is None:
            continue                    # 카운트다운, 시작 전 오류 등
        current.events.append(event)
        if kind in _END_TYPES:
            current.end = event["t"]
            current.status = kind
            current.result = event.get("result")
            trials.append(current)
            current = None
            session = {}

    if current is not None:             # 녹화가 연주 도중에 끝났다
        current.end = current.events[-1]["t"]
        trials.append(current)

    for index, trial in enumerate(trials, start=1):
        trial.index = index
    return trials


def _mean(values):
    return statistics.fmean(values) if values else None


def _median(values):
    return statistics.median(values) if values else None


def gaze_metrics(runs, start, end):
    """[start, end) 구간의 시선 지표. runs 는 시각 순 구간 목록."""
    totals = {SCREEN: 0.0, KEYS: 0.0, UNKNOWN: 0.0}
    key_dwells = []
    screen_dwells = []
    glances = 0
    last_known = None

    for run in runs:
        if run.end <= start:
            if run.label != UNKNOWN:
                last_known = run.label
            continue
        if run.start >= end:
            break

        overlap = min(run.end, end) - max(run.start, start)
        totals[run.label] += overlap
        # 구간 안에서 시작한 것만 '한 번 본 것'으로 센다 — 앞에서 이어진
        # 시선은 이전 구간의 몫이다.
        if run.start >= start:
            if run.label == KEYS:
                key_dwells.append(overlap)
                if last_known == SCREEN:
                    glances += 1
            elif run.label == SCREEN:
                screen_dwells.append(overlap)
        if run.label != UNKNOWN:
            last_known = run.label

    duration = end - start
    valid = totals[SCREEN] + totals[KEYS]
    return {
        "duration_s": duration,
        "screen_s": totals[SCREEN],
        "keys_s": totals[KEYS],
        "unknown_s": totals[UNKNOWN],
        "valid_ratio": valid / duration if duration > 0 else None,
        "screen_ratio": totals[SCREEN] / valid if valid > 0 else None,
        "glances": glances,
        "glances_per_min": glances / (valid / 60) if valid > 0 else None,
        "keys_dwell_mean_s": _mean(key_dwells),
        "keys_dwell_median_s": _median(key_dwells),
        "screen_dwell_mean_s": _mean(screen_dwells),
    }


def _finishes_step(event, practice):
    if event.get("type") != "note":
        return False
    if practice:
        # 화음을 덜 친 중간 이벤트에는 first_try 가 없다.
        return bool(event.get("pitch_ok")) and "first_try" in event
    return bool(event.get("pitch_ok")) or bool(event.get("timed_out"))


def _is_wrong_press(event):
    return (event.get("type") == "note" and not event.get("pitch_ok")
            and not event.get("timed_out"))


def note_rows(trial, runs):
    """음(걸음)마다 소요 시간·오타와 그 사이의 시선."""
    rows = []
    previous = trial.start
    wrong = 0

    for event in trial.events:
        if _is_wrong_press(event):
            wrong += 1
        if not _finishes_step(event, trial.practice):
            continue

        window = gaze_metrics(runs, previous, event["t"])
        valid = window["screen_s"] + window["keys_s"]
        rows.append({
            "trial": trial.index,
            "step": len(rows) + 1,
            "t_s": event["t"] - trial.start,
            "interval_s": event["t"] - previous,
            "expected": event.get("expected"),
            "first_try": event.get("first_try"),
            "grade": event.get("grade"),
            "diff_ms": event.get("diff_ms"),
            "wrong_presses": wrong,
            "keys_s": window["keys_s"],
            "keys_ratio": window["keys_s"] / valid if valid > 0 else None,
            "glances": window["glances"],
            "looked_down": window["keys_s"] >= MIN_DURATION,
        })
        previous = event["t"]
        wrong = 0
    return rows


def trial_summary(trial, runs):
    """시도 하나의 요약 한 줄과 음별 표를 함께 돌려준다."""
    gaze = gaze_metrics(runs, trial.start, trial.end)
    notes = note_rows(trial, runs)
    result = trial.result or {}
    intervals = [row["interval_s"] for row in notes]

    summary = {
        "trial": trial.index,
        "song": trial.song,
        "mode": trial.mode,
        "speed": trial.session.get("speed"),
        "status": trial.status,
        **gaze,
        "steps": len(notes),
        "note_time_median_s": _median(intervals),
        "steps_looked_down_ratio": (
            _mean([1.0 if row["looked_down"] else 0.0 for row in notes])),
        "wrong_presses": sum(1 for event in trial.events
                             if _is_wrong_press(event)),
        "pitch_accuracy": result.get("pitch_accuracy"),
        "timing_accuracy": result.get("timing_accuracy"),
        "overall": result.get("overall"),
    }
    return summary, notes
