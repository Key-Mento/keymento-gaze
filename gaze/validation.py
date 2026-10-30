"""신호음 검증 — 측정값이 실제로 내려다본 양에 비례하는지 확인한다.

녹화기가 무작위 시점에 신호음을 낸다.

    높은 삐  건반을 내려다본다
    낮은 삐  화면으로 돌아온다

블록마다 내려다보는 횟수를 다르게 하고(기본 4·8·12회, 블록 순서는 무작위),
한 번에 내려다보는 시간도 0.5~2.5초 사이에서 바꾼다. 신호를 실제로 낸
시각이 정답이 되고, 분석기가 정답과 측정값을 비교한다.

    검출률         신호마다 내려다본 것으로 잡혔는가
    반응 지연      신호부터 '건반'으로 바뀌기까지 (사람의 반응 시간 포함)
    거짓 내려다봄  신호가 없는데 잡힌 내려다봄 (분당)
    비례           지시 시간 ↔ 측정 시간 (신호마다)
                   지시 횟수 ↔ 측정 횟수, 지시 총 시간 ↔ 측정 총 시간 (블록마다)
                   의 상관 r 과 기울기 — 기울기 1 이면 같은 크기로 비례한다

녹화기(record_face.py)와 분석기(analyze.py)가 함께 쓴다. 카메라·MediaPipe
없이 도는 계산만 둔다.
"""

import math
import random
import statistics

import numpy as np

from gaze.classify import KEYS, MIN_DURATION
from gaze.metrics import gaze_metrics

COUNTS = (4, 8, 12)         # 블록별 내려다볼 횟수
BLOCK_SECONDS = 50.0        # 블록 하나의 길이
MIN_GAP = 1.5               # 화면으로 돌아온 뒤 다음 신호까지 최소 간격
DURATIONS = (0.5, 2.5)      # 한 번 내려다보는 시간의 범위
LEAD_IN = 3.0               # 시작 신호 뒤 첫 블록까지

# 신호 뒤 이 시간 안에 '건반'이 시작돼야 그 신호를 따른 것으로 본다.
RESPONSE = 1.5
# 올라오라는 신호 뒤에도 이만큼은 그 내려다봄의 몫으로 본다 (반응 지연).
LAG = 1.0


def make_schedule(seed, counts=COUNTS, block_seconds=BLOCK_SECONDS,
                  min_gap=MIN_GAP, durations=DURATIONS, lead_in=LEAD_IN):
    """검증 일정 — 블록 목록. 시각은 검증을 시작한 때 기준(초)."""
    rng = random.Random(seed)
    order = list(counts)
    rng.shuffle(order)

    blocks = []
    start = lead_in
    for number, count in enumerate(order, start=1):
        lengths = [round(rng.uniform(*durations), 1) for _ in range(count)]
        room = block_seconds - min_gap * (count + 1)
        if sum(lengths) > room:     # 블록에 다 안 들어가면 비율대로 줄인다
            lengths = [round(length * room / sum(lengths), 1)
                       for length in lengths]
        # 남는 시간을 신호 사이사이에 무작위로 나눠 시점을 예측하기 어렵게 한다.
        spare = max(0.0, room - sum(lengths))
        weights = [rng.random() for _ in range(count + 1)]
        total = sum(weights)
        extras = [spare * weight / total for weight in weights]

        cues = []
        cursor = start
        for index, length in enumerate(lengths):
            cursor += min_gap + extras[index]
            cues.append({"down": round(cursor, 3),
                         "up": round(cursor + length, 3)})
            cursor += length
        blocks.append({"block": number, "count": count, "start": start,
                       "end": start + block_seconds, "cues": cues})
        start += block_seconds
    return blocks


class CueRunner:
    """일정대로 신호를 낸다. 녹화 루프가 프레임마다 update() 를 부른다.

    신호를 실제로 낸 시각을 기록한다 — 분석의 정답은 계획한 시각이 아니라
    이 시각이다 (프레임 간격만큼 늦을 수 있다).

    Args:
        schedule: make_schedule() 결과.
        started:  검증을 시작한 PC 시각.
        on_cue:   신호를 낼 때 부르는 함수. 'down' / 'up' / 'done' 을 받는다.
        seed:     일정을 만든 시드 (기록용).
    """

    def __init__(self, schedule, started, on_cue, seed=None):
        self.started = started
        self.seed = seed
        self.on_cue = on_cue
        self.blocks = [dict(block, cues=[dict(cue) for cue in block["cues"]])
                       for block in schedule]
        self._queue = sorted(
            (cue[kind], kind, block_index, cue_index)
            for block_index, block in enumerate(self.blocks)
            for cue_index, cue in enumerate(block["cues"])
            for kind in ("down", "up"))
        self._next = 0
        self._end = self.blocks[-1]["end"]
        self.state = "screen"       # 지금 봐야 할 곳: 'screen' / 'keys'
        self.position = None        # (블록 번호, 블록 안 신호 번호)
        self.completed = False

    def update(self, now):
        elapsed = now - self.started
        while (self._next < len(self._queue)
               and self._queue[self._next][0] <= elapsed):
            _, kind, block_index, cue_index = self._queue[self._next]
            self._next += 1
            self.blocks[block_index]["cues"][cue_index][kind + "_at"] = now
            self.state = "keys" if kind == "down" else "screen"
            self.position = (block_index + 1, cue_index + 1)
            self.on_cue(kind)
        if not self.completed and elapsed >= self._end:
            self.completed = True
            self.on_cue("done")
        return self.state

    def describe(self):
        """미리보기에 띄울 한 줄 (ASCII)."""
        if self.completed:
            return "VALID: done"
        if self.position is None:
            return "VALID: starting"
        block, cue = self.position
        count = len(self.blocks[block - 1]["cues"])
        return f"VALID {block}/{len(self.blocks)}  cue {cue}/{count}"

    def record(self):
        """메타 파일에 남길 검증 기록 (PC 시각). 아직 안 낸 신호는 None."""
        return {
            "seed": self.seed,
            "started": self.started,
            "completed": self.completed,
            "blocks": [{
                "block": block["block"],
                "count": block["count"],
                "start": self.started + block["start"],
                "end": self.started + block["end"],
                "cues": [{"down": cue.get("down_at"), "up": cue.get("up_at"),
                          "planned_s": round(cue["up"] - cue["down"], 3)}
                         for cue in block["cues"]],
            } for block in self.blocks],
        }


def span(record):
    """검증이 차지한 구간 (시작, 끝) — 시선 분석에서 시도와 겹치지 않게 뺀다."""
    return record["started"], record["blocks"][-1]["end"] + LAG


def proportionality(planned, detected):
    """지시값 ↔ 측정값의 상관 r 과 최소제곱 기울기·절편."""
    xs = np.asarray(planned, dtype=float)
    ys = np.asarray(detected, dtype=float)
    keep = ~(np.isnan(xs) | np.isnan(ys))
    xs, ys = xs[keep], ys[keep]

    result = {"n": int(len(xs)), "r": None, "slope": None, "intercept": None}
    if len(xs) < 3 or np.std(xs) == 0:
        return result
    slope, intercept = np.polyfit(xs, ys, 1)
    result["slope"] = float(slope)
    result["intercept"] = float(intercept)
    if np.std(ys) > 0:
        result["r"] = float(np.corrcoef(xs, ys)[0, 1])
    return result


def _fired(record):
    """올라오라는 신호까지 낸 신호만 — (블록, 블록 안 번호, 신호)."""
    for block in record["blocks"]:
        for number, cue in enumerate(block["cues"], start=1):
            if cue.get("down") is not None and cue.get("up") is not None:
                yield block, number, cue


def evaluate(runs, record, response=RESPONSE, lag=LAG):
    """시선 구간(runs)을 검증 기록과 비교한다 → (신호별 줄, 블록별 줄, 요약)."""
    fired = list(_fired(record))
    if not fired:
        return [], [], None
    downs = sorted(cue["down"] for _, _, cue in fired)

    cue_rows = []
    windows = []
    for block, number, cue in fired:
        # 이 신호의 몫: 내려다보라는 신호부터 올라오라는 신호 + lag 까지.
        # 다음 신호가 그보다 먼저 오면 거기서 끊는다.
        later = [down for down in downs if down > cue["down"]]
        end = min(cue["up"] + lag, later[0] if later else math.inf)
        windows.append((cue["down"], end))
        gaze = gaze_metrics(runs, cue["down"], end)
        onset = next((run.start for run in runs
                      if run.label == KEYS
                      and cue["down"] <= run.start < cue["down"] + response),
                     None)
        cue_rows.append({
            "block": block["block"],
            "count": block["count"],
            "cue": number,
            "planned_s": cue["up"] - cue["down"],
            "detected_s": gaze["keys_s"],
            "hit": gaze["keys_s"] >= MIN_DURATION,
            "latency_s": None if onset is None else onset - cue["down"],
        })

    start = record["blocks"][0]["start"]
    stop = max(cue["up"] for _, _, cue in fired) + lag
    false_glances = sum(
        1 for run in runs
        if run.label == KEYS and start <= run.start < stop
        and not any(run.start < w_end and run.end > w_start
                    for w_start, w_end in windows))

    block_rows = []
    for block in record["blocks"]:
        cues = [cue for owner, _, cue in fired if owner is block]
        if not cues:
            continue
        gaze = gaze_metrics(runs, block["start"], block["end"] + lag)
        block_rows.append({
            "block": block["block"],
            "planned_glances": len(cues),
            "detected_glances": gaze["glances"],
            "planned_keys_s": sum(cue["up"] - cue["down"] for cue in cues),
            "detected_keys_s": gaze["keys_s"],
        })

    summary = summarize(cue_rows, block_rows, false_glances,
                        (stop - start) / 60)
    return cue_rows, block_rows, summary


def summarize(cue_rows, block_rows, false_glances, minutes):
    """신호·블록 줄(여러 녹화를 합친 것도 된다)의 요약."""
    latencies = [row["latency_s"] for row in cue_rows
                 if row["latency_s"] is not None]
    return {
        "cues": len(cue_rows),
        "hits": sum(1 for row in cue_rows if row["hit"]),
        "hit_rate": (statistics.fmean(1.0 if row["hit"] else 0.0
                                      for row in cue_rows)
                     if cue_rows else None),
        "false_glances": false_glances,
        "minutes": minutes,
        "false_per_min": false_glances / minutes if minutes > 0 else None,
        "latency_median_s": statistics.median(latencies) if latencies else None,
        "duration": proportionality([row["planned_s"] for row in cue_rows],
                                    [row["detected_s"] for row in cue_rows]),
        "count": proportionality(
            [row["planned_glances"] for row in block_rows],
            [row["detected_glances"] for row in block_rows]),
        "time": proportionality(
            [row["planned_keys_s"] for row in block_rows],
            [row["detected_keys_s"] for row in block_rows]),
    }
