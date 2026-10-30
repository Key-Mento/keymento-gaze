"""얼굴 특징 → 프레임별 시선 방향(화면 / 건반) 분류.

참가자마다 얼굴·자세·카메라 위치가 달라 고정 임계값을 쓸 수 없다. 녹화
첫머리의 보정 구간(화면 5초, 건반 5초)으로 그 사람의 두 자세를 가장 잘
가르는 선형 판별 방향(LDA)을 구하고, 모든 프레임을 그 축에 투영해
0(화면 자세) ~ 1(건반 자세) 점수로 바꾼다.

    점수 → 짧은 눈 감김(깜빡임) 비우기
         → 이력(hysteresis) 임계값으로 상태 결정
         → 짧은 끊김(깜빡임·얼굴 놓침) 메우기
         → 너무 짧은 구간(떨림) 흡수
         → 구간(Run) 목록

numpy 만 쓴다. 특징 추출(features.py, MediaPipe)과 떼어 두어 카메라·모델
없이 단위 테스트할 수 있다.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np

SCREEN = "screen"
KEYS = "keys"
UNKNOWN = "unknown"

# 판별에 쓰는 특징 — 모두 아래를 볼수록 한쪽으로 움직이는 값이다.
#   pitch      머리 끄덕임 각도(도)
#   eye_down   눈동자 아래 보기 정도 (blendshape: 아래 - 위)
#   iris_drop  눈꼬리 선 대비 홍채 중심의 높이 (눈 너비로 나눔)
#   eye_open   눈꺼풀 사이 간격 (눈 너비로 나눔) — 내려다보면 좁아진다
FEATURES = ("pitch", "eye_down", "iris_drop", "eye_open")
_OPEN = FEATURES.index("eye_open")

# 표준화할 때 거의 안 움직이는 특징이 터지지 않게 두는 최소 표준편차
_MIN_SCALE = np.array([0.5, 0.02, 0.01, 0.01])

# 보정 구간에서 이만큼은 남아야 판별 방향을 믿을 수 있다 (30fps 0.5초)
MIN_CALIB_FRAMES = 15

# 판별 방향을 구할 때 공분산에 더하는 릿지 — 특징끼리 강하게 얽혀도
# 역행렬이 튀지 않게 한다. 표준화한 단위라 0.05 면 거의 영향이 없다.
RIDGE = 0.05

# 이 시간보다 짧게 얼굴을 놓치거나 눈을 감은 구간은 앞 상태로 메우고,
# 더 길면 '알 수 없음'으로 남긴다.
MAX_GAP = 0.5

# 이보다 짧은 화면/건반 구간은 앞 구간에 흡수한다 (깜빡임·떨림).
MIN_DURATION = 0.2

# 눈 감김이 이보다 짧으면 깜빡임으로 보고 앞 상태로 메운다. 더 길면 깊게
# 내려다본 것이다 — 건반을 깊이 내려다보면 눈꺼풀이 거의 덮여 깜빡임과
# 모양이 같아진다. 첫 실측: 깜빡임 0.03~0.11초, 내려다봄 0.26~0.7초.
BLINK_MAX = 0.25

# 이력 임계값 — 화면 상태에서는 점수가 ENTER 를 넘어야 건반으로, 건반
# 상태에서는 LEAVE 아래로 내려와야 화면으로 바뀐다. 경계 근처에서 상태가
# 깜빡이는 것을 막는다.
ENTER = 0.6
LEAVE = 0.4


class CalibrationError(ValueError):
    """보정 구간으로 판별 방향을 만들 수 없을 때."""


@dataclass
class Calibration:
    center: list            # 표준화 중심 (특징별)
    scale: list             # 표준화 폭 (특징별)
    weights: list           # 판별 방향
    screen_level: float     # 화면 보정 구간의 평균 투영값
    keys_level: float       # 건반 보정 구간의 평균 투영값
    separation: float       # 두 자세가 갈리는 정도 d' (클수록 좋다)
    blink_open: float       # eye_open 이 이보다 작으면 눈이 감긴 것으로 본다
    n_screen: int
    n_keys: int

    def score(self, rows):
        """특징 행렬 (n, len(FEATURES)) → 점수 (0=화면 자세, 1=건반 자세).

        얼굴을 못 찾은 프레임은 NaN 이다. 눈 감김은 closed() 로 따로 본다.
        """
        rows = np.asarray(rows, dtype=float).reshape(-1, len(FEATURES))
        z = (rows - np.asarray(self.center)) / np.asarray(self.scale)
        projected = z @ np.asarray(self.weights)
        scores = ((projected - self.screen_level)
                  / (self.keys_level - self.screen_level))
        scores[np.isnan(rows).any(axis=1)] = np.nan
        return scores

    def closed(self, rows):
        """눈이 거의 감긴 프레임 — 깜빡임이거나 깊게 내려다본 것."""
        rows = np.asarray(rows, dtype=float).reshape(-1, len(FEATURES))
        with np.errstate(invalid="ignore"):
            return rows[:, _OPEN] < self.blink_open

    def quality(self):
        """분리도를 한 단어로 — 콘솔·보고용."""
        if self.separation >= 3:
            return "좋음"
        if self.separation >= 2:
            return "보통"
        return "낮음"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


def _drop_blinks(rows):
    """눈을 거의 감은 프레임(그 자세의 중앙값의 절반 미만)을 뺀다."""
    if len(rows) == 0:
        return rows
    typical = np.median(rows[:, _OPEN])
    return rows[rows[:, _OPEN] >= 0.5 * typical]


def _drop_outliers(rows, limit=4.0):
    """중앙값에서 크게 벗어난 프레임을 뺀다 — 보정 중 잠깐 딴 데를 본 순간."""
    if len(rows) < MIN_CALIB_FRAMES:
        return rows
    median = np.median(rows, axis=0)
    spread = np.median(np.abs(rows - median), axis=0) * 1.4826
    spread = np.maximum(spread, _MIN_SCALE)
    keep = (np.abs(rows - median) <= limit * spread).all(axis=1)
    return rows[keep]


def _clean(rows):
    rows = np.asarray(rows, dtype=float).reshape(-1, len(FEATURES))
    rows = rows[~np.isnan(rows).any(axis=1)]
    return _drop_outliers(_drop_blinks(rows))


def fit_calibration(screen_rows, keys_rows, ridge=RIDGE):
    """보정 구간의 특징으로 판별 방향을 구한다.

    Args:
        screen_rows: 화면을 보던 구간의 특징 행렬 (n, len(FEATURES)).
        keys_rows:   건반을 보던 구간의 특징 행렬.

    Raises:
        CalibrationError: 쓸 수 있는 프레임이 모자랄 때.
    """
    screen = _clean(screen_rows)
    keys = _clean(keys_rows)

    for label, rows in (("화면", screen), ("건반", keys)):
        if len(rows) < MIN_CALIB_FRAMES:
            raise CalibrationError(
                f"{label} 보정 구간에서 얼굴이 잡힌 프레임이 {len(rows)}개뿐"
                f"입니다 (최소 {MIN_CALIB_FRAMES}개). 보정을 다시 하세요.")

    both = np.vstack([screen, keys])
    center = both.mean(axis=0)
    scale = np.maximum(both.std(axis=0), _MIN_SCALE)
    z_screen = (screen - center) / scale
    z_keys = (keys - center) / scale

    dof = len(z_screen) + len(z_keys) - 2
    within = (np.cov(z_screen, rowvar=False) * (len(z_screen) - 1)
              + np.cov(z_keys, rowvar=False) * (len(z_keys) - 1)) / dof
    within += ridge * np.eye(len(FEATURES))
    weights = np.linalg.solve(within, z_keys.mean(axis=0)
                              - z_screen.mean(axis=0))

    p_screen = z_screen @ weights
    p_keys = z_keys @ weights
    screen_level = float(p_screen.mean())
    keys_level = float(p_keys.mean())
    spread = math.sqrt(((len(p_screen) - 1) * p_screen.var(ddof=1)
                        + (len(p_keys) - 1) * p_keys.var(ddof=1)) / dof)
    separation = (keys_level - screen_level) / max(spread, 1e-9)

    blink_open = 0.5 * min(screen[:, _OPEN].mean(), keys[:, _OPEN].mean())

    return Calibration(
        center=center.tolist(),
        scale=scale.tolist(),
        weights=weights.tolist(),
        screen_level=screen_level,
        keys_level=keys_level,
        separation=float(separation),
        blink_open=float(blink_open),
        n_screen=len(screen),
        n_keys=len(keys),
    )


@dataclass
class Run:
    """같은 상태가 이어진 구간. end 는 다음 구간의 첫 프레임 시각이다."""
    label: str
    start: float
    end: float
    first: int      # 첫 프레임 번호
    stop: int       # 마지막 프레임 번호 + 1

    @property
    def duration(self):
        return self.end - self.start


def to_runs(times, labels):
    """프레임별 라벨 → 구간 목록."""
    n = len(times)
    if n == 0:
        return []
    step = float(np.median(np.diff(times))) if n > 1 else 0.0

    runs = []
    first = 0
    for index in range(1, n + 1):
        if index == n or labels[index] != labels[first]:
            end = float(times[index]) if index < n else float(times[-1]) + step
            runs.append(Run(labels[first], float(times[first]), end,
                            first, index))
            first = index
    return runs


def mask_blinks(times, scores, closed, max_blink=BLINK_MAX):
    """짧은 눈 감김(깜빡임)의 점수만 NaN 으로 비운다.

    비운 자리는 label_frames 가 앞 상태로 메운다. max_blink 보다 오래 감긴
    구간은 깊게 내려다본 것이므로 점수를 그대로 둔다 — 눈꺼풀이 덮이고
    눈동자가 내려가 건반 쪽 점수가 나온다.
    """
    scores = np.array(scores, dtype=float)
    for run in to_runs(times, [bool(flag) for flag in closed]):
        if run.label and run.duration <= max_blink:
            scores[run.first:run.stop] = np.nan
    return scores


def _hysteresis(times, scores, enter, leave, max_gap):
    """점수가 있는 프레임에만 상태를 매긴다. 나머지는 UNKNOWN."""
    labels = [UNKNOWN] * len(times)
    state = None
    last_time = None

    for index, score in enumerate(scores):
        if math.isnan(score):
            continue
        if last_time is not None and times[index] - last_time > max_gap:
            state = None    # 오래 끊겼으면 이전 상태를 믿지 않고 새로 정한다
        if state is None:
            state = KEYS if score >= 0.5 else SCREEN
        elif state == SCREEN and score >= enter:
            state = KEYS
        elif state == KEYS and score <= leave:
            state = SCREEN
        labels[index] = state
        last_time = times[index]
    return labels


def _fill_short_gaps(times, labels, max_gap):
    """양옆에 상태가 있는 짧은 UNKNOWN 구간을 앞 상태로 메운다."""
    for run in to_runs(times, labels):
        if run.label != UNKNOWN or run.first == 0 or run.stop >= len(times):
            continue    # 녹화 처음·끝의 공백은 메울 근거가 없다
        gap = times[run.stop] - times[run.first - 1]
        if gap <= max_gap:
            labels[run.first:run.stop] = [labels[run.first - 1]] * (
                run.stop - run.first)
    return labels


def _absorb_short_runs(times, labels, min_duration):
    """min_duration 보다 짧은 화면/건반 구간을 이웃 구간으로 바꾼다."""
    runs = to_runs(times, labels)
    previous = None
    for position, run in enumerate(runs):
        label = run.label
        if label != UNKNOWN and run.duration < min_duration:
            following = (runs[position + 1].label
                         if position + 1 < len(runs) else None)
            neighbor = previous if previous not in (None, UNKNOWN) else (
                following if following not in (None, UNKNOWN) else None)
            if neighbor is not None:
                labels[run.first:run.stop] = [neighbor] * (run.stop - run.first)
                label = neighbor
        previous = label
    return labels


def label_frames(times, scores, enter=ENTER, leave=LEAVE, max_gap=MAX_GAP,
                 min_duration=MIN_DURATION):
    """프레임 시각·점수 → 프레임별 상태 (SCREEN / KEYS / UNKNOWN)."""
    times = np.asarray(times, dtype=float)
    scores = np.asarray(scores, dtype=float)
    labels = _hysteresis(times, scores, enter, leave, max_gap)
    labels = _fill_short_gaps(times, labels, max_gap)
    return _absorb_short_runs(times, labels, min_duration)
