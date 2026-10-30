"""시선 분류·지표·검증 단위 테스트 (카메라 없이 합성 데이터로).

    run test        (저장소 폴더에서)
"""

import io
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gaze import analyze, classify, metrics, validation  # noqa: E402
from gaze.classify import KEYS, SCREEN, UNKNOWN  # noqa: E402

FPS = 30.0


def posture(n, keys, rng):
    """한 자세의 특징 행렬 — 건반을 보면 고개·눈이 내려가고 눈이 좁아진다."""
    base = (np.array([18.0, 0.45, 0.06, 0.20]) if keys
            else np.array([3.0, -0.05, -0.05, 0.32]))
    noise = rng.normal(0, [1.5, 0.05, 0.01, 0.015], size=(n, 4))
    return base + noise


def labels_at(times, labels, t):
    return labels[int(np.searchsorted(times, t))]


class CalibrationTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.calibration = classify.fit_calibration(
            posture(120, False, self.rng), posture(120, True, self.rng))

    def test_separates_postures(self):
        screen = self.calibration.score(posture(200, False, self.rng))
        keys = self.calibration.score(posture(200, True, self.rng))
        self.assertAlmostEqual(np.nanmean(screen), 0.0, delta=0.1)
        self.assertAlmostEqual(np.nanmean(keys), 1.0, delta=0.1)
        self.assertGreater(self.calibration.separation, 3)
        self.assertEqual(self.calibration.quality(), "좋음")

    def test_missing_face_is_nan_and_closed_eyes_flagged(self):
        rows = posture(3, False, self.rng)
        rows[1, :] = np.nan                 # 얼굴 못 찾음
        rows[2, 3] = 0.02                   # 눈 감음
        scores = self.calibration.score(rows)
        self.assertFalse(math.isnan(scores[0]))
        self.assertTrue(math.isnan(scores[1]))
        self.assertFalse(math.isnan(scores[2]))
        self.assertEqual(self.calibration.closed(rows).tolist(),
                         [False, False, True])

    def test_ignores_stray_frames_in_calibration(self):
        screen = posture(120, False, self.rng)
        screen[:10] = posture(10, True, self.rng)   # 보정 중 잠깐 내려다봄
        calibration = classify.fit_calibration(screen,
                                               posture(120, True, self.rng))
        self.assertGreater(calibration.separation, 3)
        self.assertLess(calibration.n_screen, 120)

    def test_rejects_too_few_frames(self):
        with self.assertRaises(classify.CalibrationError):
            classify.fit_calibration(np.full((40, 4), np.nan),
                                     posture(120, True, self.rng))

    def test_round_trips_through_dict(self):
        restored = classify.Calibration.from_dict(self.calibration.to_dict())
        rows = posture(5, True, self.rng)
        np.testing.assert_allclose(restored.score(rows),
                                   self.calibration.score(rows))


class LabelTest(unittest.TestCase):
    def times(self, seconds):
        return np.arange(int(seconds * FPS)) / FPS

    def test_hysteresis_ignores_noise_near_threshold(self):
        times = self.times(4)
        scores = np.where(times < 2, 0.0, 1.0)
        scores[(times > 0.5) & (times < 1.5)] = 0.55    # 경계 근처 떨림
        labels = classify.label_frames(times, scores)
        self.assertEqual(labels_at(times, labels, 1.0), SCREEN)
        self.assertEqual(labels_at(times, labels, 3.0), KEYS)
        self.assertEqual(len(classify.to_runs(times, labels)), 2)

    def test_short_spike_is_absorbed(self):
        times = self.times(3)
        scores = np.zeros(len(times))
        scores[(times >= 1.0) & (times < 1.1)] = 1.0    # 0.1초 — 깜빡임 수준
        labels = classify.label_frames(times, scores)
        self.assertEqual(set(labels), {SCREEN})

    def test_real_glance_is_kept(self):
        times = self.times(3)
        scores = np.zeros(len(times))
        scores[(times >= 1.0) & (times < 1.6)] = 1.0
        runs = classify.to_runs(times, classify.label_frames(times, scores))
        self.assertEqual([run.label for run in runs], [SCREEN, KEYS, SCREEN])
        self.assertAlmostEqual(runs[1].duration, 0.6, delta=1.5 / FPS)

    def test_blink_is_masked_but_long_closure_counts_as_looking_down(self):
        times = self.times(4)
        scores = np.zeros(len(times))
        closed = np.zeros(len(times), dtype=bool)
        blink = (times >= 1.0) & (times < 1.1)          # 0.1초 깜빡임
        deep = (times >= 2.5) & (times < 3.1)           # 0.6초 깊게 내려다봄
        closed[blink | deep] = True
        scores[blink | deep] = 2.0                      # 감긴 눈은 건반 쪽 점수
        masked = classify.mask_blinks(times, scores, closed)
        self.assertTrue(np.isnan(masked[blink]).all())
        self.assertFalse(np.isnan(masked[deep]).any())

        labels = classify.label_frames(times, masked)
        self.assertEqual(labels_at(times, labels, 1.05), SCREEN)
        self.assertEqual(labels_at(times, labels, 2.8), KEYS)

    def test_short_gap_filled_long_gap_unknown(self):
        times = self.times(6)
        scores = np.zeros(len(times))
        scores[(times >= 1.0) & (times < 1.3)] = np.nan     # 0.3초
        scores[(times >= 3.0) & (times < 4.0)] = np.nan     # 1초
        labels = classify.label_frames(times, scores)
        self.assertEqual(labels_at(times, labels, 1.15), SCREEN)
        self.assertEqual(labels_at(times, labels, 3.5), UNKNOWN)
        self.assertEqual(labels_at(times, labels, 5.0), SCREEN)


def run(label, start, end):
    return classify.Run(label, start, end, 0, 0)


class GazeMetricsTest(unittest.TestCase):
    def test_counts_glances_and_dwell(self):
        runs = [run(SCREEN, 0, 10), run(KEYS, 10, 11), run(SCREEN, 11, 20),
                run(KEYS, 20, 23), run(SCREEN, 23, 30)]
        result = metrics.gaze_metrics(runs, 0, 30)
        self.assertEqual(result["glances"], 2)
        self.assertAlmostEqual(result["screen_ratio"], 26 / 30)
        self.assertAlmostEqual(result["glances_per_min"], 4.0)
        self.assertAlmostEqual(result["keys_dwell_mean_s"], 2.0)

    def test_unknown_excluded_from_ratio(self):
        runs = [run(SCREEN, 0, 5), run(UNKNOWN, 5, 7), run(KEYS, 7, 10)]
        result = metrics.gaze_metrics(runs, 0, 10)
        self.assertAlmostEqual(result["valid_ratio"], 0.8)
        self.assertAlmostEqual(result["screen_ratio"], 5 / 8)
        self.assertEqual(result["glances"], 1)   # 알 수 없음 너머의 전환도 센다

    def test_glance_started_before_window_not_counted(self):
        runs = [run(SCREEN, 0, 4), run(KEYS, 4, 7), run(SCREEN, 7, 10)]
        result = metrics.gaze_metrics(runs, 5, 10)
        self.assertEqual(result["glances"], 0)
        self.assertAlmostEqual(result["keys_s"], 2.0)


def note(t, ok, **extra):
    return {"type": "note", "t": t, "pitch_ok": ok, **extra}


class TrialTest(unittest.TestCase):
    def events(self):
        return [
            {"type": "session", "t": 0.0, "song": "twinkle", "practice": True,
             "speed": 0.35},
            {"type": "countdown", "t": 0.5, "seconds": 1},
            {"type": "start", "t": 1.0, "practice": True},
            note(3.0, True, first_try=True, expected="도4"),
            note(4.0, False, retry=True),
            note(6.0, True, first_try=False, expected="솔4"),
            {"type": "done", "t": 6.5, "result": {"pitch_accuracy": 50.0}},
            {"type": "session", "t": 10.0, "song": "abc", "practice": False},
            {"type": "start", "t": 11.0, "practice": False},
            note(12.0, True, grade="Perfect", diff_ms=40),
        ]

    def test_split_and_summarize(self):
        trials = metrics.split_trials(self.events())
        self.assertEqual([t.status for t in trials], ["done", "incomplete"])
        self.assertEqual(trials[0].song, "twinkle")
        self.assertEqual(trials[1].mode, "normal")

        runs = [run(SCREEN, 0, 4.5), run(KEYS, 4.5, 5.5), run(SCREEN, 5.5, 13)]
        summary, notes = metrics.trial_summary(trials[0], runs)
        self.assertEqual(summary["steps"], 2)
        self.assertEqual(summary["wrong_presses"], 1)
        self.assertEqual(summary["glances"], 1)
        self.assertEqual(summary["pitch_accuracy"], 50.0)

        self.assertEqual([row["wrong_presses"] for row in notes], [0, 1])
        self.assertEqual([row["looked_down"] for row in notes], [False, True])
        self.assertAlmostEqual(notes[1]["interval_s"], 3.0)


class CalibrationScheduleTest(unittest.TestCase):
    def test_each_frame_uses_latest_calibration(self):
        fitted = [(1, {"screen": [10, 15]}, None),
                  (2, {"screen": [40, 45]}, None)]
        active = analyze._active(np.array([0, 12, 39, 40, 90]), fitted)
        self.assertEqual(active.tolist(), [0, 0, 0, 1, 1])

    def test_free_spans_skip_calibrations_and_short_gaps(self):
        busy = [(0, 10), (13, 23), (66, 76)]
        spans = analyze._free_spans(0, 85, busy)
        self.assertEqual(spans, [(23, 66), (76, 85)])   # 10~13 은 너무 짧다


class ValidationScheduleTest(unittest.TestCase):
    def test_schedule_fits_blocks_with_gaps(self):
        blocks = validation.make_schedule(seed=1)
        self.assertEqual(sorted(block["count"] for block in blocks),
                         sorted(validation.COUNTS))
        self.assertEqual(blocks, validation.make_schedule(seed=1))  # 재현 가능
        for block in blocks:
            self.assertEqual(len(block["cues"]), block["count"])
            previous = block["start"]
            for cue in block["cues"]:
                self.assertGreaterEqual(cue["down"] - previous,
                                        validation.MIN_GAP - 1e-6)
                self.assertGreater(cue["up"], cue["down"])
                self.assertLessEqual(cue["up"] - cue["down"],
                                     validation.DURATIONS[1] + 1e-6)
                previous = cue["up"]
            self.assertLessEqual(previous, block["end"])

    def test_runner_fires_every_cue_then_done(self):
        fired = []
        schedule = validation.make_schedule(seed=2)
        runner = validation.CueRunner(schedule, 1000.0, fired.append, seed=2)
        now = 1000.0
        while not runner.completed:
            now += 1 / FPS
            runner.update(now)
        cues = sum(block["count"] for block in schedule)
        self.assertEqual(fired.count("down"), cues)
        self.assertEqual(fired.count("up"), cues)
        self.assertEqual(fired[-1], "done")
        record = runner.record()
        self.assertTrue(record["completed"])
        for block in record["blocks"]:
            for cue in block["cues"]:
                self.assertGreaterEqual(cue["down"], record["started"])
                self.assertAlmostEqual(cue["up"] - cue["down"],
                                       cue["planned_s"], delta=2 / FPS)


def follow(record, latency=0.4, skip=(), extra=()):
    """신호를 따라 내려다본 사람의 시선 구간을 만든다."""
    edges = []
    for block in record["blocks"]:
        for cue in block["cues"]:
            if (block["block"], cue["down"]) not in skip:
                edges.append((cue["down"] + latency, cue["up"] + latency))
    edges = sorted(edges + list(extra))
    runs, cursor = [], record["started"]
    for start, end in edges:
        runs.append(run(SCREEN, cursor, start))
        runs.append(run(KEYS, start, end))
        cursor = end
    runs.append(run(SCREEN, cursor, record["blocks"][-1]["end"] + 5))
    return runs


class ValidationEvaluateTest(unittest.TestCase):
    def record(self):
        runner = validation.CueRunner(validation.make_schedule(seed=3),
                                      500.0, lambda kind: None, seed=3)
        now = 500.0
        while not runner.completed:
            now += 1 / FPS
            runner.update(now)
        return runner.record()

    def test_perfect_follower_is_proportional(self):
        record = self.record()
        cues, blocks, summary = validation.evaluate(runs=follow(record),
                                                    record=record)
        self.assertEqual(summary["hit_rate"], 1.0)
        self.assertEqual(summary["false_glances"], 0)
        self.assertAlmostEqual(summary["latency_median_s"], 0.4, delta=0.01)
        self.assertGreater(summary["duration"]["r"], 0.99)
        self.assertAlmostEqual(summary["duration"]["slope"], 1.0, delta=0.05)
        self.assertEqual([row["detected_glances"] for row in blocks],
                         [row["planned_glances"] for row in blocks])

    def test_missed_cue_and_stray_glance_are_counted(self):
        record = self.record()
        block = record["blocks"][0]
        missed = block["cues"][0]
        # 신호 없이 내려다봄 — 앞 신호의 여유(LAG)가 끝난 뒤, 다음 신호
        # (최소 MIN_GAP 뒤) 전에 넣는다.
        stray_at = block["cues"][1]["up"] + validation.LAG + 0.1
        runs = follow(record, skip={(block["block"], missed["down"])},
                      extra=[(stray_at, stray_at + 0.3)])
        cues, blocks, summary = validation.evaluate(runs, record)
        self.assertEqual(summary["hits"], summary["cues"] - 1)
        self.assertEqual(summary["false_glances"], 1)


class AppEventsTest(unittest.TestCase):
    """Keymento 를 고치지 않고 이벤트를 받는 부분 — 시각 어림·세션 정보."""

    @classmethod
    def setUpClass(cls):
        from gaze import record_face
        cls.module = record_face

    def poller(self, payloads):
        events = self.module.AppEvents("http://fake")
        events._fetch = lambda since: payloads.pop(0)
        events._out = io.StringIO()
        return events

    @staticmethod
    def state(seq, events, state="playing"):
        return {"seq": seq, "events": events, "state": state,
                "selected": "abc", "songs": [{"id": "abc", "name": "Abc"}],
                "active_practice": True, "effective_speed": 0.35,
                "input": "local", "progress": {}}

    def lines(self, events):
        return [json.loads(line) for line in
                events._out.getvalue().splitlines()]

    def test_estimates_time_and_adds_session(self):
        payloads = [
            self.state(5, [[5, {"type": "done"}]], "idle"),     # 켜기 전 기록
            self.state(6, [[6, {"type": "countdown", "seconds": 1}]],
                       "countdown"),
            self.state(8, [[7, {"type": "start"}],
                           [8, {"type": "note", "pitch_ok": True}]]),
            self.state(9, [[9, {"type": "done", "result": {}}]], "done"),
            self.state(10, [[10, {"type": "countdown", "seconds": 1}]],
                       "countdown"),
        ]
        events = self.poller(payloads)
        for now in (100.0, 100.2, 100.25, 100.3, 105.0):
            events._poll(now)

        lines = self.lines(events)
        self.assertEqual([line["type"] for line in lines],
                         ["session", "countdown", "start", "note", "done",
                          "session", "countdown"])
        session = lines[0]
        self.assertEqual((session["song"], session["song_name"],
                          session["practice"], session["speed"]),
                         ("abc", "Abc", True, 0.35))
        self.assertAlmostEqual(lines[1]["t"], 100.1)
        self.assertAlmostEqual(lines[2]["t"], 100.225)
        self.assertTrue(lines[2]["t_estimated"])

    def test_keymento_restart_is_followed(self):
        payloads = [
            self.state(9, [], "idle"),
            self.state(2, [], "playing"),       # 번호가 줄었다 = 다시 켰다
            self.state(2, [[1, {"type": "countdown", "seconds": 1}],
                           [2, {"type": "start"}]]),
        ]
        events = self.poller(payloads)
        events._poll(10.0)
        events._poll(10.2)
        self.assertEqual([line["type"] for line in self.lines(events)],
                         ["session", "countdown", "start"])


if __name__ == "__main__":
    unittest.main()
