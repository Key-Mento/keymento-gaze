"""실험용 얼굴 녹화기 — 시선 분석(analyze.py)에 쓸 자료를 남긴다.

연주자 얼굴을 정면에서 찍는 웹캠(화면 바로 위)으로 녹화하면서, 같은 PC
에서 돌고 있는 Keymento 의 판정 이벤트도 함께 받아 둔다. Keymento 코드는
건드리지 않는다 — 웹 UI 가 브라우저에 주는 상태(/api/state)를 똑같이 읽는다.

    _face.mp4      얼굴 영상
    _frames.csv    프레임마다 찍은 PC 시각
    _events.jsonl  판정 이벤트 (세션 정보·음마다 판정·완주)
    _meta.json     참가자·조건·카메라·보정 구간·검증 기록

분석은 실험이 끝난 뒤 따로 한다. 녹화만 하므로 AR 창이 느려지지 않고,
AR 창이 없는 악보 조건(Keymento 의 run nocam)에서도 똑같이 쓴다.

실행 (keymento-gaze 폴더에서, Keymento 를 먼저 켜 둔다):
    run record --participant P01 --condition AR
    run record --participant P01 --condition sheet --camera 1
    run validate --participant P01          신호음 검증 — 창에서 V

창 안에서 쓰는 키:
    R      녹화 시작
    C      보정 — 삐(화면 5초) → 삐삐(건반 5초) → 삐-(끝). 다시 누르면 다시 잰다
    V      검증 — 높은 삐에 내려다보고 낮은 삐에 화면으로 (약 2분 30초).
           보정 전이면 보정부터 한다
    ESC/Q  저장하고 종료
"""

import argparse
import json
import math
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

# Windows 기본 카메라 백엔드(MSMF)의 하드웨어 변환은 일부 웹캠을 여는 데
# 십수 초씩 걸리게 한다. cv2 를 불러오기 전에 꺼야 한다.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2  # noqa: E402

TOOL_DIR = Path(__file__).resolve().parents[1]
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from gaze import features, validation  # noqa: E402
from gaze.recording import DATA_DIR, RecordingFiles, write_meta  # noqa: E402

WINDOW_NAME = "Keymento Face Recorder"
FONT = cv2.FONT_HERSHEY_SIMPLEX
HELP_LINE = "R: record  C: calibrate  V: validate  ESC: save & quit"

# 신호음 (주파수 Hz, 길이 ms). 주파수 0 은 쉼.
# 높은 음(1400)은 늘 '건반', 낮은 음(1000)은 늘 '화면'이다.
BEEP_SCREEN = [(1000, 200)]
BEEP_KEYS = [(1400, 150), (0, 100), (1400, 150)]
BEEP_DONE = [(800, 600)]
BEEP_DOWN = [(1400, 150)]
BEEP_UP = [(1000, 150)]
BEEP_START = [(1000, 100), (0, 80), (1000, 100), (0, 80), (1000, 100)]


def beep(pattern):
    """신호음을 울린다. 녹화 루프가 멈추지 않게 따로 돈다."""
    def play():
        try:
            import winsound
        except ImportError:         # Windows 가 아니면 터미널 벨로 대신한다
            print("\a", end="", flush=True)
            return
        for frequency, ms in pattern:
            if not frequency:
                time.sleep(ms / 1000)
                continue
            try:
                winsound.Beep(frequency, ms)
            except RuntimeError:    # 소리 장치가 없다 — 콘솔 안내만 남는다
                return

    threading.Thread(target=play, daemon=True).start()


class Calibration:
    """보정 순서 — 화면을 보는 구간과 건반을 보는 구간을 차례로 잰다.

    연주자에게 미리 알려 둔다: 삐 한 번이면 화면 가운데, 삐삐면 손을
    건반에 올리고 칠 때처럼 건반을, 길게 삐- 하면 끝.
    """

    def __init__(self, seconds):
        self.seconds = seconds
        self.started = None
        self.completed = []         # [{"screen": [시작, 끝], "keys": [...]}]
        self._keys_announced = False

    @property
    def running(self):
        return self.started is not None

    def start(self, now):
        if self.started is not None:
            print("보정을 처음부터 다시 잽니다.")
        self.started = now
        self._keys_announced = False
        beep(BEEP_SCREEN)
        print(f"보정: 화면 가운데를 보게 하세요 ({self.seconds:g}초)")

    def update(self, now):
        """지금 단계 ('screen' / 'keys' / None) 와 남은 초."""
        if self.started is None:
            return None, 0.0
        elapsed = now - self.started
        if elapsed < self.seconds:
            return "screen", self.seconds - elapsed

        if not self._keys_announced:
            self._keys_announced = True
            beep(BEEP_KEYS)
            print(f"보정: 손을 건반에 올리고 칠 때처럼 건반을 보게 하세요 "
                  f"({self.seconds:g}초)")
        if elapsed < 2 * self.seconds:
            return "keys", 2 * self.seconds - elapsed

        middle = self.started + self.seconds
        self.completed.append({
            "screen": [self.started, middle],
            "keys": [middle, middle + self.seconds],
        })
        self.started = None
        beep(BEEP_DONE)
        print(f"보정 완료 ({len(self.completed)}번째).")
        return None, 0.0


class AppEvents(threading.Thread):
    """Keymento 웹 UI(/api/state)를 폴링해 판정 이벤트를 파일로 옮긴다.

    Keymento 를 고치지 않고 브라우저와 같은 상태를 읽는다. 그래서
    - 이벤트에 발생 시각이 없다. '지난 폴링과 이번 폴링 사이의 가운데'로
      어림한다(t_estimated). 연주 중에는 0.05초마다 물으므로 오차는
      ±0.025초 안팎이다. 서버가 시각("t")을 주면 그것을 쓴다.
    - 곡·모드·속도는 상태에서 읽어 세션이 시작될 때 'session' 줄을
      직접 만들어 넣는다.

    켜자마자 폴링을 시작해 미리보기에 앱 상태를 보여 주고, begin() 뒤에
    나온 이벤트만 파일에 쓴다 — 녹화 전 이벤트는 영상이 없는 구간이다.
    """

    FAST = 0.05     # 카운트다운·연주 중
    SLOW = 0.2      # 그 밖

    def __init__(self, base_url):
        super().__init__(daemon=True)
        self.base_url = base_url.rstrip("/")
        self.connected = False
        self.status = "not connected"
        self.count = 0
        self._seq = None            # 서버에서 마지막으로 본 이벤트 번호
        self._asked = None          # 지난번에 성공한 폴링을 보낸 시각
        self._session_open = False
        self._interval = self.SLOW
        self._out = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    def begin(self, path):
        """여기부터 나오는 이벤트를 path 에 쓴다."""
        if not self.connected:
            print("[!] Keymento 에 연결되지 않았습니다. 앱을 켜면 그때부터 "
                  "판정 이벤트를 받습니다.")
        elif self.status.startswith(("playing", "countdown")):
            print("[!] 연주가 이미 진행 중입니다. 이번 시도는 앞부분이 "
                  "빠집니다 — 녹화를 먼저 시작한 뒤 연주를 시작하세요.")
        with self._lock:
            self._out = open(path, "a", encoding="utf-8")

    def stop(self):
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=3)
        with self._lock:
            if self._out is not None:
                self._out.close()
                self._out = None

    def run(self):
        while not self._stop_event.is_set():
            self._poll()
            self._stop_event.wait(self._interval)
        self._poll()                # 멈추기 직전에 남은 이벤트까지

    def _fetch(self, since):
        url = f"{self.base_url}/api/state?since={since}"
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def _poll(self, now=None):
        asked = time.time() if now is None else now
        payload = self._fetch(self._seq or 0)
        if payload is None:
            if self.connected:
                print("[!] Keymento 와 연결이 끊겼습니다.")
            self.connected = False
            self.status = "not connected"
            self._interval = self.SLOW
            return

        if not self.connected:
            print(f"Keymento 연결됨: {self.base_url}")
        self.connected = True
        self.status = self._describe(payload)
        self._interval = (self.FAST if payload.get("state")
                          in ("countdown", "playing") else self.SLOW)

        # 이번에 처음 보는 이벤트는 지난 폴링과 이번 폴링 사이에 났다.
        estimate = asked if self._asked is None else (self._asked + asked) / 2
        self._asked = asked

        seq = int(payload.get("seq", 0))
        if self._seq is None:
            self._seq = seq         # 켜기 전에 쌓인 이벤트는 받지 않는다
            return
        if seq < self._seq:         # Keymento 를 다시 켰다 — 처음부터 받는다
            self._seq = 0
            self._session_open = False
            payload = self._fetch(0)
            if payload is None:
                return

        with self._lock:
            for number, event in payload.get("events", []):
                if number <= self._seq:
                    continue
                self._seq = number
                self._handle(event, payload, estimate)
            if self._out is not None:
                self._out.flush()

    def _handle(self, event, payload, estimate):
        if "t" not in event:
            event["t"] = estimate
            event["t_estimated"] = True

        kind = event.get("type")
        if kind == "session":       # Keymento 가 세션 정보를 직접 줄 때
            self._session_open = True
        elif kind in ("countdown", "start") and not self._session_open:
            self._session_open = True
            self._write(self._session_from(payload, event["t"]))
        self._write(event)
        if kind in ("done", "aborted", "error"):
            self._session_open = False

    @staticmethod
    def _session_from(payload, t):
        """상태에서 곡·모드·속도를 읽어 세션 줄을 만든다. 연주 중에는 설정을
        바꿀 수 없으므로 지금 선택된 곡이 연주 중인 곡이다."""
        song = payload.get("selected")
        names = {item.get("id"): item.get("name")
                 for item in payload.get("songs") or []}
        practice = payload.get("active_practice")
        if practice is None:
            practice = payload.get("practice")
        return {
            "type": "session",
            "t": t,
            "song": song,
            "song_name": names.get(song, song),
            "practice": bool(practice),
            "speed": payload.get("effective_speed"),
            "input": payload.get("input"),
            "from_state": True,
        }

    def _write(self, event):
        if self._out is None:
            return                  # 아직 녹화 전
        self._out.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.count += 1
        self._announce(event)

    @staticmethod
    def _describe(payload):
        state = payload.get("state", "?")
        progress = payload.get("progress") or {}
        if state in ("playing", "countdown"):
            mode = "practice" if payload.get("active_practice") else "normal"
            return (f"{state} {progress.get('index', 0)}/"
                    f"{progress.get('total', 0)} ({mode})")
        return state

    @staticmethod
    def _announce(event):
        kind = event.get("type")
        if kind == "session":
            mode = "연습" if event.get("practice") else "일반"
            print(f"세션: {event.get('song_name') or event.get('song')} "
                  f"({mode} 모드)")
        elif kind == "start":
            print("연주 시작")
        elif kind == "done":
            result = event.get("result") or {}
            print(f"완주 — 정확도 {result.get('pitch_accuracy')}%, "
                  f"오타 {result.get('wrong_attempts')}회")
        elif kind == "aborted":
            print("연주 중단")


class FaceCheck:
    """화각 확인용 — 분석과 같은 모델로 얼굴이 잡히는지 가끔 본다."""

    def __init__(self, every=10):
        self._landmarker = features.create_landmarker(video=False,
                                                      details=False)
        self._every = every
        self._count = 0
        self.box = None

    def update(self, frame):
        if self._count % self._every == 0:
            self.box = features.face_box(self._landmarker, frame)
        self._count += 1
        return self.box

    def close(self):
        self._landmarker.close()


class FpsMeter:
    """최근 프레임 간격으로 실제 fps 를 잰다 — 영상 파일에 적을 값."""

    def __init__(self, size=60):
        self._times = []
        self._size = size

    def tick(self, now):
        self._times.append(now)
        del self._times[:-self._size]

    @property
    def fps(self):
        if len(self._times) < 10:
            return None
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


def open_face_camera(index, width, height):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"카메라 {index}번을 열 수 없습니다. --camera 로 "
                           "다른 번호를 주세요 (AR 이 쓰는 카메라와 달라야 함).")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    for _ in range(10):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return cap, frame
    cap.release()
    raise RuntimeError(f"카메라 {index}번이 열렸지만 영상이 오지 않습니다. "
                       "AR 앱 등 다른 프로그램이 이미 쓰고 있을 수 있습니다.")


class Recorder:
    """영상·프레임 시각·이벤트·메타 파일을 함께 쓴다."""

    def __init__(self, args, frame_size, fps, events):
        self.files = RecordingFiles.new(args.participant, args.condition,
                                        root=args.out)
        self.files.stem.parent.mkdir(parents=True, exist_ok=True)
        width, height = frame_size

        video = self.files.path("video")
        self.writer = cv2.VideoWriter(str(video),
                                      cv2.VideoWriter.fourcc(*"mp4v"),
                                      fps, (width, height))
        if not self.writer.isOpened():
            # mp4 코덱이 없으면 MJPG AVI 로 대신한다 (파일이 커진다).
            video = video.with_suffix(".avi")
            self.writer = cv2.VideoWriter(str(video),
                                          cv2.VideoWriter.fourcc(*"MJPG"),
                                          fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError("영상 파일을 만들 수 없습니다 (코덱 없음).")

        self.frames = open(self.files.path("frames"), "w", encoding="utf-8")
        self.frames.write("frame,t\n")
        self.count = 0
        self.started = time.time()
        self.meta = {
            "participant": args.participant,
            "condition": args.condition,
            "note": args.note,
            "created": datetime.now().isoformat(timespec="seconds"),
            "camera": args.camera,
            "width": width,
            "height": height,
            "fps": round(fps, 2),
            "video": video.name,
            "app_url": args.app_url,
            "start_time": self.started,
            "end_time": None,
            "frames": 0,
            "calibrations": [],
            "status": "recording",
        }
        write_meta(self.files, self.meta)

        self.events = events
        self.events.begin(self.files.path("events"))

    def write(self, frame, now):
        self.writer.write(frame)
        self.frames.write(f"{self.count},{now:.6f}\n")
        self.count += 1
        if self.count % 30 == 0:
            self.frames.flush()

    def save(self, **fields):
        """메타 정보 일부를 바꿔 바로 저장한다 (보정·검증 기록)."""
        self.meta.update(fields)
        write_meta(self.files, self.meta)

    def close(self):
        self.events.stop()
        self.writer.release()
        self.frames.close()
        self.save(end_time=time.time(), frames=self.count,
                  events=self.events.count, status="complete")


def draw_preview(frame, recorder, look, box, app_status, fps, calib_text,
                 valid_text):
    """미리보기 화면. look 은 지금 연주자가 봐야 할 곳과 남은 초 (없으면 None)."""
    view = frame.copy()
    height, width = view.shape[:2]
    scale = width / 640

    def text(message, position, color=(255, 255, 255), size=0.55, thick=1):
        # 테두리는 같은 굵기로 조금씩 비켜 그린다. 굵기를 키워 그리면
        # OpenCV 5 의 글꼴은 글자 폭까지 넓어져 테두리가 어긋난다.
        x, y = position
        for dx in (-2, 0, 2):
            for dy in (-2, 0, 2):
                if dx or dy:
                    cv2.putText(view, message, (x + dx, y + dy), FONT,
                                size * scale, (0, 0, 0), thick, cv2.LINE_AA)
        cv2.putText(view, message, position, FONT, size * scale, color,
                    thick, cv2.LINE_AA)

    if box is not None:
        x, y, w, h = box
        cv2.rectangle(view, (x, y), (x + w, y + h), (0, 220, 0), 2)
    else:
        text("NO FACE", (int(width / 2 - 50 * scale), int(60 * scale)),
             (0, 0, 255), 0.8, 2)

    if recorder is not None:
        elapsed = time.time() - recorder.started
        cv2.circle(view, (int(18 * scale), int(20 * scale)), int(8 * scale),
                   (0, 0, 255), -1)
        text(f"REC {int(elapsed // 60):02d}:{int(elapsed % 60):02d}",
             (int(32 * scale), int(27 * scale)), (0, 0, 255), 0.65, 2)
    else:
        text("PREVIEW - press R to record", (int(10 * scale), int(27 * scale)),
             (0, 255, 255), 0.6, 2)

    if fps:
        text(f"{fps:4.1f} fps", (width - int(95 * scale), int(27 * scale)))

    if look is not None:
        target, left = look
        message = "LOOK AT SCREEN" if target == "screen" else "LOOK AT KEYS"
        if left is not None:
            message += f"  {math.ceil(left)}"
        text(message, (int(width / 2 - 150 * scale), int(height / 2)),
             (0, 255, 255), 1.0, 2)

    calib_color = (0, 220, 0) if "done" in calib_text else (0, 165, 255)
    text(calib_text, (int(10 * scale), height - int(64 * scale)), calib_color)
    if valid_text:
        text(valid_text, (int(250 * scale), height - int(64 * scale)),
             (0, 255, 255))
    text(f"APP: {app_status}", (int(10 * scale), height - int(40 * scale)),
         (255, 255, 255) if app_status != "not connected" else (0, 165, 255))
    text(HELP_LINE, (int(10 * scale), height - int(12 * scale)),
         (200, 200, 200), 0.5)
    return view


def parse_args():
    parser = argparse.ArgumentParser(
        description="실험용 얼굴 녹화 — 시선 분석(analyze.py)에 쓸 영상과 "
                    "판정 이벤트를 남긴다")
    parser.add_argument("--participant", required=True,
                        help="참가자 번호 (예: P01)")
    parser.add_argument("--condition", required=True,
                        help="실험 조건 (예: AR, sheet, validation)")
    parser.add_argument("--camera", type=int, default=0,
                        help="얼굴을 찍는 카메라 번호 (기본: 0 — AR 은 1번을 "
                             "먼저 쓰므로 노트북 내장 카메라가 보통 0번)")
    parser.add_argument("--width", type=int, default=640,
                        help="녹화 가로 해상도 (기본: 640)")
    parser.add_argument("--height", type=int, default=480,
                        help="녹화 세로 해상도 (기본: 480)")
    parser.add_argument("--calib-seconds", type=float, default=5.0,
                        help="보정 단계 하나의 길이(초) (기본: 5)")
    parser.add_argument("--seed", type=int, default=None,
                        help="검증 신호 일정의 시드 (기본: 무작위, 메타에 남음)")
    parser.add_argument("--app-url", default="http://127.0.0.1:8321",
                        help="Keymento 웹 UI 주소 (기본: http://127.0.0.1:8321)")
    parser.add_argument("--out", default=str(DATA_DIR),
                        help="저장 폴더 (기본: data)")
    parser.add_argument("--note", default="",
                        help="메모 (예: '안경 착용') — 메타 파일에 남는다")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        cap, frame = open_face_camera(args.camera, args.width, args.height)
    except RuntimeError as exc:
        print(f"[!] {exc}")
        return

    height, width = frame.shape[:2]
    print(f"얼굴 카메라: {args.camera}번 ({width}x{height})")
    print("연주자 얼굴이 정면으로, 화면 가운데쯤 오게 카메라를 맞추세요.")
    print("  R      녹화 시작")
    print("  C      보정 (삐: 화면 → 삐삐: 건반 → 삐-: 끝)")
    print("  V      검증 (높은 삐: 건반 보기 → 낮은 삐: 화면 보기, 약 2분 30초)")
    print("  ESC/Q  저장하고 종료")

    calibration = Calibration(args.calib_seconds)
    face_check = FaceCheck()
    meter = FpsMeter()
    events = AppEvents(args.app_url)
    events.start()
    recorder = None
    runner = None                   # 검증 신호 진행
    validation_pending = False      # 보정이 끝나면 검증을 시작한다
    saved_calibrations = 0
    failures = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)

    def start_recording():
        nonlocal recorder
        fps = meter.fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
        recorder = Recorder(args, (width, height), fps, events)
        print(f"녹화 시작: {recorder.files.stem}")

    def on_cue(kind):
        if kind == "down":
            beep(BEEP_DOWN)
        elif kind == "up":
            beep(BEEP_UP)
        else:
            beep(BEEP_DONE)
            print("검증 완료. ESC 로 저장하고 끝내세요.")

    def start_validation():
        nonlocal runner
        seed = args.seed if args.seed is not None else random.randrange(10**6)
        runner = validation.CueRunner(validation.make_schedule(seed),
                                      time.time(), on_cue, seed=seed)
        recorder.save(validation=runner.record())
        beep(BEEP_START)
        seconds = runner.blocks[-1]["end"]
        print(f"검증 시작 (약 {seconds / 60:.1f}분). 높은 삐에 건반을 "
              f"내려다보고, 낮은 삐에 화면으로 돌아오게 하세요.")

    try:
        while True:
            ok, frame = cap.read()
            now = time.time()
            if not ok or frame is None:
                failures += 1
                if failures > 30:
                    print("[!] 카메라 영상이 끊겼습니다. 저장하고 끝냅니다.")
                    break
                continue
            failures = 0
            meter.tick(now)

            if recorder is not None:
                recorder.write(frame, now)

            phase, left = calibration.update(now)
            done = len(calibration.completed)
            if recorder is not None and done != saved_calibrations:
                recorder.save(calibrations=list(calibration.completed))
                saved_calibrations = done
                if validation_pending:
                    validation_pending = False
                    start_validation()

            look = (phase, left) if phase else None
            if runner is not None and not runner.completed:
                runner.update(now)
                if runner.completed:
                    recorder.save(validation=runner.record())
                else:
                    look = (runner.state, None)

            view = draw_preview(
                frame, recorder, look, face_check.update(frame),
                events.status, meter.fps,
                f"CALIB: {done} done" if done else "CALIB: none - press C",
                runner.describe() if runner else "")
            cv2.imshow(WINDOW_NAME, view)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                break
            # 창을 X 로 닫으면 ESC 와 같게 — 그대로 두면 다음 imshow 가 창을
            # 다시 띄워 꺼진 줄 알았던 녹화가 계속 돈다.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

            validating = runner is not None and not runner.completed
            if key in (ord("r"), ord("R")) and recorder is None:
                start_recording()
            if key in (ord("c"), ord("C")):
                if validating:
                    print("검증 중에는 보정할 수 없습니다.")
                else:
                    if recorder is None:
                        start_recording()
                    calibration.start(time.time())
            if key in (ord("v"), ord("V")):
                if validating:
                    print("이미 검증 중입니다.")
                elif calibration.running:
                    validation_pending = True
                    print("보정이 끝나면 검증을 시작합니다.")
                else:
                    if recorder is None:
                        start_recording()
                    if calibration.completed:
                        start_validation()
                    else:
                        validation_pending = True
                        print("먼저 보정합니다. 끝나면 검증이 이어집니다.")
                        calibration.start(time.time())
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:     # 영상 파일을 만들 수 없을 때
        print(f"[!] {exc}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_check.close()
        if recorder is None:
            events.stop()
        else:
            if runner is not None:
                recorder.save(validation=runner.record())
            recorder.close()
            _report(recorder, calibration, runner)


def _report(recorder, calibration, runner):
    seconds = time.time() - recorder.started
    print(f"\n저장: {recorder.files.stem}_*")
    print(f"  {recorder.count}프레임 · {seconds:.0f}초 · 판정 이벤트 "
          f"{recorder.events.count}개 · 보정 {len(calibration.completed)}번")
    if runner is not None and not runner.completed:
        print("[!] 검증을 끝까지 하지 않았습니다. 낸 신호까지만 분석합니다.")
    if not calibration.completed:
        print("[!] 보정을 하지 않았습니다. 분석할 때 같은 참가자의 다른 "
              "녹화로 보정해야 합니다 (run analyze --calib-from ...).")
    if recorder.events.count == 0 and runner is None:
        print("[!] 판정 이벤트가 없습니다. Keymento 가 켜져 있었는지, "
              "--app-url 이 맞는지 확인하세요.")


if __name__ == "__main__":
    main()
