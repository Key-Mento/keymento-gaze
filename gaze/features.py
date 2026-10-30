"""얼굴 영상 → 프레임별 시선 특징 (MediaPipe Face Landmarker).

이 저장소의 venv(.venv)에서 쓴다. mediapipe 가 함께 까는
opencv-contrib-python 이 Keymento venv 의 opencv-python 과 충돌하기 때문이다.
녹화기도 화각 확인(face_box)에 같은 모델을 쓴다.

프레임마다 뽑는 값 (classify.FEATURES 순서와 맞춘다)
    pitch      머리 끄덕임 각도(도). 얼굴 변환 행렬에서 구한다.
    eye_down   blendshape eyeLookDown − eyeLookUp (양 눈 평균)
    iris_drop  눈꼬리를 잇는 선에서 홍채 중심까지의 수직 거리 / 눈 너비
    eye_open   위·아래 눈꺼풀 간격 / 눈 너비
그리고 확인용으로 yaw, roll, blink(eyeBlink 평균)를 함께 남긴다.

부호는 신경 쓰지 않아도 된다 — 보정(classify.fit_calibration)이 참가자마다
'건반 쪽'이 어느 방향인지 데이터에서 정한다.
"""

import math
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from gaze.classify import FEATURES
from gaze.recording import TOOL_DIR

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = TOOL_DIR / "models" / "face_landmarker.task"

COLUMNS = ("frame", "t", "face") + FEATURES + ("yaw", "roll", "blink")

# MediaPipe 얼굴 메시(홍채 포함 478점) 번호. '오른쪽'은 찍힌 사람 기준이다.
#   (바깥 눈꼬리, 안쪽 눈꼬리, 위 눈꺼풀, 아래 눈꺼풀, 홍채 중심)
_EYES = (
    (33, 133, 159, 145, 468),     # 오른쪽 눈
    (263, 362, 386, 374, 473),    # 왼쪽 눈
)


def ensure_model(path=MODEL_PATH):
    """얼굴 모델이 없으면 내려받는다 (약 3.6MB, 처음 한 번)."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"MediaPipe 얼굴 모델을 내려받습니다 (처음 한 번): {MODEL_URL}")
    tmp = path.with_name(path.name + ".part")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    tmp.replace(path)
    return path


def create_landmarker(model_path=None, video=True, details=True):
    """얼굴 landmarker.

    Args:
        video:   True 면 영상 프레임을 이어서 추적(VIDEO), False 면 낱장(IMAGE).
        details: blendshape·얼굴 변환 행렬까지 낼지. 화각 확인에는 필요 없다.
    """
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(ensure_model(model_path or MODEL_PATH))),
        running_mode=(vision.RunningMode.VIDEO if video
                      else vision.RunningMode.IMAGE),
        num_faces=1,
        output_face_blendshapes=details,
        output_facial_transformation_matrixes=details,
    )
    return vision.FaceLandmarker.create_from_options(options)


def face_box(landmarker, frame):
    """낱장(IMAGE) landmarker 로 얼굴 테두리 (x, y, w, h). 없으면 None."""
    height, width = frame.shape[:2]
    image = mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = landmarker.detect(image)
    if not result.face_landmarks:
        return None
    xs = [point.x for point in result.face_landmarks[0]]
    ys = [point.y for point in result.face_landmarks[0]]
    left, top = int(min(xs) * width), int(min(ys) * height)
    return (left, top, int(max(xs) * width) - left,
            int(max(ys) * height) - top)


def head_angles(matrix):
    """얼굴 변환 행렬 → (pitch, yaw, roll) 도."""
    rotation = np.asarray(matrix, dtype=float)[:3, :3]
    rotation = rotation / np.linalg.norm(rotation, axis=0)   # 배율을 뺀다
    pitch = math.degrees(math.atan2(rotation[2, 1], rotation[2, 2]))
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, -rotation[2, 0]))))
    roll = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    return pitch, yaw, roll


def eye_geometry(points):
    """랜드마크 (478, 2) 픽셀 좌표 → (iris_drop, eye_open) 양 눈 평균."""
    drops = []
    openings = []
    for outer, inner, upper, lower, iris in _EYES:
        a, b = points[outer], points[inner]
        width = np.linalg.norm(b - a)
        if width < 1e-6:
            continue
        along = (b - a) / width
        # 눈꼬리 선에 수직이면서 화면 아래를 향하는 방향 — 고개를 갸웃해도
        # (roll) 눈 기준의 '아래'를 잰다.
        down = np.array([-along[1], along[0]])
        if down[1] < 0:
            down = -down
        drops.append(float((points[iris] - (a + b) / 2) @ down) / width)
        openings.append(float(np.linalg.norm(points[lower] - points[upper]))
                        / width)
    if not drops:
        return math.nan, math.nan
    return float(np.mean(drops)), float(np.mean(openings))


def _blendshape(scores, name):
    return scores.get(name + "Left", 0.0), scores.get(name + "Right", 0.0)


def features_from_result(result, width, height):
    """FaceLandmarkerResult → 특징 dict. 얼굴이 없으면 None."""
    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]
    points = np.array([(p.x * width, p.y * height) for p in landmarks])
    iris_drop, eye_open = eye_geometry(points)

    pitch = yaw = roll = math.nan
    if result.facial_transformation_matrixes:
        pitch, yaw, roll = head_angles(result.facial_transformation_matrixes[0])

    scores = {}
    if result.face_blendshapes:
        scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
    down = np.mean(_blendshape(scores, "eyeLookDown"))
    up = np.mean(_blendshape(scores, "eyeLookUp"))
    blink = np.mean(_blendshape(scores, "eyeBlink"))

    return {
        "pitch": pitch,
        "eye_down": float(down - up),
        "iris_drop": iris_drop,
        "eye_open": eye_open,
        "yaw": yaw,
        "roll": roll,
        "blink": float(blink),
    }


def extract(video_path, frame_times, on_progress=None):
    """영상의 모든 프레임에서 특징을 뽑는다.

    Args:
        video_path:  녹화 영상.
        frame_times: 프레임별 PC 시각(초). 녹화기가 프레임마다 적은 값.
        on_progress: (처리한 프레임 수, 전체) 를 받는 함수. 선택.

    Returns:
        COLUMNS 를 키로 갖는 dict 목록. 얼굴이 없던 프레임은 값이 NaN.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

    rows = []
    total = len(frame_times)
    last_ms = -1
    try:
        with create_landmarker() as landmarker:
            for index, t in enumerate(frame_times):
                ok, frame = cap.read()
                if not ok:
                    break
                height, width = frame.shape[:2]
                image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # VIDEO 모드는 시각이 계속 늘어나야 한다 (ms 정수).
                ms = max(last_ms + 1, int(round((t - frame_times[0]) * 1000)))
                last_ms = ms
                found = features_from_result(
                    landmarker.detect_for_video(image, ms), width, height)

                row = {"frame": index, "t": t, "face": int(found is not None)}
                for column in COLUMNS[3:]:
                    row[column] = found[column] if found else math.nan
                rows.append(row)

                if on_progress is not None and (index % 100 == 0
                                                or index == total - 1):
                    on_progress(index + 1, total)
    finally:
        cap.release()

    if len(rows) < total:
        print(f"  ⚠️ 영상 프레임({len(rows)})이 시각 기록({total})보다 "
              f"적습니다. 있는 만큼만 씁니다.")
    return rows
