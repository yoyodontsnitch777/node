from __future__ import annotations

import copy
import math
import pickle
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import cv2
import torch


# ============================================================
# ComfyUI Node (pose_data + PKL)
# ============================================================

_GLOBAL_LOCK = threading.Lock()


class KPSSmoothPoseDataAndRender:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "filter_extra_people": ("BOOLEAN", {"default": True}),
                "smooth_alpha": ("FLOAT", {"default": 0.7, "min": 0.01, "max": 0.99, "step": 0.01}),
                "gap_frames": ("INT", {"default": 12, "min": 0, "max": 100, "step": 1}),
                "min_run_frames": ("INT", {"default": 3, "min": 1, "max": 60, "step": 1}),
                "conf_thresh_body": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "conf_thresh_hands": ("FLOAT", {"default": 0.60, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "POSEDATA")
    RETURN_NAMES = ("IMAGE", "pose_data")
    FUNCTION = "run"
    CATEGORY = "posedata"

    def run(self, pose_data, **kwargs):
        filter_extra_people = bool(kwargs.get("filter_extra_people", True))

        smooth_alpha = float(kwargs.get("smooth_alpha", 0.7))
        gap_frames = int(kwargs.get("gap_frames", 12))
        min_run_frames = int(kwargs.get("min_run_frames", 2))

        conf_thresh_body = float(kwargs.get("conf_thresh_body", 0.20))
        conf_thresh_hands = float(kwargs.get("conf_thresh_hands", 0.50))
        conf_thresh_face = 0.20

        force_body_18 = bool(kwargs.get("force_body_18", False))

        pose_data = _coerce_pose_data_to_obj(pose_data)
        frames_json_like, meta_ref = _pose_data_to_kps_frames(pose_data, force_body_18=force_body_18)

        with _GLOBAL_LOCK:
            old = _snapshot_tunable_globals()
            try:
                globals()["CONF_GATE_BODY"] = conf_thresh_body
                globals()["CONF_GATE_HAND"] = conf_thresh_hands
                globals()["CONF_GATE_FACE"] = conf_thresh_face

                globals()["ALPHA_BODY"] = smooth_alpha
                globals()["SUPER_SMOOTH_ALPHA"] = smooth_alpha
                globals()["MAX_GAP_FRAMES"] = gap_frames
                globals()["MIN_RUN_FRAMES"] = min_run_frames
                globals()["DENSE_SUPER_SMOOTH_ALPHA"] = smooth_alpha
                globals()["DENSE_MAX_GAP_FRAMES"] = gap_frames
                globals()["DENSE_MIN_RUN_FRAMES"] = min_run_frames
                globals()["FILTER_EXTRA_PEOPLE"] = filter_extra_people

                smoothed_frames = smooth_KPS_json_obj(
                    frames_json_like,
                    keep_face_untouched=False,
                    keep_hands_untouched=False,
                    filter_extra_people=filter_extra_people,
                )
            finally:
                _restore_tunable_globals(old)

        out_pose_data = _kps_frames_to_pose_data(pose_data, smoothed_frames, meta_ref, force_body_18=force_body_18)

        w, h = _extract_canvas_wh(smoothed_frames, default_w=720, default_h=1280)
        frames_np = []
        for fr in smoothed_frames:
            if isinstance(fr, dict) and fr.get("people"):
                img = _draw_pose_frame_full(
                    w,
                    h,
                    fr["people"][0],
                    conf_thresh_body=conf_thresh_body,
                    conf_thresh_hands=conf_thresh_hands,
                    conf_thresh_face=conf_thresh_face,
                )
            else:
                img = np.zeros((h, w, 3), dtype=np.uint8)
            frames_np.append(img)

        frames_t = torch.from_numpy(np.stack(frames_np, axis=0)).float() / 255.0
        return (frames_t, out_pose_data)


# ============================================================
# PKL / pose_data IO
# ============================================================


class _PoseDummyObj:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        elif isinstance(state, (list, tuple)) and len(state) == 2 and isinstance(state[0], dict):
            self.__dict__.update(state[0])
            if isinstance(state[1], dict):
                self.__dict__.update(state[1])
            else:
                self.__dict__["_slotstate"] = state[1]
        else:
            self.__dict__["_state"] = state


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        if module.startswith("numpy._globals"):
            module = module.replace("numpy._globals", "numpy", 1)
        if name in {"AAPoseMeta"}:
            return _PoseDummyObj
        try:
            return super().find_class(module, name)
        except Exception:
            return _PoseDummyObj


def _load_pose_data_pkl(path: str) -> Any:
    with open(path, "rb") as f:
        return _SafeUnpickler(f).load()


def _coerce_pose_data_to_obj(pd: Any) -> Any:
    if isinstance(pd, str):
        return _load_pose_data_pkl(pd)
    if isinstance(pd, dict) and "pose_data" in pd:
        return pd["pose_data"]
    return pd


def _as_attr(x: Any, key: str, default=None):
    if isinstance(x, dict):
        return x.get(key, default)
    return getattr(x, key, default)


def _set_attr(x: Any, key: str, value: Any):
    if isinstance(x, dict):
        x[key] = value
    else:
        setattr(x, key, value)


def _xy_p_to_flat(xy: Optional[np.ndarray], p: Optional[np.ndarray]) -> Optional[List[float]]:
    if xy is None:
        return None
    arr = np.asarray(xy)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    N = arr.shape[0]
    if p is None:
        pp = np.ones((N,), dtype=np.float32)
    else:
        pp = np.asarray(p).reshape(-1)
        if pp.shape[0] != N:
            pp = np.ones((N,), dtype=np.float32)
    out: List[float] = []
    for i in range(N):
        out.extend([float(arr[i, 0]), float(arr[i, 1]), float(pp[i])])
    return out


def _flat_to_xy_p(flat: Optional[List[float]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not isinstance(flat, list) or len(flat) % 3 != 0:
        return None, None
    N = len(flat) // 3
    xy = np.zeros((N, 2), dtype=np.float32)
    p = np.zeros((N,), dtype=np.float32)
    for i in range(N):
        xy[i, 0] = float(flat[3 * i + 0])
        xy[i, 1] = float(flat[3 * i + 1])
        p[i] = float(flat[3 * i + 2])
    return xy, p


def _pose_data_to_kps_frames(pose_data: Any, *, force_body_18: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pose_metas = _as_attr(pose_data, "pose_metas", None)
    if pose_metas is None:
        pose_metas = _as_attr(pose_data, "frames", None)
    if pose_metas is None or not isinstance(pose_metas, list):
        raise ValueError("pose_data does not contain 'pose_metas' list.")
    frames: List[Dict[str, Any]] = []
    for meta in pose_metas:
        h = _as_attr(meta, "height", 1280)
        w = _as_attr(meta, "width", 720)
        kps_body = _as_attr(meta, "kps_body", None)
        kps_body_p = _as_attr(meta, "kps_body_p", None)
        kps_face = _as_attr(meta, "kps_face", None)
        kps_face_p = _as_attr(meta, "kps_face_p", None)
        kps_lhand = _as_attr(meta, "kps_lhand", None)
        kps_lhand_p = _as_attr(meta, "kps_lhand_p", None)
        kps_rhand = _as_attr(meta, "kps_rhand", None)
        kps_rhand_p = _as_attr(meta, "kps_rhand_p", None)

        pose_flat = _xy_p_to_flat(kps_body, kps_body_p)
        face_flat = _xy_p_to_flat(kps_face, kps_face_p)
        lh_flat = _xy_p_to_flat(kps_lhand, kps_lhand_p)
        rh_flat = _xy_p_to_flat(kps_rhand, kps_rhand_p)

        if force_body_18 and isinstance(pose_flat, list) and len(pose_flat) >= 18 * 3:
            pose_flat = pose_flat[: 18 * 3]

        person = {
            "pose_keypoints_2d": pose_flat if pose_flat is not None else [],
            "face_keypoints_2d": face_flat if face_flat is not None else [],
            "hand_left_keypoints_2d": lh_flat,
            "hand_right_keypoints_2d": rh_flat,
        }
        frame = {"people": [person], "canvas_height": int(h), "canvas_width": int(w)}
        frames.append(frame)

    meta_ref = {"pose_metas": pose_metas, "len": len(pose_metas)}
    return frames, meta_ref


def _kps_frames_to_pose_data(
    pose_data_in: Any, frames_kps: List[Dict[str, Any]], meta_ref: Dict[str, Any], *, force_body_18: bool
) -> Any:
    out_pd = copy.deepcopy(pose_data_in)
    pose_metas_out = _as_attr(out_pd, "pose_metas", None)
    if pose_metas_out is None:
        pose_metas_out = meta_ref.get("pose_metas")
    if pose_metas_out is None or not isinstance(pose_metas_out, list):
        raise ValueError("Failed to locate pose_metas in output pose_data.")

    T = min(len(pose_metas_out), len(frames_kps))
    for t in range(T):
        meta = pose_metas_out[t]
        fr = frames_kps[t]
        people = fr.get("people", []) if isinstance(fr, dict) else []
        p0 = people[0] if people else None
        if not isinstance(p0, dict):
            continue

        pose_flat = p0.get("pose_keypoints_2d")
        face_flat = p0.get("face_keypoints_2d")
        lh_flat = p0.get("hand_left_keypoints_2d")
        rh_flat = p0.get("hand_right_keypoints_2d")

        if force_body_18 and isinstance(pose_flat, list) and len(pose_flat) >= 18 * 3:
            pose_flat = pose_flat[: 18 * 3]

        body_xy, body_p = _flat_to_xy_p(pose_flat if isinstance(pose_flat, list) else None)
        face_xy, face_p = _flat_to_xy_p(face_flat if isinstance(face_flat, list) else None)
        lh_xy, lh_p = _flat_to_xy_p(lh_flat if isinstance(lh_flat, list) else None)
        rh_xy, rh_p = _flat_to_xy_p(rh_flat if isinstance(rh_flat, list) else None)

        if body_xy is not None and body_p is not None:
            _set_attr(meta, "kps_body", body_xy.astype(np.float32, copy=False))
            _set_attr(meta, "kps_body_p", body_p.astype(np.float32, copy=False))
        if face_xy is not None and face_p is not None:
            _set_attr(meta, "kps_face", face_xy.astype(np.float32, copy=False))
            _set_attr(meta, "kps_face_p", face_p.astype(np.float32, copy=False))
        if lh_xy is not None and lh_p is not None:
            _set_attr(meta, "kps_lhand", lh_xy.astype(np.float32, copy=False))
            _set_attr(meta, "kps_lhand_p", lh_p.astype(np.float32, copy=False))
        if rh_xy is not None and rh_p is not None:
            _set_attr(meta, "kps_rhand", rh_xy.astype(np.float32, copy=False))
            _set_attr(meta, "kps_rhand_p", rh_p.astype(np.float32, copy=False))

        if isinstance(fr, dict):
            if "canvas_width" in fr:
                _set_attr(meta, "width", int(fr["canvas_width"]))
            if "canvas_height" in fr:
                _set_attr(meta, "height", int(fr["canvas_height"]))

    _set_attr(out_pd, "pose_metas", pose_metas_out)
    return out_pd


def _extract_canvas_wh(data: Any, default_w: int, default_h: int) -> Tuple[int, int]:
    w, h = int(default_w), int(default_h)
    if isinstance(data, list):
        for fr in data:
            if isinstance(fr, dict) and "canvas_width" in fr and "canvas_height" in fr:
                try:
                    w = int(fr["canvas_width"])
                    h = int(fr["canvas_height"])
                    break
                except Exception:
                    pass
    return w, h


# ============================================================
# === START: smooth_KPS_json.py logic (ported as-is)
# ============================================================

ROOTSCALE_CARRY_ENABLED = True
CARRY_MAX_FRAMES = 48
CARRY_MIN_ANCHORS = 2
CARRY_ANCHOR_JOINTS = [0, 1, 2, 5, 3, 6, 4, 7]
CARRY_CONF_GATE = 0.20

FILTER_EXTRA_PEOPLE = True
MAIN_PERSON_MODE = "longest_track"
TRACK_MATCH_MIN_PX = 80.0
TRACK_MATCH_FACTOR = 3.0
TRACK_MAX_FRAME_GAP = 32

SPATIAL_OUTLIER_FIX = True
BONE_MAX_FACTOR = 2.3
TORSO_RADIUS_FACTOR = 4.0

ALPHA_BODY = 0.70
MAX_STEP_BODY = 60.0
VEL_ALPHA = 0.45
EPS = 0.3
CONF_GATE_BODY = 0.20
CONF_FLOOR_BODY = 0.00

TRACK_DIST_PENALTY = 1.5
FACE_WEIGHT_IN_SCORE = 0.15
HAND_WEIGHT_IN_SCORE = 0.35

ALLOW_DISAPPEAR_JOINTS = {3, 4, 6, 7}

GAP_FILL_ENABLED = True
MAX_GAP_FRAMES = 12
MIN_RUN_FRAMES = 2

TORSO_SYNC_ENABLED = True
TORSO_JOINTS = {1, 2, 5, 8, 11}
TORSO_LOOKAHEAD_FRAMES = 32

SUPER_SMOOTH_ENABLED = True
SUPER_SMOOTH_ALPHA = 0.7
SUPER_SMOOTH_MIN_CONF = 0.20

MEDIAN3_ENABLED = True
FACE_SMOOTH_ENABLED = True
HANDS_SMOOTH_ENABLED = False

CONF_GATE_FACE = 0.20
CONF_GATE_HAND = 0.50

HAND_MIN_POINTS_PRESENT = 7
MIN_HAND_RUN_FRAMES = 6

DENSE_GAP_FILL_ENABLED = False
DENSE_MAX_GAP_FRAMES = 8
DENSE_MIN_RUN_FRAMES = 2
DENSE_MEDIAN3_ENABLED = False
DENSE_SUPER_SMOOTH_ENABLED = False
DENSE_SUPER_SMOOTH_ALPHA = 0.7


def _snapshot_tunable_globals() -> Dict[str, Any]:
    keys = [
        "FILTER_EXTRA_PEOPLE",
        "SUPER_SMOOTH_ALPHA",
        "MAX_GAP_FRAMES",
        "MIN_RUN_FRAMES",
        "DENSE_SUPER_SMOOTH_ALPHA",
        "DENSE_MAX_GAP_FRAMES",
        "DENSE_MIN_RUN_FRAMES",
        "ALPHA_BODY",
        "CONF_GATE_BODY",
        "CONF_GATE_HAND",
        "CONF_GATE_FACE",
    ]
    return {k: globals().get(k) for k in keys}


def _restore_tunable_globals(old: Dict[str, Any]) -> None:
    for k, v in old.items():
        globals()[k] = v


def _is_valid_xyc(x: float, y: float, c: float) -> bool:
    if c is None or c <= 0:
        return False
    if x == 0 and y == 0:
        return False
    if math.isnan(x) or math.isnan(y) or math.isnan(c):
        return False
    return True


def _reshape_keypoints_2d(arr: List[float]) -> List[Tuple[float, float, float]]:
    if arr is None:
        return []
    out = []
    for i in range(0, len(arr), 3):
        out.append((float(arr[i]), float(arr[i + 1]), float(arr[i + 2])))
    return out


def _flatten_keypoints_2d(kps: List[Tuple[float, float, float]]) -> List[float]:
    out: List[float] = []
    for x, y, c in kps:
        out.extend([float(x), float(y), float(c)])
    return out


def _sum_conf(arr: Optional[List[float]], sample_step: int = 1) -> float:
    if not arr:
        return 0.0
    s = 0.0
    for i in range(2, len(arr), 3 * sample_step):
        try:
            c = float(arr[i])
        except Exception:
            c = 0.0
        if c > 0:
            s += c
    return s


def _body_center_from_pose(pose_arr: Optional[List[float]]) -> Optional[Tuple[float, float]]:
    if not pose_arr:
        return None
    kps = _reshape_keypoints_2d(pose_arr)
    idxs = [2, 5, 8, 11, 1]
    pts = []
    for idx in idxs:
        if idx < len(kps) and _is_valid_xyc(*kps[idx]):
            pts.append((kps[idx][0], kps[idx][1]))
    if not pts:
        for x, y, c in kps:
            if _is_valid_xyc(x, y, c):
                pts.append((x, y))
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _choose_single_person(
    people: List[Dict[str, Any]], prev_center: Optional[Tuple[float, float]]
) -> Optional[Dict[str, Any]]:
    if not people:
        return None
    best = None
    best_score = -1e18
    for p in people:
        pose = p.get("pose_keypoints_2d")
        score = _sum_conf(pose)
        score += FACE_WEIGHT_IN_SCORE * _sum_conf(p.get("face_keypoints_2d"), 4)
        score += HAND_WEIGHT_IN_SCORE * (
            _sum_conf(p.get("hand_left_keypoints_2d"), 2) + _sum_conf(p.get("hand_right_keypoints_2d"), 2)
        )
        center = _body_center_from_pose(pose)
        if prev_center is not None and center is not None:
            score -= TRACK_DIST_PENALTY * _dist(prev_center, center)
        if score > best_score:
            best_score = score
            best = p
    return best


@dataclass
class _Track:
    frames: Dict[int, Dict[str, Any]]
    centers: Dict[int, Tuple[float, float]]
    last_t: int
    last_center: Tuple[float, float]


def _estimate_torso_scale(pose: List[Tuple[float, float, float]]) -> Optional[float]:
    def dist(i, k):
        if i >= len(pose) or k >= len(pose):
            return None
        if not _is_valid_xyc(*pose[i]) or not _is_valid_xyc(*pose[k]):
            return None
        return math.hypot(pose[i][0] - pose[k][0], pose[i][1] - pose[k][1])

    cand = [c for c in [dist(2, 5), dist(8, 11), dist(1, 8), dist(1, 11)] if c is not None and c > 1e-3]
    if not cand:
        return None
    return float(sum(cand) / len(cand))


def _track_match_threshold_from_pose(pose_arr: Optional[List[float]]) -> float:
    if isinstance(pose_arr, list):
        s = _estimate_torso_scale(_reshape_keypoints_2d(pose_arr))
        if s is not None:
            return max(float(TRACK_MATCH_MIN_PX), float(TRACK_MATCH_FACTOR) * float(s))
    return float(max(TRACK_MATCH_MIN_PX, 120.0))


def _build_tracks_over_video(frames_data: List[Any]) -> List[_Track]:
    tracks: List[_Track] = []
    for t, frame in enumerate(frames_data):
        if not isinstance(frame, dict):
            continue
        people = frame.get("people", [])
        if not isinstance(people, list) or not people:
            continue

        cand = []
        for i, p in enumerate(people):
            if not isinstance(p, dict):
                continue
            c = _body_center_from_pose(p.get("pose_keypoints_2d"))
            if c is not None:
                cand.append((i, p, c))
        if not cand:
            continue

        used = set()
        track_order = sorted(range(len(tracks)), key=lambda k: tracks[k].last_t, reverse=True)
        for k in track_order:
            tr = tracks[k]
            if (t - tr.last_t) > int(TRACK_MAX_FRAME_GAP):
                continue
            best_idx, best_d = None, 1e18
            for i, p, cc in cand:
                if i in used:
                    continue
                thr = _track_match_threshold_from_pose(p.get("pose_keypoints_2d"))
                d = _dist(tr.last_center, cc)
                if d <= thr and d < best_d:
                    best_d = d
                    best_idx = i
            if best_idx is not None:
                i, p, cc = next(x for x in cand if x[0] == best_idx)
                used.add(i)
                tr.frames[t], tr.centers[t], tr.last_t, tr.last_center = p, cc, t, cc
        for i, p, cc in cand:
            if i not in used:
                tracks.append(_Track(frames={t: p}, centers={t: cc}, last_t=t, last_center=cc))
    return tracks


def _track_presence_score(tr: _Track) -> Tuple[int, float, float]:
    face_sum, body_sum = 0.0, 0.0
    for p in tr.frames.values():
        face_sum += _sum_conf(p.get("face_keypoints_2d"), 4)
        body_sum += _sum_conf(p.get("pose_keypoints_2d"), 1)
    return (len(tr.frames), face_sum, body_sum)


def _pick_main_track(tracks: List[_Track]) -> Optional[_Track]:
    if not tracks:
        return None
    best, best_key = None, (-1, -1e18, -1e18)
    for tr in tracks:
        key = _track_presence_score(tr)
        if key > best_key:
            best_key, best = key, tr
    return best


@dataclass
class BodyState:
    last_xy: List[Optional[Tuple[float, float]]]
    last_v: List[Tuple[float, float]]

    def __init__(self, joints: int):
        self.last_xy = [None] * joints
        self.last_v = [(0.0, 0.0)] * joints


def _smooth_body_pose(pose_arr: Optional[List[float]], state: BodyState) -> Optional[List[float]]:
    if pose_arr is None:
        return None
    kps = _reshape_keypoints_2d(pose_arr)
    J = len(kps)
    if len(state.last_xy) != J:
        state.last_xy = [None] * J
        state.last_v = [(0.0, 0.0)] * J

    out: List[Tuple[float, float, float]] = []
    for j in range(J):
        x, y, c = kps[j]
        last = state.last_xy[j]
        vx_last, vy_last = state.last_v[j]
        valid_in = _is_valid_xyc(x, y, c) and (c >= CONF_GATE_BODY)

        if valid_in:
            if last is None:
                state.last_xy[j] = (x, y)
                state.last_v[j] = (0.0, 0.0)
                out.append((x, y, float(c)))
                continue

            dx_raw, dy_raw = x - last[0], y - last[1]
            if abs(dx_raw) < EPS:
                dx_raw = 0.0
            if abs(dy_raw) < EPS:
                dy_raw = 0.0

            vx = VEL_ALPHA * dx_raw + (1.0 - VEL_ALPHA) * vx_last
            vy = VEL_ALPHA * dy_raw + (1.0 - VEL_ALPHA) * vy_last
            nx = ALPHA_BODY * x + (1.0 - ALPHA_BODY) * (last[0] + vx)
            ny = ALPHA_BODY * y + (1.0 - ALPHA_BODY) * (last[1] + vy)

            d = math.hypot(nx - last[0], ny - last[1])
            if d > MAX_STEP_BODY and d > 1e-6:
                scale = MAX_STEP_BODY / d
                nx = last[0] + (nx - last[0]) * scale
                ny = last[1] + (ny - last[1]) * scale
                vx, vy = nx - last[0], ny - last[1]

            state.last_xy[j], state.last_v[j] = (nx, ny), (vx, vy)
            out.append((nx, ny, float(c)))
        else:
            state.last_xy[j] = None
            state.last_v[j] = (0.0, 0.0)
            out.append((0.0, 0.0, 0.0))

    return _flatten_keypoints_2d(out)


COCO18_EDGES = [
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (8, 11),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
]
HAND21_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]

_NEIGHBORS = None


def _build_neighbors():
    global _NEIGHBORS
    if _NEIGHBORS is not None:
        return
    _NEIGHBORS = {}
    for a, b in COCO18_EDGES:
        _NEIGHBORS.setdefault(a, set()).add(b)
        _NEIGHBORS.setdefault(b, set()).add(a)


def _suppress_spatial_outliers_in_pose_arr(
    pose_arr: Optional[List[float]], *, conf_gate: float
) -> Optional[List[float]]:
    if not isinstance(pose_arr, list) or len(pose_arr) % 3 != 0:
        return pose_arr
    pose = _reshape_keypoints_2d(pose_arr)
    J = len(pose)
    center = _body_center_from_pose(pose_arr)
    scale = _estimate_torso_scale(pose)
    if center is None or scale is None:
        return pose_arr

    max_r, max_bone = TORSO_RADIUS_FACTOR * scale, BONE_MAX_FACTOR * scale
    out = [list(p) for p in pose]

    def visible(j):
        return j < J and out[j][2] >= conf_gate and not (out[j][0] == 0 and out[j][1] == 0)

    for j in range(J):
        if visible(j) and math.hypot(out[j][0] - center[0], out[j][1] - center[1]) > max_r:
            out[j] = [0.0, 0.0, 0.0]

    for a, b in COCO18_EDGES:
        if a >= J or b >= J:
            continue
        if not visible(a) or not visible(b):
            continue
        if math.hypot(out[a][0] - out[b][0], out[a][1] - out[b][1]) > max_bone:
            if out[a][2] <= out[b][2]:
                out[a] = [0.0, 0.0, 0.0]
            else:
                out[b] = [0.0, 0.0, 0.0]

    flat = []
    for p in out:
        flat.extend(p)
    return flat


def _suppress_isolated_joints_in_pose_arr(
    pose_arr: Optional[List[float]], *, conf_gate: float, keep: set[int] = None
) -> Optional[List[float]]:
    if not isinstance(pose_arr, list) or len(pose_arr) % 3 != 0:
        return pose_arr
    _build_neighbors()
    pose = _reshape_keypoints_2d(pose_arr)
    J, out = len(pose), [list(p) for p in pose]
    keep = keep or set()

    def vis(j):
        return j < J and out[j][2] >= conf_gate and not (out[j][0] == 0 and out[j][1] == 0)

    for j in range(J):
        if j in keep or not vis(j):
            continue
        if not any(n < J and vis(n) for n in _NEIGHBORS.get(j, set())):
            out[j] = [0.0, 0.0, 0.0]

    flat = []
    for p in out:
        flat.extend(p)
    return flat


def _denoise_and_fill_gaps_pose_seq(
    pose_arr_seq: List[Optional[List[float]]], *, conf_gate: float, min_run: int, max_gap: int
) -> List[Optional[List[float]]]:
    if not pose_arr_seq:
        return pose_arr_seq
    J = next(
        (len(arr) // 3 for arr in pose_arr_seq if isinstance(arr, list) and len(arr) % 3 == 0 and len(arr) > 0), None
    )
    if J is None:
        return pose_arr_seq
    T = len(pose_arr_seq)
    out_seq = [list(arr) if isinstance(arr, list) and len(arr) == J * 3 else arr for arr in pose_arr_seq]

    def is_vis(arr, j):
        return float(arr[3 * j + 2]) >= conf_gate and not (float(arr[3 * j + 0]) == 0 and float(arr[3 * j + 1]) == 0)

    for j in range(J):
        start = None
        for t in range(T + 1):
            cur = t < T and isinstance(out_seq[t], list) and is_vis(out_seq[t], j)
            if cur and start is None:
                start = t
            if not cur and start is not None:
                if (t - start) < min_run:
                    for k in range(start, t):
                        if isinstance(out_seq[k], list):
                            out_seq[k][3 * j : 3 * j + 3] = [0.0, 0.0, 0.0]
                start = None

    for j in range(J):
        t = 0
        while t < T:
            arr = out_seq[t]
            if isinstance(arr, list) and is_vis(arr, j):
                last_vis_t = t
                t += 1
                while t < T:
                    if isinstance(out_seq[t], list) and is_vis(out_seq[t], j):
                        break
                    t += 1
                if t < T and (t - last_vis_t - 1) > 0 and (t - last_vis_t - 1) <= max_gap:
                    a, b = out_seq[last_vis_t], out_seq[t]
                    ax, ay, ac = float(a[3 * j]), float(a[3 * j + 1]), float(a[3 * j + 2])
                    bx, by, bc = float(b[3 * j]), float(b[3 * j + 1]), float(b[3 * j + 2])
                    for k in range(last_vis_t + 1, t):
                        if isinstance(out_seq[k], list):
                            r = (k - last_vis_t) / (t - last_vis_t)
                            out_seq[k][3 * j : 3 * j + 3] = [ax + (bx - ax) * r, ay + (by - ay) * r, min(ac, bc)]
            else:
                t += 1
    return out_seq


def _zero_lag_ema_pose_seq(
    pose_seq: List[Optional[List[float]]], *, alpha: float, conf_gate: float
) -> List[Optional[List[float]]]:
    if not pose_seq:
        return pose_seq
    J = next((len(arr) // 3 for arr in pose_seq if isinstance(arr, list) and len(arr) % 3 == 0 and len(arr) > 0), None)
    if J is None:
        return pose_seq
    T = len(pose_seq)

    def is_vis(arr, j):
        return float(arr[3 * j + 2]) >= conf_gate and not (float(arr[3 * j + 0]) == 0 and float(arr[3 * j + 1]) == 0)

    fwd, last = [None] * T, [None] * J
    for t in range(T):
        if not isinstance(pose_seq[t], list) or len(pose_seq[t]) != J * 3:
            fwd[t] = pose_seq[t]
            continue
        out = list(pose_seq[t])
        for j in range(J):
            if is_vis(pose_seq[t], j):
                x, y = float(pose_seq[t][3 * j]), float(pose_seq[t][3 * j + 1])
                sx, sy = (
                    (x, y)
                    if last[j] is None
                    else (alpha * x + (1 - alpha) * last[j][0], alpha * y + (1 - alpha) * last[j][1])
                )
                last[j], out[3 * j], out[3 * j + 1] = (sx, sy), float(sx), float(sy)
            else:
                last[j] = None
        fwd[t] = out

    bwd, last = [None] * T, [None] * J
    for t in range(T - 1, -1, -1):
        if not isinstance(fwd[t], list) or len(fwd[t]) != J * 3:
            bwd[t] = fwd[t]
            continue
        out = list(fwd[t])
        for j in range(J):
            if is_vis(fwd[t], j):
                x, y = float(fwd[t][3 * j]), float(fwd[t][3 * j + 1])
                sx, sy = (
                    (x, y)
                    if last[j] is None
                    else (alpha * x + (1 - alpha) * last[j][0], alpha * y + (1 - alpha) * last[j][1])
                )
                last[j], out[3 * j], out[3 * j + 1] = (sx, sy), float(sx), float(sy)
            else:
                last[j] = None
        bwd[t] = out
    return bwd


def _apply_root_scale(
    pose_arr: Optional[List[float]], *, src_root, src_scale, dst_root, dst_scale
) -> Optional[List[float]]:
    if not isinstance(pose_arr, list) or len(pose_arr) % 3 != 0 or src_scale <= 1e-6 or dst_scale <= 1e-6:
        return pose_arr
    kps = _reshape_keypoints_2d(pose_arr)
    s = dst_scale / src_scale
    out = [
        (
            (dst_root[0] + (x - src_root[0]) * s, dst_root[1] + (y - src_root[1]) * s, c)
            if c > 0 and not (x == 0 and y == 0)
            else (x, y, c)
        )
        for x, y, c in kps
    ]
    return _flatten_keypoints_2d(out)


def _carry_pose_when_torso_missing(
    pose_seq: List[Optional[List[float]]], *, conf_gate, max_carry, anchor_joints, min_anchors
) -> List[Optional[List[float]]]:
    if not pose_seq:
        return pose_seq
    J = next((len(a) // 3 for a in pose_seq if isinstance(a, list) and len(a) % 3 == 0 and len(a) > 0), None)
    if J is None:
        return pose_seq
    out = [a if a is None else list(a) for a in pose_seq]
    FILL = {1, 8, 9, 10, 11, 12, 13} - set(ALLOW_DISAPPEAR_JOINTS)

    def is_vis(arr, j):
        return float(arr[3 * j + 2]) >= conf_gate and not (float(arr[3 * j]) == 0 and float(arr[3 * j + 1]) == 0)

    def rs_anchors(arr):
        pts = [(float(arr[3 * j]), float(arr[3 * j + 1])) for j in anchor_joints if j < J and is_vis(arr, j)]
        if len(pts) < min_anchors:
            return None
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        s = max(max(xs) - min(xs), max(ys) - min(ys))
        if s <= 1e-3:
            return None
        return (sum(xs) / len(pts), sum(ys) / len(pts)), float(s)

    last_good, last_rs, carry = None, None, 0
    for t, arr in enumerate(out):
        if not isinstance(arr, list) or len(arr) != J * 3:
            continue
        rs = rs_anchors(arr)
        if (
            sum(1 for j in anchor_joints if j < J and is_vis(arr, j)) >= min_anchors
            and rs
            and sum(1 for j in FILL if j < J and is_vis(arr, j)) >= 2
        ):
            last_good, last_rs, carry = list(arr), rs, max_carry
            continue
        if rs and last_good and last_rs and carry > 0:
            carried = _apply_root_scale(
                last_good, src_root=last_rs[0], src_scale=last_rs[1], dst_root=rs[0], dst_scale=rs[1]
            )
            if isinstance(carried, list) and len(carried) == J * 3:
                for j in FILL:
                    if (
                        j < J
                        and not is_vis(arr, j)
                        and (float(carried[3 * j]) != 0 or float(carried[3 * j + 1]) != 0)
                        and float(carried[3 * j + 2]) > 0
                    ):
                        arr[3 * j : 3 * j + 3] = [
                            float(carried[3 * j]),
                            float(carried[3 * j + 1]),
                            max(min(float(carried[3 * j + 2]), 0.60), conf_gate),
                        ]
                out[t], carry = arr, carry - 1
                continue
        carry = max(carry - 1, 0)
    return out


def _force_full_torso_pair(
    pose_seq: List[Optional[List[float]]],
    *,
    conf_gate,
    anchor_joints,
    min_anchors,
    max_lookback=240,
    fill_legs_with_hip=True,
    always_fill_if_one_hip=True,
) -> List[Optional[List[float]]]:
    if not pose_seq:
        return pose_seq
    J = next((len(a) // 3 for a in pose_seq if isinstance(a, list) and len(a) % 3 == 0 and len(a) > 0), None)
    if J is None:
        return pose_seq
    out = [a if a is None else list(a) for a in pose_seq]

    def is_vis(arr, j):
        return (
            j < J and float(arr[3 * j + 2]) >= conf_gate and not (float(arr[3 * j]) == 0 and float(arr[3 * j + 1]) == 0)
        )

    def rs_anchors(arr):
        pts = [(float(arr[3 * j]), float(arr[3 * j + 1])) for j in anchor_joints if j < J and is_vis(arr, j)]
        if len(pts) < min_anchors:
            return None
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        s = max(max(xs) - min(xs), max(ys) - min(ys))
        return ((sum(xs) / len(pts), sum(ys) / len(pts)), float(s)) if s > 1e-3 else None

    last_idx, last_f, last_rs = None, None, None
    for t, arr in enumerate(out):
        if not isinstance(arr, list) or len(arr) != J * 3:
            continue
        rs = rs_anchors(arr)
        r_ok, l_ok = is_vis(arr, 8), is_vis(arr, 11)
        if rs and sum(1 for j in anchor_joints if is_vis(arr, j)) >= min_anchors and r_ok and l_ok:
            last_idx, last_f, last_rs = t, list(arr), rs
            continue
        if (
            not last_f
            or not last_rs
            or (t - last_idx) > max_lookback
            or not rs
            or (r_ok and l_ok)
            or (not r_ok and not l_ok and not always_fill_if_one_hip)
        ):
            continue
        carried = _apply_root_scale(last_f, src_root=last_rs[0], src_scale=last_rs[1], dst_root=rs[0], dst_scale=rs[1])
        if not isinstance(carried, list) or len(carried) != J * 3:
            continue

        def cp(j):
            if (
                j < J
                and not is_vis(arr, j)
                and (float(carried[3 * j]) != 0 or float(carried[3 * j + 1]) != 0)
                and float(carried[3 * j + 2]) > 0
            ):
                arr[3 * j : 3 * j + 3] = [
                    float(carried[3 * j]),
                    float(carried[3 * j + 1]),
                    max(min(float(carried[3 * j + 2]), 0.60), conf_gate),
                ]

        if not r_ok:
            cp(8)
            if fill_legs_with_hip:
                cp(9)
                cp(10)
        if not l_ok:
            cp(11)
            if fill_legs_with_hip:
                cp(12)
                cp(13)
        out[t] = arr
    return out


def _median3_pose_seq(pose_seq: List[Optional[List[float]]], *, conf_gate: float) -> List[Optional[List[float]]]:
    if not pose_seq:
        return pose_seq
    J = next((len(a) // 3 for a in pose_seq if isinstance(a, list) and len(a) % 3 == 0 and len(a) > 0), None)
    if J is None:
        return pose_seq
    T = len(pose_seq)

    def is_vis(arr, j):
        return float(arr[3 * j + 2]) >= conf_gate and not (float(arr[3 * j]) == 0 and float(arr[3 * j + 1]) == 0)

    out_seq = []
    for t in range(T):
        if not isinstance(pose_seq[t], list) or len(pose_seq[t]) != J * 3:
            out_seq.append(pose_seq[t])
            continue
        out = list(pose_seq[t])
        a0, a1, a2 = pose_seq[max(0, t - 1)], pose_seq[t], pose_seq[min(T - 1, t + 1)]
        for j in range(J):
            if not is_vis(pose_seq[t], j):
                continue
            xs, ys = [], []
            for aa in (a0, a1, a2):
                if isinstance(aa, list) and len(aa) == J * 3 and is_vis(aa, j):
                    xs.append(float(aa[3 * j]))
                    ys.append(float(aa[3 * j + 1]))
            if len(xs) >= 2:
                xs.sort()
                ys.sort()
                out[3 * j], out[3 * j + 1] = float(xs[len(xs) // 2]), float(ys[len(ys) // 2])
        out_seq.append(out)
    return out_seq


def _sync_group_appearances(
    pose_arr_seq: List[Optional[List[float]]], *, group: set[int], conf_gate: float, lookahead: int
) -> List[Optional[List[float]]]:
    if not pose_arr_seq:
        return pose_arr_seq
    J = next((len(a) // 3 for a in pose_arr_seq if isinstance(a, list) and len(a) % 3 == 0 and len(a) > 0), None)
    if J is None:
        return pose_arr_seq
    T = len(pose_arr_seq)
    out = [list(a) if isinstance(a, list) and len(a) == J * 3 else a for a in pose_arr_seq]

    def is_vis(arr, j):
        return float(arr[3 * j + 2]) >= conf_gate and not (float(arr[3 * j]) == 0 and float(arr[3 * j + 1]) == 0)

    for t in range(T):
        arr = out[t]
        if not isinstance(arr, list):
            continue
        vis = {j for j in group if j < J and is_vis(arr, j)}
        if not vis:
            continue
        for j in list({j for j in group if j < J and j not in vis}):
            t2 = next(
                (
                    tt
                    for tt in range(t + 1, min(T, t + 1 + lookahead))
                    if isinstance(out[tt], list) and is_vis(out[tt], j)
                ),
                None,
            )
            if t2 is None:
                continue
            last_t = next((tb for tb in range(t - 1, -1, -1) if isinstance(out[tb], list) and is_vis(out[tb], j)), None)
            b = out[t2]
            if last_t is None:
                for k in range(t, t2):
                    if isinstance(out[k], list):
                        out[k][3 * j : 3 * j + 3] = b[3 * j : 3 * j + 3]
            else:
                a = out[last_t]
                if float(a[3 * j]) == 0 and float(a[3 * j + 1]) == 0:
                    continue
                c_fill = min(float(a[3 * j + 2]), float(b[3 * j + 2]))
                for tt in range(t, t2):
                    if isinstance(out[tt], list):
                        r = (tt - last_t) / (t2 - last_t)
                        out[tt][3 * j : 3 * j + 3] = [
                            float(a[3 * j]) + (float(b[3 * j]) - float(a[3 * j])) * r,
                            float(a[3 * j + 1]) + (float(b[3 * j + 1]) - float(a[3 * j + 1])) * r,
                            float(c_fill),
                        ]
    return out


def _count_valid_points(arr: Optional[List[float]], *, conf_gate: float) -> int:
    if not isinstance(arr, list) or len(arr) % 3 != 0:
        return 0
    return sum(
        1
        for i in range(0, len(arr), 3)
        if float(arr[i + 2]) >= conf_gate and not (float(arr[i]) == 0 and float(arr[i + 1]) == 0)
    )


def _zero_out_kps(arr: Optional[List[float]]) -> Optional[List[float]]:
    if not isinstance(arr, list) or len(arr) % 3 != 0:
        return arr
    out = list(arr)
    for i in range(0, len(out), 3):
        out[i : i + 3] = [0.0, 0.0, 0.0]
    return out


def _pin_body_wrist_to_hand(
    p_out: Dict[str, Any], *, side: str, conf_gate_body: float, conf_gate_hand: float, blend: float
) -> None:
    bw, hk = (4, "hand_right_keypoints_2d") if side == "right" else (7, "hand_left_keypoints_2d")
    pose, hand = p_out.get("pose_keypoints_2d"), p_out.get(hk)
    if not isinstance(pose, list) or not isinstance(hand, list) or len(pose) < (bw * 3 + 3) or len(hand) < 3:
        return
    hx, hy, hc = float(hand[0]), float(hand[1]), float(hand[2])
    if hc < conf_gate_hand or (hx == 0.0 and hy == 0.0):
        return
    bx, by, bc = float(pose[bw * 3]), float(pose[bw * 3 + 1]), float(pose[bw * 3 + 2])
    if bc < conf_gate_body or (bx == 0.0 and by == 0.0):
        pose[bw * 3 : bw * 3 + 3] = [hx, hy, float(max(bc, min(hc, 0.9)))]
    else:
        pose[bw * 3 : bw * 3 + 3] = [
            bx * (1.0 - blend) + hx * blend,
            by * (1.0 - blend) + hy * blend,
            float(min(bc, hc)),
        ]
    p_out["pose_keypoints_2d"] = pose


def _fix_elbow_using_wrist(p_out: Dict[str, Any], *, side: str, conf_gate: float) -> None:
    pose = p_out.get("pose_keypoints_2d")
    if not isinstance(pose, list) or len(pose) % 3 != 0:
        return
    sh, el, wr = (2, 3, 4) if side == "right" else (5, 6, 7)

    def vis(x, y, c):
        return c >= conf_gate and not (x == 0.0 and y == 0.0)

    sx, sy, sc = float(pose[3 * sh]), float(pose[3 * sh + 1]), float(pose[3 * sh + 2])
    ex, ey, ec = float(pose[3 * el]), float(pose[3 * el + 1]), float(pose[3 * el + 2])
    wx, wy, wc = float(pose[3 * wr]), float(pose[3 * wr + 1]), float(pose[3 * wr + 2])
    if not vis(sx, sy, sc) or not vis(wx, wy, wc):
        return
    if vis(ex, ey, ec):
        Lse, Lew = math.hypot(ex - sx, ey - sy), math.hypot(wx - ex, wy - ey)
    else:
        dsw = math.hypot(wx - sx, wy - sy)
        if dsw < 1e-3:
            return
        Lse, Lew = 0.55 * dsw, 0.45 * dsw
    dx, dy = wx - sx, wy - sy
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return
    d2 = max(min(d, (Lse + Lew) - 1e-3), abs(Lse - Lew) + 1e-3)
    a = (Lse * Lse - Lew * Lew + d2 * d2) / (2.0 * d2)
    h = math.sqrt(max(Lse * Lse - a * a, 0.0))
    px, py = sx + a * (dx / d), sy + a * (dy / d)
    rx, ry = -dy / d, dx / d
    e1x, e1y, e2x, e2y = px + h * rx, py + h * ry, px - h * rx, py - h * ry
    nx, ny = (
        (e1x, e1y)
        if not vis(ex, ey, ec) or math.hypot(e1x - ex, e1y - ey) <= math.hypot(e2x - ex, e2y - ey)
        else (e2x, e2y)
    )
    pose[3 * el : 3 * el + 3] = [float(nx), float(ny), float(max(min(ec, 0.8), conf_gate))]
    p_out["pose_keypoints_2d"] = pose


def _remove_short_presence_runs_kps_seq(
    seq: List[Optional[List[float]]], *, conf_gate: float, min_points_present: int, min_run: int
) -> List[Optional[List[float]]]:
    if not seq:
        return seq
    out = [None if a is None else list(a) for a in seq]
    start = None
    for t in range(len(seq) + 1):
        cur = t < len(seq) and _count_valid_points(seq[t], conf_gate=conf_gate) >= min_points_present
        if cur and start is None:
            start = t
        if not cur and start is not None:
            if (t - start) < min_run:
                for k in range(start, t):
                    out[k] = _zero_out_kps(out[k])
            start = None
    return out


def _zero_sparse_frames_kps_seq(
    seq: List[Optional[List[float]]], *, conf_gate: float, min_points_present: int
) -> List[Optional[List[float]]]:
    if not seq:
        return seq
    return [
        (
            _zero_out_kps(a)
            if isinstance(a, list) and _count_valid_points(a, conf_gate=conf_gate) < min_points_present
            else a
        )
        for a in seq
    ]


def _suppress_spatial_outliers_in_hand_arr(
    hand_arr: Optional[List[float]], *, conf_gate: float, max_bone_factor: float = 3.0
) -> Optional[List[float]]:
    if not isinstance(hand_arr, list) or len(hand_arr) % 3 != 0:
        return hand_arr
    pts = _reshape_keypoints_2d(hand_arr)
    J = len(pts)
    if J < 21:
        return hand_arr
    out = [list(p) for p in pts]

    def vis(j):
        return out[j][2] >= conf_gate and not (out[j][0] == 0 and out[j][1] == 0)

    vv = [(x, y) for x, y, c in out if c >= conf_gate and not (x == 0 and y == 0)]
    if len(vv) < 6:
        return hand_arr
    xs, ys = [p[0] for p in vv], [p[1] for p in vv]
    s = max(max(xs) - min(xs), max(ys) - min(ys))
    if s <= 1e-3:
        return hand_arr
    max_bone = max_bone_factor * s
    for a, b in HAND21_EDGES:
        if a >= J or b >= J or not vis(a) or not vis(b):
            continue
        if math.hypot(out[a][0] - out[b][0], out[a][1] - out[b][1]) > max_bone:
            if out[a][2] <= out[b][2]:
                out[a] = [0.0, 0.0, 0.0]
            else:
                out[b] = [0.0, 0.0, 0.0]
    return _flatten_keypoints_2d([(x, y, c) for x, y, c in out])


def _body_head_root_scale_from_pose(
    pose_arr: Optional[List[float]], *, conf_gate: float
) -> Optional[Tuple[Tuple[float, float], float]]:
    if not isinstance(pose_arr, list) or len(pose_arr) % 3 != 0:
        return None
    kps = _reshape_keypoints_2d(pose_arr)

    def vis(j):
        return (
            (float(kps[j][0]), float(kps[j][1]))
            if j < len(kps) and kps[j][2] >= conf_gate and not (kps[j][0] == 0 and kps[j][1] == 0)
            else None
        )

    pts = [p for p in (vis(j) for j in [0, 1, 14, 15, 16, 17]) if p is not None]
    if not pts:
        return None
    root = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    cands = [
        math.hypot(a[0] - b[0], a[1] - b[1])
        for a, b in ((vis(14), vis(15)), (vis(16), vis(17)), (vis(2), vis(5)))
        if a and b and math.hypot(a[0] - b[0], a[1] - b[1]) > 1e-3
    ]
    return (root, float(sum(cands) / len(cands))) if cands else None


def _body_wrist_root_scale_from_pose(
    pose_arr: Optional[List[float]], *, side: str, conf_gate: float
) -> Optional[Tuple[Tuple[float, float], float]]:
    if not isinstance(pose_arr, list) or len(pose_arr) % 3 != 0:
        return None
    kps = _reshape_keypoints_2d(pose_arr)
    w, e = (4, 3) if side == "right" else (7, 6)

    def vis(j):
        return (
            (float(kps[j][0]), float(kps[j][1]))
            if j < len(kps) and kps[j][2] >= conf_gate and not (kps[j][0] == 0 and kps[j][1] == 0)
            else None
        )

    pw = vis(w)
    if not pw:
        return None
    pe = vis(e)
    s = math.hypot(pw[0] - pe[0], pw[1] - pe[1]) if pe and math.hypot(pw[0] - pe[0], pw[1] - pe[1]) > 1e-3 else None
    if s is None:
        p2, p5 = vis(2), vis(5)
        if p2 and p5 and math.hypot(p2[0] - p5[0], p2[1] - p5[1]) > 1e-3:
            s = math.hypot(p2[0] - p5[0], p2[1] - p5[1])
    return (pw, float(s)) if s else None


def _smooth_dense_seq_anchored_to_body(
    dense_seq: List[Optional[List[float]]],
    body_pose_seq: List[Optional[List[float]]],
    *,
    kind: str,
    conf_gate_dense: float,
    conf_gate_body: float,
    median3: bool,
    zero_lag_alpha: float,
) -> List[Optional[List[float]]]:
    if not dense_seq:
        return dense_seq
    Jd = next((len(a) // 3 for a in dense_seq if isinstance(a, list) and len(a) % 3 == 0 and len(a) > 0), None)
    if Jd is None:
        return dense_seq
    T = len(dense_seq)
    out = [None if a is None else list(a) for a in dense_seq]
    norm_seq = [None] * T

    for t in range(T):
        arr, body = out[t], body_pose_seq[t] if t < len(body_pose_seq) else None
        if not isinstance(arr, list) or len(arr) != Jd * 3 or not isinstance(body, list):
            norm_seq[t] = arr
            continue
        rs = (
            _body_head_root_scale_from_pose(body, conf_gate=conf_gate_body)
            if kind == "face"
            else _body_wrist_root_scale_from_pose(
                body, side="left" if kind == "hand_left" else "right", conf_gate=conf_gate_body
            )
        )
        if not rs or rs[1] <= 1e-6:
            norm_seq[t] = arr
            continue
        (rx, ry), s = rs
        norm_seq[t] = [
            (x - rx) / s if i % 3 == 0 else (y - ry) / s if i % 3 == 1 else c
            for i, (x, y, c) in enumerate(zip(arr[0::3], arr[1::3], arr[2::3]))
        ]

    if median3:
        norm_seq = _median3_pose_seq(norm_seq, conf_gate=conf_gate_dense)
    norm_seq = _zero_lag_ema_pose_seq(norm_seq, alpha=zero_lag_alpha, conf_gate=conf_gate_dense)

    for t in range(T):
        arrn, body = norm_seq[t], body_pose_seq[t] if t < len(body_pose_seq) else None
        if not isinstance(arrn, list) or len(arrn) != Jd * 3 or not isinstance(body, list):
            continue
        rs = (
            _body_head_root_scale_from_pose(body, conf_gate=conf_gate_body)
            if kind == "face"
            else _body_wrist_root_scale_from_pose(
                body, side="left" if kind == "hand_left" else "right", conf_gate=conf_gate_body
            )
        )
        if not rs or rs[1] <= 1e-6:
            continue
        (rx, ry), s = rs
        for j in range(Jd):
            if (
                out[t][3 * j + 2] >= conf_gate_dense
                and arrn[3 * j + 2] >= conf_gate_dense
                and not (out[t][3 * j] == 0 and out[t][3 * j + 1] == 0)
            ):
                out[t][3 * j : 3 * j + 2] = [rx + arrn[3 * j] * s, ry + arrn[3 * j + 1] * s]
    return out


def smooth_KPS_json_obj(
    data: Any,
    *,
    keep_face_untouched: bool = True,
    keep_hands_untouched: bool = True,
    filter_extra_people: Optional[bool] = None,
) -> Any:
    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON to be a list of frames.")
    filter_extra_people = bool(FILTER_EXTRA_PEOPLE) if filter_extra_people is None else filter_extra_people
    chosen_people: List[Optional[Dict[str, Any]]] = [None] * len(data)

    if MAIN_PERSON_MODE == "longest_track":
        main_tr = _pick_main_track(_build_tracks_over_video(data))
        if main_tr:
            for t in range(len(data)):
                if t in main_tr.frames:
                    chosen_people[t] = main_tr.frames[t]
        else:
            prev_center = None
            for i, frame in enumerate(data):
                if not isinstance(frame, dict) or not frame.get("people"):
                    continue
                chosen_people[i] = _choose_single_person(frame.get("people", []), prev_center)
                if chosen_people[i]:
                    c = _body_center_from_pose(chosen_people[i].get("pose_keypoints_2d"))
                    if c:
                        prev_center = c
    else:
        prev_center = None
        for i, frame in enumerate(data):
            if not isinstance(frame, dict) or not frame.get("people"):
                continue
            chosen_people[i] = _choose_single_person(frame.get("people", []), prev_center)
            if chosen_people[i]:
                c = _body_center_from_pose(chosen_people[i].get("pose_keypoints_2d"))
                if c:
                    prev_center = c

    pose_seq = [p.get("pose_keypoints_2d") if isinstance(p, dict) else None for p in chosen_people]

    if SPATIAL_OUTLIER_FIX:
        pose_seq = [
            _suppress_spatial_outliers_in_pose_arr(arr, conf_gate=CONF_GATE_BODY) if arr else None for arr in pose_seq
        ]

    if GAP_FILL_ENABLED:
        pose_seq = _denoise_and_fill_gaps_pose_seq(
            pose_seq, conf_gate=CONF_GATE_BODY, min_run=MIN_RUN_FRAMES, max_gap=MAX_GAP_FRAMES
        )

    if TORSO_SYNC_ENABLED:
        pose_seq = _sync_group_appearances(
            pose_seq, group=TORSO_JOINTS, conf_gate=CONF_GATE_BODY, lookahead=TORSO_LOOKAHEAD_FRAMES
        )

    pose_seq = [
        _suppress_isolated_joints_in_pose_arr(arr, conf_gate=CONF_GATE_BODY, keep=TORSO_JOINTS) if arr else None
        for arr in pose_seq
    ]

    if MEDIAN3_ENABLED:
        pose_seq = _median3_pose_seq(pose_seq, conf_gate=CONF_GATE_BODY)
    if SUPER_SMOOTH_ENABLED:
        pose_seq = _zero_lag_ema_pose_seq(pose_seq, alpha=SUPER_SMOOTH_ALPHA, conf_gate=SUPER_SMOOTH_MIN_CONF)
    if ROOTSCALE_CARRY_ENABLED:
        pose_seq = _carry_pose_when_torso_missing(
            pose_seq,
            conf_gate=CARRY_CONF_GATE,
            max_carry=CARRY_MAX_FRAMES,
            anchor_joints=CARRY_ANCHOR_JOINTS,
            min_anchors=CARRY_MIN_ANCHORS,
        )
    pose_seq = _force_full_torso_pair(
        pose_seq,
        conf_gate=CARRY_CONF_GATE,
        anchor_joints=CARRY_ANCHOR_JOINTS,
        min_anchors=CARRY_MIN_ANCHORS,
        max_lookback=240,
        fill_legs_with_hip=True,
        always_fill_if_one_hip=True,
    )
    pose_seq = _denoise_and_fill_gaps_pose_seq(pose_seq, conf_gate=CONF_GATE_BODY, min_run=MIN_RUN_FRAMES, max_gap=0)

    face_seq = [p.get("face_keypoints_2d") if isinstance(p, dict) else None for p in chosen_people]
    lh_seq = [p.get("hand_left_keypoints_2d") if isinstance(p, dict) else None for p in chosen_people]
    rh_seq = [p.get("hand_right_keypoints_2d") if isinstance(p, dict) else None for p in chosen_people]

    if HANDS_SMOOTH_ENABLED and not keep_hands_untouched:
        lh_seq = [_suppress_spatial_outliers_in_hand_arr(a, conf_gate=CONF_GATE_HAND) if a else None for a in lh_seq]
        rh_seq = [_suppress_spatial_outliers_in_hand_arr(a, conf_gate=CONF_GATE_HAND) if a else None for a in rh_seq]
        lh_seq = _remove_short_presence_runs_kps_seq(
            lh_seq, conf_gate=CONF_GATE_HAND, min_points_present=HAND_MIN_POINTS_PRESENT, min_run=MIN_HAND_RUN_FRAMES
        )
        rh_seq = _remove_short_presence_runs_kps_seq(
            rh_seq, conf_gate=CONF_GATE_HAND, min_points_present=HAND_MIN_POINTS_PRESENT, min_run=MIN_HAND_RUN_FRAMES
        )
        lh_seq = _zero_sparse_frames_kps_seq(
            lh_seq, conf_gate=CONF_GATE_HAND, min_points_present=HAND_MIN_POINTS_PRESENT
        )
        rh_seq = _zero_sparse_frames_kps_seq(
            rh_seq, conf_gate=CONF_GATE_HAND, min_points_present=HAND_MIN_POINTS_PRESENT
        )
        if DENSE_GAP_FILL_ENABLED:
            lh_seq = _denoise_and_fill_gaps_pose_seq(
                lh_seq, conf_gate=CONF_GATE_HAND, min_run=DENSE_MIN_RUN_FRAMES, max_gap=DENSE_MAX_GAP_FRAMES
            )
            rh_seq = _denoise_and_fill_gaps_pose_seq(
                rh_seq, conf_gate=CONF_GATE_HAND, min_run=DENSE_MIN_RUN_FRAMES, max_gap=DENSE_MAX_GAP_FRAMES
            )
        lh_seq = _smooth_dense_seq_anchored_to_body(
            lh_seq,
            pose_seq,
            kind="hand_left",
            conf_gate_dense=CONF_GATE_HAND,
            conf_gate_body=CONF_GATE_BODY,
            median3=DENSE_MEDIAN3_ENABLED,
            zero_lag_alpha=DENSE_SUPER_SMOOTH_ALPHA,
        )
        rh_seq = _smooth_dense_seq_anchored_to_body(
            rh_seq,
            pose_seq,
            kind="hand_right",
            conf_gate_dense=CONF_GATE_HAND,
            conf_gate_body=CONF_GATE_BODY,
            median3=DENSE_MEDIAN3_ENABLED,
            zero_lag_alpha=DENSE_SUPER_SMOOTH_ALPHA,
        )

    if FACE_SMOOTH_ENABLED and not keep_face_untouched:
        if DENSE_GAP_FILL_ENABLED:
            face_seq = _denoise_and_fill_gaps_pose_seq(
                face_seq, conf_gate=CONF_GATE_FACE, min_run=DENSE_MIN_RUN_FRAMES, max_gap=DENSE_MAX_GAP_FRAMES
            )
        face_seq = _smooth_dense_seq_anchored_to_body(
            face_seq,
            pose_seq,
            kind="face",
            conf_gate_dense=CONF_GATE_FACE,
            conf_gate_body=CONF_GATE_BODY,
            median3=DENSE_MEDIAN3_ENABLED,
            zero_lag_alpha=DENSE_SUPER_SMOOTH_ALPHA,
        )

    out_frames = []
    body_state: Optional[BodyState] = None

    for i, frame in enumerate(data):
        if not isinstance(frame, dict):
            out_frames.append(frame)
            continue

        frame_out = copy.deepcopy(frame)
        chosen = chosen_people[i]

        if chosen is None:
            if filter_extra_people:
                frame_out["people"] = []
            out_frames.append(frame_out)
            body_state = None
            continue

        p_out = copy.deepcopy(chosen)
        p_out["pose_keypoints_2d"] = pose_seq[i]

        pose_arr = p_out.get("pose_keypoints_2d")
        joints = (len(pose_arr) // 3) if isinstance(pose_arr, list) else 0
        if body_state is None:
            body_state = BodyState(joints if joints > 0 else 18)

        p_out["pose_keypoints_2d"] = _smooth_body_pose(p_out.get("pose_keypoints_2d"), body_state)

        if FACE_SMOOTH_ENABLED and not keep_face_untouched:
            p_out["face_keypoints_2d"] = face_seq[i]
        else:
            p_out["face_keypoints_2d"] = chosen.get("face_keypoints_2d", p_out.get("face_keypoints_2d"))

        if HANDS_SMOOTH_ENABLED and not keep_hands_untouched:
            p_out["hand_left_keypoints_2d"], p_out["hand_right_keypoints_2d"] = lh_seq[i], rh_seq[i]
        else:
            p_out["hand_left_keypoints_2d"] = chosen.get("hand_left_keypoints_2d", p_out.get("hand_left_keypoints_2d"))
            p_out["hand_right_keypoints_2d"] = chosen.get(
                "hand_right_keypoints_2d", p_out.get("hand_right_keypoints_2d")
            )

        _pin_body_wrist_to_hand(
            p_out, side="left", conf_gate_body=CONF_GATE_BODY, conf_gate_hand=CONF_GATE_HAND, blend=1.0
        )
        _pin_body_wrist_to_hand(
            p_out, side="right", conf_gate_body=CONF_GATE_BODY, conf_gate_hand=CONF_GATE_HAND, blend=1.0
        )
        _fix_elbow_using_wrist(p_out, side="left", conf_gate=CONF_GATE_BODY)
        _fix_elbow_using_wrist(p_out, side="right", conf_gate=CONF_GATE_BODY)

        if filter_extra_people:
            frame_out["people"] = [p_out]
        else:
            orig_people = frame.get("people", [])
            if not isinstance(orig_people, list):
                frame_out["people"] = [p_out]
            else:
                replaced, new_people = False, []
                for op in orig_people:
                    if not replaced and (op is chosen):
                        new_people.append(p_out)
                        replaced = True
                    else:
                        new_people.append(copy.deepcopy(op))
                if not replaced:
                    new_people = [p_out] + [copy.deepcopy(op) for op in orig_people]
                frame_out["people"] = new_people

        out_frames.append(frame_out)

    final_body, final_lh, final_rh, final_face = [], [], [], []
    for f in out_frames:
        if f.get("people") and len(f["people"]) > 0:
            p = f["people"][0]
            final_body.append(p.get("pose_keypoints_2d"))
            final_lh.append(p.get("hand_left_keypoints_2d"))
            final_rh.append(p.get("hand_right_keypoints_2d"))
            final_face.append(p.get("face_keypoints_2d"))
        else:
            final_body.append(None)
            final_lh.append(None)
            final_rh.append(None)
            final_face.append(None)

    eff_min = max(2, MIN_RUN_FRAMES)

    final_body = _denoise_and_fill_gaps_pose_seq(final_body, conf_gate=CONF_GATE_BODY, min_run=eff_min, max_gap=0)
    final_lh = _remove_short_presence_runs_kps_seq(
        final_lh, conf_gate=CONF_GATE_HAND, min_points_present=1, min_run=eff_min
    )
    final_rh = _remove_short_presence_runs_kps_seq(
        final_rh, conf_gate=CONF_GATE_HAND, min_points_present=1, min_run=eff_min
    )
    final_face = _remove_short_presence_runs_kps_seq(
        final_face, conf_gate=CONF_GATE_FACE, min_points_present=1, min_run=eff_min
    )

    for i, f in enumerate(out_frames):
        if f.get("people") and len(f["people"]) > 0:
            f["people"][0]["pose_keypoints_2d"] = final_body[i]
            f["people"][0]["hand_left_keypoints_2d"] = final_lh[i]
            f["people"][0]["hand_right_keypoints_2d"] = final_rh[i]
            f["people"][0]["face_keypoints_2d"] = final_face[i]
    # ========================================================

    return out_frames


# ============================================================
# === START: render_pose_video.py logic (ported to frame render)
# ============================================================

OP_COLORS: List[Tuple[int, int, int]] = [
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 0, 85),
]

BODY_EDGES: List[Tuple[int, int]] = [
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
]

BODY_EDGE_COLORS = OP_COLORS[: len(BODY_EDGES)]
BODY_JOINT_COLORS = OP_COLORS

HAND_EDGES: List[Tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def _valid_pt(x: float, y: float, c: float, conf_thresh: float) -> bool:
    return (c is not None) and (c >= conf_thresh) and not (x == 0 and y == 0)


def _hsv_to_bgr(h: float, s: float, v: float) -> Tuple[int, int, int]:
    H = int(np.clip(h, 0.0, 1.0) * 179.0)
    S = int(np.clip(s, 0.0, 1.0) * 255.0)
    V = int(np.clip(v, 0.0, 1.0) * 255.0)
    hsv = np.uint8([[[H, S, V]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _looks_normalized(points: List[Tuple[float, float, float]], conf_thresh: float) -> bool:
    valid = [(x, y, c) for (x, y, c) in points if _valid_pt(x, y, c, conf_thresh)]
    if not valid:
        return False
    in01 = sum(1 for (x, y, _) in valid if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)
    return (in01 / float(len(valid))) >= 0.7


def _draw_body(
    canvas: np.ndarray, pose: List[Tuple[float, float, float]], conf_thresh: float, xinsr_stick_scaling: bool = False
) -> None:
    CH, CW = canvas.shape[:2]
    stickwidth = 2
    valid = [(x, y, c) for (x, y, c) in pose if _valid_pt(x, y, c, conf_thresh)]
    norm = False
    if valid:
        in01 = sum(1 for (x, y, _) in valid if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)
        norm = (in01 / float(len(valid))) >= 0.7

    def to_px(x: float, y: float) -> Tuple[float, float]:
        if norm:
            return x * CW, y * CH
        return x, y

    max_side = max(CW, CH)
    stick_scale = 1 if max_side < 500 else min(2 + (max_side // 1000), 7) if xinsr_stick_scaling else 1

    for idx, (a, b) in enumerate(BODY_EDGES):
        if a >= len(pose) or b >= len(pose):
            continue
        ax, ay, ac = pose[a]
        bx, by, bc = pose[b]
        if not (_valid_pt(ax, ay, ac, conf_thresh) and _valid_pt(bx, by, bc, conf_thresh)):
            continue

        ax, ay = to_px(ax, ay)
        bx, by = to_px(bx, by)
        base = BODY_EDGE_COLORS[idx] if idx < len(BODY_EDGE_COLORS) else (255, 255, 255)

        X = np.array([ay, by], dtype=np.float32)
        Y = np.array([ax, bx], dtype=np.float32)

        mX, mY = float(np.mean(X)), float(np.mean(Y))
        length = float(np.hypot(X[0] - X[1], Y[0] - Y[1]))
        if length < 1.0:
            continue

        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly(
            (int(mY), int(mX)), (int(length / 2), int(stickwidth * stick_scale)), int(angle), 0, 360, 1
        )
        cv2.fillConvexPoly(canvas, polygon, (int(base[0] * 0.6), int(base[1] * 0.6), int(base[2] * 0.6)))

    for j, (x, y, c) in enumerate(pose):
        if not _valid_pt(x, y, c, conf_thresh):
            continue
        x, y = to_px(x, y)
        col = BODY_JOINT_COLORS[j] if j < len(BODY_JOINT_COLORS) else (255, 255, 255)
        cv2.circle(canvas, (int(x), int(y)), 2, col, thickness=-1)


def _draw_hand(canvas: np.ndarray, hand: List[Tuple[float, float, float]], conf_thresh: float) -> None:
    if not hand or len(hand) < 21:
        return
    CH, CW = canvas.shape[:2]
    norm = _looks_normalized(hand, conf_thresh)

    def to_px(x: float, y: float) -> Tuple[float, float]:
        return (x * CW, y * CH) if norm else (x, y)

    n_edges = len(HAND_EDGES)
    for i, (a, b) in enumerate(HAND_EDGES):
        x1, y1, c1 = hand[a]
        x2, y2, c2 = hand[b]
        if _valid_pt(x1, y1, c1, conf_thresh) and _valid_pt(x2, y2, c2, conf_thresh):
            x1, y1 = to_px(x1, y1)
            x2, y2 = to_px(x2, y2)
            cv2.line(
                canvas,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                _hsv_to_bgr(i / float(n_edges), 1.0, 1.0),
                1,
                cv2.LINE_AA,
            )
    for x, y, c in hand:
        if _valid_pt(x, y, c, conf_thresh):
            x, y = to_px(x, y)
            cv2.circle(canvas, (int(x), int(y)), 1, (0, 0, 255), -1, cv2.LINE_AA)


def _draw_face(canvas: np.ndarray, face: List[Tuple[float, float, float]], conf_thresh: float) -> None:
    if not face:
        return
    CH, CW = canvas.shape[:2]
    norm = _looks_normalized(face, conf_thresh)

    def to_px(x: float, y: float) -> Tuple[float, float]:
        return (x * CW, y * CH) if norm else (x, y)

    for x, y, c in face:
        if _valid_pt(x, y, c, conf_thresh):
            x, y = to_px(x, y)
            cv2.circle(canvas, (int(x), int(y)), 0, (255, 255, 255), -1, cv2.LINE_AA)


def _draw_pose_frame_full(
    w: int,
    h: int,
    person: Dict[str, Any],
    conf_thresh_body: float = 0.10,
    conf_thresh_hands: float = 0.10,
    conf_thresh_face: float = 0.10,
) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    pose = _reshape_keypoints_2d(person.get("pose_keypoints_2d") or [])
    face = _reshape_keypoints_2d(person.get("face_keypoints_2d") or [])
    hand_l = _reshape_keypoints_2d(person.get("hand_left_keypoints_2d") or [])
    hand_r = _reshape_keypoints_2d(person.get("hand_right_keypoints_2d") or [])

    if pose:
        _draw_body(img, pose, conf_thresh_body)
    if hand_l:
        _draw_hand(img, hand_l, conf_thresh_hands)
    if hand_r:
        _draw_hand(img, hand_r, conf_thresh_hands)
    if face:
        _draw_face(img, face, conf_thresh_face)
    return img
