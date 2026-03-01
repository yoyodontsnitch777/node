import os
import re
import shutil
import subprocess
import time
from collections.abc import Mapping

import torch
import numpy as np

try:
    import cv2

    _has_cv2 = True
except Exception:
    _has_cv2 = False

ENCODE_ARGS = ("utf-8", "backslashreplace")


def _pick_ffmpeg_path():
    if "VHS_FORCE_FFMPEG_PATH" in os.environ:
        p = os.environ.get("VHS_FORCE_FFMPEG_PATH")
        if p:
            return p

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg is not None:
        return system_ffmpeg

    if os.path.isfile("ffmpeg"):
        return os.path.abspath("ffmpeg")
    if os.path.isfile("ffmpeg.exe"):
        return os.path.abspath("ffmpeg.exe")

    return None


ffmpeg_path = _pick_ffmpeg_path()


def get_audio(file, start_time=0, duration=0):
    if ffmpeg_path is None:
        raise Exception("ffmpeg not found. Put ffmpeg in PATH, or set VHS_FORCE_FFMPEG_PATH env var.")

    args = [ffmpeg_path, "-i", file]
    if start_time > 0:
        args += ["-ss", str(start_time)]
    if duration > 0:
        args += ["-t", str(duration)]

    try:
        # как в utils: вытаскиваем raw f32le в stdout
        res = subprocess.run(args + ["-f", "f32le", "-"], capture_output=True, check=True)
        audio = torch.frombuffer(bytearray(res.stdout), dtype=torch.float32)
        match = re.search(r", (\d+) Hz, (\w+), ", res.stderr.decode(*ENCODE_ARGS))
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to extract audio from {file}:\n" + e.stderr.decode(*ENCODE_ARGS))

    if match:
        ar = int(match.group(1))
        ac = {"mono": 1, "stereo": 2}.get(match.group(2), 2)
    else:
        ar = 44100
        ac = 2

    if audio.numel() == 0:
        empty = torch.zeros((1, 1, 0), dtype=torch.float32)
        return {"waveform": empty, "sample_rate": ar}

    audio = audio.reshape((-1, ac)).transpose(0, 1).unsqueeze(0)
    return {"waveform": audio, "sample_rate": ar}


class LazyAudioMap(Mapping):
    def __init__(self, file, start_time, duration):
        self.file = file
        self.start_time = start_time
        self.duration = duration
        self._dict = None

    def _ensure(self):
        if self._dict is None:
            self._dict = get_audio(self.file, self.start_time, self.duration)

    def __getitem__(self, key):
        self._ensure()
        return self._dict[key]

    def __iter__(self):
        self._ensure()
        return iter(self._dict)

    def __len__(self):
        self._ensure()
        return len(self._dict)


def lazy_get_audio(file, start_time=0, duration=0, **kwargs):
    return LazyAudioMap(file, start_time, duration)


def extract_first_number(s):
    match = re.search(r"\d+", s)
    return int(match.group()) if match else float("inf")


sort_methods = [
    "None",
    "Alphabetical (ASC)",
    "Alphabetical (DESC)",
    "Numerical (ASC)",
    "Numerical (DESC)",
    "Datetime (ASC)",
    "Datetime (DESC)",
]


def sort_by(items, base_path=".", method=None):
    def fullpath(x):
        return os.path.join(base_path, x)

    def get_timestamp(path):
        try:
            return os.path.getmtime(path)
        except FileNotFoundError:
            return float("-inf")

    if method == "Alphabetical (ASC)":
        return sorted(items)
    elif method == "Alphabetical (DESC)":
        return sorted(items, reverse=True)
    elif method == "Numerical (ASC)":
        return sorted(items, key=lambda x: extract_first_number(os.path.splitext(x)[0]))
    elif method == "Numerical (DESC)":
        return sorted(items, key=lambda x: extract_first_number(os.path.splitext(x)[0]), reverse=True)
    elif method == "Datetime (ASC)":
        return sorted(items, key=lambda x: get_timestamp(fullpath(x)))
    elif method == "Datetime (DESC)":
        return sorted(items, key=lambda x: get_timestamp(fullpath(x)), reverse=True)
    else:
        return items


def target_size(width, height, custom_width, custom_height, downscale_ratio=8):
    if downscale_ratio is None:
        downscale_ratio = 8

    if custom_width == 0 and custom_height == 0:
        new_w, new_h = width, height
    elif custom_height == 0:
        new_h = int(height * (custom_width / width))
        new_w = int(custom_width)
    elif custom_width == 0:
        new_w = int(width * (custom_height / height))
        new_h = int(custom_height)
    else:
        new_w, new_h = int(custom_width), int(custom_height)

    new_w = int(new_w / downscale_ratio + 0.5) * downscale_ratio
    new_h = int(new_h / downscale_ratio + 0.5) * downscale_ratio
    return new_w, new_h


def _read_frames_vhs_like(
    video_path: str,
    force_rate: float = 0,
    custom_width: int = 0,
    custom_height: int = 0,
    downscale_ratio: int = 8,
    frame_load_cap: int = 0,
    select_every_nth: int = 1,
):

    if select_every_nth is None or select_every_nth < 1:
        select_every_nth = 1

    if not _has_cv2:
        raise RuntimeError("OpenCV (cv2) not available. Install opencv-python.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened() or not cap.grab():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ok0, frame0 = cap.retrieve()
    if not ok0 or frame0 is None:
        cap.release()
        raise RuntimeError(f"Cannot retrieve first frame from: {video_path}")

    if width <= 0 or height <= 0:
        height, width = frame0.shape[:2]

    base_dt = 1.0 / float(fps)
    target_dt = base_dt if force_rate == 0 else (1.0 / float(force_rate))

    effective_dt = target_dt * float(select_every_nth)
    loaded_fps = 1.0 / effective_dt if effective_dt > 0 else float(fps)

    new_w, new_h = target_size(width, height, custom_width, custom_height, downscale_ratio)
    do_resize = (new_w != width) or (new_h != height)

    def _process_frame(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if do_resize:
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        return rgb

    frames = []

    evaluated = -1

    def _maybe_add(bgr):
        nonlocal evaluated
        evaluated += 1
        if (evaluated % select_every_nth) != 0:
            return
        frames.append(_process_frame(bgr))

    _maybe_add(frame0)

    if frame_load_cap > 0 and len(frames) >= frame_load_cap:
        cap.release()
        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
        t = torch.from_numpy(arr)
        loaded_duration = float(len(t) * effective_dt)
        start_time = 0.0
        return t, float(fps), float(loaded_fps), loaded_duration, start_time

    time_offset = target_dt
    time_offset -= target_dt

    while cap.isOpened():
        if time_offset < target_dt:
            ok = cap.grab()
            if not ok:
                break
            time_offset += base_dt
            continue

        ok, frame_bgr = cap.retrieve()
        if not ok or frame_bgr is None:
            break

        _maybe_add(frame_bgr)

        if frame_load_cap > 0 and len(frames) >= frame_load_cap:
            break

        time_offset -= target_dt

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames could be read from: {video_path}")

    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    t = torch.from_numpy(arr)

    loaded_duration = float(len(t) * effective_dt)
    start_time = 0.0
    return t, float(fps), float(loaded_fps), loaded_duration, start_time


class LoadVideoBatchListFromDir:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
                "force_rate": ("FLOAT", {"default": 0, "min": 0, "max": 120, "step": 1}),
                "width": ("INT", {"default": 720, "min": 0, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 1280, "min": 0, "max": 8192, "step": 1}),
            },
            "optional": {
                "video_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFF, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                "load_always": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "sort_method": (sort_methods,),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT")
    RETURN_NAMES = ("IMAGE", "audio", "COUNT")
    OUTPUT_IS_LIST = (True, True, False)

    FUNCTION = "load_videos"
    CATEGORY = "video"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs.get("load_always"):
            return float("NaN")
        return hash(frozenset(kwargs.items()))

    def load_videos(
        self,
        directory: str,
        force_rate: float = 0,
        width: int = 0,
        height: int = 0,
        video_load_cap: int = 0,
        frame_load_cap: int = 0,
        select_every_nth: int = 1,
        start_index: int = 0,
        load_always: bool = False,
        sort_method=None,
    ):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' cannot be found.")

        files = os.listdir(directory)
        if len(files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        valid_ext = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
        files = [
            f
            for f in files
            if os.path.isfile(os.path.join(directory, f)) and os.path.splitext(f)[1].lower() in valid_ext
        ]
        if len(files) == 0:
            raise FileNotFoundError(f"No video files in directory '{directory}' (expected: {sorted(valid_ext)}).")

        files = sort_by(files, directory, sort_method)
        files = files[start_index:]
        if video_load_cap > 0:
            files = files[:video_load_cap]

        images_list = []
        audios_list = []

        for fname in files:
            path = os.path.join(directory, fname)

            vid, source_fps, loaded_fps, loaded_duration, start_time = _read_frames_vhs_like(
                path,
                force_rate=force_rate,
                custom_width=width,
                custom_height=height,
                downscale_ratio=8,
                frame_load_cap=frame_load_cap,
                select_every_nth=select_every_nth,
            )

            images_list.append(vid)

            audio = lazy_get_audio(path, start_time, loaded_duration)
            audios_list.append(audio)

        return (images_list, audios_list, len(images_list))
