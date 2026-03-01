"""Single-file extraction of the **Video Combine** node from ComfyUI-VideoHelperSuite.

- Node class: VideoCombine
- ComfyUI registration name: VHS_VideoCombine

This is intended to be dropped into your own custom node package as a single .py file.

Notes:
- This file includes the minimal helpers that VideoCombine depends on (ffmpeg discovery,
  simple caching, format json parsing, ffmpeg/gifski subprocess pipelines, etc.).
- It assumes you're running inside ComfyUI, so core modules like `folder_paths`, `server`,
  and `comfy.utils.ProgressBar` must be available.
"""

from __future__ import annotations

import copy
import datetime
import functools
import itertools
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from string import Template
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import numpy as np
import torch
from PIL import ExifTags, Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
import server
from comfy.utils import ProgressBar

# -----------------------------------------------------------------------------
# Logging (minimal, compatible)
# -----------------------------------------------------------------------------
logger = logging.getLogger("VideoHelperSuite.VideoCombine")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(name)s] - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

ENCODE_ARGS = ("utf-8", "backslashreplace")

# -----------------------------------------------------------------------------
# Small utility helpers copied/adapted from VideoHelperSuite
# -----------------------------------------------------------------------------

BIGMAX = 2**53 - 1


class MultiInput(str):
    """ComfyUI trick: allows a single input to accept multiple types."""

    def __new__(cls, string: str, allowed_types: Any = "*"):
        res = super().__new__(cls, string)
        res.allowed_types = allowed_types
        return res

    def __ne__(self, other: Any) -> bool:
        if self.allowed_types == "*" or other == "*":
            return False
        return other not in self.allowed_types


imageOrLatent = MultiInput("IMAGE", ["IMAGE", "LATENT"])
floatOrInt = MultiInput("FLOAT", ["FLOAT", "INT"])


class ContainsAll(dict):
    """ComfyUI hidden input helper."""

    def __contains__(self, other: Any) -> bool:  # noqa: D401
        return True

    def __getitem__(self, key: Any):
        return super().get(key, (None, {}))


def cached(duration: int):
    """Time-based cache decorator (seconds)."""

    def dec(f):
        cached_ret = None
        cache_time = 0.0

        @functools.wraps(f)
        def cached_func():
            nonlocal cache_time, cached_ret
            now = time.time()
            if now > cache_time + duration or cached_ret is None:
                cache_time = now
                cached_ret = f()
            return cached_ret

        return cached_func

    return dec


def merge_filter_args(args: List[str], ftype: str = "-vf"):
    """Merge multiple -vf occurrences into one (simple, no filter_complex support)."""

    try:
        start_index = args.index(ftype) + 1
        index = start_index
        while True:
            index = args.index(ftype, index)
            args[start_index] += "," + args[index + 1]
            args.pop(index)
            args.pop(index)
    except ValueError:
        pass


# -----------------------------------------------------------------------------
# ffmpeg / gifski discovery (copied in spirit from VHS)
# -----------------------------------------------------------------------------


def _ffmpeg_suitability(path: str) -> int:
    try:
        version = subprocess.run([path, "-version"], check=True, capture_output=True).stdout.decode(*ENCODE_ARGS)
    except Exception:
        return 0

    score = 0
    simple_criterion = [("libvpx", 20), ("264", 10), ("265", 3), ("svtav1", 5), ("libopus", 1)]
    for needle, pts in simple_criterion:
        if needle in version:
            score += pts

    copyright_index = version.find("2000-2")
    if copyright_index >= 0:
        yr = version[copyright_index + 6 : copyright_index + 9]
        if yr.isnumeric():
            score += int(yr)

    return score


def _pick_ffmpeg() -> Optional[str]:
    if "VHS_FORCE_FFMPEG_PATH" in os.environ:
        return os.environ.get("VHS_FORCE_FFMPEG_PATH")

    ffmpeg_paths: List[str] = []

    # Prefer imageio-ffmpeg if available.
    try:
        from imageio_ffmpeg import get_ffmpeg_exe  # type: ignore

        ffmpeg_paths.append(get_ffmpeg_exe())
    except Exception:
        if "VHS_USE_IMAGEIO_FFMPEG" in os.environ:
            raise
        logger.warning("Failed to import imageio_ffmpeg")

    if "VHS_USE_IMAGEIO_FFMPEG" in os.environ and ffmpeg_paths:
        return ffmpeg_paths[0]

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        ffmpeg_paths.append(system_ffmpeg)

    if os.path.isfile("ffmpeg"):
        ffmpeg_paths.append(os.path.abspath("ffmpeg"))
    if os.path.isfile("ffmpeg.exe"):
        ffmpeg_paths.append(os.path.abspath("ffmpeg.exe"))

    if not ffmpeg_paths:
        logger.error("No valid ffmpeg found.")
        return None

    if len(ffmpeg_paths) == 1:
        return ffmpeg_paths[0]

    return max(ffmpeg_paths, key=_ffmpeg_suitability)


ffmpeg_path: Optional[str] = _pick_ffmpeg()


gifski_path: Optional[str] = os.environ.get("VHS_GIFSKI") or os.environ.get("JOV_GIFSKI")
if gifski_path is None:
    gifski_path = shutil.which("gifski")


# -----------------------------------------------------------------------------
# Built-in video format definitions (embedded from VideoHelperSuite/video_formats)
# -----------------------------------------------------------------------------

BUILTIN_VIDEO_FORMATS: Dict[str, Dict[str, Any]] = {
    "16bit-png": {"extension": "%03d.png", "input_color_depth": "16bit", "main_pass": ["-n", "-pix_fmt", "rgba64"]},
    "8bit-png": {"extension": "%03d.png", "main_pass": ["-n"]},
    "ProRes": {
        "audio_pass": ["-c:a", "pcm_s16le"],
        "extension": "mov",
        "extra_widgets": [["profile", ["lt", "standard", "hq", "4444", "4444xq"], {"default": "hq"}]],
        "fake_trc": "bt709",
        "main_pass": [
            "-n",
            "-c:v",
            "prores_ks",
            "-profile:v",
            [["$profile"]],
            [
                "profile",
                {
                    "1": [[]],
                    "2": [[]],
                    "3": [[]],
                    "4": [
                        "has_alpha",
                        {"False": [["-pix_fmt", "yuv444p10le"]], "True": [["-pix_fmt", "yuva444p10le"]]},
                    ],
                    "4444": [
                        "has_alpha",
                        {"False": [["-pix_fmt", "yuv444p10le"]], "True": [["-pix_fmt", "yuva444p10le"]]},
                    ],
                    "4444xq": [
                        "has_alpha",
                        {"False": [["-pix_fmt", "yuv444p10le"]], "True": [["-pix_fmt", "yuva444p10le"]]},
                    ],
                    "hq": [[]],
                    "lt": [[]],
                    "standard": [[]],
                },
            ],
            "-vf",
            "scale=out_color_matrix=bt709",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
        ],
    },
    "av1-webm": {
        "audio_pass": ["-c:a", "libopus"],
        "environment": {"SVT_LOG": "1"},
        "extension": "webm",
        "fake_trc": "bt709",
        "input_color_depth": ["input_color_depth", ["8bit", "16bit"]],
        "main_pass": [
            "-n",
            "-c:v",
            "libsvtav1",
            "-pix_fmt",
            ["pix_fmt", ["yuv420p10le", "yuv420p"]],
            "-crf",
            ["crf", "INT", {"default": 23, "max": 100, "min": 0, "step": 1}],
            "-vf",
            "scale=out_color_matrix=bt709",
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
        ],
        "save_metadata": ["save_metadata", "BOOLEAN", {"default": True}],
    },
    "ffmpeg-gif": {
        "extension": "gif",
        "main_pass": [
            "-n",
            "-filter_complex",
            [
                "dither",
                [
                    "bayer",
                    "heckbert",
                    "floyd_steinberg",
                    "sierra2",
                    "sierra2_4a",
                    "sierra3",
                    "burkes",
                    "atkinson",
                    "none",
                ],
                {"default": "sierra2_4a"},
                "[0:v] split [a][b]; [a] palettegen=reserve_transparent=on:transparency_color=ffffff "
                "[p]; [b][p] paletteuse=dither=$val",
            ],
        ],
    },
    "ffv1-mkv": {
        "audio_pass": ["-c:a", "flac"],
        "extension": "mkv",
        "input_color_depth": "16bit",
        "main_pass": [
            "-n",
            "-c:v",
            "ffv1",
            "-level",
            ["level", ["0", "1", "3"], {"default": "3"}],
            "-coder",
            ["coder", ["0", "1", "2"], {"default": "1"}],
            "-context",
            ["context", ["0", "1"], {"default": "1"}],
            "-g",
            ["gop_size", "INT", {"default": 1, "max": 300, "min": 1, "step": 1}],
            "-slices",
            ["slices", ["4", "6", "9", "12", "16", "20", "24", "30"], {"default": "16"}],
            "-slicecrc",
            ["slicecrc", ["0", "1"], {"default": "1"}],
            "-pix_fmt",
            [
                "pix_fmt",
                [
                    "rgba64le",
                    "bgra",
                    "yuv420p",
                    "yuv422p",
                    "yuv444p",
                    "yuva420p",
                    "yuva422p",
                    "yuva444p",
                    "yuv420p10le",
                    "yuv422p10le",
                    "yuv444p10le",
                    "yuv420p12le",
                    "yuv422p12le",
                    "yuv444p12le",
                    "yuv420p14le",
                    "yuv422p14le",
                    "yuv444p14le",
                    "yuv420p16le",
                    "yuv422p16le",
                    "yuv444p16le",
                    "gray",
                    "gray10le",
                    "gray12le",
                    "gray16le",
                ],
                {"default": "rgba64le"},
            ],
        ],
        "save_metadata": ["save_metadata", "BOOLEAN", {"default": True}],
        "trim_to_audio": ["trim_to_audio", "BOOLEAN", {"default": True}],
    },
    "gifski": {
        "extension": "gif",
        "gifski_pass": ["-Q", ["quality", "INT", {"default": 90, "max": 100, "min": 1, "step": 1}]],
        "main_pass": ["-pix_fmt", "yuv444p", "-vf", "scale=out_color_matrix=bt709:out_range=pc", "-color_range", "pc"],
    },
    "h264-mp4": {
        "audio_pass": ["-c:a", "aac"],
        "extension": "mp4",
        "fake_trc": "bt709",
        "main_pass": [
            "-n",
            "-c:v",
            "libx264",
            "-pix_fmt",
            ["pix_fmt", ["yuv420p", "yuv420p10le"]],
            "-crf",
            ["crf", "INT", {"default": 19, "max": 100, "min": 0, "step": 1}],
            "-vf",
            "scale=out_color_matrix=bt709",
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
        ],
        "save_metadata": ["save_metadata", "BOOLEAN", {"default": True}],
        "trim_to_audio": ["trim_to_audio", "BOOLEAN", {"default": True}],
    },
    "h265-mp4": {
        "audio_pass": ["-c:a", "aac"],
        "extension": "mp4",
        "fake_trc": "bt709",
        "main_pass": [
            "-n",
            "-c:v",
            "libx265",
            "-vtag",
            "hvc1",
            "-pix_fmt",
            ["pix_fmt", ["yuv420p10le", "yuv420p"]],
            "-crf",
            ["crf", "INT", {"default": 22, "max": 100, "min": 0, "step": 1}],
            "-preset",
            "medium",
            "-x265-params",
            "log-level=quiet",
            "-vf",
            "scale=out_color_matrix=bt709",
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
        ],
        "save_metadata": ["save_metadata", "BOOLEAN", {"default": True}],
    },
    "nvenc_av1-mp4": {
        "audio_pass": ["-c:a", "aac"],
        "bitrate": ["bitrate", "INT", {"default": 10, "max": 999, "min": 1, "step": 1}],
        "extension": "mp4",
        "fake_trc": "bt709",
        "main_pass": [
            "-n",
            "-c:v",
            "av1_nvenc",
            "-pix_fmt",
            ["pix_fmt", ["yuv420p", "p010le"]],
            "-vf",
            "scale=out_color_matrix=bt709",
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
        ],
        "megabit": ["megabit", "BOOLEAN", {"default": True}],
        "save_metadata": ["save_metadata", "BOOLEAN", {"default": True}],
    },
    "nvenc_h264-mp4": {
        "audio_pass": ["-c:a", "aac"],
        "bitrate": ["bitrate", "INT", {"default": 10, "max": 999, "min": 1, "step": 1}],
        "extension": "mp4",
        "fake_trc": "bt709",
        "main_pass": [
            "-n",
            "-c:v",
            "h264_nvenc",
            "-pix_fmt",
            ["pix_fmt", ["yuv420p", "p010le"]],
            "-vf",
            "scale=out_color_matrix=bt709",
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
        ],
        "megabit": ["megabit", "BOOLEAN", {"default": True}],
        "save_metadata": ["save_metadata", "BOOLEAN", {"default": True}],
    },
    "nvenc_hevc-mp4": {
        "audio_pass": ["-c:a", "aac"],
        "bitrate": ["bitrate", "INT", {"default": 10, "max": 999, "min": 1, "step": 1}],
        "extension": "mp4",
        "fake_trc": "bt709",
        "main_pass": [
            "-n",
            "-c:v",
            "hevc_nvenc",
            "-vtag",
            "hvc1",
            "-pix_fmt",
            ["pix_fmt", ["yuv420p", "p010le"]],
            "-vf",
            "scale=out_color_matrix=bt709",
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
        ],
        "megabit": ["megabit", "BOOLEAN", {"default": True}],
        "save_metadata": ["save_metadata", "BOOLEAN", {"default": True}],
    },
    "webm": {
        "audio_pass": ["-c:a", "libvorbis"],
        "extension": "webm",
        "fake_trc": "bt709",
        "main_pass": [
            "-n",
            "-pix_fmt",
            ["pix_fmt", ["yuv420p", "yuva420p"]],
            "-crf",
            ["crf", "INT", {"default": 20, "max": 100, "min": 0, "step": 1}],
            "-b:v",
            "0",
            "-vf",
            "scale=out_color_matrix=bt709",
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
        ],
        "save_metadata": ["save_metadata", "BOOLEAN", {"default": True}],
        "trim_to_audio": ["trim_to_audio", "BOOLEAN", {"default": True}],
    },
}


# -----------------------------------------------------------------------------
# Requeue support for BatchManager compatibility (as in VHS)
# -----------------------------------------------------------------------------

prompt_queue = server.PromptServer.instance.prompt_queue


def requeue_workflow_unchecked():
    """Requeues the current workflow without checking for multiple requeues."""

    currently_running = prompt_queue.currently_running
    value = next(iter(currently_running.values()))

    if len(value) == 6:
        (_, prompt_id, prompt, extra_data, outputs_to_execute, sensitive) = value
    else:
        (_, prompt_id, prompt, extra_data, outputs_to_execute) = value
        sensitive = {}

    prompt = prompt.copy()
    for uid in prompt:
        if prompt[uid]["class_type"] == "VHS_BatchManager":
            prompt[uid]["inputs"]["requeue"] = prompt[uid]["inputs"].get("requeue", 0) + 1

    number = -server.PromptServer.instance.number
    server.PromptServer.instance.number += 1
    prompt_id = str(server.uuid.uuid4())
    prompt_queue.put((number, prompt_id, prompt, extra_data, outputs_to_execute, sensitive))


_requeue_guard = [None, 0, 0, {}]


def requeue_workflow(requeue_required: Tuple[Any, bool] = (-1, True)):
    """Requeue once all managed outputs have finished this batch."""

    assert len(prompt_queue.currently_running) == 1
    global _requeue_guard

    value = next(iter(prompt_queue.currently_running.values()))
    if len(value) == 6:
        (run_number, _, prompt, extra_data, outputs_to_execute, _) = value
    else:
        (run_number, _, prompt, extra_data, outputs_to_execute) = value

    if _requeue_guard[0] != run_number:
        managed_outputs = 0
        for bm_uid in prompt:
            if prompt[bm_uid]["class_type"] == "VHS_BatchManager":
                for output_uid in prompt:
                    if prompt[output_uid]["class_type"] in ["VHS_VideoCombine"]:
                        for inp in prompt[output_uid]["inputs"].values():
                            if inp == [bm_uid, 0]:
                                managed_outputs += 1
        _requeue_guard = [run_number, 0, managed_outputs, {}]

    _requeue_guard[1] += 1
    _requeue_guard[3][requeue_required[0]] = requeue_required[1]

    if _requeue_guard[1] == _requeue_guard[2] and max(_requeue_guard[3].values() or [False]):
        requeue_workflow_unchecked()


if "VHS_video_formats" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["VHS_video_formats"] = ((), {".json"})
if len(folder_paths.folder_names_and_paths["VHS_video_formats"][1]) == 0:
    folder_paths.folder_names_and_paths["VHS_video_formats"][1].add(".json")


def flatten_list(l: List[Any]) -> List[Any]:
    ret: List[Any] = []
    for e in l:
        if isinstance(e, list):
            ret.extend(e)
        else:
            ret.append(e)
    return ret


def iterate_format(video_format: Dict[str, Any], for_widgets: bool = True):
    """Iterate over widget/argument definitions inside a format json."""

    def indirector(cont: Any, index: Any):
        if isinstance(cont[index], list) and (
            not for_widgets or (len(cont[index]) > 1 and not isinstance(cont[index][1], dict))
        ):
            inp = yield cont[index]
            if inp is not None:
                cont[index] = inp
                yield

    for k in video_format:
        if k == "extra_widgets":
            if for_widgets:
                yield from video_format["extra_widgets"]
        elif k.endswith("_pass"):
            for i in range(len(video_format[k])):
                yield from indirector(video_format[k], i)
            if not for_widgets:
                video_format[k] = flatten_list(video_format[k])
        else:
            yield from indirector(video_format, k)


_external_formats_dir = os.environ.get("VHS_BASE_FORMATS_DIR")
if not _external_formats_dir:
    _external_formats_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_formats")


@cached(5)
def get_video_formats():
    format_files: Dict[str, Any] = {}

    for format_name in folder_paths.get_filename_list("VHS_video_formats"):
        format_files[format_name] = folder_paths.get_full_path("VHS_video_formats", format_name)

    for k in BUILTIN_VIDEO_FORMATS.keys():
        format_files.setdefault(k, ("__embedded__", k))

    if _external_formats_dir and os.path.isdir(_external_formats_dir):
        for item in os.scandir(_external_formats_dir):
            if not item.is_file() or not item.name.endswith(".json"):
                continue
            format_files[item.name[:-5]] = item.path

    formats: List[str] = []
    format_widgets: Dict[str, List[Any]] = {}

    for format_name, src in format_files.items():
        if isinstance(src, tuple) and src[0] == "__embedded__":
            video_format = copy.deepcopy(BUILTIN_VIDEO_FORMATS[src[1]])
        else:
            with open(src, "r", encoding="utf-8") as stream:
                video_format = json.load(stream)

        video_format.pop("save_metadata", None)

        if "gifski_pass" in video_format and gifski_path is None:
            continue

        widgets = list(iterate_format(video_format))
        formats.append("video/" + format_name)
        if widgets:
            format_widgets["video/" + format_name] = widgets

    return formats, format_widgets


def apply_format_widgets(format_name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Load a format definition and fill/resolve widget-driven parameters.

    `format_name` here is the part after 'video/', i.e. the json name without extension.
    """

    if format_name in BUILTIN_VIDEO_FORMATS:
        video_format = copy.deepcopy(BUILTIN_VIDEO_FORMATS[format_name])
    else:
        external_path = None
        if _external_formats_dir:
            p = os.path.join(_external_formats_dir, format_name + ".json")
            if os.path.exists(p):
                external_path = p

        if external_path is not None:
            video_format_path = external_path
        else:
            video_format_path = folder_paths.get_full_path("VHS_video_formats", format_name)

        with open(video_format_path, "r", encoding="utf-8") as stream:
            video_format = json.load(stream)

    video_format.pop("save_metadata", None)

    for w in iterate_format(video_format):
        if w[0] not in kwargs:
            if len(w) > 2 and isinstance(w[2], dict) and "default" in w[2]:
                default = w[2]["default"]
            else:
                if isinstance(w[1], list):
                    default = w[1][0]
                else:
                    default = {"BOOLEAN": False, "INT": 0, "FLOAT": 0, "STRING": ""}[w[1]]
            kwargs[w[0]] = default
            logger.warning(f"Missing input for {w[0]} has been set to {default}")

    wit = iterate_format(video_format, False)
    for w in wit:
        while isinstance(w, list):
            if len(w) == 1:
                w = [Template(x).substitute(**kwargs) for x in w[0]]
                break
            elif isinstance(w[1], dict):
                w = w[1][str(kwargs[w[0]])]
            elif len(w) > 3:
                w = Template(w[3]).substitute(val=kwargs[w[0]])
            else:
                w = str(kwargs[w[0]])
        wit.send(w)

    video_format["save_metadata"] = "False"

    return video_format


# -----------------------------------------------------------------------------
# Tensor -> bytes helpers
# -----------------------------------------------------------------------------


def tensor_to_int(tensor: torch.Tensor, bits: int) -> np.ndarray:
    arr = tensor.cpu().numpy() * (2**bits - 1) + 0.5
    return np.clip(arr, 0, (2**bits - 1))


def tensor_to_shorts(tensor: torch.Tensor) -> np.ndarray:
    return tensor_to_int(tensor, 16).astype(np.uint16)


def tensor_to_bytes(tensor: torch.Tensor) -> np.ndarray:
    return tensor_to_int(tensor, 8).astype(np.uint8)


# -----------------------------------------------------------------------------
# ffmpeg / gifski pipeline processes (generators)
# -----------------------------------------------------------------------------


def ffmpeg_process(
    args: List[str], video_format: Dict[str, Any], video_metadata: Dict[str, Any], file_path: str, env: Dict[str, str]
):
    res = None
    frame_data = yield
    total_frames_output = 0

    if video_format.get("save_metadata", "False") != "False":
        os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
        metadata_path = os.path.join(folder_paths.get_temp_directory(), "metadata.txt")

        def escape_ffmpeg_metadata(key: str, value: Any) -> str:
            value = str(value)
            value = value.replace("\\", "\\\\")
            value = value.replace(";", "\\;")
            value = value.replace("#", "\\#")
            value = value.replace("=", "\\=")
            value = value.replace("\n", "\\\n")
            return f"{key}={value}"

        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(";FFMETADATA1\n")
            if "prompt" in video_metadata:
                f.write(escape_ffmpeg_metadata("prompt", json.dumps(video_metadata["prompt"])) + "\n")
            if "workflow" in video_metadata:
                f.write(escape_ffmpeg_metadata("workflow", json.dumps(video_metadata["workflow"])) + "\n")
            for k, v in video_metadata.items():
                if k not in ["prompt", "workflow"]:
                    f.write(escape_ffmpeg_metadata(k, json.dumps(v)) + "\n")

        m_args = (
            args[:1]
            + ["-i", metadata_path]
            + args[1:]
            + ["-metadata", "creation_time=now", "-movflags", "use_metadata_tags"]
        )

        with subprocess.Popen(m_args + [file_path], stderr=subprocess.PIPE, stdin=subprocess.PIPE, env=env) as proc:
            try:
                while frame_data is not None:
                    proc.stdin.write(frame_data)
                    frame_data = yield
                    total_frames_output += 1
                proc.stdin.flush()
                proc.stdin.close()
                res = proc.stderr.read()
            except BrokenPipeError:
                err = proc.stderr.read()
                if os.path.exists(file_path):
                    raise Exception("An error occurred in the ffmpeg subprocess:\n" + err.decode(*ENCODE_ARGS))
                print(err.decode(*ENCODE_ARGS), end="", file=sys.stderr)
                logger.warning("An error occurred when saving with metadata")

    if res != b"":
        with subprocess.Popen(args + [file_path], stderr=subprocess.PIPE, stdin=subprocess.PIPE, env=env) as proc:
            try:
                while frame_data is not None:
                    proc.stdin.write(frame_data)
                    frame_data = yield
                    total_frames_output += 1
                proc.stdin.flush()
                proc.stdin.close()
                res = proc.stderr.read()
            except BrokenPipeError:
                res = proc.stderr.read()
                raise Exception("An error occurred in the ffmpeg subprocess:\n" + res.decode(*ENCODE_ARGS))

    yield total_frames_output
    if res and len(res) > 0:
        print(res.decode(*ENCODE_ARGS), end="", file=sys.stderr)


def gifski_process(
    args: List[str],
    dimensions: Tuple[int, int],
    frame_rate: float,
    video_format: Dict[str, Any],
    file_path: str,
    env: Dict[str, str],
):
    if gifski_path is None:
        raise ProcessLookupError("gifski is required for this output format but was not found")

    frame_data = yield
    with subprocess.Popen(
        args + video_format["main_pass"] + ["-f", "yuv4mpegpipe", "-"],
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=env,
    ) as procff:
        with subprocess.Popen(
            [gifski_path]
            + video_format["gifski_pass"]
            + ["-W", f"{dimensions[0]}", "-H", f"{dimensions[1]}"]
            + ["-r", f"{frame_rate}"]
            + ["-q", "-o", file_path, "-"],
            stderr=subprocess.PIPE,
            stdin=procff.stdout,
            stdout=subprocess.PIPE,
            env=env,
        ) as procgs:
            try:
                while frame_data is not None:
                    procff.stdin.write(frame_data)
                    frame_data = yield
                procff.stdin.flush()
                procff.stdin.close()
                resff = procff.stderr.read()
                resgs = procgs.stderr.read()
                outgs = procgs.stdout.read()
            except BrokenPipeError:
                procff.stdin.close()
                resff = procff.stderr.read()
                resgs = procgs.stderr.read()
                raise Exception(
                    "An error occurred while creating gifski output\n"
                    "Make sure you are using gifski --version >=1.32.0\n"
                    + "ffmpeg: "
                    + resff.decode(*ENCODE_ARGS)
                    + "\n"
                    + "gifski: "
                    + resgs.decode(*ENCODE_ARGS)
                )

    if resff and len(resff) > 0:
        print(resff.decode(*ENCODE_ARGS), end="", file=sys.stderr)
    if resgs and len(resgs) > 0:
        print(resgs.decode(*ENCODE_ARGS), end="", file=sys.stderr)
    if outgs and len(outgs) > 0:
        print(outgs.decode(*ENCODE_ARGS))


def to_pingpong(inp: Any):
    if not hasattr(inp, "__getitem__"):
        inp = list(inp)
    yield from inp
    for i in range(len(inp) - 2, 0, -1):
        yield inp[i]


# -----------------------------------------------------------------------------
# VideoCombine node
# -----------------------------------------------------------------------------


class TSVideoCombineNoMetadata:
    @classmethod
    def INPUT_TYPES(cls):
        ffmpeg_formats, format_widgets = get_video_formats()
        format_widgets["image/webp"] = [["lossless", "BOOLEAN", {"default": True}]]
        return {
            "required": {
                "images": (imageOrLatent,),
                "frame_rate": (floatOrInt, {"default": 8, "min": 1, "step": 1}),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "filename_prefix": ("STRING", {"default": "AnimateDiff"}),
                "format": (
                    ["image/gif", "image/webp"] + ffmpeg_formats,
                    {"formats": format_widgets},
                ),
                "pingpong": ("BOOLEAN", {"default": False}),
                "save_output": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "meta_batch": ("VHS_BatchManager",),
                "vae": ("VAE",),
            },
            "hidden": ContainsAll(
                {
                    "prompt": "PROMPT",
                    "extra_pnginfo": "EXTRA_PNGINFO",
                    "unique_id": "UNIQUE_ID",
                }
            ),
        }

    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    OUTPUT_NODE = True
    CATEGORY = "Video Combine"
    FUNCTION = "combine_video"

    def combine_video(
        self,
        frame_rate: int,
        loop_count: int,
        images=None,
        latents=None,
        filename_prefix: str = "AnimateDiff",
        format: str = "image/gif",
        pingpong: bool = False,
        save_output: bool = True,
        prompt=None,
        extra_pnginfo=None,
        audio=None,
        unique_id=None,
        manual_format_widgets=None,
        meta_batch=None,
        vae=None,
        **kwargs,
    ):
        if latents is not None:
            images = latents
        if images is None:
            return ((save_output, []),)

        if vae is not None:
            if isinstance(images, dict):
                images = images["samples"]
            else:
                vae = None

        if isinstance(images, torch.Tensor) and images.size(0) == 0:
            return ((save_output, []),)

        num_frames = len(images)
        pbar = ProgressBar(num_frames)

        if vae is not None:
            downscale_ratio = getattr(vae, "downscale_ratio", 8)
            width = images.size(-1) * downscale_ratio
            height = images.size(-2) * downscale_ratio
            frames_per_batch = (1920 * 1080 * 16) // (width * height) or 1

            def batched(it, n):
                while batch := tuple(itertools.islice(it, n)):
                    yield batch

            def batched_decode(latents_iter, vae_obj, fpb):
                for batch in batched(iter(latents_iter), fpb):
                    latent_batch = torch.from_numpy(np.array(batch))
                    yield from vae_obj.decode(latent_batch)

            images = batched_decode(images, vae, frames_per_batch)
            first_image = next(images)
            images = itertools.chain([first_image], images)
            while len(first_image.shape) > 3:
                first_image = first_image[0]
        else:
            first_image = images[0]
            images = iter(images)

        output_dir = folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()
        (full_output_folder, filename, _, subfolder, _) = folder_paths.get_save_image_path(filename_prefix, output_dir)
        output_files: List[str] = []

        metadata = PngInfo()
        video_metadata: Dict[str, Any] = {}
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
            video_metadata["prompt"] = json.dumps(prompt)

        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                video_metadata[x] = extra_pnginfo[x]
            extra_options = extra_pnginfo.get("workflow", {}).get("extra", {})
        else:
            extra_options = {}

        metadata.add_text("CreationTime", datetime.datetime.now().isoformat(" ")[:19])

        if meta_batch is not None and unique_id in getattr(meta_batch, "outputs", {}):
            (counter, output_process) = meta_batch.outputs[unique_id]
        else:
            max_counter = 0
            matcher = re.compile(f"{re.escape(filename)}_(\\d+)\\D*\\..+", re.IGNORECASE)
            for existing_file in os.listdir(full_output_folder):
                match = matcher.fullmatch(existing_file)
                if match:
                    file_counter = int(match.group(1))
                    max_counter = max(max_counter, file_counter)
            counter = max_counter + 1
            output_process = None

        first_image_file = f"{filename}_{counter:05}.png"
        png_path = os.path.join(full_output_folder, first_image_file)
        if extra_options.get("VHS_MetadataImage", True) is not False:
            Image.fromarray(tensor_to_bytes(first_image)).save(png_path, pnginfo=metadata, compress_level=4)
        output_files.append(png_path)

        format_type, format_ext = format.split("/")

        if format_type == "image":
            if meta_batch is not None:
                raise Exception("Pillow('image/') formats are not compatible with batched output")

            image_kwargs: Dict[str, Any] = {}
            if format_ext == "gif":
                image_kwargs["disposal"] = 2
            if format_ext == "webp":
                exif = Image.Exif()
                exif[ExifTags.IFD.Exif] = {36867: datetime.datetime.now().isoformat(" ")[:19]}
                image_kwargs["exif"] = exif
                image_kwargs["lossless"] = kwargs.get("lossless", True)

            out_file = f"{filename}_{counter:05}.{format_ext}"
            out_path = os.path.join(full_output_folder, out_file)

            if pingpong:
                images = to_pingpong(images)

            def frames_gen(images_iter):
                for i in images_iter:
                    pbar.update(1)
                    yield Image.fromarray(tensor_to_bytes(i))

            frames = frames_gen(images)
            next(frames).save(
                out_path,
                format=format_ext.upper(),
                save_all=True,
                append_images=frames,
                duration=round(1000 / frame_rate),
                loop=loop_count,
                compress_level=4,
                **image_kwargs,
            )
            output_files.append(out_path)
            file_for_preview = out_file

        else:
            if ffmpeg_path is None:
                raise ProcessLookupError(
                    "ffmpeg is required for video outputs and could not be found.\n"
                    "In order to use video outputs, you must either:\n"
                    "- Install imageio-ffmpeg with pip,\n"
                    "- Place a ffmpeg executable in the ComfyUI working directory, or\n"
                    "- Install ffmpeg and add it to the system path."
                )

            if manual_format_widgets is not None:
                logger.warning(
                    "Format args can now be passed directly. The manual_format_widgets argument is deprecated."
                )
                kwargs.update(manual_format_widgets)

            has_alpha = first_image.shape[-1] == 4
            kwargs["has_alpha"] = has_alpha

            video_format = apply_format_widgets(format_ext, kwargs)
            dim_alignment = video_format.get("dim_alignment", 2)

            if (first_image.shape[1] % dim_alignment) or (first_image.shape[0] % dim_alignment):
                to_pad = (-first_image.shape[1] % dim_alignment, -first_image.shape[0] % dim_alignment)
                padding = (
                    to_pad[0] // 2,
                    to_pad[0] - to_pad[0] // 2,
                    to_pad[1] // 2,
                    to_pad[1] - to_pad[1] // 2,
                )
                padfunc = torch.nn.ReplicationPad2d(padding)

                def pad(image):
                    image = image.permute((2, 0, 1))
                    padded = padfunc(image.to(dtype=torch.float32))
                    return padded.permute((1, 2, 0))

                images = map(pad, images)
                dimensions = (
                    -first_image.shape[1] % dim_alignment + first_image.shape[1],
                    -first_image.shape[0] % dim_alignment + first_image.shape[0],
                )
                logger.warning("Output images were not of valid resolution; padding was applied")
            else:
                dimensions = (first_image.shape[1], first_image.shape[0])

            if pingpong:
                if meta_batch is not None:
                    logger.error("pingpong is incompatible with batched output")
                images = to_pingpong(images)
                if num_frames > 2:
                    num_frames = num_frames + (num_frames - 2)
                    pbar.total = num_frames

            loop_args = ["-vf", f"loop=loop={loop_count}:size={num_frames}"] if loop_count > 0 else []

            if video_format.get("input_color_depth", "8bit") == "16bit":
                images = map(tensor_to_shorts, images)
                i_pix_fmt = "rgba64" if has_alpha else "rgb48"
            else:
                images = map(tensor_to_bytes, images)
                i_pix_fmt = "rgba" if has_alpha else "rgb24"

            out_file = f"{filename}_{counter:05}.{video_format['extension']}"
            out_path = os.path.join(full_output_folder, out_file)

            bitrate_arg: List[str] = []
            bitrate = video_format.get("bitrate")
            if bitrate is not None:
                if video_format.get("megabit") == "True":
                    bitrate_arg = ["-b:v", str(bitrate) + "M"]
                else:
                    bitrate_arg = ["-b:v", str(bitrate) + "K"]

            args = [
                ffmpeg_path,
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                i_pix_fmt,
                "-color_range",
                "pc",
                "-colorspace",
                "rgb",
                "-color_primaries",
                "bt709",
                "-color_trc",
                video_format.get("fake_trc", "iec61966-2-1"),
                "-s",
                f"{dimensions[0]}x{dimensions[1]}",
                "-r",
                str(frame_rate),
                "-i",
                "-",
            ] + loop_args

            images = map(lambda x: x.tobytes(), images)

            env = os.environ.copy()
            if "environment" in video_format:
                env.update(video_format["environment"])

            if "pre_pass" in video_format:
                if meta_batch is not None:
                    raise Exception("Formats requiring pre_pass are incompatible with Batch Manager")
                images = [b"".join(images)]
                os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
                in_args_len = args.index("-i") + 2
                pre_pass_args = args[:in_args_len] + video_format["pre_pass"]
                merge_filter_args(pre_pass_args)
                try:
                    subprocess.run(pre_pass_args, input=images[0], env=env, capture_output=True, check=True)
                except subprocess.CalledProcessError as e:
                    raise Exception("An error occurred in the ffmpeg prepass:\n" + e.stderr.decode(*ENCODE_ARGS))

            if "inputs_main_pass" in video_format:
                in_args_len = args.index("-i") + 2
                args = args[:in_args_len] + video_format["inputs_main_pass"] + args[in_args_len:]

            if output_process is None:
                if "gifski_pass" in video_format:
                    format = "image/gif"
                    output_process = gifski_process(args, dimensions, frame_rate, video_format, out_path, env)
                    audio = None
                else:
                    args += video_format["main_pass"] + bitrate_arg
                    merge_filter_args(args)
                    output_process = ffmpeg_process(args, video_format, video_metadata, out_path, env)

                output_process.send(None)
                if meta_batch is not None:
                    meta_batch.outputs[unique_id] = (counter, output_process)

            for image_bytes in images:
                pbar.update(1)
                output_process.send(image_bytes)

            if meta_batch is not None:
                requeue_workflow((meta_batch.unique_id, not meta_batch.has_closed_inputs))

            if meta_batch is None or meta_batch.has_closed_inputs:
                try:
                    total_frames_output = output_process.send(None)
                    output_process.send(None)
                except StopIteration:
                    total_frames_output = num_frames

                if meta_batch is not None:
                    meta_batch.outputs.pop(unique_id, None)
                    if len(meta_batch.outputs) == 0:
                        meta_batch.reset()
            else:
                return {"ui": {"unfinished_batch": [True]}, "result": ((save_output, []),)}

            output_files.append(out_path)

            a_waveform = None
            if audio is not None:
                try:
                    a_waveform = audio["waveform"]
                except Exception:
                    a_waveform = None

            if a_waveform is not None:
                output_with_audio = f"{filename}_{counter:05}-audio.{video_format['extension']}"
                output_with_audio_path = os.path.join(full_output_folder, output_with_audio)

                if "audio_pass" not in video_format:
                    logger.warning("Selected video format does not have explicit audio support")
                    video_format["audio_pass"] = ["-c:a", "libopus"]

                channels = audio["waveform"].size(1)
                min_audio_dur = total_frames_output / frame_rate + 1
                apad = (
                    []
                    if video_format.get("trim_to_audio", "True") != "False"
                    else ["-af", f"apad=whole_dur={min_audio_dur}"]
                )

                mux_args = (
                    [
                        ffmpeg_path,
                        "-v",
                        "error",
                        "-n",
                        "-i",
                        out_path,
                        "-ar",
                        str(audio["sample_rate"]),
                        "-ac",
                        str(channels),
                        "-f",
                        "f32le",
                        "-i",
                        "-",
                        "-c:v",
                        "copy",
                    ]
                    + video_format["audio_pass"]
                    + apad
                    + ["-shortest", output_with_audio_path]
                )

                audio_data = audio["waveform"].squeeze(0).transpose(0, 1).numpy().tobytes()
                merge_filter_args(mux_args, "-af")
                try:
                    res = subprocess.run(mux_args, input=audio_data, env=env, capture_output=True, check=True)
                except subprocess.CalledProcessError as e:
                    raise Exception("An error occurred in the ffmpeg subprocess:\n" + e.stderr.decode(*ENCODE_ARGS))

                if res.stderr:
                    print(res.stderr.decode(*ENCODE_ARGS), end="", file=sys.stderr)

                output_files.append(output_with_audio_path)
                file_for_preview = output_with_audio
            else:
                file_for_preview = out_file

        if extra_options.get("VHS_KeepIntermediate", True) is False:
            for intermediate in output_files[1:-1]:
                if os.path.exists(intermediate):
                    os.remove(intermediate)

        preview = {
            "filename": file_for_preview,
            "subfolder": subfolder,
            "type": "output" if save_output else "temp",
            "format": format,
            "frame_rate": frame_rate,
            "workflow": first_image_file,
            "fullpath": output_files[-1],
        }
        if num_frames == 1 and "png" in format and "%03d" in file_for_preview:
            preview["format"] = "image/png"
            preview["filename"] = file_for_preview.replace("%03d", "001")

        return {"ui": {"gifs": [preview]}, "result": ((save_output, output_files),)}
