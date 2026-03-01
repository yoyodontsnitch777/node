import os
import re
import uuid
import shutil


def extract_first_number(s: str):
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


def _safe_list_files(directory: str):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


def _format_name(index: int, digits: int, prefix: str, ext: str):
    """
    ext ожидается как ".png"/".jpg"/".jpeg" (с точкой).
    ВАЖНО: underscore после номера ВСЕГДА, потом расширение как есть.
    Пример: prefix_0001_.png
    """
    num = str(index).zfill(digits)
    left = f"{prefix}_" if prefix else ""
    return f"{left}{num}_{ext}"


def _index_taken(directory: str, digits: int, prefix: str, index: int) -> bool:
    """
    Проверяем, занят ли номер index ЛЮБЫМ расширением в папке.
    Т.е. если есть prefix_0001_.png, то prefix_0001_.jpg уже нельзя.
    """
    num = str(index).zfill(digits)
    left = f"{prefix}_" if prefix else ""
    start = f"{left}{num}_"

    try:
        entries = os.listdir(directory)
    except FileNotFoundError:
        return False

    for f in entries:
        p = os.path.join(directory, f)
        if os.path.isfile(p) and f.startswith(start):
            return True
    return False


def _find_next_free_index(directory: str, digits: int, prefix: str, start_from: int = 1) -> int:
    idx = max(1, int(start_from))
    while _index_taken(directory, digits, prefix, idx):
        idx += 1
    return idx


class RenameFilesInDir:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            },
            "optional": {
                "output_directory": ("STRING", {"default": ""}),
                "sort_method": (sort_methods,),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                "files_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "prefix": ("STRING", {"default": ""}),
                "digits": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("COUNT",)
    FUNCTION = "run"
    CATEGORY = "InspirePack/files"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def run(
        self,
        directory: str,
        output_directory: str = "",
        sort_method=None,
        start_index: int = 0,
        files_load_cap: int = 0,
        prefix: str = "",
        digits: int = 4,
    ):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' cannot be found.")

        files = _safe_list_files(directory)
        if not files:
            return (0,)

        files = sort_by(files, directory, sort_method)
        files = files[start_index:]

        if files_load_cap > 0:
            files = files[:files_load_cap]

        if not files:
            return (0,)

        inplace = (output_directory is None) or (str(output_directory).strip() == "")

        if not inplace:
            os.makedirs(output_directory, exist_ok=True)

        count = 0

        # ---------- COPY MODE ----------
        if not inplace:
            for fname in files:
                src = os.path.join(directory, fname)
                _, ext = os.path.splitext(fname)  # ext = ".png" / ".jpg" / ...

                next_idx = _find_next_free_index(output_directory, digits, prefix, start_from=1)
                new_name = _format_name(next_idx, digits, prefix, ext)

                dst = os.path.join(output_directory, new_name)
                shutil.copy2(src, dst)
                count += 1

            return (count,)

        # ---------- INPLACE RENAME ----------
        temp_map = []
        used_temp = set()

        def _make_temp_name(old_name: str):
            while True:
                t = f"__tmp__{uuid.uuid4().hex}__{old_name}"
                if t not in used_temp and not os.path.exists(os.path.join(directory, t)):
                    used_temp.add(t)
                    return t

        # phase1 -> temp
        for fname in files:
            old_path = os.path.join(directory, fname)
            tmp = _make_temp_name(fname)
            tmp_path = os.path.join(directory, tmp)

            os.rename(old_path, tmp_path)
            temp_map.append((tmp, fname))

        # phase2 -> final
        for tmp, original_name in temp_map:
            tmp_path = os.path.join(directory, tmp)
            _, ext = os.path.splitext(original_name)

            next_idx = _find_next_free_index(directory, digits, prefix, start_from=1)
            new_name = _format_name(next_idx, digits, prefix, ext)

            new_path = os.path.join(directory, new_name)
            os.rename(tmp_path, new_path)
            count += 1

        return (count,)
