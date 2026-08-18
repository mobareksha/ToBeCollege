"""识别并消除 16 行 10 列屏幕网格中总和为 10 的矩形区域。

启动后由用户框选网格区域。程序识别每个单元格中的数字，并拖动鼠标框选总和为 10
的矩形。使用 --preview 可以只检查识别结果，不控制鼠标。
"""

from __future__ import annotations

import argparse
import ctypes
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import cv2
    import mss
    import numpy as np
    import pyautogui
except ModuleNotFoundError as exc:
    requirements = Path(__file__).with_name("requirements.txt")
    raise SystemExit(
        f"Missing Python package: {exc.name}. Install dependencies with:\n"
        f'python -m pip install -r "{requirements}"'
    ) from exc


@dataclass(frozen=True)
class Region:
    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True)
class Match:
    first: tuple[int, int]
    second: tuple[int, int]
    gap: int


def make_dpi_aware() -> None:
    """确保 Windows 下截图像素坐标与鼠标坐标一致。"""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def capture(region: Region | None = None) -> np.ndarray:
    with mss.MSS() as grabber:
        monitor = (
            {
                "left": region.left,
                "top": region.top,
                "width": region.width,
                "height": region.height,
            }
            if region
            else grabber.monitors[0]
        )
        return cv2.cvtColor(np.asarray(grabber.grab(monitor)), cv2.COLOR_BGRA2BGR)


def select_region() -> Region:
    screenshot = capture()
    x, y, width, height = map(
        int,
        cv2.selectROI(
            "Drag over the complete grid, then press Enter",
            screenshot,
            showCrosshair=True,
            fromCenter=False,
        ),
    )
    cv2.destroyAllWindows()
    # Windows 异步关闭 OpenCV 窗口，需要等待桌面合成器移除框选窗口后再截图。
    for _ in range(5):
        cv2.waitKey(1)
        time.sleep(0.15)
    if width <= 0 or height <= 0:
        raise SystemExit("No game region selected.")
    # 多显示器环境中的虚拟桌面起点可能是负坐标。
    with mss.MSS() as grabber:
        virtual = grabber.monitors[0]
    return Region(x + virtual["left"], y + virtual["top"], width, height)


def calibrate_grid(
    image: np.ndarray, region: Region, rows: int, cols: int
) -> tuple[np.ndarray, Region]:
    """检测白色数字块，并计算框选区域内真正的网格边界。"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    saturation = hsv[:, :, 1]

    green = cv2.inRange(hsv, np.array((30, 55, 25)), np.array((105, 255, 255)))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    green_contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if green_contours:
        gx, gy, green_width, green_height = cv2.boundingRect(
            max(green_contours, key=cv2.contourArea)
        )
    else:
        gx, gy, green_width, green_height = 0, 0, image.shape[1], image.shape[0]

    white = np.where((gray > 170) & (saturation < 90), 255, 0).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pitch_x_guess = green_width / cols
    pitch_y_guess = green_height / rows
    boxes = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        center_x, center_y = x + width / 2, y + height / 2
        inside_green = gx <= center_x <= gx + green_width and gy <= center_y <= gy + green_height
        tile_sized = (
            pitch_x_guess * 0.35 <= width <= pitch_x_guess * 1.1
            and pitch_y_guess * 0.35 <= height <= pitch_y_guess * 1.1
        )
        if inside_green and tile_sized and 0.45 <= width / height <= 1.8:
            boxes.append((x, y, width, height))

    if len(boxes) < rows * cols:
        print(f"Grid calibration found {len(boxes)}/{rows * cols} white tiles; using selected ROI.")
        return image, region

    # 保留面积最接近中位数的 160 个候选块，再根据中心坐标排列成行和列。
    median_area = float(np.median([width * height for _, _, width, height in boxes]))
    boxes.sort(key=lambda box: abs(box[2] * box[3] - median_area))
    boxes = boxes[: rows * cols]
    boxes.sort(key=lambda box: box[1] + box[3] / 2)
    row_groups = [boxes[index * cols : (index + 1) * cols] for index in range(rows)]
    if any(len(group) != cols for group in row_groups):
        return image, region
    for group in row_groups:
        group.sort(key=lambda box: box[0] + box[2] / 2)

    x_centers = np.median(
        [[box[0] + box[2] / 2 for box in group] for group in row_groups], axis=0
    )
    y_centers = np.array(
        [np.median([box[1] + box[3] / 2 for box in group]) for group in row_groups]
    )
    pitch_x = float(np.median(np.diff(x_centers)))
    pitch_y = float(np.median(np.diff(y_centers)))
    left = max(0, round(x_centers[0] - pitch_x / 2))
    top = max(0, round(y_centers[0] - pitch_y / 2))
    right = min(image.shape[1], round(x_centers[-1] + pitch_x / 2))
    bottom = min(image.shape[0], round(y_centers[-1] + pitch_y / 2))
    if right <= left or bottom <= top:
        return image, region

    calibrated = Region(region.left + left, region.top + top, right - left, bottom - top)
    print(f"Grid calibrated inside selection: offset=({left},{top}), size={right-left}x{bottom-top}")
    return image[top:bottom, left:right], calibrated


def save_region(path: Path, region: Region) -> None:
    path.write_text(json.dumps(region.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")


def load_region(path: Path) -> Region | None:
    if not path.exists():
        return None
    try:
        return Region(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None


REFERENCE_GRID = (
    (4, 8, 2, 6, 1, 4, 3, 4, 5, 2),
    (7, 3, 3, 3, 5, 3, 7, 9, 8, 2),
    (3, 8, 6, 4, 9, 8, 6, 3, 6, 3),
    (3, 6, 8, 9, 3, 7, 1, 3, 1, 9),
    (2, 4, 8, 5, 1, 4, 1, 7, 3, 1),
    (3, 9, 1, 9, 1, 7, 5, 4, 2, 7),
    (9, 3, 6, 5, 3, 1, 5, 9, 9, 8),
    (8, 4, 4, 4, 3, 3, 4, 8, 1, 7),
    (1, 8, 7, 2, 4, 4, 4, 5, 6, 6),
    (9, 7, 7, 2, 5, 6, 2, 3, 1, 1),
    (8, 1, 5, 6, 6, 8, 8, 7, 7, 1),
    (5, 3, 7, 3, 6, 2, 4, 1, 6, 5),
    (3, 8, 9, 3, 5, 4, 1, 6, 5, 2),
    (2, 5, 1, 4, 1, 4, 5, 1, 2, 7),
    (9, 1, 7, 3, 7, 8, 5, 5, 5, 2),
    (8, 3, 4, 1, 1, 5, 9, 1, 4, 6),
)


def crop_cell(image: np.ndarray, row: int, col: int, rows: int, cols: int) -> np.ndarray:
    height, width = image.shape[:2]
    y1, y2 = round(row * height / rows), round((row + 1) * height / rows)
    x1, x2 = round(col * width / cols), round((col + 1) * width / cols)
    pad_x = max(2, round((x2 - x1) * 0.08))
    pad_y = max(2, round((y2 - y1) * 0.08))
    return image[
        max(0, y1 - pad_y) : min(height, y2 + pad_y),
        max(0, x1 - pad_x) : min(width, x2 + pad_x),
    ]


def normalize_digit(cell: np.ndarray) -> np.ndarray | None:
    """提取数字字形，并归一化为不受原始尺寸影响的二值图。"""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    tile_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(tile_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        tile = max(contours, key=cv2.contourArea)
        x, y, width, height = cv2.boundingRect(tile)
        if width >= 5 and height >= 5:
            cell = cell[y : y + height, x : x + width]

    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    saturation = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)[:, :, 1]
    # 黑色数字的亮度和饱和度较低，绿色棋盘的饱和度较高；即使白块被框选边缘截断，
    # 也能排除绿色背景。
    digit_mask = np.where((gray < 140) & (saturation < 110), 255, 0).astype(np.uint8)
    digit_contours, _ = cv2.findContours(digit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_contours = [contour for contour in digit_contours if cv2.contourArea(contour) >= 2]
    if not digit_contours:
        return None
    x, y, width, height = cv2.boundingRect(max(digit_contours, key=cv2.contourArea))
    glyph = digit_mask[y : y + height, x : x + width]

    canvas = np.zeros((48, 36), dtype=np.uint8)
    scale = min(30 / max(1, width), 40 / max(1, height))
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    glyph = cv2.resize(glyph, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
    left = (canvas.shape[1] - resized_width) // 2
    top = (canvas.shape[0] - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = glyph
    return canvas


def hog_feature(glyph: np.ndarray) -> np.ndarray:
    descriptor = cv2.HOGDescriptor((36, 48), (12, 12), (6, 6), (6, 6), 9)
    return descriptor.compute(glyph).ravel().astype(np.float32)


def build_template_bank(path: Path) -> dict[int, list[np.ndarray]]:
    reference = cv2.imread(str(path))
    if reference is None:
        raise SystemExit(f"Reference board image not found: {path}")
    bank: dict[int, list[np.ndarray]] = {digit: [] for digit in range(1, 10)}
    for scale in (0.5, 0.7, 1.0, 1.5, 2.0):
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        scaled = cv2.resize(reference, None, fx=scale, fy=scale, interpolation=interpolation)
        samples: dict[int, list[np.ndarray]] = {digit: [] for digit in range(1, 10)}
        for row in range(16):
            for col in range(10):
                glyph = normalize_digit(crop_cell(scaled, row, col, 16, 10))
                if glyph is not None:
                    samples[REFERENCE_GRID[row][col]].append(hog_feature(glyph))
        for digit, features in samples.items():
            if features:
                bank[digit].append(np.mean(features, axis=0))
    return bank


def classify_digit(glyph: np.ndarray, bank: dict[int, list[np.ndarray]]) -> tuple[int, float]:
    query = hog_feature(glyph)
    query_norm = float(np.linalg.norm(query)) or 1.0
    best_digit, best_score = 1, -1.0
    for digit, templates in bank.items():
        for sample in templates:
            score = float(np.dot(query, sample) / (query_norm * (np.linalg.norm(sample) or 1.0)))
            if score > best_score:
                best_digit, best_score = digit, score
    return best_digit, best_score * 100


def recognize_grid(
    image: np.ndarray, rows: int, cols: int, margin: float
) -> tuple[list[list[int | None]], list[list[float]]]:
    del margin
    grid: list[list[int | None]] = []
    confidence_grid: list[list[float]] = []
    bank = build_template_bank(Path(__file__).with_name("digit_reference.png"))

    for row in range(rows):
        values: list[int | None] = []
        confidences: list[float] = []
        for col in range(cols):
            glyph = normalize_digit(crop_cell(image, row, col, rows, cols))
            if glyph is None:
                value, confidence = None, -1.0
            else:
                value, confidence = classify_digit(glyph, bank)
            values.append(value)
            confidences.append(confidence)
        grid.append(values)
        confidence_grid.append(confidences)
    return grid, confidence_grid


def find_matches(
    grid: list[list[int | None]], gap: int, diagonal: bool
) -> Iterable[Match]:
    rows, cols = len(grid), len(grid[0])
    distance = gap + 1
    directions = [(0, 1), (1, 0)]
    if diagonal:
        directions += [(1, 1), (1, -1)]

    for row in range(rows):
        for col in range(cols):
            first = grid[row][col]
            if first is None:
                continue
            for dr, dc in directions:
                other_row, other_col = row + dr * distance, col + dc * distance
                if not (0 <= other_row < rows and 0 <= other_col < cols):
                    continue
                second = grid[other_row][other_col]
                between_is_empty = all(
                    grid[row + dr * step][col + dc * step] is None
                    for step in range(1, distance)
                )
                if between_is_empty and second is not None and first + second == 10:
                    yield Match((row, col), (other_row, other_col), gap)


def find_boxes(grid: list[list[int | None]], max_size: int = 10) -> Iterable[Match]:
    """按尺寸从小到大的顺序生成数字总和为 10 的矩形。"""
    rows, cols = len(grid), len(grid[0])
    values = np.array([[value or 0 for value in row] for row in grid], dtype=np.int16)
    prefix = np.pad(values.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    for size in range(1, max_size + 1):
        if size == 1:
            dimensions = [(1, 1)]
        elif size == 2:
            dimensions = [(1, 2), (2, 1), (2, 2)]
        else:
            dimensions = [(size, 1), (1, size)]
            for inner in range(2, size):
                dimensions.extend(((inner, size), (size, inner)))
            dimensions.append((size, size))
        for height, width in dimensions:
            if height > rows or width > cols:
                continue
            sums = (
                prefix[height:, width:]
                - prefix[:-height, width:]
                - prefix[height:, :-width]
                + prefix[:-height, :-width]
            )
            for row, col in np.argwhere(sums == 10):
                yield Match(
                    (int(row), int(col)),
                    (int(row + height - 1), int(col + width - 1)),
                    size - 1,
                )


def cleared_cell_count(grid: list[list[int | None]], match: Match) -> int:
    return sum(
        grid[row][col] is not None
        for row in range(match.first[0], match.second[0] + 1)
        for col in range(match.first[1], match.second[1] + 1)
    )


def clear_box(grid: list[list[int | None]], match: Match) -> list[list[int | None]]:
    result = [row[:] for row in grid]
    for row in range(match.first[0], match.second[0] + 1):
        for col in range(match.first[1], match.second[1] + 1):
            result[row][col] = None
    return result


def choose_best_box(
    grid: list[list[int | None]],
    max_size: int,
    lookahead: int,
    branch_width: int,
) -> Match | None:
    """通过多步前瞻选择预计能清除最多格子的操作。"""
    cache: dict[tuple[tuple[tuple[int | None, ...], ...], int], tuple[float, Match | None]] = {}

    def search(state: list[list[int | None]], depth: int) -> tuple[float, Match | None]:
        key = (tuple(tuple(row) for row in state), depth)
        if key in cache:
            return cache[key]

        matches = list(find_boxes(state, max_size))
        if not matches:
            result = (0.0, None)
            cache[key] = result
            return result
        if depth == 0:
            occupied = {
                (row, col)
                for match in matches
                for row in range(match.first[0], match.second[0] + 1)
                for col in range(match.first[1], match.second[1] + 1)
                if state[row][col] is not None
            }
            result = (0.02 * len(occupied) + 0.002 * len(matches), None)
            cache[key] = result
            return result

        ranked = sorted(
            matches,
            key=lambda match: (
                cleared_cell_count(state, match),
                -((match.second[0] - match.first[0] + 1) * (match.second[1] - match.first[1] + 1)),
            ),
            reverse=True,
        )[:branch_width]
        best_score = -1.0
        best_match = ranked[0]
        for match in ranked:
            removed = cleared_cell_count(state, match)
            future_score, _ = search(clear_box(state, match), depth - 1)
            score = removed + future_score
            if score > best_score:
                best_score = score
                best_match = match
        result = (best_score, best_match)
        cache[key] = result
        return result

    return search(grid, lookahead)[1]


def build_optimized_plan(
    grid: list[list[int | None]],
    max_size: int,
    max_actions: int,
    rollouts: int,
    candidate_pool: int,
) -> list[Match]:
    """模拟多条完整消除路线，并保留清除格子最多的一条。"""
    rng = np.random.default_rng(20260816)
    best_plan: list[Match] = []
    best_cleared = -1
    initial_count = sum(value is not None for row in grid for value in row)

    for rollout in range(max(1, rollouts)):
        state = [row[:] for row in grid]
        plan: list[Match] = []
        for _ in range(max_actions):
            matches = list(find_boxes(state, max_size))
            if not matches:
                break
            if rollout == 0:
                selected = matches[0]
            else:
                ranked = sorted(
                    matches,
                    key=lambda match: (
                        cleared_cell_count(state, match),
                        -((match.second[0] - match.first[0] + 1) * (match.second[1] - match.first[1] + 1)),
                    ),
                    reverse=True,
                )[:candidate_pool]
                weights = np.linspace(1.0, 0.2, len(ranked), dtype=np.float64)
                weights /= weights.sum()
                selected = ranked[int(rng.choice(len(ranked), p=weights))]
            state = clear_box(state, selected)
            plan.append(selected)

        cleared = initial_count - sum(value is not None for row in state for value in row)
        if (cleared, len(plan)) > (best_cleared, len(best_plan)):
            best_cleared = cleared
            best_plan = plan
    return best_plan


def drag_match(
    region: Region,
    rows: int,
    cols: int,
    match: Match,
    duration: float,
    inset: float,
) -> None:
    min_row = min(match.first[0], match.second[0])
    max_row = max(match.first[0], match.second[0])
    min_col = min(match.first[1], match.second[1])
    max_col = max(match.first[1], match.second[1])
    cell_width = region.width / cols
    cell_height = region.height / rows
    start = (
        region.left + round((min_col + inset) * cell_width),
        region.top + round((min_row + inset) * cell_height),
    )
    end = (
        region.left + round((max_col + 1 - inset) * cell_width),
        region.top + round((max_row + 1 - inset) * cell_height),
    )
    pyautogui.moveTo(*start, duration=0.08)
    pyautogui.dragTo(*end, duration=duration, button="left")


def format_grid(grid: list[list[int | None]]) -> str:
    return "\n".join(" ".join("." if value is None else str(value) for value in row) for row in grid)


def format_array(grid: list[list[int | None]]) -> str:
    return "[\n" + ",\n".join(
        "  [" + ", ".join("null" if value is None else str(value) for value in row) + "]"
        for row in grid
    ) + "\n]"


def preview(image: np.ndarray, grid: list[list[int | None]], rows: int, cols: int) -> None:
    canvas = image.copy()
    height, width = canvas.shape[:2]
    for row in range(rows):
        for col in range(cols):
            x1, x2 = round(col * width / cols), round((col + 1) * width / cols)
            y1, y2 = round(row * height / rows), round((row + 1) * height / rows)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 190, 80), 1)
            label = "." if grid[row][col] is None else str(grid[row][col])
            cv2.putText(
                canvas,
                label,
                (x1 + 4, y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
    cv2.imshow("Recognition preview - Q/Esc to quit, any other key to continue", canvas)


def show_capture_review(selected: np.ndarray, calibrated: np.ndarray) -> None:
    panels = []
    for image, label in ((selected, "SELECTED ROI"), (calibrated, "CALIBRATED 10x16 GRID")):
        scale = min(1.0, 560 / image.shape[0], 620 / image.shape[1])
        resized = cv2.resize(
            image,
            (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        panel = cv2.copyMakeBorder(resized, 34, 4, 4, 4, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        cv2.putText(panel, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        panels.append(panel)

    target_height = max(panel.shape[0] for panel in panels)
    panels = [
        cv2.copyMakeBorder(panel, 0, target_height - panel.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        for panel in panels
    ]
    review = cv2.hconcat(panels)
    window = "Captured image review - press any key to continue"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.imshow(window, review)
    cv2.resizeWindow(window, min(review.shape[1], 1250), min(review.shape[0], 700))
    cv2.waitKey(0)
    cv2.destroyWindow(window)


class ChineseArgumentParser(argparse.ArgumentParser):
    def format_help(self) -> str:
        return super().format_help().replace("usage:", "用法:").replace("options:", "选项:")


def parse_args() -> argparse.Namespace:
    parser = ChineseArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("-h", "--help", action="help", help="显示帮助信息并退出")
    parser.add_argument("--rows", type=int, default=16, help="网格行数，默认 16")
    parser.add_argument("--cols", type=int, default=10, help="网格列数，默认 10")
    parser.add_argument("--preview", action="store_true", help="只预览识别结果，不控制鼠标")
    parser.add_argument("--diagonal", action="store_true", help="允许处理对角线方向")
    parser.add_argument("--margin", type=float, default=0.18, help="数字裁剪边距比例，默认 0.18")
    parser.add_argument("--box-inset", type=float, default=0.05, help="鼠标框选向内缩进比例，默认 0.05")
    parser.add_argument("--delay", type=float, default=0.08, help="每次消除后的等待秒数，默认 0.08")
    parser.add_argument("--drag-duration", type=float, default=0.08, help="鼠标拖框持续秒数，默认 0.08")
    parser.add_argument("--max-box-size", type=int, default=10, help="矩形最大边长，默认 10")
    parser.add_argument("--rollouts", type=int, default=100, help="开始前模拟的完整路线数量，默认 30")
    parser.add_argument("--candidate-pool", type=int, default=3, help="每步参与优化的候选数量，默认 6")
    parser.add_argument("--countdown", type=int, default=3, help="开始控制鼠标前的倒计时秒数，默认 3")
    parser.add_argument("--max-actions", type=int, default=999, help="最多执行的消除次数，默认 50")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0 <= args.margin < 0.45):
        raise SystemExit("--margin must be between 0 and 0.45")
    if not (0 <= args.box_inset < 0.5):
        raise SystemExit("--box-inset must be between 0 and 0.5")
    if args.rollouts < 1 or args.candidate_pool < 1:
        raise SystemExit("--rollouts and --candidate-pool must be positive")
    make_dpi_aware()
    pyautogui.FAILSAFE = True  # 将鼠标移到屏幕左上角可紧急停止。
    region = select_region()

    actions = 0
    print(f"Region: {region}; grid: {args.rows} rows x {args.cols} columns")
    selected_image = capture(region)
    cv2.imwrite(str(Path(__file__).with_name("last_capture.png")), selected_image)
    first_image, region = calibrate_grid(selected_image, region, args.rows, args.cols)
    cv2.imwrite(str(Path(__file__).with_name("last_grid.png")), first_image)
    show_capture_review(selected_image, first_image)
    first_grid, _ = recognize_grid(first_image, args.rows, args.cols, args.margin)
    print("\nFirst recognition result (16 rows x 10 columns):\n" + format_array(first_grid))
    missing = sum(value is None for row in first_grid for value in row)
    if missing:
        print(f"WARNING: {missing} cells were not recognized. The initial board is expected to be full.")
    if not args.preview:
        answer = input("\nIs this recognition correct? Type y to start: ").strip().lower()
        if answer != "y":
            print("Cancelled. No mouse action was performed.")
            return
    else:
        preview(first_image, first_grid, args.rows, args.cols)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # 棋盘只截图并识别一次。此后以内存矩阵为准，被消除的位置设为 None。
    grid = [row[:] for row in first_grid]
    planning_started = time.perf_counter()
    plan = build_optimized_plan(
        grid,
        args.max_box_size,
        args.max_actions,
        args.rollouts,
        args.candidate_pool,
    )
    simulated = [row[:] for row in grid]
    for match in plan:
        simulated = clear_box(simulated, match)
    expected_cleared = sum(value is not None for row in grid for value in row) - sum(
        value is not None for row in simulated for value in row
    )
    print(
        f"Optimized {len(plan)} actions clearing {expected_cleared} cells "
        f"in {time.perf_counter() - planning_started:.3f}s."
    )
    if not plan:
        print(f"No rectangle with edge size 1..{args.max_box_size} sums to 10.")
        return
    for remaining in range(args.countdown, 0, -1):
        print(f"Mouse control starts in {remaining}...")
        time.sleep(1)

    for selected in plan:
        print("\nRecognized grid:\n" + format_grid(grid))
        box_height = selected.second[0] - selected.first[0] + 1
        box_width = selected.second[1] - selected.first[1] + 1
        print(
            f"Dragging {box_height}x{box_width}: "
            f"{selected.first} -> {selected.second}"
        )
        drag_match(
            region,
            args.rows,
            args.cols,
            selected,
            args.drag_duration,
            args.box_inset,
        )
        # 总和为 10 的矩形内所有数字都会消失。
        for row in range(selected.first[0], selected.second[0] + 1):
            for col in range(selected.first[1], selected.second[1] + 1):
                grid[row][col] = None
        actions += 1
        time.sleep(args.delay)

    print(f"Completed optimized plan: {actions} actions, {expected_cleared} cells cleared.")


if __name__ == "__main__":
    main()
