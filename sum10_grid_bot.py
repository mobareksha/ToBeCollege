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
    (8, 7, 7, 2, 7, 8, 1, 7, 2, 5),
    (9, 3, 9, 9, 6, 6, 5, 1, 7, 8),
    (8, 7, 9, 8, 6, 5, 2, 1, 6, 1),
    (9, 8, 3, 6, 5, 6, 2, 9, 3, 5),
    (6, 6, 5, 2, 8, 2, 4, 9, 5, 4),
    (3, 3, 1, 4, 3, 9, 4, 4, 8, 8),
    (6, 8, 7, 3, 6, 3, 4, 1, 8, 3),
    (6, 4, 1, 1, 9, 6, 6, 8, 2, 2),
    (1, 3, 8, 1, 5, 2, 7, 8, 5, 7),
    (5, 5, 3, 9, 3, 2, 2, 3, 2, 3),
    (5, 3, 9, 5, 5, 2, 8, 7, 8, 1),
    (2, 1, 5, 2, 1, 9, 3, 1, 9, 2),
    (1, 7, 2, 3, 7, 9, 5, 1, 5, 6),
    (5, 2, 4, 9, 3, 4, 3, 1, 4, 7),
    (3, 3, 3, 2, 4, 4, 6, 8, 2, 2),
    (4, 2, 9, 8, 3, 9, 6, 5, 4, 1),
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


@dataclass(frozen=True)
class _PlanMove:
    mask: int
    match: Match
    counts: tuple[int, ...]
    size: int


@dataclass(frozen=True)
class _PlanState:
    moves: tuple[_PlanMove, ...]
    counts: tuple[int, ...]
    occupied: int
    covered: int
    isolated_high: int
    fragile_high: int
    shortage: float


def _inventory_shortage(counts: tuple[int, ...]) -> float:
    """Soft heuristic only: allocate scarce complements to 9, 8, 7, then 6.

    Includes multi-cell complements (8=1+1, 7=2+1, etc.). Geometry may
    prevent these combinations; this is not a proof of solvability/death.
    Surplus small digits are not penalized just for outnumbering big ones.
    """
    stock = list(counts)
    cost = 0.0
    recipes = {
        9: ((1,),),
        8: ((2,), (1, 1)),
        7: ((3,), (2, 1), (1, 1, 1)),
        6: ((4,), (3, 1), (2, 2), (2, 1, 1), (1, 1, 1, 1)),
    }
    for high, weight in ((9, 2.6), (8, 2.0), (7, 1.6), (6, 1.0)):
        needed = stock[high]
        for recipe in recipes[high]:
            demand = {digit: recipe.count(digit) for digit in set(recipe)}
            supplied = min([needed] + [stock[d] // n for d, n in demand.items()])
            for digit, count in demand.items():
                stock[digit] -= supplied * count
            needed -= supplied
        cost += weight * needed
    return cost


class _PlanSearch:
    """Fixed-value board: the occupied-cell bitmask uniquely identifies a state."""

    def __init__(self, grid: list[list[int | None]], max_size: int,
                 cache_size: int = 16000) -> None:
        from functools import lru_cache

        self.rows, self.cols = len(grid), len(grid[0])
        self.values = tuple(value or 0 for row in grid for value in row)
        self.initial = sum(1 << i for i, value in enumerate(self.values) if value)
        self.digit_masks = tuple(
            sum(1 << i for i, value in enumerate(self.values) if value == digit)
            for digit in range(10)
        )
        self.high_mask = self.digit_masks[7] | self.digit_masks[8] | self.digit_masks[9]
        # Precompute every permitted rectangle; one vectorized prefix-sum
        # lookup replaces hundreds of separate numpy operations per state.
        rectangles = []
        for height in range(1, min(max_size, self.rows) + 1):
            for width in range(1, min(max_size, self.cols) + 1):
                for top in range(self.rows - height + 1):
                    for left in range(self.cols - width + 1):
                        rectangles.append((top, left, top + height, left + width))
        self.bounds = np.asarray(rectangles, dtype=np.intp).T
        self.rectangle_masks = tuple(
            sum(((1 << (right - left)) - 1) << (row * self.cols + left)
                for row in range(top, bottom))
            for top, left, bottom, right in rectangles
        )
        self.analyze = lru_cache(maxsize=cache_size)(self._analyze)
        self.inventory = lru_cache(maxsize=4096)(_inventory_shortage)

    def counts(self, mask: int) -> tuple[int, ...]:
        return tuple((mask & digit_mask).bit_count() for digit_mask in self.digit_masks)

    def _analyze(self, mask: int) -> _PlanState:
        values = np.fromiter(
            (value if mask & (1 << i) else 0 for i, value in enumerate(self.values)),
            dtype=np.int16, count=len(self.values),
        ).reshape(self.rows, self.cols)
        prefix = np.pad(values.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
        top, left, bottom, right = self.bounds
        sums = prefix[bottom, right] - prefix[top, right] - prefix[bottom, left] + prefix[top, left]
        unique: dict[int, _PlanMove] = {}
        covered = twice = 0
        for index in np.flatnonzero(sums == 10):
            removed = mask & self.rectangle_masks[int(index)]
            if removed in unique:
                continue
            # Shrink empty borders. Equivalent rectangles clear the same cells,
            # so search them once and keep a compact drag target.
            cells = []
            bits = removed
            while bits:
                bit = bits & -bits
                cells.append(divmod(bit.bit_length() - 1, self.cols))
                bits ^= bit
            r1, r2 = min(r for r, _ in cells), max(r for r, _ in cells)
            c1, c2 = min(c for _, c in cells), max(c for _, c in cells)
            move = _PlanMove(removed, Match((r1, c1), (r2, c2), max(r2-r1, c2-c1)),
                             self.counts(removed), len(cells))
            unique[removed] = move
            twice |= covered & removed
            covered |= removed
        counts = self.counts(mask)
        return _PlanState(tuple(unique.values()), counts, mask.bit_count(), covered.bit_count(),
                          (mask & self.high_mask & ~covered).bit_count(),
                          (mask & self.high_mask & covered & ~twice).bit_count(),
                          self.inventory(counts))

    def rank_moves(self, state: _PlanState, policy: int = 0) -> list[_PlanMove]:
        def score(move: _PlanMove) -> tuple[float, int, int]:
            remaining = tuple(a-b for a, b in zip(state.counts, move.counts))
            shortage_change = self.inventory(remaining) - state.shortage
            high = move.counts[7] + 1.3*move.counts[8] + 1.6*move.counts[9]
            # policy 0: balanced; 1: total-clear; 2: protect complements for
            # remaining 7/8/9.  All three still compete on final cells cleared.
            if policy == 0:
                priority = 0.40 * move.size + 1.40 * high - 2.20 * shortage_change
            elif policy == 1:
                priority = 1.00 * move.size + 0.15 * high - 0.35 * shortage_change
            else:
                priority = 0.12 * move.size + 2.35 * high - 5.00 * shortage_change
            area = ((move.match.second[0]-move.match.first[0]+1)
                    * (move.match.second[1]-move.match.first[1]+1))
            return priority, -area, -move.mask
        return sorted(state.moves, key=score, reverse=True)

    def evaluate(self, mask: int, state: _PlanState, policy: int = 0) -> float:
        cleared = self.initial.bit_count() - state.occupied
        high_cleared = (self.initial & self.high_mask).bit_count() - sum(state.counts[7:10])
        # Uncovered != permanently dead: penalize only softly. Counts of
        # currently legal moves are secondary to distinct covered cells.
        if policy == 1:
            return cleared + 0.32*state.covered - 0.35*state.shortage
        if policy == 2:
            return (cleared + 0.35*state.covered + 1.20*high_cleared
                    - 3.10*state.shortage - 0.42*state.isolated_high
                    - 0.12*state.fragile_high)
        return (cleared + 0.60*state.covered + 0.65*high_cleared
                - 1.7*state.shortage - 0.18*state.isolated_high
                - 0.06*state.fragile_high + 0.015*min(len(state.moves), 80))


def build_optimized_plan(
    grid: list[list[int | None]],
    max_size: int,
    max_actions: int,
    rollouts: int,
    candidate_pool: int,
    *,
    planning_seconds: float = 20.0,
    beam_width: int = 24,
    verbose: bool = True,
) -> list[Match]:
    """Anytime portfolio: resource-aware rollouts, diverse beam, exact endgame.

    Heuristic weights are engineering defaults, not fitted empirical truths.
    Only actual cleared-cell count selects the returned route; high-digit
    removal breaks ties. Time limits/beam pruning prevent a global guarantee.
    The input board and all screenshot/OCR/mouse functions remain untouched.
    """
    import math
    import random

    if max_size < 1 or max_actions < 0 or rollouts < 1 or candidate_pool < 1 or beam_width < 1:
        raise ValueError("Invalid search budget: positive sizes/widths, nonnegative max_actions required")
    if not math.isfinite(planning_seconds) or planning_seconds <= 0:
        raise ValueError("planning_seconds must be finite and positive")
    if not grid or not grid[0]:
        return []
    if any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("Grid must be rectangular")
    if any(value is not None and (type(value) is not int or not 1 <= value <= 9)
           for row in grid for value in row):
        raise ValueError("Grid cells must be integers 1..9 or null/None")
    if max_actions == 0 or not any(value is not None for row in grid for value in row):
        return []
    started = time.perf_counter()
    # A node budget makes the route sequence independent of incidental timing
    # (terminal output, CPU scheduling and cache warm-up).
    state_budget = max(4_000, int(planning_seconds * 4_000))
    engine = _PlanSearch(grid, max_size)
    initial_count = engine.initial.bit_count()
    max_actions = min(max_actions, sum(engine.values) // 10)
    rng = random.Random(20260916)
    best_plan: tuple[_PlanMove, ...] = ()
    best_key = (0, 0, 0, 0)
    last_report = started

    def out_of_budget(limit: int | None = None) -> bool:
        return engine.analyze.cache_info().misses >= (state_budget if limit is None else limit)

    def remember(mask: int, plan: tuple[_PlanMove, ...]) -> None:
        nonlocal best_plan, best_key
        state = engine.analyze(mask)
        key = (initial_count-mask.bit_count(),
               ((engine.initial ^ mask) & engine.high_mask).bit_count(),
               -round(state.shortage * 100), -state.isolated_high)
        if key > best_key:
            best_key, best_plan = key, plan

    def report(stage: str) -> None:
        nonlocal last_report
        now = time.perf_counter()
        if verbose and now-last_report >= 2:
            info = engine.analyze.cache_info()
            print(f"Planning... {stage}, {now-started:.1f}s, best={best_key[0]}/{initial_count}, "
                  f"states={info.misses}, cache hits={info.hits}", flush=True)
            last_report = now

    if verbose:
        print(f"Planning... fixed search budget {planning_seconds:g}; safe high-digit search", flush=True)

    # First obtain complete feasible routes. Reserve most time for beam search.
    rollout_state_limit = max(1_000, int(state_budget * 0.40))
    for trial in range(rollouts):
        if out_of_budget(rollout_state_limit) and trial >= 2:
            break
        mask, plan = engine.initial, ()
        policy = (0, 0, 1, 2, 2)[trial % 5]
        while len(plan) < max_actions and not out_of_budget():
            state = engine.analyze(mask)
            if not state.moves:
                break
            ranked = engine.rank_moves(state, policy)
            if trial < 2:
                move = ranked[0]
            else:
                if policy == 3:
                    pool = ranked[:min(len(ranked), max(24, candidate_pool * 4))]
                    weights = [1 / ((index + 1) ** 0.35) for index in range(len(pool))]
                else:
                    pool = ranked[:candidate_pool]
                    weights = range(len(pool), 0, -1)
                move = rng.choices(pool, weights=weights, k=1)[0]
            mask ^= move.mask
            plan += (move,)
            remember(mask, plan)
            report("rollouts")
        if best_key[0] == initial_count or out_of_budget():
            break

    # Keep half the beam under each evaluation to reduce one-heuristic bias.
    frontier = [(engine.initial, ())]
    exact_cache: dict[tuple[int, int], tuple[_PlanMove, ...]] = {}
    exact_nodes = 0

    class _SearchLimit(Exception):
        pass

    def exact(mask: int, remaining: int) -> tuple[_PlanMove, ...]:
        nonlocal exact_nodes
        key = mask, remaining
        if key in exact_cache:
            return exact_cache[key]
        if out_of_budget() or exact_nodes >= 12000:
            raise _SearchLimit
        exact_nodes += 1
        result: tuple[_PlanMove, ...] = ()
        result_key = (0, 0)
        if remaining:
            for move in engine.rank_moves(engine.analyze(mask)):
                tail = exact(mask ^ move.mask, remaining-1)
                candidate = (move,) + tail
                cleared = sum(item.size for item in candidate)
                high = sum(sum(item.counts[7:10]) for item in candidate)
                if (cleared, high) > result_key:
                    result, result_key = candidate, (cleared, high)
                if cleared == mask.bit_count():
                    break
        # Cache only completed subproblems; interrupted results aren't exact.
        exact_cache[key] = result
        return result

    for depth in range(max_actions):
        if out_of_budget() or best_key[0] == initial_count or not frontier:
            break
        children: dict[int, tuple[_PlanMove, ...]] = {}
        for mask, plan in frontier:
            if out_of_budget():
                break
            state = engine.analyze(mask)
            if not state.moves:
                continue
            if state.occupied <= 16 and exact_nodes < 12000:
                try:
                    tail = exact(mask, max_actions-len(plan))
                    final_mask = mask
                    for move in tail:
                        final_mask ^= move.mask
                    remember(final_mask, plan+tail)
                    continue
                except _SearchLimit:
                    pass
            branch = max(4, candidate_pool)
            if state.occupied <= 32:
                branch *= 2
            ranked = engine.rank_moves(state)
            selected = {move.mask: move for move in ranked[:branch]}
            for move in engine.rank_moves(state, 1)[:max(2, branch//3)]:
                selected.setdefault(move.mask, move)
            for move in engine.rank_moves(state, 2)[:max(2, branch//3)]:
                selected.setdefault(move.mask, move)
            for move in selected.values():
                child = mask ^ move.mask
                if child not in children:
                    child_plan = plan + (move,)
                    children[child] = child_plan
                    remember(child, child_plan)
            report(f"beam depth={depth+1}")
        candidates = []
        for mask, plan in children.items():
            if out_of_budget():
                break
            state = engine.analyze(mask)
            if state.moves and len(plan) < max_actions:
                candidates.append((
                    mask, plan,
                    engine.evaluate(mask, state),
                    engine.evaluate(mask, state, 1),
                    engine.evaluate(mask, state, 2),
                ))
        width = beam_width * (2 if candidates and min(item[0].bit_count() for item in candidates) <= 32 else 1)
        chosen = {}
        # The main beam retains V3's balance: half resource-aware, then
        # score-first and complement-protection paths.
        for slot, target in ((2, (width + 1)//2), (3, (3*width + 3)//4), (4, width)):
            for item in sorted(candidates, key=lambda item: item[slot], reverse=True):
                chosen.setdefault(item[0], (item[0], item[1]))
                if len(chosen) >= target:
                    break
        frontier = list(chosen.values())

    # The main V3 beam may exhaust its preferred branches well before the time
    # budget. Use only that otherwise-idle time for broad portfolio rollouts.
    # Because remember() accepts strictly better actual clears first, these
    # routes cannot replace the V3 result merely for looking different.
    escape_attempts = 0
    while not out_of_budget() and best_key[0] < initial_count:
        mask, plan = engine.initial, ()
        while len(plan) < max_actions and not out_of_budget():
            state = engine.analyze(mask)
            if not state.moves:
                break
            ranked = engine.rank_moves(state, 0)
            # Include moves well outside the regular candidate pool. The mild
            # decay still avoids turning every rescue route into pure noise.
            pool = ranked[:min(len(ranked), max(30, candidate_pool * 5))]
            weights = [1 / ((index + 1) ** 0.30) for index in range(len(pool))]
            move = rng.choices(pool, weights=weights, k=1)[0]
            mask ^= move.mask
            plan += (move,)
            remember(mask, plan)
        escape_attempts += 1
        report(f"portfolio escape={escape_attempts}")

    # Independent replay: no stale-mask or illegal rectangle can reach mouse control.
    result = [move.match for move in best_plan]
    validate_plan(grid, result, max_size, max_actions)
    if verbose:
        info = engine.analyze.cache_info()
        print(f"Plan ready: {len(result)} actions, {best_key[0]}/{initial_count} cells; "
              f"{time.perf_counter()-started:.2f}s, {info.misses} states, {info.hits} cache hits.", flush=True)
    return result


def validate_plan(grid: list[list[int | None]], plan: list[Match],
                  max_size: int, max_actions: int) -> list[list[int | None]]:
    """Replay using original rectangle semantics; raise before any mouse action."""
    state = [row[:] for row in grid]
    if len(plan) > max_actions:
        raise ValueError("Plan exceeds max_actions")
    for index, move in enumerate(plan, 1):
        r1, c1 = move.first
        r2, c2 = move.second
        if not (0 <= r1 <= r2 < len(state) and 0 <= c1 <= c2 < len(state[0])):
            raise ValueError(f"Invalid rectangle in action {index}")
        if max(r2-r1+1, c2-c1+1) > max_size:
            raise ValueError(f"Oversized rectangle in action {index}")
        total = sum(state[r][c] or 0 for r in range(r1, r2+1) for c in range(c1, c2+1))
        if total != 10:
            raise ValueError(f"Action {index} sums to {total}, not 10")
        state = clear_box(state, move)
    return state


def _run_offline(args: argparse.Namespace) -> None:
    payload = json.loads(args.grid_json.read_text(encoding="utf-8-sig"))
    grid = payload["grid"] if isinstance(payload, dict) else payload
    plan = build_optimized_plan(grid, args.max_box_size, args.max_actions, args.rollouts,
                                args.candidate_pool, planning_seconds=args.planning_seconds,
                                beam_width=args.beam_width)
    remaining = validate_plan(grid, plan, args.max_box_size, args.max_actions)
    cleared = sum(value is not None for row in grid for value in row) - sum(
        value is not None for row in remaining for value in row)
    result = {"cleared": cleared, "remaining_grid": remaining,
              "actions": [{"first": move.first, "second": move.second} for move in plan]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.plan_json:
        args.plan_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")



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
    parser.add_argument("--auto-start", action="store_true", help="框选后跳过复核、确认和倒计时，直接执行路线")
    parser.add_argument("--min-score", type=int, help="预测清除格数低于此值时自动退出，不控制鼠标")
    parser.add_argument("--min-ocr-confidence", type=float, default=88.0,
                        help="自动模式接受的最低数字识别置信度，默认 88")
    parser.add_argument("--max-digit-count", type=int, default=32,
                        help="同一个数字允许出现的最大次数，默认 32；用于拦截读成一片 5 的错误")
    parser.add_argument("--select-region", action="store_true", help="重新框选并保存棋盘位置")
    parser.add_argument("--diagonal", action="store_true", help="允许处理对角线方向")
    parser.add_argument("--margin", type=float, default=0.18, help="数字裁剪边距比例，默认 0.18")
    parser.add_argument("--box-inset", type=float, default=0.05, help="鼠标框选向内缩进比例，默认 0.05")
    parser.add_argument("--delay", type=float, default=0.08, help="每次消除后的等待秒数，默认 0.08")
    parser.add_argument("--drag-duration", type=float, default=0.08, help="鼠标拖框持续秒数，默认 0.08")
    parser.add_argument("--max-box-size", type=int, default=10, help="矩形最大边长，默认 10")
    parser.add_argument("--rollouts", type=int, default=100, help="最多模拟的完整路线数，默认 100，受计算时限约束")
    parser.add_argument("--candidate-pool", type=int, default=6, help="每步优先候选数量，默认 6")
    parser.add_argument("--countdown", type=int, default=3, help="开始控制鼠标前的倒计时秒数，默认 3")
    parser.add_argument("--max-actions", type=int, default=999, help="最多执行的消除次数，默认 999")
    parser.add_argument("--planning-seconds", type=float, default=5.0, help="搜索强度，默认 5；每单位约分析 4,000 个棋盘状态")
    parser.add_argument("--beam-width", type=int, default=24, help="保留的搜索分支数量，默认 24，残局加倍")
    parser.add_argument("--grid-json", type=Path, help="从 JSON 棋盘离线求解，不截图、不控制鼠标")
    parser.add_argument("--plan-json", type=Path, help="离线模式的路线输出文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0 <= args.margin < 0.45):
        raise SystemExit("--margin must be between 0 and 0.45")
    if not (0 <= args.box_inset < 0.5):
        raise SystemExit("--box-inset must be between 0 and 0.5")
    if args.rollouts < 1 or args.candidate_pool < 1:
        raise SystemExit("--rollouts and --candidate-pool must be positive")
    import math
    if args.max_box_size < 1 or args.max_actions < 0 or args.beam_width < 1:
        raise SystemExit("Invalid max-box-size, max-actions or beam-width")
    if args.min_score is not None and not (0 <= args.min_score <= args.rows * args.cols):
        raise SystemExit("--min-score must be between 0 and rows * cols")
    if not (0 <= args.min_ocr_confidence <= 100):
        raise SystemExit("--min-ocr-confidence must be between 0 and 100")
    if not (1 <= args.max_digit_count <= args.rows * args.cols):
        raise SystemExit("--max-digit-count must be between 1 and rows * cols")
    if not math.isfinite(args.planning_seconds) or args.planning_seconds <= 0:
        raise SystemExit("--planning-seconds must be finite and positive")
    if args.grid_json:
        _run_offline(args)
        return
    if args.plan_json:
        raise SystemExit("--plan-json requires --grid-json")
    make_dpi_aware()
    pyautogui.FAILSAFE = True  # 将鼠标移到屏幕左上角可紧急停止。
    region_path = Path(__file__).with_name("grid_region.json")
    region = None if args.select_region else load_region(region_path)
    if region is None:
        print("Select the grid once; its location will be reused on later runs.")
        region = select_region()
    else:
        print(f"Reusing saved grid region: {region}. Use --select-region to change it.")

    actions = 0
    print(f"Region: {region}; grid: {args.rows} rows x {args.cols} columns")
    selected_image = capture(region)
    cv2.imwrite(str(Path(__file__).with_name("last_capture.png")), selected_image)
    first_image, region = calibrate_grid(selected_image, region, args.rows, args.cols)
    save_region(region_path, region)
    cv2.imwrite(str(Path(__file__).with_name("last_grid.png")), first_image)
    if not args.auto_start and args.min_score is None:
        show_capture_review(selected_image, first_image)
    first_grid, confidence_grid = recognize_grid(first_image, args.rows, args.cols, args.margin)
    print("\nFirst recognition result (16 rows x 10 columns):\n" + format_array(first_grid))
    Path(__file__).with_name("last_grid.json").write_text(
        json.dumps(first_grid, ensure_ascii=False, indent=2), encoding="utf-8")
    digit_counts = {digit: sum(value == digit for row in first_grid for value in row) for digit in range(1, 10)}
    minimum_confidence = min(confidence for row in confidence_grid for confidence in row)
    Path(__file__).with_name("last_recognition.json").write_text(
        json.dumps(
            {"grid": first_grid, "confidence": confidence_grid, "digit_counts": digit_counts},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Recognition check: lowest confidence {minimum_confidence:.1f}; digit counts {digit_counts}")
    missing = sum(value is None for row in first_grid for value in row)
    if missing:
        print(f"WARNING: {missing} cells were not recognized. The initial board is expected to be full.")
    recognition_warnings: list[str] = []
    if minimum_confidence < args.min_ocr_confidence:
        recognition_warnings.append(
            f"lowest confidence {minimum_confidence:.1f} is below {args.min_ocr_confidence:.1f}"
        )
    overloaded = [str(digit) for digit, count in digit_counts.items() if count > args.max_digit_count]
    if overloaded:
        recognition_warnings.append(
            "unusually many reads of digit(s) " + ", ".join(overloaded)
        )
    if recognition_warnings:
        print("WARNING: recognition looks unreliable: " + "; ".join(recognition_warnings))
        if args.min_score is not None or args.auto_start:
            print("No planning or mouse action was performed. Re-run with --select-region after checking the board.")
            return
    if args.preview:
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
        planning_seconds=args.planning_seconds,
        beam_width=args.beam_width,
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
    if args.min_score is not None:
        print(f"Predicted score: {expected_cleared}; target: {args.min_score}.")
        if expected_cleared < args.min_score:
            print("Below target. No mouse action was performed.")
            return
    if args.auto_start:
        print("Auto-start enabled: executing the planned route now.")
    else:
        answer = input("\nIs this recognition correct? Type y to execute this plan: ").strip().lower()
        if answer != "y":
            print("Cancelled. No mouse action was performed.")
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
