#!/usr/bin/env python3
import argparse
import random
import subprocess
import sys
import time
import os
from pathlib import Path
# from tkinter import ANCHOR

import cv2
import numpy as np
import pyautogui
from PIL import Image
from pyscreeze import Box
from pyscreeze import _load_cv2 as pyscreeze_load_cv2

# Here be magic numbers!
# ---- CONFIG ----
ANCHOR_IMAGE_PATH = "menu_anchor.png"
# Vertical offset for the anchor match, as a fraction of the anchor image height.
# 0.0 = very top of the anchor, 0.5 = halfway down, 1.0 = bottom, etc.
ANCHOR_OFFSET_X = 4.3 #控制锚点偏移的X轴 - 如果代码看不到按钮，微调这个数值。我测试的过程中 4.0-4.4 好像都行。
ANCHOR_OFFSET_Y = 16.3 #控制锚点偏移的Y轴 - 如果代码看不到按钮，微调这个数值。我测试的过程中 15-17 好像都行。
# Haystack rectangle relative to the TOP-LEFT of the matched anchor image (after applying ANCHOR_OFFSET_Y).
# dx = (haystack_left - anchor_left), dy = (haystack_top - anchor_top), then width × height.
SEARCH_REGION = (200, 1)  # width, height 截图区宽度和高度度，单位：像素
CONFIDENCE = 0.85 # 匹配的置信度，0.0-1.0，越高越精确，但也会越慢。建议别碰
ANCHOR_CONFIDENCE = 0.8 # 锚点匹配的置信度，0.0-1.0，越高越精确，但也会越慢。建议别碰
ANCHOR_FIND_TIMEOUT = 15 # 锚点匹配的超时时间，单位：秒 15秒的时间要是找不到锚点基本上可以判断为代码别的地方出问题了。
ANCHOR_REFRESH_EVERY = 1.0 # 锚点匹配的刷新时间，单位：秒 1秒刷新一次，要是锚点匹配失败，会自动刷新。

IGNORE_RGB = (140,165,69) # 提升至xx等级的盒子的颜色

CLICKS = 5
BUTTON = "left"
RETRY_EVERY = 0.1 # 程序刷新速度，单位：秒 0.1秒刷新一次，要是找不到按钮，会自动刷新。注意，这不是真的速度，是检查间隔。每次运行之间的间隔。不要设定的太小，不然程序要是坚信一个莫名其妙的像素是个按钮并且疯狂点击，你就抢不回你鼠标的控制权了。。。

TIMEOUT = 6 * 60 * 60 # 最高连续运行时间。6小时。
MOVE_TO_BEFORE_CLICK = True # 鼠标移动到目标位置再点击，要是为False，会直接点击。改成False的话，程序会直接点击，不会移动鼠标。极大增加封号概率，不过会让你点的很快。
MOVE_DURATION_SEC = (0.1, 0.15)  # 鼠标移动到目标位置的时间，单位：秒 0.1-0.15秒随机，要是为0，会直接点击。
MOVE_TWEEN = pyautogui.easeInOutQuad # 别碰

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05


def move_mouse_to_target(x: float, y: float) -> None:
    if not MOVE_TO_BEFORE_CLICK:
        return
    dur = random.uniform(MOVE_DURATION_SEC[0], MOVE_DURATION_SEC[1])
    pyautogui.moveTo(x, y, duration=dur, tween=MOVE_TWEEN)


def load_needle_base(path: Path) -> Image.Image:
    im = Image.open(path)
    if im.mode in ("RGBA", "LA"):
        rgba = im.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        im = bg
    else:
        im = im.convert("RGB")
    return im

def _match_template_max(hay: np.ndarray, tpl: np.ndarray) -> tuple[float, tuple[int, int]]:
    res = cv2.matchTemplate(hay, tpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return max_val, max_loc


def locate_needle_multiscale(
    base_needle: Image.Image,
    haystack: Image.Image,
    scales: tuple[float, ...],
    confidence: float,
) -> tuple[Box | None, float]:
    """Best TM_CCOEFF_NORMED across scales; grayscale + BGR (Preview/color mgmt can favor one)."""
    hay_g = pyscreeze_load_cv2(haystack, grayscale=True)
    hay_c = pyscreeze_load_cv2(haystack, grayscale=False)
    hw, hh = haystack.size
    best_val = -1.0
    best_box: Box | None = None
    for s in scales:
        w, h = base_needle.size
        nw = max(1, int(round(w * s)))
        nh = max(1, int(round(h * s)))
        if nw > hw or nh > hh:
            continue
        needle = (
            base_needle
            if abs(s - 1.0) < 1e-6 and nw == w and nh == h
            else base_needle.resize((nw, nh), Image.Resampling.LANCZOS)
        )
        n_g = pyscreeze_load_cv2(needle, grayscale=True)
        n_c = pyscreeze_load_cv2(needle, grayscale=False)
        g_val, g_loc = _match_template_max(hay_g, n_g)
        c_val, c_loc = _match_template_max(hay_c, n_c)
        if c_val > g_val:
            max_val, max_loc = c_val, c_loc
        else:
            max_val, max_loc = g_val, g_loc
        if max_val > best_val:
            best_val = max_val
            x, y = max_loc
            best_box = Box(x, y, nw, nh)
    if best_box is not None and best_val >= confidence:
        return best_box, best_val
    return None, best_val


def haystack_screen_rect(anchor_box: Box) -> tuple[int, int, int, int]:
    """Absolute screen rect (left, top, w, h) from anchor top-left + SEARCH_REGION dx,dy."""
    w, h = SEARCH_REGION
    left = int(round(anchor_box.left/2))
    top = int(round(anchor_box.top/2))
    return left, top, w, h


def preview_search_region(out: Path) -> None:
    anchor_image = Path(ANCHOR_IMAGE_PATH)
    if not anchor_image.exists():
        print(f"Anchor image not found: {anchor_image}")
        sys.exit(1)
    anchor_box = locate_anchor(anchor_image, ANCHOR_FIND_TIMEOUT)
    left, top, width, height = haystack_screen_rect(anchor_box)
    shot = pyautogui.screenshot(region=(left, top, width, height))
    shot.save(out)
    print(
        f"Anchor TL ({anchor_box.left}, {anchor_box.top}); "
        f"haystack left={left} top={top} width={width} height={height} "
        f"(right={left + width} bottom={top + height})"
    )
    print(f"Saved snapshot: {out.resolve()}")
    if sys.platform == "darwin":
        subprocess.run(["open", str(out)], check=False)


def capture_live_haystack(out_hay: Path) -> None:
    """Capture the exact anchor-relative search region used at runtime."""
    anchor_image = Path(ANCHOR_IMAGE_PATH)
    if not anchor_image.exists():
        print(f"Anchor image not found: {anchor_image}")
        sys.exit(1)
    anchor_box = locate_anchor(anchor_image, ANCHOR_FIND_TIMEOUT)
    left, top, width, height = haystack_screen_rect(anchor_box)
    print(f"Left: {left}, Top: {top}, Width: {width}, Height: {height}")
    haystack = pyautogui.screenshot(region=(left, top, width, height))
    haystack.save(out_hay)
    print(
        f"Anchor TL ({anchor_box.left}, {anchor_box.top}); "
        f"haystack = ({left}, {top}, {width}, {height})"
    )
    print(f"Saved live haystack: {out_hay.resolve()}")
    if sys.platform == "darwin":
        subprocess.run(["open", str(out_hay)], check=False)


def locate_anchor(anchor_image: Path, timeout_s: float) -> Box:
    start = time.time()
    while True:
        try:
            box = pyautogui.locateOnScreen(str(anchor_image), confidence=ANCHOR_CONFIDENCE)
            # print(f"Anchor found at: {box} x{box.left/2} y{box.top}")
        except pyautogui.ImageNotFoundException:
            box = None
        if box:
            # apply a tunable vertical offset within the anchor image
            box = Box(
                box.left + int(box.width * ANCHOR_OFFSET_X),
                box.top + int(box.height * ANCHOR_OFFSET_Y),
                box.width,
                box.height,
            )
            return box
        if time.time() - start > timeout_s:
            raise RuntimeError(
                f"Could not find anchor image '{anchor_image}' within {timeout_s:.0f}s "
                f"(confidence {ANCHOR_CONFIDENCE})."
            )
        time.sleep(0.2)

def tune_offset_y() -> None:
    """Interactively try different ANCHOR_OFFSET_Y values and capture screenshots."""
    anchor_image = Path(ANCHOR_IMAGE_PATH)
    if not anchor_image.exists():
        print(f"Anchor image not found: {anchor_image}")
        sys.exit(1)

    print(f"Locating anchor for tuning: {anchor_image} ...")
    base_anchor = locate_anchor(anchor_image, ANCHOR_FIND_TIMEOUT)
    print(f"Base anchor box: {base_anchor}")
    print("Enter a vertical offset fraction (e.g. 0.0, 0.25, 0.5). 'q' to quit.")

    while True:
        raw = input(f"OFFSET_Y fraction (10 <-> -10, 'q' to quit). Current OFFSET_Y: {ANCHOR_OFFSET_Y} ").strip()
        if raw.lower() in ("q", "quit", "exit"):
            break
        try:
            offset_y = float(raw)
        except ValueError:
            print("Please enter a number like 0.25, or 'q' to quit.")
            continue

        temp_box = Box(
            base_anchor.left,
            base_anchor.top + int(base_anchor.height * offset_y),
            base_anchor.width,
            base_anchor.height,
        )
        left, top, w, h = haystack_screen_rect(temp_box)
        h = 30
        print(f"Using OFFSET_Y={ANCHOR_OFFSET_Y:.3f}, rect=({left}, {top}, {w}, {h})")
        shot = pyautogui.screenshot(region=(left, top, w, h))
        out = Path(f"tune_offset_y_{ANCHOR_OFFSET_Y:.3f}.png")
        shot.save(out)
        print(f"Saved {out.resolve()}")
        if sys.platform == "darwin":
            subprocess.run(["open", str(out)], check=False)


def main():
    anchor_image = Path(ANCHOR_IMAGE_PATH)
    if not anchor_image.exists():
        print(f"Anchor image not found: {anchor_image}")
        sys.exit(1)

    width, height = SEARCH_REGION
    print(f"Locating anchor: {anchor_image} ...")
    anchor_box = locate_anchor(anchor_image, ANCHOR_FIND_TIMEOUT)
    left, top, _, _ = haystack_screen_rect(anchor_box)
    print(
        f"Anchor TL ({anchor_box.left}, {anchor_box.top}); "
        
        f"haystack screen TL=({left}, {top}) size {width}x{height}"
    )
    last_anchor_refresh = 0.0
    start = time.time()

    while True:
        print(f"Searching for hand... {time.strftime('%Y-%m-%d %H:%M:%S')}")
        now = time.time()
        if now - last_anchor_refresh >= ANCHOR_REFRESH_EVERY:
            try:
                anchor_box = locate_anchor(anchor_image, 1.2)
                left, top, _, _ = haystack_screen_rect(anchor_box)
            except RuntimeError:
                pass
            last_anchor_refresh = now

        haystack = pyautogui.screenshot(region=(left, top, width, height))

        # Debug: print the color of each pixel across this 1-pixel-high scanline.
        # if height == 1:
            # row_colors = [haystack.getpixel((x, 0)) for x in range(width)]
            # print(f"Scanline colors ({width} px): {row_colors}")

        # Look for any pixel whose RGB is OUTSIDE the specified ranges:
        #   R ∈ [170, 190], G ∈ [170, 190], B ∈ [70, 90]
        click_pos: tuple[int, int] | None = None

        screen_blocked = False
        # if the first 5 pixels are all not in range, set screen_blocked to True
        for x in range(5):
            y = 0 if height == 1 else height // 2
            pixel = haystack.getpixel((x, y))
            if isinstance(pixel, tuple):
                r, g, b = pixel[:3]
            else:
                r = g = b = int(pixel)
            in_range = (130 <= r <= 255) and (130 <= g <= 255) and (70 <= b <= 255)
            if not in_range:
                screen_blocked = True
                break

        for x in range(width):
            y = 0 if height == 1 else height // 2
            pixel = haystack.getpixel((x, y))
            # Handle RGB or RGBA; drop alpha if present.
            if isinstance(pixel, tuple):
                r, g, b = pixel[:3]
            else:
                # Fallback, though Pillow should always return a tuple here.
                r = g = b = int(pixel)
            in_range = (110 <= r <= 255) and (130 <= g <= 255) and (70 <= b <= 255)
            if not in_range and (r, g, b) != IGNORE_RGB:
                screen_x = left + x + random.randint(0, 10)
                screen_y = top + y + random.randint(0, 10)
                click_pos = (screen_x, screen_y)
                print(f"Found out-of-range pixel at x={x}, color=({r},{g},{b}); screen=({screen_x},{screen_y})")
                break

        if click_pos is not None and not screen_blocked:
            os.system('afplay /System/Library/Sounds/Glass.aiff &')
            screen_x, screen_y = click_pos
            # click 1-3 times randomly (ease mouse to target on first click only)
            n_clicks = random.randint(1, CLICKS)
            for i in range(n_clicks):
                if i == 0:
                    move_mouse_to_target(screen_x, screen_y)
                pyautogui.click(screen_x, screen_y, 1, 0, button=BUTTON)
                time.sleep(random.uniform(0.01, 0.2))
            # return

        if time.time() - start > TIMEOUT:
            print("Timed out without finding the icon.")
            sys.exit(2)

        # time.sleep(RETRY_EVERY)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Click when a template appears in SEARCH_REGION.")
    parser.add_argument(
        "--preview",
        metavar="PNG",
        nargs="?",
        const="search_region_preview.png",
        help="Save a screenshot of SEARCH_REGION to this file (default: search_region_preview.png) and exit.",
    )
    parser.add_argument(
        "--debug-live-haystack",
        metavar="PNG",
        nargs="?",
        const="debug_live_haystack.png",
        help="Find anchor, capture the live anchor-relative haystack image, and exit.",
    )
    parser.add_argument(
        "--tune-offset-y",
        action="store_true",
        help="Interactively try OFFSET_Y values (fraction of anchor height) and capture screenshots.",
    )
    args = parser.parse_args()
    if args.tune_offset_y:
        tune_offset_y()
    elif args.debug_live_haystack is not None:
        capture_live_haystack(Path(args.debug_live_haystack))
    elif args.preview is not None:
        preview_search_region(Path(args.preview))
    else:
        main()