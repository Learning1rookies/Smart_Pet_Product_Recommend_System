from __future__ import annotations

import argparse
import io
import json
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageChops
from playwright.sync_api import Page, sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = PROJECT_ROOT / "assets"
FULL_FLOW_QUERY = "我要购买智能宠物喂食器"
QUICK_FLOW_QUERY = "我想买智能宠物摄像头，预算200元以内，关注画质/夜视和稳定/卡顿，没有特别避免。"
MEMORY_QUERY = "我上一次购买的产品是什么？"
EDGE_CANDIDATES = (
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
)


def find_edge() -> Path:
    for candidate in EDGE_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Microsoft Edge was not found in a standard installation directory.")


class GifRecorder:
    def __init__(self, page: Page, frame_dir: Path):
        self.page = page
        self.frame_dir = frame_dir
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        for old_frame in self.frame_dir.glob("frame-*.png"):
            old_frame.unlink()
        self.duration_path = self.frame_dir / "durations.json"
        self.duration_path.unlink(missing_ok=True)
        self.frame_paths: list[Path] = []
        self.durations: list[int] = []

    def capture(self, duration_ms: int = 650) -> None:
        image = Image.open(io.BytesIO(self.page.screenshot())).convert("RGB")
        if self.frame_paths:
            with Image.open(self.frame_paths[-1]) as previous:
                if ImageChops.difference(previous.convert("RGB"), image).getbbox() is None:
                    self.durations[-1] += duration_ms
                    self.duration_path.write_text(json.dumps(self.durations), encoding="utf-8")
                    return
        frame_path = self.frame_dir / f"frame-{len(self.frame_paths):03d}.png"
        image.save(frame_path, format="PNG", optimize=True)
        self.frame_paths.append(frame_path)
        self.durations.append(duration_ms)
        self.duration_path.write_text(json.dumps(self.durations), encoding="utf-8")

    def save(self, output: Path) -> None:
        if not self.frame_paths:
            raise RuntimeError("No demo frames were captured.")
        output.parent.mkdir(parents=True, exist_ok=True)
        palette_frames: list[Image.Image] = []
        try:
            for frame_path in self.frame_paths:
                with Image.open(frame_path) as frame:
                    palette_frames.append(
                        frame.convert("RGB").quantize(
                            colors=96,
                            method=Image.Quantize.MEDIANCUT,
                            dither=Image.Dither.FLOYDSTEINBERG,
                        )
                    )
            palette_frames[0].save(
                output,
                save_all=True,
                append_images=palette_frames[1:],
                duration=self.durations,
                loop=0,
                optimize=True,
                disposal=2,
            )
        finally:
            for frame in palette_frames:
                frame.close()

    def cleanup(self) -> None:
        for frame_path in self.frame_paths:
            frame_path.unlink(missing_ok=True)
        self.duration_path.unlink(missing_ok=True)
        try:
            self.frame_dir.rmdir()
        except OSError:
            pass


def prepare_page(page: Page, url: str) -> None:
    page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    page.locator("textarea").wait_for(state="visible", timeout=60_000)
    page.add_style_tag(
        content="""
        [data-testid="stHeader"], [data-testid="stToolbar"],
        #MainMenu, footer { display: none !important; }
        .stMainBlockContainer { max-width: 1080px !important; padding-top: 18px !important; }
        """
    )
    page.wait_for_timeout(800)


def fill_in_steps(page: Page, recorder: GifRecorder, text: str) -> None:
    textarea = page.locator("textarea")
    stops = sorted({max(1, round(len(text) * ratio)) for ratio in (0.22, 0.45, 0.7, 1.0)})
    for end in stops:
        textarea.fill(text[:end])
        recorder.capture(430)
    recorder.capture(850)


def wait_for_visible(
    page: Page,
    recorder: GifRecorder,
    locator: object,
    *,
    timeout_seconds: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    next_capture = 0.0
    while time.monotonic() < deadline:
        if locator.count() and locator.first.is_visible():
            return
        now = time.monotonic()
        if now >= next_capture:
            recorder.capture(520)
            next_capture = now + 2.5
        page.wait_for_timeout(300)
    raise TimeoutError("The expected Streamlit control did not become visible in time.")


def select_option(page: Page, recorder: GifRecorder, index: int = 0) -> str:
    select = page.locator('[data-baseweb="select"]').last
    select.click()
    page.wait_for_timeout(200)
    for _ in range(index + 1):
        page.keyboard.press("ArrowDown")
    page.keyboard.press("Enter")
    page.wait_for_timeout(450)
    recorder.capture(1_100)
    return "selected"


def select_named_option(page: Page, recorder: GifRecorder, label: str) -> None:
    select = page.locator('[data-baseweb="select"]').last
    select.click()
    option = page.locator('[role="option"]').filter(has_text=label).first
    option.wait_for(state="visible", timeout=5_000)
    option.click(timeout=5_000)
    page.wait_for_timeout(450)
    recorder.capture(1_100)


def record_full_flow(page: Page, query: str, timeout_seconds: int, frame_dir: Path) -> GifRecorder:
    recorder = GifRecorder(page, frame_dir)
    textarea = page.locator("textarea")

    recorder.capture(1_300)
    fill_in_steps(page, recorder, query)
    textarea.press("Enter")

    budget_button = page.get_by_role("button", name="确认预算", exact=True)
    wait_for_visible(page, recorder, budget_button, timeout_seconds=timeout_seconds)
    recorder.capture(1_300)
    select_option(page, recorder, index=1)
    budget_button.click()

    priority_button = page.get_by_role("button", name="确认关注点", exact=True)
    wait_for_visible(page, recorder, priority_button, timeout_seconds=timeout_seconds)
    recorder.capture(1_300)
    select_option(page, recorder, index=0)
    select_option(page, recorder, index=0)
    priority_button.click()

    avoid_button = page.get_by_role("button", name="确认避免项", exact=True)
    wait_for_visible(page, recorder, avoid_button, timeout_seconds=timeout_seconds)
    recorder.capture(1_300)
    select_named_option(page, recorder, "安全风险")
    avoid_button.click()

    result = page.get_by_text("首推商品", exact=True)
    wait_for_visible(page, recorder, result, timeout_seconds=timeout_seconds)

    page.wait_for_timeout(1_500)
    recorder.capture(2_000)

    confirm_purchase = page.get_by_role("button", name="确认已购买并记住", exact=True)
    wait_for_visible(page, recorder, confirm_purchase, timeout_seconds=30)
    confirm_purchase.scroll_into_view_if_needed()
    page.wait_for_timeout(600)
    recorder.capture(1_500)
    confirm_purchase.click()

    memory_saved = page.get_by_text("已保存最近购买记录", exact=False)
    wait_for_visible(page, recorder, memory_saved, timeout_seconds=30)
    recorder.capture(1_700)

    new_session = page.get_by_role("button", name="新建会话", exact=True)
    new_session.click()
    textarea.wait_for(state="visible", timeout=30_000)
    page.wait_for_timeout(800)
    recorder.capture(1_400)

    fill_in_steps(page, recorder, MEMORY_QUERY)
    textarea.press("Enter")
    memory_answer = page.get_by_text("你上一次确认记录的已购商品是", exact=False)
    wait_for_visible(page, recorder, memory_answer, timeout_seconds=timeout_seconds)
    page.wait_for_timeout(900)
    recorder.capture(3_500)
    return recorder


def record_quick_flow(page: Page, query: str, timeout_seconds: int, frame_dir: Path) -> GifRecorder:
    page.add_style_tag(content='[data-testid="stSidebar"] { display: none !important; }')
    recorder = GifRecorder(page, frame_dir)
    textarea = page.locator("textarea")

    recorder.capture(1_300)
    fill_in_steps(page, recorder, query)
    textarea.press("Enter")

    result = page.get_by_text("首推商品", exact=True)
    wait_for_visible(page, recorder, result, timeout_seconds=timeout_seconds)
    page.wait_for_timeout(1_500)
    recorder.capture(3_000)
    return recorder


def load_recording(page: Page, frame_dir: Path) -> GifRecorder:
    recorder = GifRecorder.__new__(GifRecorder)
    recorder.page = page
    recorder.frame_dir = frame_dir
    recorder.duration_path = frame_dir / "durations.json"
    recorder.frame_paths = sorted(frame_dir.glob("frame-*.png"))
    recorder.durations = json.loads(recorder.duration_path.read_text(encoding="utf-8"))
    if len(recorder.frame_paths) != len(recorder.durations):
        raise RuntimeError("Recorded frame and duration counts do not match.")
    return recorder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record the Streamlit recommendation flow as an optimized GIF.")
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    parser.add_argument("--flow", choices=("full", "quick"), default="full")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--query")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--encode-only", action="store_true", help="Encode frames left by an interrupted recording.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or ASSET_DIR / (
        "demo_full_flow.gif" if args.flow == "full" else "demo_quick_recommendation.gif"
    )
    query = args.query or (FULL_FLOW_QUERY if args.flow == "full" else QUICK_FLOW_QUERY)
    frame_dir = ASSET_DIR / ".demo_frames"
    recorder: GifRecorder | None = None
    if args.encode_only:
        recorder = load_recording(None, frame_dir)  # type: ignore[arg-type]
        recorder.save(output)
        size_mb = output.stat().st_size / (1024 * 1024)
        print(f"demo_gif: {output}")
        print(f"frames: {len(recorder.frame_paths)}")
        print(f"size_mb: {size_mb:.2f}")
        recorder.cleanup()
        return

    with sync_playwright() as playwright:
        with tempfile.TemporaryDirectory(prefix="smart-pet-edge-", ignore_cleanup_errors=True) as profile_dir:
            context = playwright.chromium.launch_persistent_context(
                user_data_dir=profile_dir,
                executable_path=str(find_edge()),
                headless=True,
                viewport={"width": 1280, "height": 800},
                device_scale_factor=1,
            )
            page = context.pages[0]
            try:
                print("demo_stage: open_streamlit", flush=True)
                prepare_page(page, args.url)
                print("demo_stage: run_recommendation", flush=True)
                recorder = (
                    record_full_flow(page, query, args.timeout, frame_dir)
                    if args.flow == "full"
                    else record_quick_flow(page, query, args.timeout, frame_dir)
                )
                print("demo_stage: encode_gif", flush=True)
                recorder.save(output)
            finally:
                context.close()

    if recorder is None:
        raise RuntimeError("Demo recorder was not initialized.")
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"demo_gif: {output}")
    print(f"frames: {len(recorder.frame_paths)}")
    print(f"size_mb: {size_mb:.2f}")
    recorder.cleanup()


if __name__ == "__main__":
    main()
