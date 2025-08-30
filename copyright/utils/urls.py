from typing import Dict, Any
import os
from playwright.sync_api import sync_playwright

from copyright.utils.functionals import sha1_bytes

HEADLESS = True
VIEWPORT   = {"width": 1365, "height": 900}
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")
TIMEOUT_MS = 45_000

def render_and_capture_url(url: str, output_dir: str) -> Dict[str,Any]:
    out = {"url": url, "ok": False}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        context = browser.new_context(
            user_agent=USER_AGENT,
            viewport=VIEWPORT,
            locale="ko-KR",
            device_scale_factor=1.0,
            java_script_enabled=True
        )
        page = context.new_page()
        page.set_default_timeout(TIMEOUT_MS)
        page.goto(url, wait_until="domcontentloaded")
        # 스크롤 다운 & lazy-load 유도
        page.wait_for_timeout(800)
        last_height = 0
        for _ in range(50):  # 최대 50 스텝
            page.mouse.wheel(0, 1200)
            page.wait_for_timeout(350)
            curr = page.evaluate("document.scrollingElement ? document.scrollingElement.scrollHeight : document.body.scrollHeight")
            if curr == last_height: break
            last_height = curr
        # 네트워크 안정화
        try:
            page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            pass

        # 풀페이지 스크린샷
        screen_path = os.path.join(output_dir, 'screens', f"{sha1_bytes(url.encode())}_fullpage.png")
        page.screenshot(path=screen_path, full_page=True)
        out["screenshot_path"] = screen_path

    return out


if __name__ == "__main__":
    output_dir='/HDD/_projects/github/similarity_inspection/outputs'
    urls = ['https://www.11st.co.kr/products/5662267421',
            'https://item.gmarket.co.kr/Item?goodscode=4426684995']
    
    for url in urls:
        render_and_capture_url(url, output_dir=output_dir)
