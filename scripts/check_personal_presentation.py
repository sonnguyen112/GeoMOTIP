#!/usr/bin/env python3
"""Exercise the actual offline, fullscreen HTML presentation in Chromium."""
from pathlib import Path
import json
import re
import tempfile
import zipfile
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'personal_presentation'


def main():
    qa=OUT/'qa';qa.mkdir(exist_ok=True)
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=['--no-sandbox'])
        context=browser.new_context(viewport=dict(width=1600,height=900),offline=True)
        page=context.new_page();errors=[];remote=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('request',lambda r:remote.append(r.url) if r.url.startswith(('http:','https:')) else None)
        page.goto((OUT/'index.html').as_uri());page.evaluate('document.fonts.ready')
        page.wait_for_function("[...document.querySelectorAll('main>.slide img')].every(x=>x.complete&&x.naturalWidth>0)")
        assert page.locator('main>.slide').count()==31
        overflow=[]
        page.add_style_tag(content='.controls{visibility:hidden!important}')
        for i in range(31):
            page.evaluate('(i)=>showSlide(i)',i)
            found=page.locator('main>.slide.active').evaluate("s=>[...s.querySelectorAll('.text')].filter(e=>e.scrollHeight>e.clientHeight+3||e.scrollWidth>e.clientWidth+3).map(e=>e.innerText)")
            if found:overflow.append(dict(slide=i+1,text=found))
            page.locator('main>.slide.active').screenshot(path=str(qa/f'slide_{i+1:02}.png'),animations='disabled')
        assert not overflow,overflow
        page.evaluate('showSlide(0)');page.keyboard.press('ArrowRight');assert page.locator('main>.slide.active').get_attribute('id')=='slide-2'
        page.keyboard.press('Space');assert page.locator('main>.slide.active').get_attribute('id')=='slide-3'
        page.keyboard.press('Home');assert page.locator('main>.slide.active').get_attribute('id')=='slide-1'
        page.keyboard.press('End');assert page.locator('main>.slide.active').get_attribute('id')=='slide-25'
        page.keyboard.press('o');assert page.locator('#overview').evaluate("e=>e.classList.contains('open')")
        assert page.locator('.thumb').count()==31
        page.locator('.thumb').nth(13).click();assert page.locator('main>.slide.active').get_attribute('id')=='slide-14'
        page.keyboard.press('n');assert 'tiền huấn luyện' in page.locator('#notes-text').inner_text().lower() or 'GT–GT' in page.locator('#notes-text').inner_text()
        page.keyboard.press('Escape');assert page.locator('.overlay.open').count()==0
        page.evaluate('showSlide(21)');page.keyboard.press('z');assert page.locator('#zoom').evaluate("e=>e.classList.contains('open')")
        page.keyboard.press('Escape');page.keyboard.press('v')
        page.wait_for_function("document.getElementById('demo-video').readyState>=1",timeout=15000)
        duration=page.locator('#demo-video').evaluate('v=>v.duration');assert abs(duration-45)<.1
        page.wait_for_function("document.getElementById('demo-video').currentTime>0",timeout=10000)
        page.keyboard.press('Escape');assert page.locator('#demo-video').evaluate('v=>v.paused')
        page.keyboard.press('f');page.wait_for_function('!!document.fullscreenElement')
        page.evaluate('document.exitFullscreen()')
        for width,height in [(1366,768),(1280,720)]:
            page.set_viewport_size(dict(width=width,height=height))
            page.wait_for_function("()=>{const r=document.querySelector('main>.slide.active').getBoundingClientRect();return r.x>=-1&&r.y>=-1&&r.right<=innerWidth+1&&r.bottom<=innerHeight+1}")
            rect=page.locator('main>.slide.active').bounding_box()
            assert rect['x']>=-1 and rect['y']>=-1 and rect['x']+rect['width']<=width+1 and rect['y']+rect['height']<=height+1
        assert not errors,errors
        assert not remote,remote
        # Verify the portable ZIP works without any ancestor project files.
        with tempfile.TemporaryDirectory(prefix='motip-html-portable-') as tmp:
            with zipfile.ZipFile(OUT/'THUYET_TRINH_MOTIP_HTML.zip') as z:
                assert z.testzip() is None;z.extractall(tmp)
            portable=context.new_page();portable.goto((Path(tmp)/'MOTIP_HTML/index.html').as_uri())
            portable.wait_for_function("[...document.querySelectorAll('main>.slide img')].every(x=>x.complete&&x.naturalWidth>0)")
            assert portable.locator('main>.slide').count()==31
            portable.close()
        browser.close()
    result=dict(slides=31,main_slides=25,appendix=6,offline=True,remote_requests=remote,
        text_overflow=overflow,javascript_errors=errors,keyboard_navigation=True,fullscreen=True,
        overview=True,notes=True,image_zoom=True,video_playback=True,video_seconds=duration,
        responsive_viewports=['1600x900','1366x768','1280x720'],portable_zip=True,
        no_personal_pdf_export=True,no_personal_pptx_conversion=True)
    (OUT/'QA.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
