"""Capture actual Streamlit demo using local Chrome headless/CDP; no API call.
Requires websocket-client (screenshot tooling only), Chrome or Edge on Windows.
"""
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import websocket

BASE = Path(__file__).resolve().parents[1]


def main():
    browser = next((p for p in [Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")] if p.exists()), None)
    if browser is None:
        raise RuntimeError("Chrome/Edge not found")
    env = os.environ.copy()
    env["DEEPSEEK_API_KEY"] = "fake-demo-session-token"  # fictional, child process only
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    processes = []
    try:
        processes.append(subprocess.Popen([sys.executable, "-m", "streamlit", "run", "tools/demo_page.py",
            "--server.address", "127.0.0.1", "--server.port", "18502", "--server.headless", "true",
            "--browser.gatherUsageStats", "false", "--server.fileWatcherType", "none"], cwd=BASE, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=flags))
        processes.append(subprocess.Popen([str(browser), "--headless=new", "--disable-gpu", "--no-first-run",
            "--remote-debugging-port=19222", "--remote-debugging-address=127.0.0.1",
            "--user-data-dir=" + str(BASE / ".demo-browser"), "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env, creationflags=flags))
        for _ in range(60):
            if any(p.poll() is not None for p in processes):
                for p in processes:
                    if p.poll() is not None:
                        print("Preview child exit", p.returncode, p.stderr.read().decode(errors="replace")[-1500:])
                raise RuntimeError("Preview child exited")
            try:
                urllib.request.urlopen("http://127.0.0.1:18502/_stcore/health", timeout=1).close()
                pages = json.load(urllib.request.urlopen("http://127.0.0.1:19222/json", timeout=1))
                break
            except OSError:
                time.sleep(.5)
        else:
            raise RuntimeError("Local preview did not start")
        ws = websocket.create_connection(pages[0]["webSocketDebuggerUrl"], suppress_origin=True, timeout=30)
        sequence = 0
        def cdp(method, params=None):
            nonlocal sequence
            sequence += 1
            ws.send(json.dumps({"id": sequence, "method": method, "params": params or {}}))
            while True:
                reply = json.loads(ws.recv())
                if reply.get("id") == sequence:
                    if "error" in reply:
                        raise RuntimeError("Browser command failed")
                    return reply.get("result", {})
        cdp("Emulation.setDeviceMetricsOverride", {"width": 1360, "height": 1120, "deviceScaleFactor": 1, "mobile": False})
        cdp("Page.navigate", {"url": "http://127.0.0.1:18502"})
        for _ in range(60):
            state = cdp("Runtime.evaluate", {"expression": "document.body.innerText.includes('en_p0404') || document.body.innerText.includes('PDF第404页')", "returnByValue": True})
            if state.get("result", {}).get("value"):
                break
            time.sleep(.5)
        else:
            state = cdp("Runtime.evaluate", {"expression": "document.body.innerText.slice(0, 1500)", "returnByValue": True})
            print("Demo page diagnostic:", state.get("result", {}).get("value", ""))
            raise RuntimeError("Demo content not rendered")
        cdp("Runtime.evaluate", {"expression": "document.querySelectorAll('details').forEach(d=>d.open=true)"})
        time.sleep(1)
        screenshot = cdp("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True})
        out = BASE / "docs/images/manual-qa.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(screenshot["data"]))
        ws.close()
        print("Saved docs/images/manual-qa.png (synthetic demo; no API)")
    finally:
        for process in reversed(processes):
            process.terminate()
            process.wait(timeout=15)


if __name__ == "__main__":
    main()
