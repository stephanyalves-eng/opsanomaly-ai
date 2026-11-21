"""Capture real CLI output as a two-minute edited offline terminal replay."""

import html
import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "docs/demo"
    output.mkdir(parents=True, exist_ok=True)
    scenes = []
    commands = [["scripts/generate_synthetic.py"], ["scripts/train.py"], ["scripts/evaluate.py"]]
    scenes.append(
        "OpsAnomaly AI\nTwo-minute edited terminal demonstration.\n"
        "Actual captured command output; playback pauses are edited. No real failure validation.\n"
        "Offline synthetic data, local CPU, no paid API or published service."
    )
    for command in commands:
        run = subprocess.run(
            [sys.executable, *command],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
            timeout=120,
        )
        scenes.append(
            "$ python " + " ".join(command) + "\n" + run.stdout + run.stderr + "\nExit code: 0"
        )
    scenes.append(
        "$ inspect artifacts/metadata.json\n"
        + (root / "artifacts/metadata.json").read_text(encoding="utf-8")
    )
    scenes.append(
        "Review: test scores are synthetic point injections.\n"
        "Precision/recall are not failure-diagnosis accuracy.\n"
        "Feature evidence describes robust deviations, not causes.\n"
        "Open docs/evaluation.md and docs/limitations.md for scope.\n"
        "Run python app/gradio_app.py for the interactive loopback demo."
    )
    events = [
        [float(i * 20), "o", "\u001b[2J\u001b[H" + s.replace("\n", "\r\n")]
        for i, s in enumerate(scenes)
    ]
    events.append([120.0, "o", "\r\nEnd of recorded demonstration.\r\n"])
    header = {
        "version": 2,
        "width": 120,
        "height": 45,
        "title": "OpsAnomaly AI — actual CLI capture, edited pacing",
        "duration": 120,
    }
    (output / "demo.cast").write_text(
        "\n".join(json.dumps(x) for x in [header, *events]) + "\n", encoding="utf-8"
    )
    # Self-contained player; no external scripts, telemetry or network dependency.
    escaped = json.dumps(scenes).replace("<", "\\u003c")
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><title>OpsAnomaly demo</title>
<style>body{background:#111827;color:#d1fae5;font:16px monospace;padding:2rem}pre{white-space:pre-wrap}button{font:inherit;padding:.5rem}</style>
<h1>OpsAnomaly AI: 2-minute CLI replay</h1><p>Actual execution output with edited pacing; synthetic data only.</p>
<button onclick="start()">Play / restart</button><span id="time"> 0:00 / 2:00</span><pre id="screen"></pre>
<script>const scenes=SCENES;let timer;function start(){clearInterval(timer);let t=0;function draw(){document.getElementById('screen').textContent=scenes[Math.min(5,Math.floor(t/20))];document.getElementById('time').textContent=' '+Math.floor(t/60)+':'+String(t%60).padStart(2,'0')+' / 2:00';}draw();timer=setInterval(()=>{t++;draw();if(t>=120)clearInterval(timer);},1000);}document.getElementById('screen').textContent=scenes[0];</script></html>"""
    (output / "index.html").write_text(page.replace("SCENES", escaped), encoding="utf-8")
    print(html.escape(str(output)))


if __name__ == "__main__":
    main()
