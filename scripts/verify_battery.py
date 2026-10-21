#!/usr/bin/env python3
"""Maya drop-in battery v2 for :8012 — auth, alias, decode speed, prefill speed, vision, tool calls."""
import json, time, base64, io, urllib.request, urllib.error

BASE = "http://127.0.0.1:8012"
import os
KEY = os.environ.get("MAYA_API_KEY", "change-me")
results = []

def req(path, body=None, key=KEY, timeout=600):
    r = urllib.request.Request(BASE + path)
    if key: r.add_header("Authorization", "Bearer " + key)
    data = json.dumps(body).encode() if body is not None else None
    if data: r.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(r, data, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode() or "{}"), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}"), time.time() - t0
    finally:
        pass
    # unreachable

def req2(path, body=None, key=KEY, timeout=600):
    r = urllib.request.Request(BASE + path)
    if key: r.add_header("Authorization", "Bearer " + key)
    data = json.dumps(body).encode() if body is not None else None
    if data: r.add_header("Content-Type", "application/json")
    t0 = time.time()
    try:
        with urllib.request.urlopen(r, data, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode() or "{}"), time.time() - t0
    except urllib.error.HTTPError as e:
        try: b = json.loads(e.read().decode() or "{}")
        except Exception: b = {}
        return e.code, b, time.time() - t0

def chat(prompt, max_tokens=400, model="qwen3.8-27b", extra=None, timeout=900):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.7}
    if extra: body.update(extra)
    return req2("/v1/chat/completions", body, timeout=timeout)

# 1. auth contract
s, b, _ = req2("/v1/models", key="wrongkey")
results.append(("wrong-key 401", "PASS" if s == 401 else f"FAIL got {s}"))
s, b, _ = req2("/v1/models")
results.append(("/v1/models 200", "PASS" if s == 200 and any("glm" in m.get("id", "").lower() for m in b.get("data", [])) else f"FAIL {s} {str(b)[:120]}"))

# 2. warmup gen (first answers slowest)
s, b, dt = chat("Write one sentence about the ocean.", max_tokens=60)
results.append(("warmup gen", f"{s}, {dt:.1f}s" if s == 200 else f"FAIL {s} {str(b)[:150]}"))

# 3. decode speed — 512 tokens (alias model name deliberately)
s, b, dt = chat("Write a detailed 400-word story about a lighthouse keeper who discovers something strange. Do not stop early.", max_tokens=512)
if s == 200:
    ctok = b.get("usage", {}).get("completion_tokens", 0)
    results.append(("decode 512tok (alias qwen3.8-27b)", f"PASS {ctok} tok in {dt:.1f}s = {ctok/dt:.1f} tok/s"))
else:
    results.append(("decode 512tok", f"FAIL {s} {str(b)[:200]}"))

# 4. prefill speed
filler = ("The quarterly report covers inventory levels across seven regional warehouses. " * 120)
for label, reps in (("prefill ~2.6K", 1), ("prefill ~8K", 3)):
    s, b, dt = chat(filler * reps + "\n\nSummarize the above text in one sentence.", max_tokens=80)
    if s == 200:
        ptok = b.get("usage", {}).get("prompt_tokens", 0)
        results.append((label, f"PASS {ptok} tok prompt, total {dt:.1f}s (~{ptok/max(dt-2,0.1):.0f} tok/s incl gen)"))
    else:
        results.append((label, f"FAIL {s} {str(b)[:150]}"))

# 5. vision — rendered text card
from PIL import Image, ImageDraw
img = Image.new("RGB", (640, 160), "white")
d = ImageDraw.Draw(img)
d.text((20, 60), "WARD 7 ROOM 12B MARLINE", fill="black")
buf = io.BytesIO(); img.save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode()
s, b, dt = req2("/v1/chat/completions", {"model": "qwen3.8-27b", "max_tokens": 300,
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
        {"type": "text", "text": "Read the exact text in this image."}]}]}, timeout=300)
txt = (b.get("choices", [{}])[0].get("message", {}).get("content") or "") if s == 200 else ""
ok = s == 200 and "MARLINE" in txt.upper()
results.append(("vision text card", f"PASS {dt:.0f}s" if ok else f"{'FAIL' if s!=200 else 'MISS'} {s} {txt[:80]}"))

# 6. tool calls
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
s, b, dt = chat("What is the weather in San Diego right now? Use the tool.", max_tokens=300,
                extra={"tools": tools, "tool_choice": "auto"})
msg = b.get("choices", [{}])[0].get("message", {}) if s == 200 else {}
tc = msg.get("tool_calls")
ok = s == 200 and tc and tc[0]["function"]["name"] == "get_weather"
results.append(("tool_call auto", f"PASS {dt:.0f}s -> {tc[0]['function']['arguments'] if tc else ''}" if ok else f"FAIL {s} {str(msg)[:150]}"))

print("\n=== MAYA 8012 BATTERY v2 ===")
for name, res in results:
    print(f"{name:36s} {res}")
