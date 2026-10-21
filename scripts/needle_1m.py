#!/usr/bin/env python3
"""900K-token needle test against Maya 1M on :8012. Writes result JSON + log."""
import json, time, random, urllib.request

random.seed(20261020)
BASE, KEY = "http://127.0.0.1:8012", os.environ.get("MAYA_API_KEY", "change-me")
NEEDLE = ("Maintenance log entry 4471, dated October 20: the lighthouse vault passphrase "
          "for the annual audit is MARLINE-7741-QUARTZ.")
QUESTION = "According to maintenance log entry 4471, what is the lighthouse vault passphrase for the annual audit? State it exactly."

subjects = ["the harbor master", "the night crew", "the ferry pilot", "the dock office", "the coast guard",
            "the fuel depot", "the crane operator", "the customs clerk", "the weather station", "the shipyard"]
verbs = ["logged", "reported", "recorded", "noted", "filed", "annotated", "confirmed", "summarized"]
objects = ["a shift change", "an inventory count", "a tide reading", "a fuel transfer", "a radar calibration",
           "a supply delivery", "a radio check", "a hull inspection", "a lantern repair", "a rope audit"]
quals = ["without incident", "ahead of schedule", "behind schedule", "pending review", "as expected",
         "under supervision", "during fog", "at first light", "before dusk", "after the storm"]

parts = []
target_chars = 3_720_000  # ~900-940K tokens at ~4 chars/token
approx = 0
i = 0
while approx < target_chars:
    i += 1
    batch = []
    for j in range(200):
        s = f"Entry {i:06d}: {random.choice(subjects)} {random.choice(verbs)} {random.choice(objects)} {random.choice(quals)}. "
        batch.append(s)
    chunk = "".join(batch)
    parts.append(chunk)
    approx += len(chunk)
# insert needle at ~93% depth
split = int(len(parts) * 0.93)
parts.insert(split, NEEDLE + "\n")
haystack = "".join(parts)
prompt = haystack + "\n\n" + QUESTION
print(f"haystack chars: {len(haystack):,} (~{len(haystack)//4:,} tokens est)", flush=True)

body = {"model": "qwen3.8-27b", "max_tokens": 1200, "temperature": 0,
        "stream": True, "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": prompt}]}
r = urllib.request.Request(BASE + "/v1/chat/completions")
r.add_header("Authorization", "Bearer " + KEY)
r.add_header("Content-Type", "application/json")

t0 = time.time()
ttft = None
content = []
usage = None
with urllib.request.urlopen(r, json.dumps(body).encode(), timeout=7200) as resp:
    for line in resp:
        line = line.decode().strip()
        if not line.startswith("data: "): continue
        payload = line[6:]
        if payload == "[DONE]": break
        try: d = json.loads(payload)
        except Exception: continue
        if d.get("usage"): usage = d["usage"]
        for ch in d.get("choices", []):
            delta = ch.get("delta", {})
            if delta.get("content"):
                if ttft is None: ttft = time.time() - t0
                content.append(delta["content"])
total = time.time() - t0
text = "".join(content)
ptok = (usage or {}).get("prompt_tokens", 0)
ctok = (usage or {}).get("completion_tokens", 0)
hit = "MARLINE-7741-QUARTZ" in text
res = {"prompt_tokens": ptok, "completion_tokens": ctok, "ttft_s": round(ttft or -1, 1),
       "total_s": round(total, 1), "prefill_tok_s": round(ptok / max(ttft or total, 0.1), 1),
       "decode_tok_s": round(ctok / max(total - (ttft or 0), 0.1), 1) if ttft else None,
       "needle_hit": hit, "answer_excerpt": text[-300:]}
json.dump(res, open("./needle-result.json", "w"), indent=1)
print(json.dumps(res, indent=1), flush=True)
