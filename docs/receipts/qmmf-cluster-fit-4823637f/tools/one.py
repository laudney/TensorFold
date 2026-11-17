#!/usr/bin/env python3
"""One /v1/completions request: one.py PROMPT_FILE_OR_TEXT MAX_TOKENS DRAFT(0/1); prints wall time and token count."""
import json, sys, time, urllib.request
p, n, draft = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1"
prompt = json.load(open(p)) if p.endswith(".json") else p
body = {"model": "local-model", "prompt": prompt, "max_tokens": n, "temperature": 0, "ignore_eos": True}
if not draft:
    body["draft"] = False
t = time.time()
req = urllib.request.Request("http://127.0.0.1:8090/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=3600))
print(f"{time.time() - t:.3f} s", r["usage"])
