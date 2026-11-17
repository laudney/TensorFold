#!/usr/bin/env python3
"""one_sha.py TEXT MAX_TOKENS: one greedy /v1/completions request with drafts off; prints wall seconds and the reply sha256."""
import hashlib, json, sys, time, urllib.request
body = {"model": "local-model", "prompt": sys.argv[1], "max_tokens": int(sys.argv[2]), "temperature": 0, "ignore_eos": True, "draft": False}
t = time.time()
req = urllib.request.Request("http://127.0.0.1:8090/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=3600))
print(f"{time.time() - t:.3f} {hashlib.sha256(r["choices"][0]["text"].encode()).hexdigest()[:12]}")
