"""Long-context probe against `nano-ai serve`: needle-in-a-haystack retrieval over real Rust source,
reporting prefill/decode speed from the server's telemetry. Usage: niah.py PORT TOKENS[,TOKENS...] [DEPTH]"""
import glob, json, os, random, sys, time, urllib.request

port, sizes = sys.argv[1], [int(s) for s in sys.argv[2].split(",")]
depth = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
files = sorted(glob.glob(os.path.expanduser("~/.cargo/registry/src/*/*/src/**/*.rs"), recursive=True))
random.Random(7).shuffle(files)
corpus, total = [], 0
for path in files:
    try:
        text = open(path, encoding="utf-8").read()
    except (UnicodeDecodeError, OSError):
        continue
    corpus.append(f"// FILE: {path.split('/src/', 1)[-1]}\n{text}\n")
    total += len(text)
    if total > 4 * max(sizes) + 100_000:
        break
blob = "".join(corpus)

for tokens in sizes:
    chars = int(tokens * 3.2)  # Rust source averages ~3.2 chars/token with this tokenizer (checked below)
    start = random.randrange(0, max(1, len(blob) - chars))
    hay = blob[start : start + chars]
    code = f"{random.randrange(1000, 9999)}-{random.choice(['KESTREL', 'OSPREY', 'HERON', 'PLOVER'])}"
    needle = f"\n// NOTE: the deployment passphrase for project Bluebird is {code}.\n"
    cut = hay.rfind("\n", 0, int(len(hay) * depth)) + 1
    prompt = (
        f"Session {random.random()}\n" + hay[:cut] + needle + hay[cut:]
        + "\n\nWhat is the deployment passphrase for project Bluebird? Answer with the passphrase only."
    )
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 48, "temperature": 0}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.time()
    try:
        rep = json.loads(urllib.request.urlopen(req, timeout=7200).read())
    except Exception as e:  # noqa: BLE001
        print(tokens, "ERROR", e)
        continue
    wall = time.time() - t0
    msg = rep["choices"][0]["message"]
    answer = (msg.get("content") or "") + " " + (msg.get("reasoning_content") or "")
    t = rep.get("telemetry", {})
    print(json.dumps({
        "target": tokens, "prompt_tokens": rep["usage"]["prompt_tokens"], "chars_per_token": round(len(prompt) / rep["usage"]["prompt_tokens"], 2),
        "depth": depth, "found": code in answer, "prefill_s": round(t.get("prefill_ms", 0) / 1e3, 2),
        "prefill_tok_s": round(t.get("prefill_tok_per_s", 0)), "decode_tok_s": round(t.get("decode_tok_per_s", 0), 1),
        "wall_s": round(wall, 1), "answer": answer.strip()[:80],
    }), flush=True)
