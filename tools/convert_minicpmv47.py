"""Convert MiniCPM-V-4.7-1B and MiniCPM-V-4.7-1B-DSpark into micro-ai's
zero-copy 16-KiB page-aligned Metal artifacts:
  - minicpm-v-4.7-1b-tern4.bin  (~374 MB: PTQ1_0 ternary embed_tokens & lm_head + INT4_G64 24 decoder layers)
  - minicpm-v-4.7-1b-int4.bin   (~540 MB: INT4_G64 everything)
  - minicpm-v-4.7-1b-vision.bin (~451 MB: PTQ1_0 vit_merger + INT4_G64 27-layer SigLIP2 & merger)
  - minicpm-v-4.7-1b-dspark.bin (~169 MB: PTQ1_0 MLP/q/fc + INT4_G64 k/v/o + BF16 markov)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

METAL_PAGE = 16 * 1024
TENSOR_ALIGN = 256

LM_MAGIC = b"MCV47\0\0\x01"
VIS_MAGIC = b"MCVIS\0\0\x01"
DSP_MAGIC = b"MCDSP\0\0\x01"

PINNED_1B_REPO = "openbmb/MiniCPM-V-4.7-1B"
PINNED_1B_REV = "bb99eab4113676f30830ffa73bac1f0b2a5f1dea"
PINNED_1B_SAFETENSORS = (
    3_109_054_480,
    "8c4921ac83182c85190ad10c3aeb118d51e385d2d98ec13b88b514bbf6cac002",
)
PINNED_1B_TOKENIZER = (
    19_992_481,
    "33861e37bb955af1e3f3061182b820f347eba2b9c2c1011c82794bf0d6e77b54",
)

PINNED_DSPARK_REPO = "openbmb/MiniCPM-V-4.7-1B-DSpark"
PINNED_DSPARK_REV = "33762d223b4be1101cf90abba1958435a9b018e0"
PINNED_DSPARK_SAFETENSORS = (
    427_102_026,
    "84222c854701adcd170f243f3dd4041f8b26c0b89089b34de2b27bc27d538196",
)


def deterministic_signs(columns: int) -> list[float]:
    """Match micro_engine::quant::deterministic_signs bit-for-bit."""
    mask = 0xFFFF_FFFF_FFFF_FFFF
    state = (columns ^ 0x9E37_79B9_7F4A_7C15) & mask
    out = []
    for _ in range(columns):
        state = (state + 0x9E37_79B9_7F4A_7C15) & mask
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58_476D_1CE4_E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D0_49BB_1331_11EB) & mask
        z = z ^ (z >> 31)
        out.append(-1.0 if (z & 1) == 0 else 1.0)
    return out


def rotate512_torch(w):
    """Signed 512-block Hadamard basis the engine rotates inputs into (`rot512_*`)."""
    import torch

    rows, cols = w.shape
    assert cols % 512 == 0
    i = torch.arange(512, device=w.device)
    bits = i[:, None] & i[None, :]
    parity = torch.zeros_like(bits)
    while bool(bits.any()):
        parity ^= bits & 1
        bits = bits >> 1
    H = (1 - 2 * parity).float() / (512**0.5)
    signs = torch.tensor(deterministic_signs(cols), dtype=torch.float32, device=w.device)
    return ((w * signs[None, :]).reshape(rows, -1, 512) @ H).reshape(rows, cols)



def pack_ptq1_rows_torch(w, rotate: bool = False):
    """Pack a 2D float32 torch tensor [rows, cols] into PTQ1_0 28-byte blocks."""
    import numpy as np
    import torch

    rows, cols = w.shape
    if rows > 32768:
        parts = [pack_ptq1_rows_torch(w[i : i + 32768], rotate=rotate) for i in range(0, rows, 32768)]
        return b"".join(parts)

    assert cols % 128 == 0
    if rotate:
        w = rotate512_torch(w)

    g = w.reshape(rows, cols // 128, 128)
    a = g.abs()
    sorted_a, _ = a.sort(-1, descending=True)
    csum = sorted_a.cumsum(-1)
    k = torch.arange(1, 129, dtype=torch.float32, device=w.device)
    best = (csum * csum / k).argmax(-1, keepdim=True)
    thresh = sorted_a.gather(-1, best)
    mask = (a >= thresh).float()
    scale = ((a * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)).to(torch.float16)
    codes = (
        (torch.sign(g) * mask * (scale.float()[..., None] != 0)).to(torch.int8) + 1
    ).cpu().numpy().astype(np.uint16)
    scale_bytes = scale.cpu().numpy().view(np.uint8).reshape(-1, 2)

    codes_flat = codes.reshape(-1, 128)
    n_blocks = codes_flat.shape[0]
    out = np.zeros((n_blocks, 28), dtype=np.uint8)
    for m in range(16):
        q = np.zeros(n_blocks, dtype=np.uint16)
        for n in range(5):
            q = q * 3 + codes_flat[:, m + n * 16]
        out[:, m] = ((q * 256 + 242) // 243).astype(np.uint8)
    for m in range(8):
        q = np.zeros(n_blocks, dtype=np.uint16)
        for n in range(5):
            q = q * 3 + codes_flat[:, 80 + m + n * 8]
        out[:, 16 + m] = ((q * 256 + 242) // 243).astype(np.uint8)
    for j in range(2):
        q = np.zeros(n_blocks, dtype=np.uint16)
        for m in range(4):
            q = q * 3 + codes_flat[:, 120 + j + m * 2]
        out[:, 24 + j] = ((q * 3 * 256 + 242) // 243).astype(np.uint8)
    out[:, 26:28] = scale_bytes
    return out.tobytes()


QUANT_STATS = {"base_mse": 0.0, "opt_mse": 0.0, "tensors": 0}


def reset_quant_stats():
    QUANT_STATS["base_mse"] = 0.0
    QUANT_STATS["opt_mse"] = 0.0
    QUANT_STATS["tensors"] = 0


def pack_int4_g64_torch(w, col_weights=None):
    """Pack a 2D float32 torch tensor [rows, cols] into INT4_G64 (codes_bytes, groups_bytes)."""
    import numpy as np
    import torch

    rows, cols = w.shape
    if rows > 32768:
        c_parts, g_parts = [], []
        for i in range(0, rows, 32768):
            c, g = pack_int4_g64_torch(w[i : i + 32768], col_weights=col_weights)
            c_parts.append(c)
            g_parts.append(g)
        return b"".join(c_parts), b"".join(g_parts)

    if cols % 64 != 0:
        pad = 64 - (cols % 64)
        w = torch.nn.functional.pad(w, (0, pad), value=0.0)
        if col_weights is not None:
            col_weights = torch.nn.functional.pad(col_weights.reshape(-1), (0, pad), value=1.0)
        cols = w.shape[1]

    g = w.reshape(rows, cols // 64, 64)
    if col_weights is not None:
        cw = col_weights.to(device=w.device, dtype=torch.float32).reshape(1, cols // 64, 64)
    else:
        cw = torch.ones((1, 1, 64), device=w.device, dtype=torch.float32)
    best_q, mx, mn, base_err, best_err = int4_g64_grid_torch(g, cw)
    QUANT_STATS["base_mse"] += float(base_err.double().sum().item())
    QUANT_STATS["opt_mse"] += float(best_err.double().sum().item())
    QUANT_STATS["tensors"] += 1
    return pack_int4_codes(best_q, mx, mn)


def int4_g64_grid_torch(g, cw):
    """FP14 (scale, bias) and codes per 64-group of g [rows, groups, 64]: clip search + WLS weighted by cw."""
    import torch

    mn_raw = g.amin(-1)
    mx_raw = g.amax(-1)
    mid = 0.5 * (mn_raw + mx_raw)
    half_span = 0.5 * (mx_raw - mn_raw)

    # Baseline unclipped (1.0, 1.0) weighted error for telemetry
    base_mn = mn_raw.to(torch.float16)
    base_mx = ((mx_raw - base_mn.float()) / 15.0).to(torch.float16)
    base_sf = base_mx.float()
    base_bf = base_mn.float()
    base_safe = torch.where(base_sf > 0, base_sf, torch.ones_like(base_sf))
    base_q = torch.round((g - base_bf[..., None]) / base_safe[..., None]).clamp(0, 15) * (base_sf[..., None] > 0)
    base_recon = base_q * base_sf[..., None] + base_bf[..., None]
    base_err = (((g - base_recon) ** 2) * cw).sum(-1)

    best_err = base_err.clone()
    best_mn = base_mn
    best_mx = base_mx
    best_q = base_q

    # Upgrade 2: 2D Asymmetric Clipping Grid Search (25 pairs)
    clips = (1.0, 0.97, 0.94, 0.91, 0.88)
    for c_lo in clips:
        c_mn = (mid - c_lo * half_span).to(torch.float16)
        bf = c_mn.float()
        for c_hi in clips:
            c_mx = (((mid + c_hi * half_span) - bf) / 15.0).to(torch.float16)
            sf = c_mx.float()
            safe = torch.where(sf > 0, sf, torch.ones_like(sf))
            q = torch.round((g - bf[..., None]) / safe[..., None]).clamp(0, 15) * (sf[..., None] > 0)
            recon = q * sf[..., None] + bf[..., None]
            err = (((g - recon) ** 2) * cw).sum(-1)
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_mn = torch.where(better, c_mn, best_mn)
            best_mx = torch.where(better, c_mx, best_mx)
            best_q = torch.where(better[..., None], q, best_q)

    # Upgrade 3: Exact Weighted Least-Squares (WLS) Scale & Bias Refinement (2 Lloyd-Max alternating steps)
    w_sum = cw.sum(-1)
    wg = (cw * g).sum(-1)
    q = best_q
    for _ in range(2):
        wq = (cw * q).sum(-1)
        wq2 = (cw * q * q).sum(-1)
        wqg = (cw * q * g).sum(-1)
        det = w_sum * wq2 - wq * wq
        valid = det > 1e-7
        safe_det = torch.where(valid, det, torch.ones_like(det))
        wls_scale = torch.where(valid, (w_sum * wqg - wq * wg) / safe_det, best_mx.float()).clamp(min=0.0).to(torch.float16)
        wls_bias = torch.where(valid, (wq2 * wg - wq * wqg) / safe_det, best_mn.float()).to(torch.float16)
        sf = wls_scale.float()
        bf = wls_bias.float()
        safe = torch.where(sf > 0, sf, torch.ones_like(sf))
        q_new = torch.round((g - bf[..., None]) / safe[..., None]).clamp(0, 15) * (sf[..., None] > 0)
        recon = q_new * sf[..., None] + bf[..., None]
        err = (((g - recon) ** 2) * cw).sum(-1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_mn = torch.where(better, wls_bias, best_mn)
        best_mx = torch.where(better, wls_scale, best_mx)
        best_q = torch.where(better[..., None], q_new, best_q)
        q = best_q

    # Upgrade 4: FP14 Scale/Bias Mantissa Rounding (zero bottom 4 noisy mantissa bits) + q Re-Snapping
    mx_u16 = best_mx.view(torch.uint16).to(torch.int32)
    mn_u16 = best_mn.view(torch.uint16).to(torch.int32)
    mx_r = ((mx_u16 + 8) & 0xFFF0).to(torch.uint16).view(torch.float16)
    mn_r = ((mn_u16 + 8) & 0xFFF0).to(torch.uint16).view(torch.float16)
    mx = torch.where(mx_r.float() > 0, mx_r, best_mx)
    mn = mn_r
    sf_f = mx.float()
    bf_f = mn.float()
    safe_f = torch.where(sf_f > 0, sf_f, torch.ones_like(sf_f))
    best_q = torch.round((g - bf_f[..., None]) / safe_f[..., None]).clamp(0, 15) * (sf_f[..., None] > 0)
    return best_q, mx, mn, base_err, best_err


def pack_int4_codes(q, mx, mn):
    """Engine INT4_G64 layout: codes two per byte (even column low nibble) + half2 (scale, bias) per group."""
    import numpy as np
    import torch

    codes = q.to(torch.uint8).cpu().numpy()
    packed_codes = (codes[..., 0::2] | (codes[..., 1::2] << 4)).tobytes()
    groups = np.stack([mx.cpu().numpy().view(np.uint16), mn.cpu().numpy().view(np.uint16)], axis=-1).tobytes()
    return packed_codes, groups


def gptq_int4_g64_torch(w, h):
    """GPTQ (Hessian h = E[x xᵀ] of the matrix input) into INT4_G64 (q, scale, bias); each group's grid is picked when reached."""
    import torch

    rows, cols = w.shape
    W = w.clone().float()
    # H = E[x xᵀ] is PSD by construction, but a rank-deficient activation set (or a dead input)
    # leaves zero eigenvalues that break the Cholesky; repair those and damp until it factors.
    H = h.to(W.device).clone().float()
    H = 0.5 * (H + H.T)
    diag = torch.diag(H)
    dead = diag <= 0
    if bool(dead.any()):
        live = diag[~dead]
        H[dead, dead] = live.mean() if live.numel() else 1.0
        W[:, dead] = 0.0
    scale = torch.diag(H).mean().clamp(min=1e-12)
    eye = torch.eye(cols, device=W.device)
    H += (0.01 * scale + 1e-9) * eye
    for extra in range(12):
        lower, info = torch.linalg.cholesky_ex(H)
        if int(info) == 0:
            break
        # Relative damping: the classic GPTQ floor, then geometric escalation for the null space.
        H += (10.0 ** (extra - 4)) * scale * eye
    else:
        raise ValueError("GPTQ Hessian could not be made positive-definite")
    # H⁻¹ = L⁻ᵀL⁻¹ with H = L Lᵀ; the upper factor of H⁻¹ is what the GPTQ recursion needs.
    Hinv_mat = torch.cholesky_inverse(lower)
    upper, info2 = torch.linalg.cholesky_ex(Hinv_mat, upper=True)
    if int(info2) == 0:
        Hinv = upper
    else:
        # Ill-conditioned inverse: any factor R with RᵀR = H⁻¹ works, so use the symmetric root.
        evals, evecs = torch.linalg.eigh(Hinv_mat)
        Hinv = evecs @ torch.diag(evals.clamp(min=1e-12).rsqrt()) @ evecs.T
    imp = torch.diag(H)
    Q = torch.zeros_like(W)
    mxs, mns = [], []
    for i1 in range(0, cols, 64):
        W1 = W[:, i1 : i1 + 64].clone()
        Hinv1 = Hinv[i1 : i1 + 64, i1 : i1 + 64]
        _, mx, mn, _, _ = int4_g64_grid_torch(W1[:, None, :], imp[i1 : i1 + 64].reshape(1, 1, 64))
        mxs.append(mx)
        mns.append(mn)
        sf, bf = mx.float()[:, 0], mn.float()[:, 0]
        safe = torch.where(sf > 0, sf, torch.ones_like(sf))
        Err1 = torch.zeros_like(W1)
        for i in range(64):
            q = torch.round((W1[:, i] - bf) / safe).clamp(0, 15) * (sf > 0)
            Q[:, i1 + i] = q
            err = (W1[:, i] - (q * sf + bf)) / Hinv1[i, i]
            W1[:, i:] -= err[:, None] * Hinv1[i, i:][None, :]
            Err1[:, i] = err
        W[:, i1 + 64 :] -= Err1 @ Hinv[i1 : i1 + 64, i1 + 64 :]
    QUANT_STATS["tensors"] += 1
    return Q.reshape(rows, cols // 64, 64), torch.cat(mxs, 1), torch.cat(mns, 1)


def int4_dequant(q, mx, mn):
    """Engine INT4_G64 weights q·scale + bias, [rows, cols] float32."""
    return (q.float() * mx.float()[..., None] + mn.float()[..., None]).reshape(q.shape[0], -1)


def awq_int4_g64_grid(w, col_weights=None):
    """The AWQ+WLS INT4_G64 grid (q, scale, bias) of `w`: the exact grid pack_int4_g64_torch picks.

    The same clip search and WLS refinement on the same column weights, so the QAT seed is the grid
    the plain int4 artifact uses, not a different one."""
    import torch

    rows, cols = w.shape
    assert rows <= 32768 and cols % 64 == 0, (rows, cols)
    g = w.reshape(rows, cols // 64, 64)
    if col_weights is None:
        cw = torch.ones((1, 1, 64), device=w.device, dtype=torch.float32)
    else:
        cw = col_weights.to(device=w.device, dtype=torch.float32).reshape(1, cols // 64, 64)
    q, mx, mn, base_err, best_err = int4_g64_grid_torch(g, cw)
    QUANT_STATS["tensors"] += 1
    QUANT_STATS["base_mse"] += float(base_err.double().sum().item())
    QUANT_STATS["opt_mse"] += float(best_err.double().sum().item())
    return q, mx, mn


def layer_awq_cw(k: str, cw_map: dict):
    """The AWQ column weights of one decoder matrix, the same ones build_artifacts uses.

    `k` is a `model.language_model.layers.<i>....weight` key and `cw_map` holds the layernorm
    column weights by their `layers.<i>....weight` key (the namespace of the language model)."""
    if k.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight", "in_proj_qkv.weight", "in_proj_z.weight")):
        suffix = "input_layernorm.weight"
    elif k.endswith(("gate_proj.weight", "up_proj.weight")):
        suffix = "post_attention_layernorm.weight"
    else:
        return None
    prefix = ".".join(k.split(".")[:4])  # model.language_model.layers.<i>
    return cw_map.get(f"layers.{prefix.split('.')[-1]}.{suffix}")


def fineweb_docs(tok):
    """fineweb-edu documents (never eval data), tokenized, each ending in EOS."""
    import datasets

    for doc in datasets.load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True):
        yield tok(doc["text"]).input_ids + [tok.eos_token_id]


def fineweb_ids(tok, ntokens: int) -> list[int]:
    """Calibration token stream: fineweb-edu documents joined."""
    ids = []
    for d in fineweb_docs(tok):
        ids += d
        if len(ids) >= ntokens:
            return ids
    raise ValueError("fineweb-edu ran out of tokens")


def glaive_objects(text: str) -> list[str]:
    """The JSON objects of glaive's `<functioncall> {...}` replies.

    A regex cannot do this: the arguments are a single-quoted JSON string, so the first `}` ends the
    object only by accident. This scans for balanced braces outside the double-quoted strings.

    Only double quotes are tracked. The single quotes wrap the arguments, and the data does not
    escape the apostrophes inside them ("aujourd'hui"), so tracking them would end the wrapper
    early and lose the object."""
    objects, i = [], 0
    while True:
        i = text.find("{", i)
        if i < 0:
            return objects
        depth, j, quoted = 0, i, False
        while j < len(text):
            ch = text[j]
            if quoted:
                if ch == "\\":
                    j += 2
                    continue
                if ch == '"':
                    quoted = False
            elif ch == '"':
                quoted = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    objects.append(text[i:j + 1])
                    break
            j += 1
        i = j + 1


def glaive_call_args(raw: str) -> tuple[str, dict] | None:
    """The (name, arguments) of one glaive `<functioncall>` object, or None when it is not a call.

    The usual form is not valid JSON as written: the arguments are a JSON string inside single
    quotes, so `json.loads` fails on the whole object and the two fields are read separately.
    """
    import json
    try:
        obj = json.loads(raw)
    except ValueError:
        obj = None
    if isinstance(obj, dict) and isinstance(obj.get("name"), str):
        args = obj.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                return None
        if isinstance(args, dict):
            return obj["name"], args
        return None
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', raw)
    args_match = re.search(r'"arguments"\s*:\s*', raw)
    if not name_match or not args_match:
        return None
    rest = raw[args_match.end():].lstrip()
    if not rest.startswith("'"):
        return None
    # The wrapper ends at the last single quote before the object's closing brace: an apostrophe
    # inside the arguments ("aujourd'hui") is not escaped, so the first quote is not the end.
    tail = rest.rfind("'")
    if tail <= 0:
        return None
    inner = rest[1:tail]
    try:
        # The wrapper is single-quoted, so an escaped apostrophe inside it is \', which JSON
        # does not accept.
        args = json.loads(inner.replace("\\'", "'"))
    except ValueError:
        return None
    if not isinstance(args, dict):
        return None
    return name_match.group(1), args


def glaive_tools(system):
    """The tool schemas of one glaive `system` turn: the JSON objects it holds."""
    import json

    dec, tools, i = json.JSONDecoder(), [], system.find("{")
    while i >= 0:
        obj, i = dec.raw_decode(system, i)
        tools.append({"type": "function", "function": obj})
        i = system.find("{", i)
    return tools


def glaive_call(chat, tools):
    """The glaive reply as the model's own tool-call turn: `<tool_call><function=NAME><parameter=…>`
    blocks. None when the reply calls no function.

    The glaive reply is `<functioncall> {"name": "f", "arguments": '{"k": v}'} <|endoftext|>`: the
    arguments are a JSON string inside single quotes, so the object is not valid JSON as written.
    """
    user = lambda text: [{"role": "user", "content": text}]
    # Only the first ASSISTANT turn belongs to this prompt; later turns are another exchange.
    reply = chat.split("ASSISTANT:", 1)[1].split("USER:", 1)[0].strip() if "ASSISTANT:" in chat else ""
    # The reply ends at the end-of-turn marker; what follows is the function's answer and the
    # next assistant turn, which are not part of this reply.
    reply = reply.split("<|endoftext|>")[0].strip()
    calls = []
    # Only the objects inside a `<functioncall>` marker are calls; the FUNCTION RESPONSE that
    # follows is a JSON object too, but it has no "name".
    marked = [obj for part in reply.split("<functioncall>")[1:] for obj in glaive_objects(part)]
    for raw in marked:
        parsed = glaive_call_args(raw)
        if parsed is not None:
            calls.append({"name": parsed[0], "arguments": parsed[1]})
    if not calls:
        return None
    prompt = user(chat.split("USER:", 1)[1].split("ASSISTANT:", 1)[0].strip())
    turn = {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]}
    return prompt + [turn]


def tool_only_ids(tok, ntokens: int, tool_rows: int = 3000, log=print) -> list[int]:
    """Calibration stream of glaive function calling only, topped up with fineweb-edu.

    The chat/math/code rows of chat_mix_ids are what taught the model to answer tool prompts in
    prose: on the 20-prompt probe every mix variant called 13-15 tools while the fineweb-only ones
    called 15-19. This stream keeps only the tool rows (the dataset's own function call as the
    reply, in the model's `<tool_call>` format) and fineweb-edu, nothing else."""
    import itertools
    import random
    import datasets

    def stream(name, split, cfg=None):
        return datasets.load_dataset(name, cfg, split=split, streaming=True).shuffle(seed=0, buffer_size=10_000)

    tool_convs = [("tools", glaive_tools(r["system"]), glaive_call(r["chat"], glaive_tools(r["system"])))
                  for r in itertools.islice((r for r in stream("glaiveai/glaive-function-calling-v2", "train")
                                             if "{" in r["system"] and "USER:" in r["chat"]), tool_rows)]
    call_convs = [c for c in tool_convs if c[2] and c[2][-1].get("tool_calls")]
    if len(call_convs) < tool_rows // 4:
        raise ValueError(f"only {len(call_convs)} tool calls parsed from {tool_rows} glaive rows")
    log("tool rows:", len(tool_convs), "of which calls:", len(call_convs))
    docs, made = [], {}
    for kind, tools, ref in call_convs:
        d = tok(tok.apply_chat_template(ref, tools=tools, tokenize=False), add_special_tokens=False).input_ids
        docs.append(d)
        made[kind] = made.get(kind, 0) + len(d)
    # Fineweb-edu tops the stream up to the requested size.
    for d in fineweb_docs(tok):
        if made.get("fineweb", 0) >= max(ntokens - sum(made.values()), ntokens * 15 // 100):
            break
        docs.append(d)
        made["fineweb"] = made.get("fineweb", 0) + len(d)
    log("calibration tokens:", made)
    random.Random(0).shuffle(docs)
    return [t for d in docs for t in d]


def chat_mix_ids(model, tok, ntokens: int, per_source: int = 300, gen_batch: int = 8, gen_budget_s: float = 2400,
                 tool_rows: int = 600, log=print) -> list[int]:
    """Calibration stream in the model's chat template, train splits only (never eval data): ultrachat_200k chat,
    gsm8k-train and MetaMathQA math, Magicoder-Evol code, glaive function calling rendered with `tools`, topped up to
    >= 15% fineweb-edu. The tool prompts keep the dataset's own function call as their reply, rendered in the model's
    tool-call format; half of the other prompts get the BF16 model's own sampled reply (a third of those with thinking
    on), the rest keep the dataset reply (thinking off). Documents are shuffled and joined.
    `tool_rows` > `per_source`: the tool rows are short (one call each) but they carry the whole
    function-calling behaviour, so they need more rows than a chat conversation to weigh the same. Only
    the glaive rows that really call a function are kept (about half of them)."""
    import itertools
    import random
    import re
    import time
    import datasets
    import torch

    def stream(name, split, cfg=None):
        return datasets.load_dataset(name, cfg, split=split, streaming=True).shuffle(seed=0, buffer_size=10_000)

    user = lambda text: [{"role": "user", "content": text}]
    pair = lambda q, a: user(q) + [{"role": "assistant", "content": a}]
    take = lambda it: itertools.islice(it, per_source)

    # Tool prompts keep the dataset's own function call as the reference: sampling the BF16 model's
    # reply teaches it to answer tool prompts in prose (the shipped artifact then calls tools 68/80
    # on BFCL, a prose-calibrated one only 30/80).
    tool_convs = [("tools", user(r["chat"].split("USER:", 1)[1].split("ASSISTANT:", 1)[0].strip()),
                   glaive_tools(r["system"]), glaive_call(r["chat"], glaive_tools(r["system"])))
                  for r in itertools.islice((r for r in stream("glaiveai/glaive-function-calling-v2", "train")
                                             if "{" in r["system"] and "USER:" in r["chat"]), tool_rows)]
    # Only the rows that really call a function are kept. The rest of the dataset answers in prose
    # ("which product do you mean?"), and 42% of them taught the model to ask instead of call:
    # with them, BFCL AST accuracy fell to 37.9% against the 62.9% of a calibration without tool
    # data at all.
    call_convs = [c for c in tool_convs if c[3] and c[3][-1].get("tool_calls")]
    if len(call_convs) < tool_rows // 2:
        raise ValueError(f"only {len(call_convs)} tool calls parsed from {tool_rows} glaive rows")
    log("tool rows:", len(tool_convs), "of which calls:", len(call_convs))
    convs = call_convs
    convs += [("chat", r["messages"][:1], None, r["messages"][:6]) for r in take(stream("HuggingFaceH4/ultrachat_200k", "train_sft"))]
    convs += [("gsm8k", user(r["question"]), None, pair(r["question"], re.sub(r"<<[^>]*>>", "", r["answer"])))
              for r in take(stream("openai/gsm8k", "train", "main"))]
    convs += [("math", user(r["query"]), None, pair(r["query"], r["response"])) for r in take(stream("meta-math/MetaMathQA", "train"))]
    convs += [("code", user(r["instruction"]), None, pair(r["instruction"], r["response"]))
              for r in take(stream("ise-uiuc/Magicoder-Evol-Instruct-110K", "train"))]
    # (kind, prompt, tools, reference conversation or None, sample a reply, thinking on); tools first, thinking last.
    # Tool prompts keep their reference (the function call itself) instead of a sampled reply.
    convs = [(k, p, t, ref, k != "tools" and (ref is None or i % 2 == 0), ref is not None and i % 6 == 0)
             for i, (k, p, t, ref) in enumerate(convs)]
    convs.sort(key=lambda c: (not c[4], c[5]))

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    docs, made, t0 = [], {}, time.time()

    def add(kind, d):
        if kind not in made:
            log("calibration sample", kind, repr(tok.decode(d)[-1500:]))
        docs.append(d)
        made[kind] = made.get(kind, 0) + len(d)

    gen = [c for c in convs if c[4]]
    for b in range(0, len(gen), gen_batch):
        batch = gen[b : b + gen_batch]
        if time.time() - t0 > gen_budget_s and batch[0][3] is not None:
            convs += [(k, p, t, ref, False, False) for k, p, t, ref, _, _ in gen[b:]]  # out of time: dataset replies
            break
        texts = [tok.apply_chat_template(p, tools=t, add_generation_prompt=True, enable_thinking=think, tokenize=False)
                 for _, p, t, _, _, think in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, do_sample=True, temperature=0.7, top_p=0.95, pad_token_id=tok.pad_token_id,
                                 max_new_tokens=1024 if batch[0][5] else 512)
        for c, ids, mask, o in zip(batch, enc.input_ids, enc.attention_mask, out[:, enc.input_ids.shape[1]:].tolist()):
            end = next((j + 1 for j, t in enumerate(o) if t in eos), len(o))
            add(c[0] + ("-think" if c[5] else "-gen"), ids[mask.bool()].tolist() + o[:end])
        log("calibration: sampled", b + len(batch), "/", len(gen), f"replies, {time.time() - t0:.0f}s")
    for k, _, t, ref, g, _ in convs:
        if not g:
            add(k, tok(tok.apply_chat_template(ref, tools=t, tokenize=False), add_special_tokens=False).input_ids)
    need = max(ntokens - sum(made.values()), ntokens * 15 // 100)
    for d in fineweb_docs(tok):
        if made.get("fineweb", 0) >= need:
            break
        add("fineweb", d)
    log("calibration tokens:", made)
    random.Random(0).shuffle(docs)
    return [t for d in docs for t in d]


def collect_hessians(model, ids: list[int], nsamples: int = 128, seqlen: int = 2048) -> dict:
    """E[x xᵀ] of every INT4 decoder matrix input, from the BF16 model on the calibration stream."""
    import torch

    device = model.device
    hess, counts, hooks = {}, {}, []
    # fp32 accumulators on the host: an fp16 GPU accumulator overflows at 65504 on the raw token
    # sums (2048 tokens x 128 samples), and 150 fp32 GPU Hessians (1.8 GB) do not fit next to the
    # fp32 model on a 15 GB T4. The per-sample xᵀx matmul stays on the GPU.
    for name, mod in model.named_modules():
        if (isinstance(mod, torch.nn.Linear) and name.startswith("model.language_model.layers.")
                and not name.endswith(("in_proj_a", "in_proj_b"))):
            key = name + ".weight"
            hess[key] = torch.zeros(mod.in_features, mod.in_features, dtype=torch.float32)
            counts[key] = 0
            def hook(_m, inp, _out, key=key):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                hess[key] += (x.T @ x).cpu()
                counts[key] += x.shape[0]
            hooks.append(mod.register_forward_hook(hook))
    with torch.no_grad():
        for s in range(nsamples):
            model(input_ids=torch.tensor([ids[s * seqlen : (s + 1) * seqlen]], device=device))
            if s % 16 == 15:
                empty_cache(device)
    for h in hooks:
        h.remove()
    # E[x xᵀ] over the tokens each matrix actually saw.
    return {k: v / max(counts[k], 1) for k, v in hess.items()}


def qat_int4_g64(lm, head, hessians: dict, ids: list[int], nsamples: int, epochs: int, batch: int = 2,
                 lr_w: float = 3e-6, lr_q: float = 3e-6, nval: int = 8, seqlen: int = 2048, rotate: bool = False,
                 nlayers: int | None = None, seed: str = "gptq", log=print):
    """Block-wise QAT (EfficientQAT Block-AP) of the text model `lm`: train each decoder layer's INT4_G64
    weights (straight-through rounding) and fp16 (scale, bias) (exact gradient through the current codes) on the
    engine grid, to match the BF16 layer's output given the output of the already-quantized layers before it (the
    student embeds through the tied INT4 `head` like the engine). Adam, cosine lr to 1/20; lrs from a 4-layer sweep
    (scale/bias carry the gain, 1e-5 starts to overfit; weight lrs >=5e-6 flipped codes at random and raised the loss).
    The last `nval` samples are held out and a layer whose held-out MSE got worse keeps the seed.
    `seed` is the grid QAT starts from: "gptq" (the Hessian-based grid) or "awq" (the clip-search + WLS grid
    the plain int4 artifact uses). The AWQ seed keeps the tool-call behaviour of the plain int4 artifact
    (BFCL AST 62.5% and 1/10 false tool calls) while the GPTQ seed drifts away from it (56.0% and 4/10).
    Returns ({key: seed (q, scale, bias)}, {key: trained (q, scale, bias)}, INT4 head, per-layer stats); with
    `rotate` the grids of 1024-wide matrices are for W·M like add_int4_g64(rotate=True). `lm` is left holding the
    trained dequantized weights."""
    import math
    import time
    import torch
    import torch.nn.functional as F
    from torch.nn.utils import parametrize

    class FakeInt4G64(torch.nn.Module):
        def __init__(self, mx, mn, rot):
            super().__init__()
            self.scale = torch.nn.Parameter(mx.float())
            self.bias = torch.nn.Parameter(mn.float())
            self.rot = rot

        def grid(self, w):
            s = self.scale + (self.scale.half().float() - self.scale).detach()  # the artifact stores fp16
            b = self.bias + (self.bias.half().float() - self.bias).detach()
            live = (s > 0)[..., None]
            # codes see (scale, bias) as constants: LSQ's gradient through round() is zero at the GPTQ init, where every
            # weight sits on its grid point
            x = (w.reshape(*s.shape, 64) - b.detach()[..., None]) / torch.where(live, s[..., None], torch.ones_like(s[..., None])).detach()
            return (x + (x.round() - x).detach()).clamp(0, 15) * live, s, b

        def forward(self, w):
            q, s, b = self.grid(w)
            w = (q * s[..., None] + b[..., None]).reshape(w.shape)
            return w if self.rot is None else w @ self.rot.T

    dev = head.device
    rot = rotate512_torch(torch.eye(1024, device=dev)) if rotate else None
    # The AWQ column weights of the decoder layernorms, by the language model's own parameter names.
    cw_map = {n: ((p.detach().float().reshape(-1) + 1.0) ** 2).clamp(min=1e-4)
              for n, p in lm.named_parameters()
              if n.endswith(("input_layernorm.weight", "post_attention_layernorm.weight"))}
    data = torch.tensor(ids[: nsamples * seqlen]).reshape(nsamples, seqlen)
    ntrain = nsamples - nval
    assert ntrain % batch == 0 and nval % batch == 0 and nval > 0

    kw = {}  # each layer's rotary/mask kwargs for a [batch, seqlen] input
    hooks = [l.register_forward_pre_hook(lambda _m, _a, k, i=i: kw.__setitem__(i, k), with_kwargs=True)
             for i, l in enumerate(lm.layers)]
    with torch.no_grad():
        lm(input_ids=data[:batch].to(dev), use_cache=False)
        g2 = lambda p: ((p.detach().float().reshape(-1) + 1.0) ** 2).clamp(min=1e-4)  # AWQ weights as in build_artifacts
        cn, c0 = g2(lm.norm.weight), g2(lm.layers[0].input_layernorm.weight)
        cw = (0.5 * cn / cn.mean() + 0.5 * c0 / c0.mean()).reshape(1, -1, 64)
        head_q = torch.cat([int4_dequant(*int4_g64_grid_torch(head[r : r + 32768].float().reshape(-1, cw.shape[1], 64), cw)[:3])
                            for r in range(0, head.shape[0], 32768)])
        xfp = torch.empty(nsamples, seqlen, head.shape[1])
        xq, y = torch.empty_like(xfp), torch.empty_like(xfp)
        for n in range(nsamples):
            xfp[n] = lm.embed_tokens(data[n].to(dev)).cpu()
            xq[n] = head_q[data[n].to(dev)].cpu()
    for h in hooks:
        h.remove()

    def run(layer, i, x, idx, grad=False):  # fp32: fp16 autocast was no faster on T4 (the DeltaNet fallback dominates)
        with torch.set_grad_enabled(grad):
            return layer(x[idx].to(dev), **kw[i])

    lm.requires_grad_(False)
    gptq, qat, stats = {}, {}, []
    for i, layer in enumerate(lm.layers[:nlayers]):
        t0 = time.time()
        for n in range(0, nsamples, batch):
            y[n : n + batch] = run(layer, i, xfp, slice(n, n + batch)).cpu()
        mods = {f"model.language_model.layers.{i}.{name}.weight": m for name, m in layer.named_modules()
                if isinstance(m, torch.nn.Linear) and not name.endswith(("in_proj_a", "in_proj_b"))}
        for k, m in mods.items():
            r = rot if m.in_features == 1024 else None
            # QAT re-runs GPTQ from CPU Hessians; plain GPTQ consumes them straight from the GPU.
            h = hessians[k].to(dev) if hessians[k].device.type == "cpu" else hessians[k]
            w0, h = (m.weight.detach(), h) if r is None else (m.weight.detach() @ r, r.T @ h @ r)
            if seed == "awq":
                # The clip-search + WLS grid of the plain int4 artifact (the AWQ column weights of this matrix).
                cw = layer_awq_cw(k, cw_map)
                q, mx, mn = awq_int4_g64_grid(w0, col_weights=cw)
            else:
                q, mx, mn = gptq_int4_g64_torch(w0, h)
            gptq[k] = (q.to(torch.uint8).cpu(), mx.cpu(), mn.cpu())
            parametrize.register_parametrization(m, "weight", FakeInt4G64(mx, mn, r))
            w = m.parametrizations.weight.original
            with torch.no_grad():
                w.copy_(int4_dequant(q, mx, mn))
                assert torch.equal(m.parametrizations.weight[0].grid(w)[0], q), k  # QAT starts exactly at the seed
            w.requires_grad_(True)
        t1 = time.time()
        fqs = [m.parametrizations.weight[0] for m in mods.values()]
        opt = torch.optim.Adam([{"params": [m.parametrizations.weight.original for m in mods.values()], "lr": lr_w},
                                {"params": [p for f in fqs for p in f.parameters()], "lr": lr_q}])
        steps = epochs * ntrain // batch
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.05 + 0.475 * (1 + math.cos(math.pi * s / steps)))
        val = lambda: sum(F.mse_loss(run(layer, i, xq, slice(n, n + batch)), y[n : n + batch].to(dev)).item()
                          for n in range(ntrain, nsamples, batch)) * batch / nval
        v0, train = val(), 0.0
        for _ in range(epochs):
            for idx in torch.randperm(ntrain).reshape(-1, batch):
                loss = F.mse_loss(run(layer, i, xq, idx, grad=True), y[idx].to(dev))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                train += loss.item() / steps
        v1 = val()
        if not v1 < v0:
            with torch.no_grad():
                for (k, m), f in zip(mods.items(), fqs):
                    m.parametrizations.weight.original.copy_(int4_dequant(*gptq[k]))
                    f.scale.copy_(gptq[k][1].float())
                    f.bias.copy_(gptq[k][2].float())
        with torch.no_grad():
            for (k, m), f in zip(mods.items(), fqs):
                w_orig = m.parametrizations.weight.original
                _, s, b = f.grid(w_orig)
                mx_h, mn_h = s.clamp(min=0).half(), b.half()
                mx_r = ((mx_h.view(torch.uint16).to(torch.int32) + 8) & 0xFFF0).to(torch.uint16).view(torch.float16)
                mn_r = ((mn_h.view(torch.uint16).to(torch.int32) + 8) & 0xFFF0).to(torch.uint16).view(torch.float16)
                mx_f = torch.where(mx_r.float() > 0, mx_r, mx_h)
                mn_f = mn_r
                sf, bf = mx_f.float(), mn_f.float()
                safe = torch.where(sf > 0, sf, torch.ones_like(sf))
                q = torch.round((w_orig.reshape(*sf.shape, 64) - bf[..., None]) / safe[..., None]).clamp(0, 15) * (sf[..., None] > 0)
                qat[k] = (q.to(torch.uint8).cpu(), mx_f.cpu(), mn_f.cpu())
                f.scale.copy_(sf)
                f.bias.copy_(bf)
                w_orig.copy_(int4_dequant(q, mx_f, mn_f))
                parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)
                m.weight.requires_grad_(False)
            for n in range(0, nsamples, batch):
                xq[n : n + batch] = run(layer, i, xq, slice(n, n + batch)).cpu()
        stats.append({"layer": i, "type": layer.block_type, "val_mse_seed": v0, "val_mse_qat": v1, "train_mse": train,
                      "seed": seed, "kept": "qat" if v1 < v0 else seed, "secs_prep": round(t1 - t0, 1),
                      "secs_train": round(time.time() - t1, 1), "target_rms": float(y[ntrain:].pow(2).mean().sqrt())})
        log("QAT", stats[-1])
        xfp, y = y, xfp
    return gptq, qat, head_q, stats



def train_zstd_dict(shuf: bytes, dict_size: int = 32768) -> bytes:
    import shutil, subprocess, tempfile

    chunk_size = 4096
    n = len(shuf)
    if n < chunk_size:
        return b""
    max_samples = min(512, n // chunk_size)
    if max_samples < 8:
        return b""
    step = (n - chunk_size) // max(1, max_samples - 1)
    samples = [shuf[i * step : i * step + chunk_size] for i in range(max_samples)]
    if shutil.which("zstd"):
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            sample_files = []
            for idx, s in enumerate(samples):
                sp = td_p / f"s_{idx:03d}.bin"
                sp.write_bytes(s)
                sample_files.append(str(sp))
            dp = td_p / "dict.bin"
            res = subprocess.run(
                ["zstd", "--train", "-q", *sample_files, "-o", str(dp), f"--maxdict={dict_size}"],
                capture_output=True,
            )
            if res.returncode == 0 and dp.exists():
                return dp.read_bytes()
    try:
        import zstandard as zstd_mod

        return zstd_mod.train_dictionary(dict_size, samples).as_bytes()
    except Exception:
        return b""


class ArtifactBuilder:
    def __init__(self, magic: bytes, fmt: str, config: dict):
        self.magic = magic
        self.fmt = fmt
        self.config = config
        self.codes_region = bytearray()
        self.aux_region = bytearray()
        self.tensors = {}

    @staticmethod
    def _align_buf(buf: bytearray, align: int = TENSOR_ALIGN):
        rem = len(buf) % align
        if rem:
            buf.extend(b"\x00" * (align - rem))

    def add_ptq1(self, name: str, w, rotate: bool = False):
        self._align_buf(self.codes_region)
        data = pack_ptq1_rows_torch(w, rotate=rotate)
        offset = len(self.codes_region)
        self.codes_region.extend(data)
        self.tensors[name] = {
            "dtype": "ptq1",
            "shape": [int(w.shape[0]), int(w.shape[1])],
            "offset": offset,
            "length": len(data),
            "aux_offset": 0,
            "aux_length": 0,
            "rotated": rotate,
        }

    def add_int4_g64(self, name: str, w, col_weights=None, hessian=None, rotate: bool = False, grid=None):
        if rotate:  # quantize W·M for inputs the engine rotates by M (signed FWHT-512); E[x'x'ᵀ] = Mᵀ H M
            import torch

            m = rotate512_torch(torch.eye(w.shape[1], device=w.device))
            w = w @ m
            hessian = None if hessian is None else m.T @ hessian.to(w.device) @ m
            col_weights = None
        self._align_buf(self.codes_region)
        if grid is not None:
            codes, groups = pack_int4_codes(*grid)
        elif hessian is not None:
            codes, groups = pack_int4_codes(*gptq_int4_g64_torch(w, hessian))
        else:
            codes, groups = pack_int4_g64_torch(w, col_weights=col_weights)
        offset = len(self.codes_region)
        self.codes_region.extend(codes)
        self._align_buf(self.aux_region)
        aux_offset = len(self.aux_region)
        self.aux_region.extend(groups)
        self.tensors[name] = {
            "dtype": "int4_g64",
            "shape": [int(w.shape[0]), int(w.shape[1])],
            "offset": offset,
            "length": len(codes),
            "aux_offset": aux_offset,
            "aux_length": len(groups),
            "rotated": rotate,
        }

    def add_bf16(self, name: str, w, rotate: bool = False):
        import numpy as np
        import torch

        if rotate:
            w = rotate512_torch(w)
        self._align_buf(self.aux_region)
        bf = w.to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().tobytes()
        offset = len(self.aux_region)
        self.aux_region.extend(bf)
        shape = [int(w.shape[0]), int(w.shape[1])] if w.ndim == 2 else [1, int(w.numel())]
        self.tensors[name] = {
            "dtype": "bf16",
            "shape": shape,
            "offset": offset,
            "length": len(bf),
            "aux_offset": 0,
            "aux_length": 0,
            "rotated": rotate,
        }

    def add_int8_row(self, name: str, w):
        """Engine INT8_ROW: symmetric per-row int8 weights [rows, cols] + F32 scales [rows] (both in the aux region)."""
        import torch

        scale = (w.abs().amax(1) / 127.0).clamp(min=1e-12)
        q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
        self._align_buf(self.aux_region)
        offset = len(self.aux_region)
        self.aux_region.extend(q.contiguous().cpu().numpy().tobytes())
        self._align_buf(self.aux_region)
        aux_offset = len(self.aux_region)
        self.aux_region.extend(scale.float().contiguous().cpu().numpy().tobytes())
        self.tensors[name] = {
            "dtype": "int8_row",
            "shape": [int(w.shape[0]), int(w.shape[1])],
            "offset": offset,
            "length": int(w.numel()),
            "aux_offset": aux_offset,
            "aux_length": int(w.shape[0]) * 4,
            "rotated": False,
        }

    def add_f32(self, name: str, w):
        import torch

        self._align_buf(self.aux_region)
        f = w.float().contiguous().cpu().numpy().tobytes()
        offset = len(self.aux_region)
        self.aux_region.extend(f)
        shape = [int(w.shape[0]), int(w.shape[1])] if w.ndim == 2 else [1, int(w.numel())]
        self.tensors[name] = {
            "dtype": "f32",
            "shape": shape,
            "offset": offset,
            "length": len(f),
            "aux_offset": 0,
            "aux_length": 0,
            "rotated": False,
        }

    # zstd off: the engine's zstd_col_delta decoder also undoes a bias prediction this encoder lacks, and fuse stores raw anyway.
    def write(self, path: Path, zstd_level: int = 0, dict_bytes: bytes | None = None) -> dict:
        import numpy as np, shutil, subprocess, tempfile

        self._align_buf(self.codes_region, TENSOR_ALIGN)
        aux_base = len(self.codes_region)
        final_tensors = {}
        for name, td in self.tensors.items():
            ntd = dict(td)
            if ntd["dtype"] in ("int4_g64", "ptq1"):
                if ntd["aux_length"] > 0:
                    ntd["aux_offset"] += aux_base
            else:
                ntd["offset"] += aux_base
                if ntd["aux_length"] > 0:
                    ntd["aux_offset"] += aux_base
            final_tensors[name] = ntd

        if (
            "model.language_model.embed_tokens.weight" in final_tensors
            and "lm_head.weight" in final_tensors
        ):
            emb_td = final_tensors["model.language_model.embed_tokens.weight"]
            head_td = final_tensors["lm_head.weight"]
            if (
                emb_td["dtype"] == head_td["dtype"]
                and emb_td["shape"] == head_td["shape"]
                and emb_td["offset"] != head_td["offset"]
            ):
                emb_td["delta_from"] = "lm_head.weight"

        payload = bytearray(self.codes_region)
        payload.extend(self.aux_region)
        self._align_buf(payload, METAL_PAGE)
        sha = hashlib.sha256(payload).hexdigest()
        payload_len = len(payload)

        compression = None
        if zstd_level > 0:
            enc_buf = bytearray(payload)
            # 1. Lossless byte-delta for correlated tensors (e.g. embed_tokens - lm_head)
            for td in final_tensors.values():
                base_name = td.get("delta_from")
                if base_name and base_name in final_tensors:
                    btd = final_tensors[base_name]
                    if td["length"] == btd["length"] and td["offset"] != btd["offset"]:
                        a = np.frombuffer(payload, dtype=np.uint8, count=td["length"], offset=td["offset"])
                        b = np.frombuffer(payload, dtype=np.uint8, count=btd["length"], offset=btd["offset"])
                        enc_buf[td["offset"] : td["offset"] + td["length"]] = ((a - b) & 0xFF).tobytes()
                    if td["aux_length"] > 0 and td["aux_length"] == btd["aux_length"] and td["aux_offset"] != btd["aux_offset"]:
                        aa = np.frombuffer(payload, dtype=np.uint8, count=td["aux_length"], offset=td["aux_offset"])
                        bb = np.frombuffer(payload, dtype=np.uint8, count=btd["aux_length"], offset=btd["aux_offset"])
                        enc_buf[td["aux_offset"] : td["aux_offset"] + td["aux_length"]] = ((aa - bb) & 0xFF).tobytes()
            # 2. Pre-invert byte_shuffle4 on delta codes and column-transposed vocabulary codes (rows >= 65536)
            def unshuf4_np(slice_bytes: bytes) -> bytes:
                n4 = len(slice_bytes) // 4
                a4 = np.frombuffer(slice_bytes, dtype=np.uint8).reshape(4, n4)
                return np.ascontiguousarray(a4.T).tobytes()

            for td in final_tensors.values():
                rows, cols = td["shape"]
                if td["dtype"] == "int4_g64":
                    rb = cols // 2
                    if td.get("delta_from"):
                        enc_buf[td["offset"] : td["offset"] + td["length"]] = unshuf4_np(
                            bytes(enc_buf[td["offset"] : td["offset"] + td["length"]])
                        )
                    elif rows >= 65536 and rows * rb == td["length"]:
                        mat = np.frombuffer(enc_buf, dtype=np.uint8, count=td["length"], offset=td["offset"]).reshape(rows, rb)
                        enc_buf[td["offset"] : td["offset"] + td["length"]] = unshuf4_np(
                            np.ascontiguousarray(mat.T).tobytes()
                        )
                elif td["dtype"] == "ptq1" and rows >= 65536 and cols % 128 == 0:
                    rb = (cols // 128) * 28
                    if rows * rb == td["length"]:
                        mat = np.frombuffer(enc_buf, dtype=np.uint8, count=td["length"], offset=td["offset"]).reshape(rows, rb)
                        enc_buf[td["offset"] : td["offset"] + td["length"]] = unshuf4_np(
                            np.ascontiguousarray(mat.T).tobytes()
                        )
            # 3. Global 4-plane byte_shuffle4 across the entire payload
            arr = np.frombuffer(enc_buf, dtype=np.uint8).reshape(-1, 4)
            shuf = np.concatenate([arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]]).tobytes()
            del enc_buf
            if dict_bytes is None:
                dict_bytes = train_zstd_dict(shuf) if (65536 <= len(shuf) < 16 * 1024 * 1024) else b""
            if shutil.which("zstd"):
                with tempfile.TemporaryDirectory() as td_dir:
                    raw_p = Path(td_dir) / "payload.bin"
                    out_p = Path(td_dir) / "payload.zst"
                    raw_p.write_bytes(shuf)
                    cmd = ["zstd", "--ultra", f"-{zstd_level}", "--long=27", "-T0", "-q", "-f", str(raw_p), "-o", str(out_p)]
                    if dict_bytes:
                        dp = Path(td_dir) / "dict.bin"
                        dp.write_bytes(dict_bytes)
                        cmd[5:5] = ["-D", str(dp)]
                    subprocess.run(cmd, check=True)
                    body = dict_bytes + out_p.read_bytes()
                    compression = "zstd_col_delta"
            else:
                try:
                    import zstandard as zstd_mod
                    dict_data = zstd_mod.ZstdCompressionDict(dict_bytes) if dict_bytes else None
                    cparams = zstd_mod.ZstdCompressionParameters.from_level(
                        min(zstd_level, 22), enable_ldm=True, window_log=27
                    )
                    cctx = zstd_mod.ZstdCompressor(compression_params=cparams, dict_data=dict_data, threads=-1)
                    body = dict_bytes + cctx.compress(shuf)
                    compression = "zstd_col_delta"
                except ImportError:
                    try:
                        from compression import zstd as py_zstd
                        dict_bytes = b""
                        body = py_zstd.compress(shuf, level=min(zstd_level, 19))
                        compression = "zstd_col_delta"
                    except ImportError:
                        dict_bytes = b""
                        body = bytes(payload)
                        compression = None
        else:
            dict_bytes = b""
            body = bytes(payload)
        header = {
            "format": self.fmt,
            "version": 1,
            "config": self.config,
            "tensors": final_tensors,
            "payload_length": payload_len,
            "payload_sha256": sha,
            "compression": compression,
            "zstd_dict_length": len(dict_bytes) if compression else 0,
        }
        header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
        payload_start = ((16 + len(header_bytes) + METAL_PAGE - 1) // METAL_PAGE) * METAL_PAGE
        pad_len = payload_start - 16 - len(header_bytes)
        with path.open("wb") as f:
            f.write(self.magic)
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            f.write(b" " * pad_len)
            f.write(body)
        return {
            "file": path.name,
            "file_bytes": payload_start + len(body),
            "payload_bytes": payload_len,
            "payload_sha256": sha,
            "compression": compression,
        }


def build_artifacts(
    st_1b_path: Path,
    st_dsp_path: Path | None,
    out_dir: Path,
    device: str = "cuda",
    int4_head: bool = True,
    high_quality: bool = False,
    rotate_all_ptq1: bool = False,
    lm_only: bool = False,
    fuse_all: bool = True,
    decoder_format: str = "int4",
    hessians: dict | None = None,
    grids: dict | None = None,
):
    import torch
    from safetensors import safe_open

    out_dir.mkdir(parents=True, exist_ok=True)
    reset_quant_stats()
    records = {}

    lm_builder = ArtifactBuilder(
        LM_MAGIC,
        "minicpm-v-4.7-1b",
        {"hidden_size": 1024, "intermediate_size": 3584, "num_layers": 24, "vocab_size": 248094, "fused": fuse_all},
    )
    vis_builder = lm_builder if fuse_all else ArtifactBuilder(
        VIS_MAGIC,
        "minicpm-v-4.7-1b-vision",
        {"hidden_size": 1152, "intermediate_size": 4304, "num_layers": 27, "patch_size": 14},
    )

    with safe_open(str(st_1b_path), framework="pt", device=device) as f:
        col_w_1b = {}
        for nk in f.keys():
            if nk.endswith(("input_layernorm.weight", "post_attention_layernorm.weight", "language_model.norm.weight")):
                gamma = f.get_tensor(nk).float().reshape(-1) + 1.0
                col_w_1b[nk] = (gamma ** 2).clamp(min=1e-4)
            elif not lm_only and nk.endswith(("layer_norm1.weight", "layer_norm2.weight")):
                gamma = f.get_tensor(nk).float().reshape(-1)
                col_w_1b[nk] = (gamma ** 2).clamp(min=1e-4)

        def get_1b_cw(k: str):
            if k in ("lm_head.weight", "model.language_model.embed_tokens.weight"):
                c_norm = col_w_1b.get("model.language_model.norm.weight")
                c_in0 = col_w_1b.get("model.language_model.layers.0.input_layernorm.weight")
                if c_norm is not None and c_in0 is not None:
                    return 0.5 * (c_norm / c_norm.mean()) + 0.5 * (c_in0 / c_in0.mean())
                return c_norm
            if k.startswith("model.language_model.layers."):
                layer_prefix = ".".join(k.split(".")[:4])
                if k.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight", "in_proj_qkv.weight", "in_proj_z.weight")):
                    return col_w_1b.get(f"{layer_prefix}.input_layernorm.weight")
                if k.endswith(("gate_proj.weight", "up_proj.weight")):
                    return col_w_1b.get(f"{layer_prefix}.post_attention_layernorm.weight")
            if k.startswith("model.vision_tower.encoder.layers."):
                layer_prefix = ".".join(k.split(".")[:5])
                if k.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight")):
                    return col_w_1b.get(f"{layer_prefix}.layer_norm1.weight")
                if k.endswith("fc1.weight"):
                    return col_w_1b.get(f"{layer_prefix}.layer_norm2.weight")
            if k.startswith("model.vision_tower.vit_merger."):
                if k.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight")):
                    return col_w_1b.get("model.vision_tower.vit_merger.layer_norm1.weight")
            return None

        has_both_tied = (
            (int4_head or high_quality)
            and "lm_head.weight" in f.keys()
            and "model.language_model.embed_tokens.weight" in f.keys()
        )
        for k in f.keys():
            if lm_only and not (k.startswith("model.language_model.") or k == "lm_head.weight"):
                continue
            if has_both_tied and k == "model.language_model.embed_tokens.weight":
                continue
            t = f.get_tensor(k).float()
            cw = get_1b_cw(k)
            if k.startswith("model.language_model.") or k == "lm_head.weight":
                if k == "model.language_model.embed_tokens.weight":
                    if int4_head or high_quality:
                        lm_builder.add_int4_g64(k, t, col_weights=cw)
                    else:
                        lm_builder.add_ptq1(k, t, rotate=rotate_all_ptq1)
                elif k == "lm_head.weight":
                    if high_quality:
                        lm_builder.add_int4_g64(k, t, col_weights=cw)
                    else:
                        lm_builder.add_ptq1(k, t, rotate=False)
                    if has_both_tied:
                        lm_builder.tensors["model.language_model.embed_tokens.weight"] = dict(
                            lm_builder.tensors["lm_head.weight"]
                        )
                elif ".in_proj_a." in k or ".in_proj_b." in k:
                    lm_builder.add_bf16(k, t)
                elif t.ndim == 2 and k.endswith(".weight"):
                    # The engine rotates inputs only for 1024-wide matrices, and its fused SwiGLU only takes INT4 gate/up.
                    if decoder_format == "bf16":
                        lm_builder.add_bf16(k, t, rotate=k.endswith(("gate_proj.weight", "up_proj.weight")))
                    elif decoder_format == "decoder-ternary" or (decoder_format == "ffn-ternary" and ".mlp." in k):
                        lm_builder.add_ptq1(k, t, rotate=t.shape[1] == 1024)
                    elif "qat" in decoder_format:
                        lm_builder.add_int4_g64(k, t, grid=grids[k], rotate=decoder_format.startswith("int4-rot") and t.shape[1] == 1024)
                    elif decoder_format == "int4-gptq-o8" and k.endswith(("o_proj.weight", "out_proj.weight")):
                        lm_builder.add_int8_row(k, t)
                    elif decoder_format in ("int4-gptq", "int4-rot-gptq", "int4-gptq-o8", "int4-gptq-mix",
                                            "int4-rot-gptq-mix", "int4-gptq-tools"):
                        rotate = decoder_format.startswith("int4-rot") and t.shape[1] == 1024
                        lm_builder.add_int4_g64(k, t, hessian=hessians[k], rotate=rotate)
                    else:
                        lm_builder.add_int4_g64(k, t, col_weights=cw)
                elif k.endswith(("input_layernorm.weight", "post_attention_layernorm.weight", "q_norm.weight", "k_norm.weight", "language_model.norm.weight")):
                    lm_builder.add_f32(k, (t + 1.0).reshape(-1))
                elif k.endswith("A_log"):
                    lm_builder.add_f32(k, (-torch.exp(t)).reshape(-1))
                else:
                    lm_builder.add_f32(k, t.reshape(-1))
            elif not lm_only and (k.startswith("model.vision_tower.") or k.startswith("model.merger.")):
                if "patch_embedding.weight" in k:
                    vis_builder.add_f32(k, t.reshape(1152, -1))
                elif "position_embedding.weight" in k:
                    vis_builder.add_bf16(k, t)
                elif k.startswith(("model.vision_tower.vit_merger.", "model.merger.mlp.")) and t.ndim == 2 and k.endswith(".weight"):
                    if high_quality:
                        vis_builder.add_int4_g64(k, t, col_weights=cw)
                    else:
                        vis_builder.add_ptq1(k, t, rotate=False)
                elif t.ndim == 2 and k.endswith(".weight"):
                    vis_builder.add_int4_g64(k, t, col_weights=cw)
                else:
                    vis_builder.add_f32(k, t.reshape(-1))
            del t
        empty_cache(device)

    if not fuse_all:
        records["lm"] = lm_builder.write(out_dir / "minicpm-v-4.7-1b-tern4.bin")
        del lm_builder
        if lm_only or st_dsp_path is None:
            return records
        records["vision"] = vis_builder.write(out_dir / "minicpm-v-4.7-1b-vision.bin")
        del vis_builder

    if not lm_only and st_dsp_path is not None:
        dsp_builder = lm_builder if fuse_all else ArtifactBuilder(
            DSP_MAGIC,
            "minicpm-v-4.7-1b-dspark",
            {"hidden_size": 1024, "intermediate_size": 3584, "num_layers": 5, "vocab_size": 248094, "markov_rank": 256},
        )
        with safe_open(str(st_dsp_path), framework="pt", device=device) as f:
            col_w_dsp = {}
            for nk in f.keys():
                if nk.endswith(("input_layernorm.weight", "post_attention_layernorm.weight", "hidden_norm.weight")):
                    gamma = f.get_tensor(nk).float().reshape(-1) + 1.0
                    col_w_dsp[nk] = (gamma ** 2).clamp(min=1e-4)

            def get_dsp_cw(k: str):
                if k.startswith("layers."):
                    layer_prefix = ".".join(k.split(".")[:2])
                    if k.endswith("q_proj.weight"):
                        return col_w_dsp.get(f"{layer_prefix}.input_layernorm.weight")
                    if k.endswith(("k_proj.weight", "v_proj.weight")):
                        in_cw = col_w_dsp.get(f"{layer_prefix}.input_layernorm.weight")
                        hid_cw = col_w_dsp.get("hidden_norm.weight")
                        if in_cw is not None and hid_cw is not None:
                            return 0.5 * (in_cw + hid_cw)
                        return in_cw if in_cw is not None else hid_cw
                    if k.endswith(("gate_proj.weight", "up_proj.weight")):
                        return col_w_dsp.get(f"{layer_prefix}.post_attention_layernorm.weight")
                return None

            for k in f.keys():
                t = f.get_tensor(k).float()
                cw = get_dsp_cw(k)
                if "markov_w" in k:
                    dsp_builder.add_int4_g64(k, t, col_weights=cw)
                elif any(x in k for x in ["fc.weight", ".q_proj.", ".mlp."]):
                    if high_quality:
                        dsp_builder.add_int4_g64(k, t, col_weights=cw)
                    else:
                        dsp_builder.add_ptq1(k, t, rotate=False)
                elif k == "confidence_head.proj.weight" or k == "confidence_head.proj.bias":
                    dsp_builder.add_f32(k, t.reshape(-1))
                elif t.ndim == 2 and k.endswith(".weight"):
                    dsp_builder.add_int4_g64(k, t, col_weights=cw)
                else:
                    dsp_builder.add_f32(k, t.reshape(-1))
                del t
        if not fuse_all:
            records["dspark"] = dsp_builder.write(out_dir / "minicpm-v-4.7-1b-dspark.bin")

    if fuse_all:
        records["fused"] = lm_builder.write(out_dir / "minicpm-v-4.7-1b-tern4.bin")

    if QUANT_STATS["base_mse"] > 0:
        red = 100.0 * (1.0 - QUANT_STATS["opt_mse"] / QUANT_STATS["base_mse"])
        records["quant_stats"] = {
            "tensors": QUANT_STATS["tensors"],
            "base_weighted_mse": round(QUANT_STATS["base_mse"], 4),
            "opt_weighted_mse": round(QUANT_STATS["opt_mse"], 4),
            "mse_reduction_pct": round(red, 2),
        }
        print(f"[AWQ+WLS] Quantized {QUANT_STATS['tensors']} INT4_G64 tensors: weighted MSE {QUANT_STATS['base_mse']:.4f} -> {QUANT_STATS['opt_mse']:.4f} (-{red:.2f}%)", flush=True)
    return records


KAGGLE_CONVERT_BODY = r'''
import hashlib, json, subprocess, sys, time, urllib.request
from pathlib import Path

T0 = time.time()
OUT = Path("/kaggle/working")
OUT.mkdir(parents=True, exist_ok=True)
WORK = Path("/kaggle/temp")
WORK.mkdir(parents=True, exist_ok=True)

def log(*a):
    print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)


def empty_cache(device="cuda"):
    """Release the cached allocator memory of the current device."""
    import torch

    name = str(device)
    if name.startswith("cuda"):
        torch.cuda.empty_cache()
    elif name.startswith("mps"):
        torch.mps.empty_cache()

import torch

def fetch_verified(repo, rev, fname, expected_size, expected_sha):
    import os
    for candidate in [
        WORK / f"{repo.replace('/', '_')}_{fname}",
        WORK / fname,
        Path("models") / repo.split("/")[-1].lower() / fname,
    ]:
        if candidate.exists() and candidate.stat().st_size == expected_size:
            return candidate
    dest = WORK / f"{repo.replace('/', '_')}_{fname}"
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = f"{endpoint}/{repo}/resolve/{rev}/{fname}"
    log("Fetching", url)
    h = hashlib.sha256()
    n = 0
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "micro-ai/1"}), timeout=120) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 22):
            h.update(chunk)
            f.write(chunk)
            n += len(chunk)
    assert n == expected_size and h.hexdigest() == expected_sha, (fname, n, h.hexdigest())
    log("Verified", fname, n, expected_sha[:16])
    return dest

st_1b = fetch_verified(PINNED_1B_REPO, PINNED_1B_REV, "model.safetensors", *PINNED_1B_SAFETENSORS)
st_dsp = fetch_verified(PINNED_DSPARK_REPO, PINNED_DSPARK_REV, "model.safetensors", *PINNED_DSPARK_SAFETENSORS)
self_test()
device = globals().get("BUILD_DEVICE") or ("cuda" if torch.cuda.is_available()
                                           else "mps" if torch.backends.mps.is_available() else "cpu")
formats = globals().get("BUILD_FORMATS", ["int4"])
hessians = grids = None
if any("gptq" in f or "qat" in f for f in formats):
    import os
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=5.18.0", "lm-eval"], check=True)
    import transformers
    from huggingface_hub import snapshot_download
    model_dir = Path(snapshot_download(PINNED_1B_REPO, revision=PINNED_1B_REV, ignore_patterns=["*.safetensors"], local_dir=WORK / "hf-1b"))
    if not (model_dir / "model.safetensors").exists():
        os.symlink(st_1b, model_dir / "model.safetensors")
    model = transformers.AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.float32).to(device).eval()
    tok = transformers.AutoTokenizer.from_pretrained(model_dir)
    qat_args = globals().get("QAT_ARGS", {"nsamples": 512, "epochs": 4})
    mix = any(f.endswith("-mix") for f in formats)
    tools = any(f.endswith("-tools") for f in formats)
    assert sum(bool(f.endswith("-mix")) for f in formats if "gptq" in f or "qat" in f) <= 1, "one calibration set per job"
    assert not (mix and tools), "one calibration set per job"
    reuse = sorted(Path("/kaggle/input").glob("**/calib_ids.json"))  # an earlier -mix job attached as a kernel source
    if mix and reuse:
        ids = json.loads(reuse[0].read_text())
        log("Calibration reused from", reuse[0])
    elif mix:
        ids = chat_mix_ids(model, tok, max(128, qat_args["nsamples"]) * 2048, log=log)
        (OUT / "calib_ids.json").write_text(json.dumps(ids))
    elif tools:
        ids = tool_only_ids(tok, max(128, qat_args["nsamples"]) * 2048, log=log)
        (OUT / "calib_ids.json").write_text(json.dumps(ids))
    else:
        ids = fineweb_ids(tok, max(128, qat_args["nsamples"]) * 2048)
    log("Collecting GPTQ Hessians (128 x 2048 calibration tokens) on", device)
    hessians = collect_hessians(model, ids)
    log("Hessians:", len(hessians))
    qat_fmt = next((f for f in formats if "qat" in f), None)
    if qat_fmt:
        rotate = qat_fmt.startswith("int4-rot")
        seed = "awq" if qat_fmt.endswith("-awq") else "gptq"
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        hessians = {k: v.cpu() for k, v in hessians.items()}  # GPU room for training and the 248k-vocab eval logits
        params = dict(model.named_parameters())
        orig = {k: params[k].detach().to("cpu", torch.bfloat16) for k in hessians}  # the checkpoint is BF16
        log("QAT", qat_fmt, qat_args, "seed", seed)
        gptq, grids, head_q, stats = qat_int4_g64(model.model.language_model, model.lm_head.weight, hessians, ids,
                                                  rotate=rotate, seed=seed, log=log, **qat_args)
        report = {"args": qat_args, "layers": stats, "qat_secs": round(time.time() - T0), "wikitext_word_ppl": {}}
        with torch.no_grad():  # the artifact embeds through the tied INT4 head
            params["lm_head.weight"].copy_(head_q)
            params["model.language_model.embed_tokens.weight"].copy_(head_q)
        hflm = HFLM(pretrained=model, tokenizer=tok, backend="causal", batch_size=1, max_length=4096)
        m = rotate512_torch(torch.eye(1024))

        def torch_weight(w):  # BF16 tensor, or GPTQ grid (of W·M when rotated)
            if not isinstance(w, tuple):
                return w
            d = int4_dequant(*w)
            return d @ m.T if rotate and d.shape[1] == 1024 else d

        # The seed grid as its own reference point: for the AWQ seed it is the plain int4 grid.
        seed_name = "int4-gptq" if seed == "gptq" else "int4-awq"
        for name, weights in ((qat_fmt, {}), (seed_name, gptq), ("bf16+int4_head_tied", orig)):
            with torch.no_grad():
                for k, w in weights.items():
                    params[k].copy_(torch_weight(w))
            try:
                r = lm_eval.simple_evaluate(model=hflm, tasks=["wikitext"], num_fewshot=0)["results"]["wikitext"]
                report["wikitext_word_ppl"][name] = r["word_perplexity,none"]
                log("wikitext", name, r["word_perplexity,none"])
            except (torch.OutOfMemoryError, RuntimeError) as e:
                # The 248k-vocab eval logits need several GiB; the artifact matters more than this
                # convenience metric (perplexity is measured locally against the served artifact).
                if "out of memory" not in str(e).lower():
                    raise
                log("wikitext", name, "skipped (out of memory):", e)
                empty_cache(device)
            (OUT / "qat.json").write_text(json.dumps(report, indent=2) + "\n")
        del hflm
    del model
    torch.cuda.empty_cache()
for fmt in formats:
    out = OUT / fmt if len(formats) > 1 else OUT
    log("Building fused artifact (decoder", fmt, ") on", device)
    records = build_artifacts(st_1b, st_dsp, out, device=device, int4_head=True, high_quality=True, decoder_format=fmt, hessians=hessians, grids=grids)
    (out / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    log("Done:", fmt, records)
'''


def self_test() -> None:
    """Check the glaive parser against the shapes the dataset really has.

    The parser silently matched nothing once (the arguments are a single-quoted JSON string, and
    the rows with an unescaped apostrophe broke the brace scan), and the calibration it produced
    taught the model to answer tool prompts in prose. It runs on every conversion."""
    cases = [
        # The usual form: the arguments are a JSON string inside single quotes.
        ('{"name": "calculate_bmi", "arguments": \'{"weight": 70, "height": 1.75}\'}',
         ("calculate_bmi", {"weight": 70, "height": 1.75})),
        # An apostrophe inside the arguments is not escaped, so the wrapper ends at the LAST quote.
        ('{"name": "sentiment", "arguments": \'{"text": "aujourd\'hui", "lang": "French"}\'}',
         ("sentiment", {"text": "aujourd'hui", "lang": "French"})),
        # An escaped apostrophe.
        ('{"name": "sentiment", "arguments": \'{"text": "aujourd\\\'hui"}\'}',
         ("sentiment", {"text": "aujourd'hui"})),
        # Nested objects in the arguments.
        ('{"name": "f", "arguments": \'{"q": "x", "o": {"deep": true}}\'}',
         ("f", {"q": "x", "o": {"deep": True}})),
        # Plain JSON arguments.
        ('{"name": "weather", "arguments": {"city": "Paris"}}', ("weather", {"city": "Paris"})),
        # Not a call.
        ('{"bmi": 22.86}', None),
    ]
    for raw, want in cases:
        got = glaive_call_args(raw)
        if got != want:
            raise AssertionError(f"glaive_call_args({raw!r}) = {got!r}, want {want!r}")

    # glaive_call must return the prompt plus the call turn, and None for a reply that calls nothing.
    chat = ("USER: What's the weather in Paris?\nASSISTANT: <functioncall> "
            '{"name": "get_weather", "arguments": \'{"city": "Paris"}\'} <|endoftext|>')
    turn = glaive_call(chat, [{"type": "function", "function": {"name": "get_weather"}}])
    if turn is None or len(turn) != 2 or turn[0]["role"] != "user" or turn[0]["content"] != "What's the weather in Paris?":
        raise AssertionError(f"glaive_call returned {turn!r}")
    call = turn[1].get("tool_calls")
    if not call or call[0]["function"]["name"] != "get_weather" or call[0]["function"]["arguments"] != {"city": "Paris"}:
        raise AssertionError(f"glaive_call returned {turn!r}")
    if glaive_call("USER: Hi\nASSISTANT: <functioncall> {\"bmi\": 22.86} <|endoftext|>", []) is not None:
        raise AssertionError("glaive_call accepted a reply that calls nothing")

    # The calibration helpers must be callable: tool_only_ids used a `user` lambda that only
    # chat_mix_ids defined, and the kernel died with a NameError before it built anything.
    for fn in (tool_only_ids, chat_mix_ids):
        if not callable(fn):
            raise AssertionError(f"{fn.__name__} is not callable")


def stage_kaggle_converter(kernel: str, output_dir: Path, formats: list[str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).read_text(encoding="utf-8")
    script = source.split("def stage_kaggle_converter")[0] + f"\nBUILD_FORMATS = {formats!r}\n" + KAGGLE_CONVERT_BODY
    (output_dir / "convert.py").write_text(script, encoding="utf-8")
    meta = {
        "id": kernel,
        "title": kernel.split("/")[1],
        "code_file": "convert.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (output_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-kaggle", help="owner/slug for private Kaggle conversion job")
    parser.add_argument("--output", type=Path, default=Path(".scratch/kaggle-convert"))
    parser.add_argument("--st-1b", type=Path, help="Path to MiniCPM-V-4.7-1B model.safetensors")
    parser.add_argument("--st-dsp", type=Path, help="Path to MiniCPM-V-4.7-1B-DSpark model.safetensors")
    parser.add_argument("--device", default="cuda", help="Torch device for quantization (default: cuda)")
    parser.add_argument(
        "--int4-head",
        action="store_true",
        default=True,
        help="Quantize embed_tokens.weight and lm_head.weight with unrotated INT4_G64 (0.48 KL / 73.5%% top-1)",
    )
    parser.add_argument(
        "--high-quality",
        action="store_true",
        default=True,
        help="Upgrade embed_tokens, lm_head, vision vit_merger.linear_1, and dspark fc/q_proj to INT4_G64",
    )
    parser.add_argument(
        "--rotate-all-ptq1",
        action="store_true",
        default=False,
        help="Apply FWHT-512 rotation to vision and dspark PTQ1_0 projections",
    )
    parser.add_argument(
        "--decoder-format",
        nargs="+",
        choices=["int4", "bf16", "ffn-ternary", "decoder-ternary", "int4-gptq", "int4-rot-gptq", "int4-gptq-o8", "int4-qat",
                 "int4-qat-awq", "int4-gptq-mix", "int4-rot-gptq-mix", "int4-qat-mix", "int4-rot-qat-mix",
                 "int4-gptq-tools", "int4-qat-tools"],
        default=["int4"],
        help="Decoder linear weights: int4 (AWQ+WLS INT4_G64), bf16, ffn-ternary (PTQ1 gate/up/down), "
        "decoder-ternary (PTQ1 everywhere int4 is used), int4-gptq (GPTQ on fineweb-edu Hessians), int4-rot-gptq "
        "(GPTQ with 1024-wide inputs FWHT-rotated), int4-gptq-o8 (int4-gptq with INT8_ROW attention o_proj and "
        "DeltaNet out_proj) or int4-qat (int4-gptq refined by block-wise QAT, see qat_int4_g64; int4-rot-qat from "
        "int4-rot-gptq); int4-qat-awq is the same QAT seeded from the plain int4 (AWQ+WLS) grid instead of the "
        "GPTQ grid, which keeps the tool-call behaviour of the plain int4 artifact; -tools (int4-gptq-tools, "
        "int4-qat-tools) calibrates on the glaive function-calling rows and fineweb-edu only, without the "
        "chat/math/code rows that taught the model to answer tool prompts in prose; -mix (int4-gptq-mix, "
        "int4-rot-gptq-mix, int4-qat-mix, int4-rot-qat-mix) calibrates on "
        "chat_mix_ids instead of fineweb-edu (one calibration per job; a job reuses /kaggle/input/*/calib_ids.json from "
        "an earlier -mix job attached as a kernel source); gptq/qat formats need --stage-kaggle; several build into "
        "<output>/<format>/",
    )
    args = parser.parse_args()
    if args.stage_kaggle:
        print(stage_kaggle_converter(args.stage_kaggle, args.output, args.decoder_format))
    elif args.st_1b and args.st_dsp:
        if any("gptq" in f or "qat" in f for f in args.decoder_format):
            parser.error("gptq/qat formats calibrate the HF model; build them with --stage-kaggle")
        self_test()
        for fmt in args.decoder_format:
            records = build_artifacts(
                args.st_1b,
                args.st_dsp,
                args.output / fmt if len(args.decoder_format) > 1 else args.output,
                device=args.device,
                int4_head=args.int4_head,
                high_quality=args.high_quality,
                rotate_all_ptq1=args.rotate_all_ptq1,
                decoder_format=fmt,
            )
            print(json.dumps(records, indent=2))


if __name__ == "__main__":
    import os
    if os.path.exists("/kaggle/working"):
        exec(KAGGLE_CONVERT_BODY, globals())
    else:
        main()
