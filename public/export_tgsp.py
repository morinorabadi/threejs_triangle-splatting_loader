#!/usr/bin/env python3
"""
TGSP v1 exporter
- Reads a PyTorch .pth state_dict produced by triangle-splatting training
- Writes a .tgsp container with: JSON header + binary TOC + tensor blobs
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch


# ----------------------------
# TGSP format constants
# ----------------------------
MAGIC = b"TGSP"
VERSION = 1

# dtype codes (extend later as needed)
DTYPE_CODES = {
    "f16": 1,
    "f32": 2,
    "u8": 3,
    "i32": 4,
    "i64": 5,
}
CODE_TO_DTYPE = {v: k for k, v in DTYPE_CODES.items()}

# Fixed header: magic(4) + version(u32) + header_len(u32) + toc_len(u32) + reserved(16)
FILE_HEADER_STRUCT = struct.Struct("<4sIII16s")  # total 4+4+4+4+16 = 32 bytes

# TOC entry (fixed 32 bytes):
# name_hash(u64), offset(u64), nbytes(u64), dtype_code(u32), reserved(u32)
TOC_ENTRY_STRUCT = struct.Struct("<QQQI I")  # 8+8+8+4+4 = 32 bytes


def fnv1a64(data: bytes) -> int:
    """FNV-1a 64-bit hash, stable across languages."""
    h = 0xCBF29CE484222325
    prime = 0x100000001B3
    for b in data:
        h ^= b
        h = (h * prime) & 0xFFFFFFFFFFFFFFFF
    return h


def torch_to_numpy(
    t: Any,
    *,
    target: Optional[str] = None,
) -> np.ndarray:
    """
    Convert a torch Tensor (or array-like) to numpy with optional dtype casting.
    target: one of {"f16","f32","u8","i32","i64"} or None to keep dtype (float->f32).
    """
    if isinstance(t, torch.Tensor):
        x = t.detach().cpu()
        # Default float tensors to f32 if no explicit target:
        if target is None and x.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            x = x.float()
        elif target == "f16":
            x = x.half()
        elif target == "f32":
            x = x.float()
        elif target == "u8":
            x = x.to(torch.uint8)
        elif target == "i32":
            x = x.to(torch.int32)
        elif target == "i64":
            x = x.to(torch.int64)
        return x.contiguous().numpy()
    # numpy already?
    arr = np.asarray(t)
    if target is None:
        if np.issubdtype(arr.dtype, np.floating):
            return arr.astype(np.float32, copy=False)
        return arr
    if target == "f16":
        return arr.astype(np.float16, copy=False)
    if target == "f32":
        return arr.astype(np.float32, copy=False)
    if target == "u8":
        return arr.astype(np.uint8, copy=False)
    if target == "i32":
        return arr.astype(np.int32, copy=False)
    if target == "i64":
        return arr.astype(np.int64, copy=False)
    raise ValueError(f"Unknown target dtype: {target}")


def numpy_dtype_tag(a: np.ndarray) -> str:
    """Map numpy dtype to our dtype tags."""
    if a.dtype == np.float16:
        return "f16"
    if a.dtype == np.float32:
        return "f32"
    if a.dtype == np.uint8:
        return "u8"
    if a.dtype == np.int32:
        return "i32"
    if a.dtype == np.int64:
        return "i64"
    # Fall back to f32 for other floats; otherwise error
    if np.issubdtype(a.dtype, np.floating):
        return "f32"
    raise TypeError(f"Unsupported dtype for export: {a.dtype}")


@dataclass
class TensorRecord:
    name: str
    array: np.ndarray
    dtype_tag: str
    shape: Tuple[int, ...]
    nbytes: int
    name_hash: int
    offset: int = 0  # set during layout


def build_records(
    sd: Dict[str, Any],
    *,
    cast_float16: bool,
    include_keys: Optional[List[str]] = None,
    export_opacity_alpha: bool = True,
) -> Tuple[List[TensorRecord], Dict[str, Any]]:
    """
    Build tensor records for export. Also returns a 'params' dict for JSON header.
    """
    # Decide which tensors to include
    all_tensor_keys = [k for k, v in sd.items() if isinstance(v, torch.Tensor)]
    keys = include_keys if include_keys is not None else all_tensor_keys

    # Ensure stable ordering for deterministic files
    keys = [k for k in keys if k in sd and isinstance(sd[k], torch.Tensor)]
    keys.sort()

    records: List[TensorRecord] = []

    # Export rule: floats -> f16 if cast_float16 else f32, ints keep their widths if possible
    for k in keys:
        t = sd[k]
        if not isinstance(t, torch.Tensor):
            continue

        if t.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            target = "f16" if cast_float16 else "f32"
        elif t.dtype in (torch.uint8,):
            target = "u8"
        elif t.dtype in (torch.int32,):
            target = "i32"
        elif t.dtype in (torch.int64,):
            target = "i64"
        else:
            # Many training pipelines only use float + int; others are rare.
            # Best-effort: cast unknown numeric types to f32.
            target = "f32"

        arr = torch_to_numpy(t, target=target)
        dtype_tag = numpy_dtype_tag(arr)

        rec = TensorRecord(
            name=k,
            array=arr,
            dtype_tag=dtype_tag,
            shape=tuple(arr.shape),
            nbytes=int(arr.nbytes),
            name_hash=fnv1a64(k.encode("utf-8")),
        )
        records.append(rec)

    # Optional export-side alpha baking for easier cross-viewer consistency.
    # Keep raw `opacity` logits and add `opacity_alpha` as preferred render alpha.
    if (
        export_opacity_alpha
        and "opacity" in sd
        and isinstance(sd["opacity"], torch.Tensor)
        and not any(r.name == "opacity_alpha" for r in records)
    ):
        alpha_t = torch.sigmoid(sd["opacity"].detach().cpu().float())
        alpha_target = "f16" if cast_float16 else "f32"
        alpha_arr = torch_to_numpy(alpha_t, target=alpha_target)
        alpha_rec = TensorRecord(
            name="opacity_alpha",
            array=alpha_arr,
            dtype_tag=numpy_dtype_tag(alpha_arr),
            shape=tuple(alpha_arr.shape),
            nbytes=int(alpha_arr.nbytes),
            name_hash=fnv1a64(b"opacity_alpha"),
        )
        records.append(alpha_rec)
        records.sort(key=lambda r: r.name)

    # Pull non-tensor params you care about (present in your example)
    params: Dict[str, Any] = {}
    if "active_sh_degree" in sd:
        # often a python int or 0-d tensor
        val = sd["active_sh_degree"]
        if isinstance(val, torch.Tensor):
            params["active_sh_degree"] = int(val.detach().cpu().item())
        else:
            params["active_sh_degree"] = int(val)
    # You can add more semantic notes (very helpful for future loaders)
    # We don't assume whether opacity is logits or sigmoid; record your intent.
    params.setdefault("opacity_semantics", "logits")
    if any(r.name == "opacity_alpha" for r in records):
        params.setdefault("opacity_alpha_semantics", "alpha_linear")
        params.setdefault("preferred_opacity_tensor", "opacity_alpha")
    else:
        params.setdefault("preferred_opacity_tensor", "opacity")
    params.setdefault("sigma_semantics", "gaussian_stddev")
    params.setdefault("space", "world")

    return records, params


def write_tgsp(
    pth_path: str,
    out_path: str,
    *,
    cast_float16: bool = True,
    include_non_tensor_meta: bool = True,
    export_opacity_alpha: bool = True,
) -> None:
    sd = torch.load(pth_path, map_location="cpu", weights_only=True)
    if not isinstance(sd, dict):
        raise ValueError(f"Expected state_dict dict, got {type(sd)}")

    records, params = build_records(
        sd,
        cast_float16=cast_float16,
        export_opacity_alpha=export_opacity_alpha,
    )

    if len(records) == 0:
        raise ValueError("No tensor records found to export.")

    # Build JSON header
    header: Dict[str, Any] = {
        "asset": "triangle-splat",
        "format": "TGSP",
        "version": VERSION,
        "endianness": "little",
        "created_unix": int(time.time()),
        "source": {
            "path": os.path.basename(pth_path),
        },
        "tensors": [
            {
                "name": r.name,
                "dtype": r.dtype_tag,
                "shape": list(r.shape),
                "name_hash_fn": "fnv1a64",
                "name_hash": f"0x{r.name_hash:016x}",
            }
            for r in records
        ],
        "params": params if include_non_tensor_meta else {},
    }

    header_bytes = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    # TOC layout: offsets relative to the start of the data section (right after TOC)
    toc_len = len(records) * TOC_ENTRY_STRUCT.size

    # Compute offsets
    offset = 0
    for r in records:
        r.offset = offset
        # Align each tensor blob to 16 bytes (nice for GPU uploads later)
        offset += r.nbytes
        pad = (-offset) % 16
        offset += pad

    # Write file
    reserved = b"\x00" * 16
    with open(out_path, "wb") as f:
        # Placeholder header (we already know lengths, so no need for seek patching)
        f.write(FILE_HEADER_STRUCT.pack(MAGIC, VERSION, len(header_bytes), toc_len, reserved))

        # JSON header
        f.write(header_bytes)

        # TOC
        for r in records:
            dtype_code = DTYPE_CODES.get(r.dtype_tag)
            if dtype_code is None:
                raise ValueError(f"No dtype code for {r.dtype_tag}")
            f.write(
                TOC_ENTRY_STRUCT.pack(
                    r.name_hash,
                    r.offset,
                    r.nbytes,
                    dtype_code,
                    0,  # reserved
                )
            )

        # Data blobs
        data_section_start = f.tell()
        for r in records:
            # Ensure we are at the expected offset
            cur = f.tell()
            expected = data_section_start + r.offset
            if cur != expected:
                if cur > expected:
                    raise RuntimeError(f"Writer advanced past expected offset for {r.name}")
                f.write(b"\x00" * (expected - cur))

            f.write(r.array.tobytes(order="C"))

            # pad so that (pos - data_section_start) is 16B-aligned
            cur2 = f.tell()
            rel = cur2 - data_section_start
            pad = (-rel) % 16
            if pad:
                f.write(b"\x00" * pad)

    print(f"[TGSP] Wrote: {out_path}")
    print(f"[TGSP] Tensors: {len(records)} | float16={'yes' if cast_float16 else 'no'}")
    has_opacity_alpha = any(r.name == "opacity_alpha" for r in records)
    print(f"[TGSP] opacity_alpha: {'yes' if has_opacity_alpha else 'no'}")
    total_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[TGSP] Size: {total_mb:.2f} MiB")


def main():
    ap = argparse.ArgumentParser(description="Export triangle-splatting .pth to .tgsp")
    ap.add_argument("pth", help="Path to .pth checkpoint/state_dict")
    ap.add_argument("out", help="Output .tgsp path")
    ap.add_argument("--f32", action="store_true", help="Store float tensors as float32 (default: float16)")
    ap.add_argument("--no-meta", action="store_true", help="Do not include non-tensor metadata like active_sh_degree")
    ap.add_argument(
        "--no-opacity-alpha",
        action="store_true",
        help="Do not export baked opacity_alpha tensor (default: export it)",
    )
    args = ap.parse_args()

    write_tgsp(
        args.pth,
        args.out,
        cast_float16=not args.f32,
        include_non_tensor_meta=not args.no_meta,
        export_opacity_alpha=not args.no_opacity_alpha,
    )


if __name__ == "__main__":
    main()
