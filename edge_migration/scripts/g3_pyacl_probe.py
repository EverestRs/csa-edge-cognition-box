#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""pyACL 约定探针（会商项A配套）：分阶段子进程隔离，segfault 不伤主进程。

每个阶段作为独立子进程运行自身（--stage N），崩溃被捕获并记录；产出
results/edge_m0/pyacl_probe.json 描述盒上 CANN 的真实接口约定——
inference_backend.AscendOMBackend 的自适应 shim 与人工判读都以它为准。

用法（盒上）:
  python edge_migration/scripts/g3_pyacl_probe.py --om models/detector/yolov8n.om
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT = HERE.parents[2]


def _describe(obj) -> str:
    return f"{type(obj).__name__}:{repr(obj)[:160]}"


def stage_api_surface() -> dict:
    out: dict = {"stage": "api_surface"}
    try:
        import acl  # type: ignore
        out["import_ok"] = True
        mdl_names = [n for n in dir(acl.mdl)]
        want = ["load_from_file", "create_desc", "get_input_dims", "get_input_shape",
                "get_input_shape_by_index", "get_dim_by_index", "get_input_data_type",
                "get_input_dtype", "get_input_data_type_by_index", "get_output_num",
                "get_output_num_by_desc", "get_output_size", "get_output_size_by_index",
                "get_output_dims", "get_output_shape_by_index", "create_dataset",
                "add_dataset_buffer", "execute", "unload", "get_model_desc"]
        out["mdl_present"] = {w: (w in mdl_names) for w in want}
        out["rt_present"] = {w: hasattr(acl.rt, w) for w in
                             ["malloc", "free", "memcpy_h2d", "memcpy_d2h",
                              "set_device", "create_context", "reset_device"]}
        consts = {k: int(getattr(acl, k)) for k in
                  ["DT_FLOAT", "DT_HALF", "DT_INT8", "DT_UINT8", "DT_INT32"]
                  if isinstance(getattr(acl, k, None), int)}
        out["dtype_consts"] = consts
        out["raw_get_version"] = _describe(acl.get_version()) if hasattr(acl, "get_version") else "n/a"
    except Exception as exc:
        out["import_ok"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def stage_load(om: str) -> dict:
    out: dict = {"stage": "load"}
    try:
        import acl  # type: ignore
        rc = acl.init()
        out["init"] = rc
        out["set_device"] = _describe(acl.rt.set_device(0))
        res = acl.mdl.load_from_file(om)
        out["load_from_file_raw"] = _describe(res)
        # 归一取 model_id：两种序都试
        mid = None
        if isinstance(res, tuple) and len(res) == 2:
            a, b = res
            if isinstance(a, int) and isinstance(b, int):
                mid = b if a == 0 else a
            elif isinstance(b, int):
                mid = b
            elif isinstance(a, int):
                mid = a
        out["model_id_inferred"] = mid
        dres = acl.mdl.create_desc()
        out["create_desc_raw"] = _describe(dres)
        desc = dres[1] if isinstance(dres, tuple) and len(dres) == 2 else dres
        out["desc"] = _describe(desc)
        # 关键探测：dims 函数族（只传 desc——旧接口传 model_id 会 segfault，已知）
        for name in ("get_input_dims", "get_input_shape_by_index", "get_input_shape"):
            fn = getattr(acl.mdl, name, None)
            if fn is None:
                out[f"{name}"] = "absent"
                continue
            try:
                r = fn(desc, 0) if name.endswith("by_index") else fn(desc)
                out[name] = _describe(r)
            except Exception as exc:
                out[name] = f"error: {exc}"
        for name in ("get_output_num", "get_output_num_by_desc"):
            fn = getattr(acl.mdl, name, None)
            if fn is None:
                continue
            try:
                out[name] = _describe(fn(desc))
            except Exception as exc:
                out[name] = f"error: {exc}"
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def stage_execute(om: str) -> dict:
    out: dict = {"stage": "execute"}
    try:
        import acl  # type: ignore
        import numpy as np
        acl.init()
        acl.rt.set_device(0)
        res = acl.mdl.load_from_file(om)
        a, b = res if isinstance(res, tuple) else (0, res)
        mid = b if isinstance(a, int) and a == 0 else (a if isinstance(a, int) else b)
        dres = acl.mdl.create_desc()
        desc = dres[1] if isinstance(dres, tuple) and len(dres) == 2 else dres
        dims = None
        for name in ("get_input_dims", "get_input_shape_by_index"):
            fn = getattr(acl.mdl, name, None)
            if fn is None:
                continue
            try:
                dims = fn(desc, 0) if "index" in name else fn(desc)
                break
            except Exception:
                continue
        # 归一 shape
        dv = dims[1] if isinstance(dims, tuple) else dims
        shape = ()
        if isinstance(dv, dict) and "dims" in dv:
            shape = tuple(int(x) for x in dv["dims"])
        elif isinstance(dv, dict) and "dim_count" in dv:
            getter = getattr(acl.mdl, "get_dim_by_index", None)
            if getter:
                shape = tuple(int(_pair_get(getter(dv, i))) for i in range(int(dv["dim_count"])))
        out["input_shape"] = list(shape)
        n = 1
        for s in shape:
            n *= max(int(s), 1)
        in_bytes = n * 4
        mres = acl.rt.malloc(in_bytes, 0)
        ptr = mres[1] if isinstance(mres, tuple) and isinstance(mres[0], int) and mres[0] == 0 else \
            (mres[0] if isinstance(mres, tuple) else mres)
        host = np.zeros(shape if shape else (1, 3, 224, 224), dtype=np.float32)
        out["memcpy_h2d"] = _describe(acl.rt.memcpy_h2d(int(ptr), in_bytes,
                                                        int(host.__array_interface__["data"][0]),
                                                        in_bytes))
        ds_in = acl.mdl.create_dataset()
        ds_in = ds_in[1] if isinstance(ds_in, tuple) else ds_in
        added = False
        for key in ("buf", "buffer"):
            try:
                r = acl.mdl.add_dataset_buffer(ds_in, {key: ptr, "size": in_bytes})
                if (r[0] if isinstance(r, tuple) and isinstance(r[0], int) else r) == 0:
                    added = True
                    out["dataset_key"] = key
                    break
            except Exception:
                continue
        if not added:
            out["add_buffer_fail"] = True
            return out
        ds_out = acl.mdl.create_dataset()
        ds_out = ds_out[1] if isinstance(ds_out, tuple) else ds_out
        eres = acl.mdl.execute(mid, ds_in, ds_out)
        out["execute"] = _describe(eres)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _pair_get(res):
    if isinstance(res, tuple) and len(res) == 2:
        return res[1] if res[0] == 0 else res[0]
    return res


def run_child(stage: int, om: str) -> dict:
    if stage == 0:
        return stage_api_surface()
    if stage == 1:
        return stage_load(om)
    if stage == 2:
        return stage_execute(om)
    return {"stage": stage, "error": "unknown"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--om", default="models/detector/yolov8n.om")
    ap.add_argument("--out", default="results/edge_m0/pyacl_probe.json")
    ap.add_argument("--stage", type=int, default=-1)
    a = ap.parse_args()

    if a.stage >= 0:  # 子进程：直进指定阶段，输出 JSON 到 stdout
        print(json.dumps(run_child(a.stage, str(PROJECT / a.om)), ensure_ascii=False))
        return 0

    os.chdir(PROJECT)
    report: dict = {"probe": "pyacl", "ts": time.time(), "om": a.om, "stages": {}}
    for stage, name in ((0, "api_surface"), (1, "load"), (2, "execute")):
        try:
            r = subprocess.run([sys.executable, str(HERE), "--om", a.om, "--stage", str(stage)],
                               capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                report["stages"][name] = {"crashed": True, "returncode": r.returncode,
                                          "stderr_tail": (r.stderr or "")[-300:]}
            else:
                report["stages"][name] = json.loads(r.stdout.strip().splitlines()[-1])
        except subprocess.TimeoutExpired:
            report["stages"][name] = {"crashed": True, "reason": "timeout_60s"}
        except Exception as exc:
            report["stages"][name] = {"crashed": True, "reason": str(exc)}

    out = PROJECT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False)[:800])
    print(f"[pyacl_probe] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
