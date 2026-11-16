#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G4 VDEC 硬件解码可行性探针（方案 B）。

目标：验证板载 DVPP VDEC 能否把 H.264 解码从 4×A55 软解卸载到硬件，
量化 (1) 硬件解码吞吐 (2) 相比 cv2 软解释放的 CPU 时间。
不碰主路径，纯独立探针。

依赖：华为官方 acllite（/usr/local/Ascend/thirdpart/aarch64/acllite）的 DvppVdec，
底层是 pyACL 的 acl.media.vdec_* + dvpp_malloc。demux 用 ffmpeg CLI 出 annexb，
Python 侧按 NAL 切访问单元(AU)，假设每帧单 slice（本项目片源满足）。

用法: ~/.conda/envs/csa/bin/python g4_vdec_probe.py --src <mp4> [--frames N] [--out json]
"""
import argparse
import json
import os
import subprocess
import sys
import time
import resource

ACLLITE_DIR = "/usr/local/Ascend/thirdpart/aarch64/acllite"
if ACLLITE_DIR not in sys.path:
    sys.path.insert(0, ACLLITE_DIR)

import numpy as np  # noqa: E402
import cv2  # noqa: E402


def _cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def ffprobe_size(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0", path
    ]).decode().strip().split(",")
    return int(out[0]), int(out[1])


def to_annexb(src, dst):
    subprocess.check_call([
        "ffmpeg", "-y", "-loglevel", "error", "-i", src,
        "-c", "copy", "-bsf:v", "h264_mp4toannexb", "-f", "h264", dst
    ])


def split_access_units(raw):
    """按 start code 切 NAL，再按'非VCL前缀 + 单个VCL slice'聚成 AU（假设单 slice/帧）。"""
    n = len(raw)
    starts = []
    i = 0
    while i + 3 <= n:
        if raw[i] == 0 and raw[i + 1] == 0 and (raw[i + 2] == 1 or (i + 3 < n and raw[i + 3] == 1 and raw[i + 2] == 0)):
            sc_len = 4 if (i + 3 < n and raw[i + 2] == 0 and raw[i + 3] == 1) else 3
            starts.append((i, sc_len))
            i += sc_len
        else:
            i += 1
    nals = []
    for idx, (pos, sc_len) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else n
        body = raw[pos + sc_len:end]
        if not body:
            continue
        nal_type = body[0] & 0x1F
        nals.append((nal_type, raw[pos:end]))
    aus = []
    buf = b""
    for nal_type, nal in nals:
        if nal_type in (7, 8, 6, 9):        # SPS/PPS/SEI/AUD 前缀
            buf += nal
        elif nal_type in (1, 5):            # VCL slice → 一帧
            aus.append(buf + nal)
            buf = b""
    return aus


_MP_G = {}


def _pool_init(barrier, ncpu):
    _MP_G["barrier"] = barrier
    _MP_G["ncpu"] = ncpu


def _mp_worker(args):
    """独立进程跑一路 VDEC 解码（各自 acl context，绕开 GIL），返回统计。
    消抖动三招：绑独立核（避免 4 进程抢 CPU）+ 栅栏同步起跑（真正并发）+ init 预热在计时外。"""
    cid, aus, w, h, max_au = args
    barrier = _MP_G["barrier"]
    ncpu = _MP_G["ncpu"]
    import os as _os
    import time as _time
    import acl
    import constants as const
    import dvpp_vdec as dvpp_vdec
    try:
        _os.sched_setaffinity(0, {cid % max(1, ncpu)})  # 每路钉一个独立核
    except (OSError, AttributeError):
        pass
    ret = acl.init()
    if ret:
        return {"channel": cid, "error": f"acl.init={ret}"}
    acl.rt.set_device(0)
    ctx, _ = acl.rt.create_context(0)
    acl.rt.set_context(ctx)
    in_buf, r = acl.media.dvpp_malloc(max_au + 1024)
    if r or not in_buf:
        return {"channel": cid, "error": f"malloc {r}"}
    vdec = dvpp_vdec.DvppVdec(cid, w, h, const.ENTYPE_H264_HIGH, ctx)
    if vdec.init() != const.SUCCESS:
        return {"channel": cid, "error": "vdec_init_failed"}
    got = fed = 0
    # 预热：先解 20 帧让 DVPP/时钟稳定，不计入
    warm = aus[:20] if len(aus) > 40 else aus[:1]
    for au in warm:
        ptr = acl.util.bytes_to_ptr(au)
        if acl.rt.memcpy(in_buf, len(au), ptr, len(au), const.ACL_MEMCPY_HOST_TO_DEVICE):
            break
        if vdec.process(in_buf, len(au), [cid, -1]):
            break
    for _ in range(len(warm)):
        _, image = vdec.read(no_wait=True)
        if image is None:
            break
    barrier.wait()  # 所有路 init+预热完成，同时起跑
    t0 = _time.perf_counter()
    for au in aus:
        ptr = acl.util.bytes_to_ptr(au)
        if acl.rt.memcpy(in_buf, len(au), ptr, len(au), const.ACL_MEMCPY_HOST_TO_DEVICE):
            break
        if vdec.process(in_buf, len(au), [cid, fed]):
            break
        fed += 1
        while got < fed:
            _, image = vdec.read(no_wait=True)
            if image is None:
                break
            got += 1
    feed_wall = _time.perf_counter() - t0
    while got < fed:
        _, image = vdec.read(no_wait=True)
        if image is None:
            break
        got += 1
    vdec.destroy()
    return {"channel": cid, "fed": fed, "decoded": got, "feed_wall": round(feed_wall, 3)}


def run_multichannel_mp(aus, w, h, max_au, nchan):
    """多进程并发：每进程一路 VDEC + 独立核 + 起跑栅栏，聚合吞吐反映硬件多通道真实上限。"""
    import multiprocessing as mp
    ncpu = os.cpu_count() or 4
    ctxm = mp.get_context("fork")  # fork：Barrier 经 initializer 继承（spawn 无法 pickle 同步原语）
    barrier = ctxm.Barrier(nchan)
    jobs = [(c, aus, w, h, max_au) for c in range(nchan)]
    t0 = time.perf_counter()
    with ctxm.Pool(nchan, initializer=_pool_init, initargs=(barrier, ncpu)) as pool:
        results = pool.map(_mp_worker, jobs)
    wall = time.perf_counter() - t0
    ok = [r for r in results if "decoded" in r]
    total_frames = sum(r["decoded"] for r in ok)
    max_feed = max((r["feed_wall"] for r in ok), default=0)
    return {
        "mode": "multiprocess_synced",
        "channels": nchan, "per_channel": results, "channels_succeeded": len(ok),
        "total_decoded_frames": total_frames, "concurrent_wall_s": round(wall, 3),
        "aggregate_fps": round(total_frames / max_feed, 1) if max_feed > 0 else 0,
        "per_channel_fps": [round(r["decoded"] / r["feed_wall"], 1) for r in ok],
    }


def run_multichannel(aus, w, h, max_au, nchan):
    """N 路并发解码（各自 context + channel_id），测聚合吞吐判断硬件是否线性扩展。"""
    import threading
    import acl
    import constants as const
    import dvpp_vdec as dvpp_vdec

    out = {}

    def worker(cid):
        try:
            ctx, _ = acl.rt.create_context(0)
            acl.rt.set_context(ctx)
            in_buf, r = acl.media.dvpp_malloc(max_au + 1024)
            if r or not in_buf:
                out[cid] = {"error": f"malloc {r}"}
                return
            vdec = dvpp_vdec.DvppVdec(cid, w, h, const.ENTYPE_H264_HIGH, ctx)
            if vdec.init() != const.SUCCESS:
                out[cid] = {"error": "vdec_init_failed"}
                return
            got = fed = 0
            t0 = time.perf_counter()
            for au in aus:
                ptr = acl.util.bytes_to_ptr(au) if "bytes_to_ptr" in dir(acl.util) \
                    else acl.util.numpy_to_ptr(np.frombuffer(au, np.uint8))
                rr = acl.rt.memcpy(in_buf, len(au), ptr, len(au), const.ACL_MEMCPY_HOST_TO_DEVICE)
                if rr:
                    break
                if vdec.process(in_buf, len(au), [cid, fed]):
                    break
                fed += 1
                while got < fed:
                    _, image = vdec.read(no_wait=True)
                    if image is None:
                        break
                    got += 1
            feed_wall = time.perf_counter() - t0
            while got < fed:
                _, image = vdec.read(no_wait=True)
                if image is None:
                    break
                got += 1
            vdec.destroy()
            out[cid] = {"fed": fed, "decoded": got, "feed_wall": round(feed_wall, 3)}
        except Exception as exc:  # noqa: BLE001
            out[cid] = {"error": str(exc)}

    cpu0 = _cpu()
    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(c,)) for c in range(nchan)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    cpu = _cpu() - cpu0

    per = {c: out.get(c, {}) for c in range(nchan)}
    ok = {c: v for c, v in per.items() if "decoded" in v}
    total_frames = sum(v["decoded"] for v in ok.values())
    max_feed = max((v["feed_wall"] for v in ok.values()), default=0)
    return {
        "channels": nchan, "per_channel": per,
        "channels_succeeded": len(ok), "total_decoded_frames": total_frames,
        "concurrent_wall_s": round(wall, 3), "aggregate_fps": round(total_frames / wall, 1) if wall > 0 else 0,
        "total_cpu_s": round(cpu, 3), "total_cpu_cores": round(cpu / wall, 2) if wall > 0 else None,
        "cpu_s_per_frame": round(cpu / total_frames * 1000, 2) if total_frames else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--out", default="results/edge_test/g4_vdec_probe.json")
    ap.add_argument("--channels", type=int, default=1, help="并发解码路数（>1 走多路聚合测试）")
    args = ap.parse_args()

    import acl  # noqa: E402  (需先 source CANN)
    import constants as const  # noqa: E402
    import dvpp_vdec as dvpp_vdec  # noqa: E402

    work = "/tmp/g4_vdec_probe"
    os.makedirs(work, exist_ok=True)
    h264 = os.path.join(work, "probe.h264")
    w, h = ffprobe_size(args.src)
    to_annexb(args.src, h264)
    raw = open(h264, "rb").read()
    aus = split_access_units(raw)[:args.frames]
    max_au = max(len(a) for a in aus)
    print(f"[g4] {args.src} {w}x{h} 取到 {len(aus)} 个 AU，最大 {max_au}B")

    # 多路并发：在 acl.init 之前 fork，子进程各自 init，避免继承父设备句柄
    if args.channels > 1:
        mc = run_multichannel_mp(aus, w, h, max_au, args.channels)
        mc["src"] = args.src
        mc["size"] = [w, h]
        mc["frames_per_channel"] = len(aus)
        os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
        with open(args.out, "w") as f:
            json.dump(mc, f, ensure_ascii=False, indent=1)
        print(json.dumps(mc, ensure_ascii=False, indent=1))
        sys.exit(0)

    ret = acl.init()
    assert ret == 0, f"acl.init={ret}"
    acl.rt.set_device(0)

    ctx, _ = acl.rt.create_context(0)
    acl.rt.set_context(ctx)

    in_buf, mret = acl.media.dvpp_malloc(max_au + 1024)
    assert mret == 0 and in_buf, f"dvpp_malloc={mret}"

    vdec = dvpp_vdec.DvppVdec(0, w, h, const.ENTYPE_H264_HIGH, ctx)
    if vdec.init() != const.SUCCESS:
        print("[g4] VDEC init 失败（硬件解码不可用），退出")
        sys.exit(3)

    # ── 硬件解码计时（拆 feed / drain，暴露硬件真实能力）──
    cpu0 = _cpu()
    t_start = time.perf_counter()
    got = 0
    first_yuv = None
    fed = 0
    first_decode_lat = None
    for au in aus:
        ptr = acl.util.bytes_to_ptr(au) if "bytes_to_ptr" in dir(acl.util) \
            else acl.util.numpy_to_ptr(np.frombuffer(au, np.uint8))
        r = acl.rt.memcpy(in_buf, len(au), ptr, len(au), const.ACL_MEMCPY_HOST_TO_DEVICE)
        if r:
            print(f"[g4] memcpy→dvpp 失败 {r}"); break
        if vdec.process(in_buf, len(au), [0, fed]):
            print(f"[g4] process 失败 @{fed}"); break
        fed += 1
        # 边喂边抽（回调线程异步解码，队列上限64）
        while got < fed:
            r2, image = vdec.read(no_wait=True)
            if image is None:
                break
            if first_yuv is None:
                first_yuv = image.byte_data_to_np_array().astype(np.uint8).reshape(-1).copy()
                first_decode_lat = time.perf_counter() - t_start
            got += 1
    t_feed_end = time.perf_counter()
    got_at_feed_end = got
    pending_at_feed_end = fed - got
    # 末段非阻塞抽干：不发 EOS 时 B 帧重排缓冲的最后几帧不会 flush，
    # 用阻塞读会撞满 READ_TIMEOUT（假象成"解码慢"）；硬件速率以 feed 窗口为准。
    while got < fed:
        r2, image = vdec.read(no_wait=True)
        if image is None:
            break
        got += 1
    t_drain_end = time.perf_counter()
    feed_wall = t_feed_end - t_start
    drain_tail = t_drain_end - t_feed_end
    hw_wall = t_drain_end - t_start
    hw_cpu = _cpu() - cpu0
    # 喂数窗口内硬件已完成 got_at_feed_end 帧 → 该速率是硬件解码吞吐（feed-bound 下界）
    hw_decode_fps = got_at_feed_end / feed_wall if feed_wall > 0 else 0
    decode_kept_up = pending_at_feed_end < 64  # 硬件基本跟上喂数，说明解码非瓶颈
    vdec.destroy()

    # ── cv2 软解基线（同帧数）──
    cpu1 = _cpu()
    ts = time.perf_counter()
    cap = cv2.VideoCapture(args.src)
    sw_frames = 0
    while sw_frames < fed:
        ok, _ = cap.read()
        if not ok:
            break
        sw_frames += 1
    sw_wall = time.perf_counter() - ts
    sw_cpu = _cpu() - cpu1
    cap.release()

    # 首帧存证：NV12 → BGR
    frame_ok = False
    if first_yuv is not None and first_yuv.size >= w * h * 3 // 2:
        nv12 = first_yuv[:w * h * 3 // 2].reshape(h * 3 // 2, w)
        bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
        cv2.imwrite(os.path.join(work, "g4_vdec_frame0.jpg"), bgr)
        frame_ok = True

    res = {
        "src": args.src, "size": [w, h], "fed": fed, "hw_decoded": got,
        "hw_wall_s": round(hw_wall, 3), "hw_fps": round(got / hw_wall, 2) if hw_wall > 0 else 0,
        "hw_ms_per_frame": round(hw_wall / got * 1000, 2) if got else None,
        "hw_cpu_s": round(hw_cpu, 3), "hw_cpu_cores": round(hw_cpu / hw_wall, 2) if hw_wall > 0 else None,
        "feed_wall_s": round(feed_wall, 3), "feed_fps": round(fed / feed_wall, 1) if feed_wall > 0 else 0,
        "drain_tail_s": round(drain_tail, 3),
        "got_at_feed_end": got_at_feed_end, "pending_at_feed_end": pending_at_feed_end,
        "hw_decode_fps": round(hw_decode_fps, 1), "decode_kept_up": decode_kept_up,
        "first_decode_lat_ms": round(first_decode_lat * 1000, 1) if first_decode_lat else None,
        "sw_cv2_frames": sw_frames, "sw_wall_s": round(sw_wall, 3),
        "sw_fps": round(sw_frames / sw_wall, 2) if sw_wall > 0 else 0,
        "sw_cpu_s": round(sw_cpu, 3), "sw_cpu_cores": round(sw_cpu / sw_wall, 2) if sw_wall > 0 else None,
        "first_frame_saved": frame_ok, "frame_path": os.path.join(work, "g4_vdec_frame0.jpg"),
        "verdict": "PASS" if (got >= fed * 0.9 and frame_ok and hw_cpu < sw_cpu) else "CHECK",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
    with open(args.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    acl.rt.destroy_context(ctx)
    acl.rt.set_device(0)  # noqa
    print(f"[g4] 证据写入 {args.out}；首帧 {res['frame_path']}")


if __name__ == "__main__":
    main()
