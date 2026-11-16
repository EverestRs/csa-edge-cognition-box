from __future__ import annotations

"""视频流读取四档后端（T2.2，01 §2.2）：cv 软解 / ffmpeg 管道 / vdec 硬解 / auto 自动降级。

- 目标是"解码 CPU 预算"：A=cv（子码流）→ B=ffmpeg 管道 → V=vdec（昇腾 DVPP 硬件 H.264 解码，
  显式 source_backend=vdec 才启用，见 §8.5 g4 探针实证 464fps/省 63% CPU）；
  auto 在 cv 打不开或持续读失败时自动切 ffmpeg；
- read() -> (ok, frame, frame_id) 契约与断线重连语义与旧版完全一致；
- ffmpeg 后端经 rawvideo/bgr24 管道出帧，分辨率固定；vdec 后端 ffmpeg 仅 demux，解码走 ASIC。
"""

import os
import shutil
import subprocess
import time
from collections import deque
from typing import Any

import cv2
import numpy as np

# 昇腾 DVPP VDEC 硬件解码后端所需（仅盒上、显式 source_backend=vdec 时启用）
# acllite 包内用 `import constants` 等平铺导入，需把 acllite 目录本身加入 sys.path
_ACLLLITE_DIR = "/usr/local/Ascend/thirdpart/aarch64/acllite"
_VDEC_CHN_SEQ = 0


class _CvBackend:
    name = "cv"

    def __init__(self, url: str):
        self.cap = cv2.VideoCapture(url)
        if url.startswith("rtsp://"):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    def opened(self) -> bool:
        return bool(self.cap.isOpened())

    def fps(self) -> float:
        try:
            v = float(self.cap.get(cv2.CAP_PROP_FPS) or 0)
        except Exception:
            v = 0.0
        return v if v > 1.0 else 0.0

    def read(self):
        return self.cap.read()

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass


class _FfmpegBackend:
    name = "ffmpeg"

    def __init__(self, url: str, width: int = 1280, height: int = 720):
        self.url = url
        self.width, self.height = int(width), int(height)
        self._frame_bytes = self.width * self.height * 3
        self._proc = None
        self._start()

    def _cmd(self) -> list[str]:
        ffmpeg = shutil.which("ffmpeg")
        base = [ffmpeg or "ffmpeg", "-loglevel", "error"]
        url = self.url
        is_device = url.isdigit()
        if url.startswith("rtsp://"):
            base += ["-rtsp_transport", "tcp"]
        elif not url.startswith(("http", "/")) and not is_device:
            base += ["-f", "lavfi"]  # 合成测试源（testsrc=…），与 g2 探针口径一致
        base += ["-i", url, "-vf", f"scale={self.width}:{self.height}",
                 "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
        return base

    def _start(self) -> bool:
        try:
            self._proc = subprocess.Popen(self._cmd(), stdout=subprocess.PIPE,
                                          stderr=subprocess.DEVNULL)
            return self._proc.poll() is None
        except Exception:
            self._proc = None
            return False

    def opened(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def fps(self) -> float:
        return 0.0  # 管道无元数据；上层默认 25

    def read(self):
        if not self.opened() or self._proc is None or self._proc.stdout is None:
            return False, None
        buf = self._proc.stdout.read(self._frame_bytes)
        if len(buf) < self._frame_bytes:
            return False, None
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(self.height, self.width, 3)
        return True, frame

    def release(self):
        try:
            if self._proc is not None:
                if self._proc.stdout is not None:
                    self._proc.stdout.close()
                self._proc.terminate()
                self._proc.wait(timeout=3)
        except Exception:
            try:
                if self._proc is not None:
                    self._proc.kill()
            except Exception:
                pass
        self._proc = None


class _VdecBackend:
    """昇腾 DVPP VDEC 硬件 H.264 解码后端（盒上，显式 source_backend=vdec）。

    链路：ffmpeg 只 demux（-c copy -bsf h264_mp4toannexb）出 annexb → 流式切访问单元(AU)
    → acllite.DvppVdec 硬解出 NV12 → cv2 转 BGR。解码卸载到 ASIC，A55 仅承担 demux+转色。
    假设每帧单 slice（本项目片源/多数摄像头子码流满足）；不满足时表现为丢帧而非崩溃。
    """

    name = "vdec"

    def __init__(self, url: str):
        global _VDEC_CHN_SEQ
        self.url = url
        self.ok = False
        self._buf = b""
        self._au_q: deque[bytes] = deque()
        self._prefix = b""          # 未遇到 VCL 前的 SPS/PPS/SEI 前缀
        self._eof = False
        self._seq = 0
        self._ffmpeg = None
        self._vdec = None
        self._ctx = None
        self.width, self.height = 0, 0
        self._fps = 0.0
        self._acl = None
        self._const = None
        self._in_buf = None
        self._max_au = 2 * 1024 * 1024
        self._vpc_ok = False
        self._vpc_warned = False
        self._vpc_ctx = None
        self._vpc_desc = None
        self._stream = None
        self._out_buf = None
        self._out_desc = None
        self._vpc_in_desc = None
        try:
            self._setup()
        except Exception as exc:  # noqa: BLE001
            print(f"[vdec] 初始化失败: {exc}")
            self._cleanup()
            self.ok = False

    # ── 探测分辨率/帧率 ──
    def _probe(self):
        ffprobe = shutil.which("ffprobe") or "ffprobe"
        cmd = [ffprobe, "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height,r_frame_rate",
               "-of", "default=noprint_wrappers=1", self.url]
        out = subprocess.check_output(cmd, timeout=20).decode()
        kv = {}
        for line in out.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                kv[k.strip()] = v.strip()
        self.width, self.height = int(kv["width"]), int(kv["height"])
        rr = kv.get("r_frame_rate", "25/1")
        num, _, den = rr.partition("/")
        try:
            self._fps = float(num) / float(den or 1)
        except (ValueError, ZeroDivisionError):
            self._fps = 25.0

    def _setup(self):
        global _VDEC_CHN_SEQ
        import sys
        self._probe()
        if self.width <= 0 or self.height <= 0:
            raise RuntimeError("ffprobe 未取到分辨率")
        # ffmpeg demux → annexb h264 管道
        ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        cmd = [ffmpeg, "-loglevel", "error"]
        if self.url.startswith("rtsp://"):
            cmd += ["-rtsp_transport", "tcp"]
        cmd += ["-i", self.url, "-c:v", "copy", "-bsf:v", "h264_mp4toannexb",
                "-f", "h264", "pipe:1"]
        self._ffmpeg = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        # acl + acllite DvppVdec
        if _ACLLLITE_DIR not in sys.path:
            sys.path.insert(0, _ACLLLITE_DIR)
        import acl
        import constants as const
        import dvpp_vdec as dvpp_vdec
        from acllite_image import AclLiteImage
        self._acl, self._const = acl, const
        self._AclLiteImage = AclLiteImage
        acl.init()               # 引擎进程已 init，重复调用无副作用
        ctx, r = acl.rt.create_context(0)
        if r:
            raise RuntimeError(f"create_context={r}")
        self._ctx = ctx
        acl.rt.set_context(ctx)
        buf, r = acl.media.dvpp_malloc(self._max_au)
        if r or not buf:
            raise RuntimeError(f"dvpp_malloc={r}")
        self._in_buf = buf
        _VDEC_CHN_SEQ += 1
        self._chn = _VDEC_CHN_SEQ % 32
        self._vdec = dvpp_vdec.DvppVdec(self._chn, self.width, self.height,
                                        const.ENTYPE_H264_HIGH, ctx)
        if self._vdec.init() != const.SUCCESS:
            raise RuntimeError("DvppVdec.init 失败")
        # VPC 硬件转色默认关闭：acllite.DvppVdec 回调线程与 VPC 存在 context 死结（见 §8.6，
        # 累计 100000/107020/507001 + 回调需 ACL 托管线程）。设 CSA_VDEC_VPC=1 可作实验性开启
        # （预期失败并自愈回退 cv2），供赛后同步原生重写时参考；生产保持关闭，转色走 cv2。
        self._vpc_ok = os.getenv("CSA_VDEC_VPC") == "1" and self._setup_vpc(acl, const)
        self.ok = True

    def _setup_vpc(self, acl, const) -> bool:
        """建 VPC 硬件转色通道 YUV420SP→BGR888（失败则回退 cv2.cvtColor）。
        注意：与 acllite.DvppVdec 的回调线程存在 context 死结（共用→107020 竞态，
        分开→507001 跨 context 读不到 vdec 输出缓冲），纯 Python 下无法两全，
        故 VPC 实际会失败并自愈回退 cv2；真正生效需改用同步原生 vdec（弃用回调线程）。"""
        try:
            self._sw = (self.width + 15) // 16 * 16
            self._sh = (self.height + 1) // 2 * 2
            self._bgr_size = self._sw * self._sh * 3
            self._vpc_desc = acl.media.dvpp_create_channel_desc()
            if acl.media.dvpp_create_channel(self._vpc_desc):
                return False
            self._stream, r = acl.rt.create_stream()
            if r:
                return False
            self._out_buf, r = acl.media.dvpp_malloc(self._bgr_size)
            if r or not self._out_buf:
                return False
            self._out_desc = acl.media.dvpp_create_pic_desc()
            acl.media.dvpp_set_pic_desc_data(self._out_desc, self._out_buf)
            acl.media.dvpp_set_pic_desc_format(self._out_desc, const.PIXEL_FORMAT_BGR_888)
            # 官方约束：convert_color 的 outputDesc width/height/stride 必须为 0（几何继承输入）
            acl.media.dvpp_set_pic_desc_size(self._out_desc, self._bgr_size)
            self._vpc_in_desc = acl.media.dvpp_create_pic_desc()
            self._d2h = getattr(const, "ACL_MEMCPY_DEVICE_TO_HOST", 2)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[vdec] VPC 转色初始化失败，回退 cv2: {exc}")
            return False

    def opened(self) -> bool:
        return bool(self.ok and self._vdec is not None
                    and (self._ffmpeg is None or self._ffmpeg.poll() is None))

    def fps(self) -> float:
        return self._fps if self._fps and self._fps > 1 else 0.0

    # ── 流式 AU 切分 ──
    @staticmethod
    def _find_starts(b: bytes) -> list[tuple[int, int]]:
        out = []
        i, n = 0, len(b)
        while i + 3 <= n:
            if b[i] == 0 and b[i + 1] == 0 and b[i + 2] == 1:
                out.append((i, 3))
                i += 3
            elif i + 4 <= n and b[i] == 0 and b[i + 1] == 0 and b[i + 2] == 0 and b[i + 3] == 1:
                out.append((i, 4))
                i += 4
            else:
                i += 1
        return out

    def _fill(self) -> bool:
        """从 ffmpeg 管道再读一块；返回是否还有数据。"""
        if self._eof or self._ffmpeg is None or self._ffmpeg.stdout is None:
            return False
        chunk = self._ffmpeg.stdout.read(1 << 16)
        if not chunk:
            self._eof = True
            return False
        self._buf += chunk
        return True

    def _next_au(self):
        while True:
            if self._au_q:
                return self._au_q.popleft()
            starts = self._find_starts(self._buf)
            if len(starts) < 2:
                if not self._fill():
                    return None
                continue
            nals = []
            for i in range(len(starts) - 1):
                p, sc = starts[i]
                end = starts[i + 1][0]
                body = self._buf[p + sc:end]
                if body:
                    nals.append((body[0] & 0x1F, self._buf[p:end]))
            tail = self._buf[starts[-1][0]:]
            aus = []
            cur = self._prefix
            for t, nal in nals:
                if t in (7, 8, 6, 9):
                    cur += nal
                elif t in (1, 5):
                    cur += nal
                    aus.append(cur)
                    cur = b""
            self._prefix = cur
            self._buf = tail
            if aus:
                self._au_q.extend(aus)
                return self._au_q.popleft()

    def _feed(self, au: bytes) -> bool:
        acl = self._acl
        ptr = acl.util.bytes_to_ptr(au)
        if acl.rt.memcpy(self._in_buf, len(au), ptr, len(au),
                         self._const.ACL_MEMCPY_HOST_TO_DEVICE):
            return False
        return self._vdec.process(self._in_buf, len(au), [self._chn, self._seq]) == 0

    def _pop_frame(self):
        _, image = self._vdec.read(no_wait=True)
        if image is None:
            return None
        if self._vpc_ok:
            try:
                acl = self._acl
                acl.rt.set_context(self._ctx)  # 共用 vdec context；VPC 会因回调线程竞态失败→自愈回退 cv2
                in_desc = self._vpc_in_desc
                acl.media.dvpp_set_pic_desc_data(in_desc, image.data())
                acl.media.dvpp_set_pic_desc_format(in_desc, self._const.PIXEL_FORMAT_YUV_SEMIPLANAR_420)
                acl.media.dvpp_set_pic_desc_width(in_desc, self.width)
                acl.media.dvpp_set_pic_desc_height(in_desc, self.height)
                acl.media.dvpp_set_pic_desc_width_stride(in_desc, self._sw)
                acl.media.dvpp_set_pic_desc_height_stride(in_desc, self._sh)
                acl.media.dvpp_set_pic_desc_size(in_desc, image.size)
                rc = acl.media.dvpp_vpc_convert_color_async(self._vpc_desc, in_desc, self._out_desc, self._stream)
                sr = acl.rt.synchronize_stream(self._stream)
                if rc == 0 and sr == 0:
                    out_img = self._AclLiteImage(
                        self._out_buf, self.width, self.height, self._sw, self._sh,
                        self._bgr_size, self._const.MEMORY_DVPP)
                    host = out_img.byte_data_to_np_array().astype(np.uint8)
                    bgr = host[:self._bgr_size].reshape(self._sh, self._sw, 3)[:self.height, :self.width]
                    return np.ascontiguousarray(bgr)
                if not self._vpc_warned:
                    print(f"[vdec] VPC convert rc={rc} sync={sr}，回退 cv2")
                    self._vpc_warned = True
                    self._vpc_ok = False
            except Exception as exc:  # noqa: BLE001
                if not self._vpc_warned:
                    print(f"[vdec] VPC 异常 {exc}，回退 cv2")
                    self._vpc_warned = True
                    self._vpc_ok = False
        arr = image.byte_data_to_np_array().astype(np.uint8)
        need = self.width * self.height * 3 // 2
        if arr.size < need:
            return None
        nv12 = arr[:need].reshape(self.height * 3 // 2, self.width)
        return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)

    def read(self):
        if not self.opened():
            return False, None
        # 喂 AU 直到有解码帧产出（B 帧重排有 1~2 帧延迟）
        for _ in range(64):
            au = self._next_au()
            if au is None:
                break
            if not self._feed(au):
                break
            self._seq += 1
            frame = self._pop_frame()
            if frame is not None:
                return True, frame
        frame = self._pop_frame()
        return (True, frame) if frame is not None else (False, None)

    def _cleanup(self):
        try:
            if self._vdec is not None:
                self._vdec.destroy()
        except Exception:
            pass
        try:
            if self._vpc_ok and self._acl is not None:
                if self._vpc_in_desc is not None:
                    self._acl.media.dvpp_destroy_pic_desc(self._vpc_in_desc)
                if self._out_desc is not None:
                    self._acl.media.dvpp_destroy_pic_desc(self._out_desc)
                if self._out_buf is not None:
                    self._acl.media.dvpp_free(self._out_buf)
                if self._stream is not None:
                    self._acl.rt.destroy_stream(self._stream)
                if self._vpc_desc is not None:
                    self._acl.media.dvpp_destroy_channel(self._vpc_desc)
                if self._vpc_ctx is not None:
                    self._acl.rt.destroy_context(self._vpc_ctx)
        except Exception:
            pass
        try:
            if self._ffmpeg is not None:
                if self._ffmpeg.stdout is not None:
                    self._ffmpeg.stdout.close()
                self._ffmpeg.terminate()
                self._ffmpeg.wait(timeout=3)
        except Exception:
            try:
                if self._ffmpeg is not None:
                    self._ffmpeg.kill()
            except Exception:
                pass
        try:
            if self._ctx is not None and self._acl is not None:
                self._acl.rt.destroy_context(self._ctx)
        except Exception:
            pass
        self._vdec = None
        self._ffmpeg = None

    def release(self):
        self._cleanup()
        self.ok = False


class VideoStream:
    """视频流读取 facade：本地文件 / RTSP / USB 索引，带断线重连与三档后端。"""

    def __init__(self, stream_url: str, *, source_backend: str = "auto",
                 resolution: tuple[int, int] | list[int] | None = None):
        self.stream_url = str(stream_url)
        self.frame_id = 0
        self.is_rtsp = self.stream_url.startswith("rtsp://") or self.stream_url.startswith("http://")
        self._reconnect_delay = 2.0
        self._max_reconnect_delay = 30.0
        self._backend_name = str(source_backend or "auto")
        self._resolution = (int(resolution[0]), int(resolution[1])) if resolution else (1280, 720)
        self._cv_failed = False
        self.backend = self._open()
        self.fps = self.backend.fps()
        if not self.fps or self.fps <= 0:
            self.fps = 25.0

    # ── 后端选择 ─────────────────────────────────────────────
    def _open(self):
        if self._backend_name == "vdec":
            vb = _VdecBackend(self._resolve_url())
            if vb.opened():
                return vb
            vb.release()
            # 显式 vdec 失败：不静默降级，返回未 opened 的后让上层按失败处理
            return vb
        want_ffmpeg = self._backend_name == "ffmpeg" or (
            self._backend_name == "auto" and self._cv_failed)
        if not want_ffmpeg:
            backend = _CvBackend(self._resolve_url())
            if backend.opened():
                return backend
            self._cv_failed = True
            if self._backend_name == "cv":
                return backend  # 显式 cv：不降级（上层按 opened 处理）
        return _FfmpegBackend(self._resolve_url(), *self._resolution)

    def _resolve_url(self) -> str:
        # ffmpeg 管道无法读 USB 索引；cv 路径原样
        return self.stream_url

    # ── 重连（仅 RTSP；ffmpeg 重启进程，cv 重开捕获） ──────────
    def _reconnect(self) -> bool:
        if not self.is_rtsp:
            return False
        time.sleep(min(self._reconnect_delay, self._max_reconnect_delay))
        self._reconnect_delay = min(self._reconnect_delay * 1.5, self._max_reconnect_delay)
        self.backend.release()
        self.backend = self._open()
        if self.backend.opened():
            self._reconnect_delay = 2.0
            return True
        return False

    def read(self):
        if not self.backend.opened():
            if self.is_rtsp and self._reconnect():
                pass
            else:
                return False, None, self.frame_id

        ok, frame = self.backend.read()
        if not ok:
            if self.is_rtsp and self._reconnect():
                ok, frame = self.backend.read()
                if ok:
                    self.frame_id += 1
                    return True, frame, self.frame_id
            return False, None, self.frame_id

        self.frame_id += 1
        self._reconnect_delay = 2.0
        return True, frame, self.frame_id

    def release(self):
        self.backend.release()

    # ── 诊断 ─────────────────────────────────────────────────
    def info(self) -> dict[str, Any]:
        return {"backend": getattr(self.backend, "name", "?"), "url": self.stream_url,
                "frames_read": self.frame_id, "fps": self.fps}
