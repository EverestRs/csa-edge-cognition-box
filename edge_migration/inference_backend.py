from __future__ import annotations

"""推理后端抽象：ONNX Runtime (CPU/CUDA) 与昇腾 OM (pyACL) 统一接口。

对齐 src/core/detector.py 的 session 创建风格（:110-120 provider 选择、
graph optimization 级联），但把"模型可用性"收敛到本模块：
create_backend 永不抛异常——失败返回 NullBackend（run 输出空列表），
由调用方按 is_real / last_error 走既有的优雅降级路径。
"""

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - 开发机通常可用
    ort = None


class BackendUnavailable(RuntimeError):
    """后端不可用（缺依赖/加载失败）。内部使用，不越过 create_backend 边界。"""


# 同一进程内 detector + pose 可能各建一个 AscendOMBackend；
# pyACL 的 acl.init 不能重复调用，这里做进程级引用计数，只初始化一次。
_ACL_INIT_LOCK = threading.Lock()
_ACL_INIT_COUNT = 0


# --------------------------------------------------------------------------- #
# ONNX Runtime 后端
# --------------------------------------------------------------------------- #
class OnnxRuntimeBackend:
    kind = "onnxruntime"

    def __init__(self, path: str, runtime: dict[str, Any] | None = None):
        self.path = str(path)
        self.last_error = ""
        self.is_real = False
        self._session = None
        self._input_name = ""
        self._input_shape: tuple[Any, ...] = ()
        self._output_names: list[str] = []
        if ort is None:
            self.last_error = "onnxruntime_not_installed"
            return
        try:
            self._session = self._create_session(runtime or {})
            inp = self._session.get_inputs()[0]
            self._input_name = str(inp.name)
            self._input_shape = tuple(inp.shape)
            self._output_names = [str(o.name) for o in self._session.get_outputs()]
            self.is_real = True
        except Exception as exc:  # 加载失败 → 记录错误，run 返回空
            self._session = None
            self.is_real = False
            self.last_error = f"onnx_session_create_failed: {exc}"

    def _create_session(self, runtime: dict[str, Any]):
        """镜像 detector 的 provider 选择与优化级联（简化版）。"""
        available = set(ort.get_available_providers())
        provider = str(runtime.get("execution_provider", "CPUExecutionProvider"))
        device = str(runtime.get("device", "cpu")).lower()
        providers: list[str] = []
        if provider in available:
            providers.append(provider)
        elif device in {"cuda", "gpu"} and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        if "CPUExecutionProvider" in available and "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")
        if not providers:
            providers = ["CPUExecutionProvider"]

        options = ort.SessionOptions()
        intra = int(runtime.get("intra_op_num_threads", 0) or 0)
        inter = int(runtime.get("inter_op_num_threads", 0) or 0)
        if intra > 0:
            options.intra_op_num_threads = intra
        if inter > 0:
            options.inter_op_num_threads = inter
        options.enable_mem_pattern = bool(runtime.get("enable_mem_pattern", True))
        return ort.InferenceSession(self.path, sess_options=options, providers=providers)

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        if self._session is None:
            return []
        return list(self._session.run(self._output_names, {self._input_name: tensor}))

    @property
    def input_name(self) -> str:
        return self._input_name

    @property
    def input_shape(self) -> tuple[Any, ...]:
        return self._input_shape


# --------------------------------------------------------------------------- #
# 昇腾 OM 后端（pyACL）—— 仅在盒上可运行；x86 上 import acl 失败即降级
# 约定自适应（FIELD_READOUT 会商项A）：盒上 CANN 6.2.RC2 为旧式 pyACL——
#   load_from_file→(model_id, rc)、create_desc 直返指针、dims 函数名 get_input_dims；
#   新式为 (rc, value)。_pair/_call 两种约定统一归一，函数名按候选探测；
#   旧接口传参错型会 segfault → 只调用"名字存在"的接口，绝不猜签名。
# --------------------------------------------------------------------------- #
_DTYPE_BY_ACL = {}  # 延迟填充：{acl.DT_FLOAT: np.float32, ...}


def _pair(res):
    """归一 pyACL 返回约定：(rc, value) / (value, rc) / 直返 value → (rc, value)。"""
    if isinstance(res, tuple) and len(res) == 2:
        a, b = res
        ai, bi = isinstance(a, int), isinstance(b, int)
        if ai and bi:
            if a == 0:
                return 0, b
            if b == 0:
                return 0, a
            return a, b  # 双非零：保守按 rc 在前（调用方检查 rc!=0 即报错）
        if ai:
            return a, b  # (rc, value)
        if bi:
            return 0 if b == 0 else 1, a  # (value, rc)
    return 0, res


def _acl_memcpy(acl, dst: int, dst_bytes: int, src: int, src_bytes: int, h2d: bool) -> int:
    """盒上适配(2026-11-05 会商项A): 旧式 pyACL 无 memcpy_h2d/d2h，只有 5 参 memcpy
    (dst, dbytes, src, sbytes, kind)；新式走 memcpy_h2d/d2h 4 参。自动按存在的函数分派。"""
    named = getattr(acl.rt, "memcpy_h2d" if h2d else "memcpy_d2h", None)
    if named is not None:
        try:
            return _rc(named(int(dst), int(dst_bytes), int(src), int(src_bytes)))
        except TypeError:
            pass
    kind = getattr(acl, "ACL_MEMCPY_HOST_TO_DEVICE" if h2d else "ACL_MEMCPY_DEVICE_TO_HOST",
                   1 if h2d else 2)  # 官方 constants: H2D=1, D2H=2
    return _rc(acl.rt.memcpy(int(dst), int(dst_bytes), int(src), int(src_bytes), int(kind)))


def _rc(res) -> int:
    """返回值归一状态码：int 直返 / (rc, value) / (value, rc)。"""
    if isinstance(res, int):
        return res
    if isinstance(res, tuple) and len(res) == 2:
        rc, _v = _pair(res)
        return int(rc) if isinstance(rc, int) else 0
    return 0


def _normalize_dims(dims, mdl) -> tuple[int, ...]:
    """把 dims 结构（dict dim_count/dims、get_dim_by_index 结构体、纯 list）归一为 shape。"""
    rc, d = _pair(dims)
    if d is None:
        return ()
    if isinstance(d, (list, tuple)):
        return tuple(int(x) for x in d)
    if isinstance(d, dict):
        if "dims" in d and isinstance(d["dims"], (list, tuple)):
            return tuple(int(x) for x in d["dims"])
        count = int(d.get("dim_count", 0) or 0)
        out = []
        getter = getattr(mdl, "get_dim_by_index", None)
        for i in range(count):
            if getter is None:
                return ()
            rc2, dim = _pair(getter(d, i))
            if rc2 != 0:
                return ()
            out.append(int(dim))
        return tuple(out)
    return ()


class AscendOMBackend:
    kind = "ascend_om"

    def __init__(self, path: str, *, device_id: int = 0,
                 input_shape: tuple[int, ...] | None = None):
        self.path = str(path)
        self.device_id = int(device_id)
        self.is_real = False
        self.last_error = ""
        self.kind = "ascend_om"
        self.input_name = "input"
        self.api_detected: dict[str, str] = {}  # 探测到的旧/新式接口名（供探针对照）
        self._acl = None
        self._model_id = None
        self._desc = None
        self._tls = threading.local()  # 每执行线程的上下文就绪标记（pyACL 上下文线程局部）
        # 设备缓冲复用：懒预分配一次、逐帧复用（省去每帧 malloc/free+dataset 重建，约半帧延迟）。
        # 引擎为共享单例、流水线工作线程并发调用，故用锁串行化共享缓冲的 memcpy+execute。
        self._buf_lock = threading.Lock()
        self._dev_in = None
        self._ds_in = None
        self._dev_outs: list[tuple[int, int, tuple[int, ...]]] = []
        self._ds_out = None
        self._input_shape = input_shape
        self._input_size = 0
        self._input_dtype = np.float32
        self._output_specs: list[tuple[int, np.dtype, tuple[int, ...]]] = []
        self._acl_init_acquired = False
        try:
            self._load()
        except BackendUnavailable as exc:
            self.last_error = str(exc)
            self._release_acl_init()
        except Exception as exc:  # pyACL 接口差异兜底：任何加载错误都不越界
            self.last_error = f"om_load_failed: {exc}"
            self._release_acl_init()

    def _load(self) -> None:
        global _ACL_INIT_COUNT
        try:
            import acl  # type: ignore
        except Exception as exc:
            raise BackendUnavailable(f"pyacl_unavailable: {exc}") from exc
        if not Path(self.path).exists():
            raise BackendUnavailable(f"om_file_missing: {self.path}")

        self._acl = acl
        with _ACL_INIT_LOCK:
            if _ACL_INIT_COUNT == 0:
                if acl.init() != 0:
                    raise BackendUnavailable("acl_init_failed")
            _ACL_INIT_COUNT += 1
        self._acl_init_acquired = True
        acl.rt.set_device(self.device_id)

        # ── 加载（新旧约定自适应） ──
        rc, model_id = _pair(acl.mdl.load_from_file(self.path))
        if rc != 0 or model_id is None:
            raise BackendUnavailable(f"mdl_load_failed rc={rc}")
        self._model_id = model_id
        rc, desc = _pair(acl.mdl.create_desc())
        if desc is None or rc != 0:
            raise BackendUnavailable(f"mdl_create_desc_failed rc={rc}")
        self._desc = desc

        # 盒上适配(2026-11-05 会商项A): 旧式 pyACL 的 create_desc() 只造空壳，
        # 必须 get_desc(desc, model_id) 把模型元数据灌进去（官方 sample 同款步骤），
        # 否则 get_num_outputs/get_*_dims 全部返回 0/空。
        _bind = getattr(acl.mdl, "get_desc", None)
        if _bind is not None:
            try:
                rc_bind = _rc(_bind(desc, model_id))  # 此板裸 int 返回，_rc 归一
                if rc_bind != 0:
                    raise BackendUnavailable(f"mdl_get_desc_failed rc={rc_bind}")
            except BackendUnavailable:
                raise
            except Exception as exc:
                raise BackendUnavailable(f"mdl_get_desc_failed: {exc}")

        for key, npd in (("DT_FLOAT", np.float32), ("DT_HALF", np.float16),
                         ("DT_INT8", np.int8), ("DT_UINT8", np.uint8),
                         ("DT_INT32", np.int32)):
            const = getattr(acl, key, None)
            if const is not None:
                _DTYPE_BY_ACL[int(const)] = npd

        # ── 输入 dims（按存在函数名探测；只传 desc，旧接口传 model_id 会 segfault）──
        shape: tuple[int, ...] = ()
        for fn_name, args in (("get_input_dims", (desc, 0)),      # CANN 6.2 盒上：双参形态
                              ("get_input_dims", (desc,)),
                              ("get_input_shape_by_index", (desc, 0)),
                              ("get_input_shape", (desc,))):
            fn = getattr(acl.mdl, fn_name, None)
            if fn is None:
                continue
            try:
                shape = _normalize_dims(fn(*args), acl.mdl)
            except Exception:
                continue
            if shape:
                self.api_detected["dims_fn"] = fn_name
                break
        if self._input_shape is None or not self._input_shape:
            self._input_shape = shape
        if not self._input_shape:
            raise BackendUnavailable("om_input_shape_unknown")

        dt = None
        for fn_name, args in (("get_input_data_type", (desc, 0)),   # CANN 6.2 盒上：双参形态
                              ("get_input_data_type", (desc,)),
                              ("get_input_dtype", (desc,)),
                              ("get_input_data_type_by_index", (desc, 0))):
            fn = getattr(acl.mdl, fn_name, None)
            if fn is None:
                continue
            try:
                dt = _pair(fn(*args))[1]
            except Exception:
                continue
            if isinstance(dt, int):
                self.api_detected["dtype_fn"] = fn_name
                break
        np_dtype = _DTYPE_BY_ACL.get(int(dt), np.float32) if isinstance(dt, int) else np.float32
        n = 1
        for d in self._input_shape:
            n *= max(int(d), 1)
        self._input_size = int(n * np.dtype(np_dtype).itemsize)
        self._input_dtype = np_dtype

        # ── 输出规格 ──
        n_out = 0
        for fn_name, args in (("get_num_outputs", (desc,)),        # CANN 6.2 盒上真实名
                              ("get_output_num", (desc,)),
                              ("get_output_num_by_desc", (desc,))):
            fn = getattr(acl.mdl, fn_name, None)
            if fn is None:
                continue
            try:
                rc2, v = _pair(fn(*args))
                if rc2 == 0 and isinstance(v, int) and v >= 1:
                    n_out = v
                    self.api_detected["out_num_fn"] = fn_name
                    break
            except Exception:
                continue
        size_fn = next((getattr(acl.mdl, nm, None) for nm in
                        ("get_output_size", "get_output_size_by_index")
                        if getattr(acl.mdl, nm, None) is not None), None)
        o_dims_fn = next((getattr(acl.mdl, nm, None) for nm in
                          ("get_output_dims", "get_output_shape_by_index", "get_output_shape")
                          if getattr(acl.mdl, nm, None) is not None), None)
        if size_fn is not None and n_out:
            for idx in range(n_out):
                size, dims = 0, ()
                try:
                    rc3, v = _pair(size_fn(desc, idx))
                    size = int(v) if rc3 == 0 and isinstance(v, int) else 0
                except Exception:
                    pass
                if o_dims_fn is not None:
                    try:
                        dims = _normalize_dims(o_dims_fn(desc, idx), acl.mdl)
                    except Exception:
                        pass
                self._output_specs.append((size, np.float32, dims))
        else:
            raise BackendUnavailable("om_output_specs_unknown")
        self.is_real = True

    def _ensure_thread_context(self) -> None:
        # pyACL 上下文线程局部：模型在主线程加载，但引擎在工作线程 execute，
        # 工作线程无上下文会静默返回空输出。每个执行线程首次 run 前建本线程上下文。
        if getattr(self._tls, "ctx_ready", False):
            return
        if self._acl is not None:
            self._acl.rt.set_device(self.device_id)
        self._tls.ctx_ready = True

    def _malloc(self, size: int):
        # 盒上适配: flag 必须用 NORMAL_ONLY=2（0 是 HUGE_FIRST，DK A2 hugepages
        # 常驻占满时会拿不到可执行内存/地址异常，官方 sample 同款）
        acl = self._acl
        flag = getattr(acl, "ACL_MEM_MALLOC_NORMAL_ONLY", 2)
        rc, ptr = _pair(acl.rt.malloc(size, int(flag)))
        return (ptr if rc == 0 and ptr is not None else None)

    def _add_buf(self, dataset, ptr, size: int) -> bool:
        acl = self._acl
        cdb = getattr(acl, "create_data_buffer", None)   # 官方 sample 路径
        if cdb is not None:
            # 此板 create_data_buffer 返回裸句柄 int；元组形态按约定归一取 value
            try:
                res = cdb(int(ptr), int(size))
                buf = res if isinstance(res, int) else _pair(res)[1]
                return buf is not None and _rc(acl.mdl.add_dataset_buffer(dataset, buf)) == 0
                # 注: 有 cdb 时禁止回退裸 ptr——实测会"假成功"绑入垃圾句柄→device 145
            except Exception:
                return False
        for keys in ({"buf": ptr, "size": size}, {"buffer": ptr, "size": size}):
            try:
                if _rc(acl.mdl.add_dataset_buffer(dataset, keys)) == 0:
                    return True
            except Exception:
                continue
        try:  # 无 create_data_buffer 的旧接口: 两参形态
            return _rc(acl.mdl.add_dataset_buffer(dataset, ptr)) == 0
        except Exception:
            return False

    def _init_buffers(self) -> bool:
        """懒预分配设备输入/输出缓冲与 dataset（仅一次），失败清理返回 False。"""
        acl = self._acl
        dev_in = self._malloc(self._input_size)
        if dev_in is None:
            return False
        ds_in = acl.mdl.create_dataset()
        if isinstance(ds_in, tuple):
            ds_in = _pair(ds_in)[1]
        if not self._add_buf(ds_in, dev_in, self._input_size):
            acl.rt.free(dev_in)
            return False
        dev_outs: list[tuple[int, int, tuple[int, ...]]] = []
        ds_out = acl.mdl.create_dataset()
        if isinstance(ds_out, tuple):
            ds_out = _pair(ds_out)[1]
        for size, _dtype, dims in self._output_specs:
            size = max(int(size), 1)
            buf = self._malloc(size)
            if buf is None or not self._add_buf(ds_out, buf, size):
                if buf is not None:
                    acl.rt.free(buf)
                break
            dev_outs.append((buf, size, dims))
        if len(dev_outs) != len(self._output_specs):
            for buf, _s, _d in dev_outs:
                try:
                    acl.rt.free(buf)
                except Exception:
                    pass
            try:
                acl.rt.free(dev_in)
            except Exception:
                pass
            return False
        self._dev_in, self._ds_in = dev_in, ds_in
        self._dev_outs, self._ds_out = dev_outs, ds_out
        return True

    def _free_buffers(self) -> None:
        acl = self._acl
        if acl is None:
            return
        for buf, _s, _d in self._dev_outs:
            try:
                acl.rt.free(buf)
            except Exception:
                pass
        if self._dev_in is not None:
            try:
                acl.rt.free(self._dev_in)
            except Exception:
                pass
        self._dev_in, self._ds_in = None, None
        self._dev_outs, self._ds_out = [], None

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        acl = self._acl
        if acl is None or not self.is_real:
            return []
        self._ensure_thread_context()
        try:
            host_in = np.ascontiguousarray(tensor)
            with self._buf_lock:
                if self._dev_in is None:
                    if not self._init_buffers():
                        self.last_error = "om_buffer_alloc_failed"
                        return []
                rc = _acl_memcpy(acl, int(self._dev_in), self._input_size,
                                 int(host_in.__array_interface__["data"][0]),
                                 int(host_in.nbytes), h2d=True)
                if rc != 0:
                    return []
                rc = _rc(acl.mdl.execute(self._model_id, self._ds_in, self._ds_out))
                results: list[np.ndarray] = []
                if rc == 0:
                    for (buf, size, dims), (_s, dtype, _d) in zip(self._dev_outs, self._output_specs):
                        host = np.empty(size // np.dtype(dtype).itemsize, dtype=dtype)
                        rc2 = _acl_memcpy(acl, int(host.__array_interface__["data"][0]),
                                          int(host.nbytes), int(buf), int(size), h2d=False)
                        if rc2 == 0:
                            results.append(host.reshape(dims) if dims else host)
                return results
        except Exception as exc:  # execute 路径任何异常都降级为空输出
            self.last_error = f"om_run_failed: {exc}"
            return []

    def _release_acl_init(self) -> None:
        """释放本后端的 acl.init 引用；最后一个后端释放时才 reset/finalize。"""
        global _ACL_INIT_COUNT
        if not getattr(self, "_acl_init_acquired", False):
            return
        with _ACL_INIT_LOCK:
            if _ACL_INIT_COUNT > 0:
                _ACL_INIT_COUNT -= 1
            last = _ACL_INIT_COUNT == 0
        self._acl_init_acquired = False
        if last and self._acl is not None:
            try:
                self._acl.rt.reset_device(self.device_id)
                self._acl.finalize()
            except Exception:
                pass

    def unload(self) -> None:
        try:
            with self._buf_lock:
                self._free_buffers()
            if self._acl is not None and self._model_id is not None:
                self._acl.mdl.unload(self._model_id)
                self._model_id = None
            self._release_acl_init()
        except Exception:
            pass
        self.is_real = False


# --------------------------------------------------------------------------- #
# 空后端（保持接口一致，供降级路径）
# --------------------------------------------------------------------------- #
class NullBackend:
    kind = "null"

    def __init__(self, reason: str):
        self.is_real = False
        self.last_error = reason
        self.input_name = "images"
        self.input_shape = ()

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        return []


def create_backend(path: str, *, fallback_onnx: str | None = None,
                   runtime: dict[str, Any] | None = None,
                   device_id: int = 0) -> Any:
    """按扩展名分流创建后端；永不抛异常。

    链路：.om → AscendOMBackend；失败且给了 fallback_onnx → ORT 兜底；
    .onnx → ORT 直连；一切失败 → NullBackend(调用方按 is_real=False 走空结果降级)。
    """
    p = str(path)
    last = ""
    if p.endswith(".om"):
        om = AscendOMBackend(p, device_id=device_id)
        if om.is_real:
            return om
        last = om.last_error
        if fallback_onnx:
            fb = OnnxRuntimeBackend(fallback_onnx, runtime)
            fb.last_error = f"[om_fallback] {last} | {fb.last_error}".strip(" |")
            if fb.is_real:
                return fb
            return NullBackend(fb.last_error or last)
        return NullBackend(last or "om_unavailable")
    if p.endswith(".onnx"):
        b = OnnxRuntimeBackend(p, runtime)
        return b if b.is_real else NullBackend(b.last_error)
    return NullBackend(f"unknown_model_ext: {p}")


# --------------------------------------------------------------------------- #
# ORT 兼容适配层：让 detector/pose_estimator 的现有后处理代码直接吃 OM 输出
# --------------------------------------------------------------------------- #
class _TensorMeta:
    def __init__(self, name: str, shape: tuple | list):
        self.name = str(name)
        self.shape = list(shape)


class OMSessionAdapter:
    """暴露 ORT session 最小 API 面（get_inputs/get_outputs/run）。

    run(output_names, feed) 忽略 output_names 与键名（OM 按位置绑定），
    取 feed 首个张量执行——与 detector/pose 的单输入调用方式匹配。
    """

    kind = "om_session_adapter"

    def __init__(self, backend: "AscendOMBackend"):
        self._backend = backend
        self.last_error = ""

    def get_inputs(self) -> list[_TensorMeta]:
        shape = tuple(getattr(self._backend, "_input_shape", None) or ())
        return [_TensorMeta(getattr(self._backend, "input_name", "input") or "input", shape)]

    def get_outputs(self) -> list[_TensorMeta]:
        specs = getattr(self._backend, "_output_specs", []) or []
        return [_TensorMeta(f"output{i}", dims) for i, (_s, _dt, dims) in enumerate(specs)]

    def run(self, _output_names, feed: dict) -> list[np.ndarray]:
        tensor = next(iter(feed.values()))
        return self._backend.run(tensor)

    @property
    def is_real_model_loaded(self) -> bool:
        return bool(self._backend.is_real)

    def close(self) -> None:
        self._backend.unload()


def ort_compatible_om_session(om_path: str, *, device_id: int = 0
                              ) -> tuple["OMSessionAdapter | None", str]:
    """返回 (ORT 兼容会话, 错误串)；OM 不可用时 (None, reason)——由调用方走 ONNX 回退。"""
    backend = AscendOMBackend(om_path, device_id=device_id)
    if backend.is_real:
        return OMSessionAdapter(backend), ""
    return None, (backend.last_error or "om_unavailable")
