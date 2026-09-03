"""Choose a local inference device without hard-coding GPU availability."""


class DeviceError(RuntimeError):
    """Raised when the requested local inference device is unavailable."""


def resolve_device(preferred: str = "auto") -> str:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    try:
        import torch
    except ImportError as exc:
        raise DeviceError(
            "PyTorch 未安装，无法运行本地 Embedding/Reranker。"
        ) from exc

    normalized = preferred.strip().lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise DeviceError("配置要求 CUDA，但 PyTorch 未检测到可用 NVIDIA GPU。")
    if normalized not in {"cuda", "cpu"}:
        raise DeviceError("RETRIEVAL_DEVICE 必须是 auto、cuda 或 cpu")
    return normalized
