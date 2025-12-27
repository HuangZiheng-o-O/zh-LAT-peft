from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

try:
    import torch
except Exception:  # pragma: no cover - torch is expected to be available in runtime envs
    torch = None  # type: ignore


_MIB = 1024 ** 2


def _bytes_to_mib(value: int) -> float:
    return round(value / _MIB, 3)


def _reset_peak_stats() -> List["torch.device"]:
    if torch is None or not torch.cuda.is_available():
        return []
    devices = [torch.device(f"cuda:{idx}") for idx in range(torch.cuda.device_count())]
    for device in devices:
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            # Best-effort reset; skip devices that are not initialized yet
            pass
    return devices


def _collect_stats(devices: List["torch.device"]) -> Optional[Dict]:
    if torch is None or not devices:
        return None

    per_device = []
    for device in devices:
        try:
            torch.cuda.synchronize(device)
            allocated = torch.cuda.max_memory_allocated(device)
            reserved = torch.cuda.max_memory_reserved(device)
            device_name = ""
            try:
                device_name = torch.cuda.get_device_name(device)
            except Exception:
                pass
        except Exception:
            continue

        per_device.append(
            {
                "device": f"{device.type}:{device.index if device.index is not None else 0}",
                "name": device_name,
                "max_memory_allocated_mb": _bytes_to_mib(allocated),
                "max_memory_reserved_mb": _bytes_to_mib(reserved),
            }
        )

    if not per_device:
        return None

    max_alloc = max(item["max_memory_allocated_mb"] for item in per_device)
    max_reserved = max(item["max_memory_reserved_mb"] for item in per_device)
    stats = {
        "unit": "MiB",
        "measurement": "torch.cuda.max_memory_{allocated,reserved} after reset_peak_memory_stats()",
        "max_memory_allocated_mb": round(max_alloc, 3),
        "max_memory_reserved_mb": round(max_reserved, 3),
        "per_device": per_device,
    }
    return stats


def _update_metadata_file(output_dir: Path, payload: Dict) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    param_file = output_dir / "parameter_counts.json"
    data: Dict = {}
    if param_file.exists():
        try:
            with open(param_file, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
                if isinstance(loaded, dict):
                    data = loaded
        except Exception:
            data = {}

    data.update(payload)
    with open(param_file, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


@contextmanager
def gpu_memory_tracker(output_dir: str | Path):
    """
    Context manager that resets CUDA peak memory statistics and records max usage
    into parameter_counts.json once the wrapped block finishes.
    """
    devices = _reset_peak_stats()
    try:
        yield
    finally:
        stats = _collect_stats(devices)
        if stats:
            _update_metadata_file(Path(output_dir), {"gpu_memory": stats})
            print(
                f"[gpu-memory] peak_alloc={stats['max_memory_allocated_mb']} MiB, "
                f"peak_reserved={stats['max_memory_reserved_mb']} MiB "
                f"(saved to parameter_counts.json)"
            )
