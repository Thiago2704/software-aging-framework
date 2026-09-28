from __future__ import annotations

import psutil


DEFAULT_RESOURCE_LIMITS: dict[str, float] = {
    "CPU": 100.0,
    "Mem": 16000000.0,
    "Swap": 8000000.0,
    "DiskSpace": 500000000.0,
}


def get_default_limits() -> dict[str, float]:
    """Retorna uma cópia dos limites usados no replay e como fallback."""
    return DEFAULT_RESOURCE_LIMITS.copy()


def get_system_limits() -> dict[str, float]:
    """Descobre os limites físicos do host na unidade usada pelo monitor."""
    mib = 1024**1
    swap_total = psutil.swap_memory().total / mib

    return {
        "CPU": 100.0,
        "Mem": psutil.virtual_memory().total / mib,
        "Swap": max(swap_total, 1.0),
        "DiskSpace": psutil.disk_usage("/").total / mib,
    }
