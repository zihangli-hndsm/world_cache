from __future__ import annotations

import json
import platform
import sys
from pathlib import Path


def main() -> int:
    report: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    failures: list[str] = []

    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        report["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
            x = torch.randn(512, 512, device="cuda")
            report["cuda_smoke_sum"] = float((x @ x).sum().item())
        else:
            failures.append("PyTorch cannot access CUDA")
    except Exception as exc:  # diagnostic script must emit a report on failure
        failures.append(f"torch: {exc!r}")

    for module in ("torchvision", "numpy", "yaml", "pytest", "timm", "hydra"):
        try:
            imported = __import__(module)
            report[module] = getattr(imported, "__version__", "installed")
        except Exception as exc:
            failures.append(f"{module}: {exc!r}")

    try:
        import habitat_sim

        report["habitat_sim"] = getattr(habitat_sim, "__version__", "installed")
    except Exception as exc:
        report["habitat_sim"] = f"not available: {exc!r}"

    report["failures"] = failures
    output = Path("outputs/reports/environment_report.txt")
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

