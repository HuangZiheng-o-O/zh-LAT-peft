import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import yaml


def _load_email_cfg(yaml_path: Optional[str]) -> Optional[dict]:
    """
    Load email notifier config from YAML. Returns None when not found or invalid.
    """
    try:
        if yaml_path:
            p = Path(yaml_path)
        else:
            # default to repo-local dangerous/email_notify.yaml
            p = Path("dangerous/email_notify.yaml")
        if not p.is_file():
            return None
        with open(p, "r") as f:
            cfg = yaml.safe_load(f) or {}
        # basic validation
        required = ["sender_email", "receiver_email", "password", "smtp_server", "port"]
        if not all(k in cfg and cfg[k] for k in required):
            return None
        return cfg
    except Exception:
        return None


def build_email_callback(cfg: dict):
    """
    Construct EmailCallback from swanlab plugin (returns None when unavailable).
    """
    try:
        from swanlab.plugin.notification import EmailCallback  # type: ignore
    except Exception:
        return None
    return EmailCallback(
        sender_email=str(cfg["sender_email"]),
        receiver_email=str(cfg["receiver_email"]),
        password=str(cfg["password"]),
        smtp_server=str(cfg["smtp_server"]),
        port=int(cfg.get("port", 587)),
        language=str(cfg.get("language", "zh")),
    )


def send_event_email(event: str, group: Optional[str] = None, details: Optional[str] = None, yaml_path: Optional[str] = None) -> bool:
    """
    Fire-and-forget email for an event (STARTED / FINISHED / FAILED / INTERRUPTED).
    Returns True on best-effort success, False otherwise.
    """
    cfg = _load_email_cfg(yaml_path or os.environ.get("SWANLAB_EMAIL_YAML"))
    if not cfg:
        return False
    cb = build_email_callback(cfg)
    if cb is None:
        return False
    try:
        subject = f"SwanLab | {event}" + (f" | {group}" if group else "")
        content = details or ""
        cb.send_email(subject=subject, content=content)
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Send SwanLab email notification (no training code changes).")
    ap.add_argument("--yaml", type=str, default=os.environ.get("SWANLAB_EMAIL_YAML", ""), help="Path to email_notify.yaml")
    ap.add_argument("--event", type=str, required=True, choices=["STARTED", "FINISHED", "FAILED", "INTERRUPTED"])
    ap.add_argument("--group", type=str, default="", help="Optional group tag (e.g., suite/round/seed/data/output_dir)")
    ap.add_argument("--extra", type=str, default="", help="Optional extra text")
    args = ap.parse_args()
    ok = send_event_email(args.event, args.group, args.extra, yaml_path=args.yaml)
    print(f"[email_notify] event={args.event} group={args.group!r} sent={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())


