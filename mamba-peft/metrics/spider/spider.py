from metrics.spider.evaluation import build_foreign_key_map_from_json, evaluate
from utils.utils import flatten_dict
import os
from pathlib import Path
from typing import Optional


class SpiderMetric():
    def __init__(self) -> None:
        # Configure NLTK search path from environment if provided
        self._configure_nltk_from_env()
        # Priority:
        # 1) SPIDER_LOCAL_DIR (if set and exists)
        # 2) /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/spider_data (if exists)
        # 3) fallback to data/xlangai_spider/spider
        root_env = os.environ.get("SPIDER_LOCAL_DIR")
        explicit_root = Path("/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/spider_data")
        root_path: Optional[Path] = None

        if root_env and Path(root_env).exists():
            root_path = Path(root_env)
        elif explicit_root.exists():
            root_path = explicit_root

        if root_path is not None:
            self.db_dir = str(root_path / "database")
            self.table = str(root_path / "tables.json")
        else:
            self.db_dir = "data/xlangai_spider/spider/database"
            self.table = "data/xlangai_spider/spider/tables.json"

        self.etype = "all"
        self.kmaps = build_foreign_key_map_from_json(self.table)

    def _configure_nltk_from_env(self) -> None:
        """
        If NLTK_DATA is defined, ensure nltk searches that directory first.
        This allows users to place/download tokenizers (e.g., punkt, punkt_tab) into a project-local path.
        """
        nltk_path = os.environ.get("NLTK_DATA")
        if not nltk_path:
            return
        try:
            import nltk  # type: ignore
            from nltk import data as nltk_data  # type: ignore
            p = Path(nltk_path)
            if not p.exists():
                # Create the directory if it does not exist to avoid surprises when users download resources into it.
                p.mkdir(parents=True, exist_ok=True)
            # Prepend to search path so it takes precedence over system locations.
            if nltk_path not in nltk_data.path:
                nltk_data.path.insert(0, nltk_path)
        except Exception:
            # Best-effort: if nltk is unavailable or any error occurs, we don't block initialization here.
            pass

    def compute(self, predictions, references):
        out = evaluate(references, predictions, self.db_dir, self.etype, self.kmaps)
        out = flatten_dict(out, sep="/")
        out = {k: v for k, v in out.items()
               if not any(excl in k for excl in ["/partial", "/count"])}
        return out