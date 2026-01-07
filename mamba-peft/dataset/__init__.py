import datetime

def _debug_print(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[DEBUG][{ts}] [dataset/__init__] {msg}", flush=True)

_debug_print("START importing dataset package...")

_debug_print("  Importing numpy...")
import numpy as np
_debug_print("  numpy... OK")

_debug_print("  Importing AlpacaDataModule...")
from dataset.alpaca import AlpacaDataModule
_debug_print("  AlpacaDataModule... OK")

_debug_print("  Importing AlpacaEvalDataModule...")
from dataset.alpaca_eval import AlpacaEvalDataModule
_debug_print("  AlpacaEvalDataModule... OK")

_debug_print("  Importing ArcDataModule...")
from dataset.arc import ArcDataModule
_debug_print("  ArcDataModule... OK")

_debug_print("  Importing BoolQDataModule...")
from dataset.boolq import BoolQDataModule
_debug_print("  BoolQDataModule... OK")

_debug_print("  Importing CifarDataModule...")
from dataset.cifar import CifarDataModule
_debug_print("  CifarDataModule... OK")

_debug_print("  Importing DartDataModule...")
from dataset.dart_data import DartDataModule
_debug_print("  DartDataModule... OK")

_debug_print("  Importing GlueDataModule (this may trigger evaluate.load)...")
from dataset.glue import GlueDataModule
_debug_print("  GlueDataModule... OK")

_debug_print("  Importing MmluDataModule...")
from dataset.mmlu import MmluDataModule
_debug_print("  MmluDataModule... OK")

_debug_print("  Importing MmluZeroShotDataModule...")
from dataset.mmlu_zero_shot import MmluZeroShotDataModule
_debug_print("  MmluZeroShotDataModule... OK")

_debug_print("  Importing MnistDataModule...")
from dataset.mnist import MnistDataModule
_debug_print("  MnistDataModule... OK")

_debug_print("  Importing PiqaDataModule...")
from dataset.piqa import PiqaDataModule
_debug_print("  PiqaDataModule... OK")

_debug_print("  Importing SocialIQADataModule...")
from dataset.social_iqa import SocialIQADataModule
_debug_print("  SocialIQADataModule... OK")

_debug_print("  Importing HellaSwagDataModule...")
from dataset.hellaswag import HellaSwagDataModule
_debug_print("  HellaSwagDataModule... OK")

_debug_print("  Importing WinoGrandeDataModule...")
from dataset.winogrande import WinoGrandeDataModule
_debug_print("  WinoGrandeDataModule... OK")

_debug_print("  Importing OpenBookQADataModule...")
from dataset.openbookqa import OpenBookQADataModule
_debug_print("  OpenBookQADataModule... OK")

_debug_print("  Importing RandomDataModule...")
from dataset.random_data import RandomDataModule
_debug_print("  RandomDataModule... OK")

_debug_print("  Importing SamSumDataModule...")
from dataset.samsum_data import SamSumDataModule
_debug_print("  SamSumDataModule... OK")

_debug_print("  Importing SpiderDataModule...")
from dataset.spider_data import SpiderDataModule
_debug_print("  SpiderDataModule... OK")

_debug_print("DONE importing dataset package")


def load_dataset(data, tokenizer, split, return_module=False, **kwargs):
    if data.startswith("glue"):
        glue, name, *subset_size = data.split("_")

        if len(subset_size) > 0:
            subset_size = int(subset_size[0])

            if split == "val":
                subset_size = int(0.1 * subset_size)
        else:
            subset_size = None

        data_module = GlueDataModule(
            tokenizer=tokenizer,
            name=name,
            split=split,
            subset_size=subset_size,
            has_test_split=glue.endswith("-tvt"),
            **kwargs
        )
    elif data == "alpaca_eval":
        data_module = AlpacaEvalDataModule(
            tokenizer=tokenizer,
            split=split,
            **kwargs
        )
    elif data.startswith("alpaca"):
        alpaca, *subset_size = data.split("_")

        if len(subset_size) > 0:
            subset_size = int(subset_size[0])

            if split == "val":
                subset_size = int(0.1 * subset_size)
        else:
            subset_size = None

        data_module = AlpacaDataModule(
            tokenizer=tokenizer,
            split=split,
            subset_size=subset_size,
            **kwargs
        )        
    elif data.startswith("dart"):
        alpaca, *subset_size = data.split("_")

        if len(subset_size) > 0:
            subset_size = int(subset_size[0])

            if split == "val":
                subset_size = int(0.1 * subset_size)
        else:
            subset_size = None

        data_module = DartDataModule(
            tokenizer=tokenizer,
            split=split,
            subset_size=subset_size,
            **kwargs
        )   
    elif data.startswith("random"):
        _, seqlen = data.split("_")
        seqlen = int(seqlen)

        data_module = RandomDataModule(
            tokenizer=tokenizer,
            split=split,
            seqlen=seqlen,
            **kwargs
        )
    elif data.startswith("spider"):
        hardness = None
        if data.endswith("_hard_extra"):
            data = data[:-len("_hard_extra")]
            hardness = ["hard", "extra"]

        spider, *subset_size = data.split("_")

        if len(subset_size) > 0:
            subset_size = int(subset_size[0])

            if split == "val":
                subset_size = int(0.1 * subset_size)
        else:
            subset_size = None

        data_module = SpiderDataModule(
            tokenizer=tokenizer,
            split=split,
            subset_size=subset_size,
            hardness=hardness,
            has_test_split=spider.endswith("-tvt"),
            **kwargs
        )
    elif data == "mmlu_zero_shot":
        data_module = MmluZeroShotDataModule(
            tokenizer=tokenizer,
            split=split,
            **kwargs
        )
    elif data.startswith("mmlu"):
        alpaca, *subset_size = data.split("_")

        if len(subset_size) > 0:
            subset_size = int(subset_size[0])

            if split == "val":
                subset_size = int(0.1 * subset_size)
        else:
            subset_size = None
        # assert split == "val"
        # mmlu, split = data.split("_")
        
        data_module = MmluDataModule(
            tokenizer=tokenizer,
            split=split,
            subset_size=subset_size,
            **kwargs
        )
    elif data.startswith("samsum"):
        samsum, *subset_size = data.split("_")

        if len(subset_size) > 0:
            subset_size = int(subset_size[0])

            if split in ("val", "test"):
                subset_size = int(0.1 * subset_size)
        else:
            subset_size = None

        data_module = SamSumDataModule(
            tokenizer=tokenizer,
            split=split,
            subset_size=subset_size,
            **kwargs
        )
    elif data.startswith("arc"):
        # Require explicit ARC variant: 'arc-easy' or 'arc-challenge'
        if data in ("arc-easy", "arc-challenge", "arc_easy", "arc_challenge"):
            arc_name = (
                data.replace("_", "-")  # arc_easy -> arc-easy
            )
        else:
            raise ValueError(
                "ARC dataset requires data to be 'arc-easy' or 'arc-challenge' (got: %s)" % data
            )
        data_module = ArcDataModule(
            tokenizer=tokenizer,
            name=arc_name,
            split=split,
            **kwargs
        )
    elif data == "piqa":
        data_module = PiqaDataModule(
            tokenizer=tokenizer,
            split=split,
            **kwargs
        )
    elif data == "boolq":
        data_module = BoolQDataModule(
            tokenizer=tokenizer,
            split=split,
            **kwargs
        )
    elif data in ("social_iqa", "siqa"):
        data_module = SocialIQADataModule(
            tokenizer=tokenizer,
            split=split,
            **kwargs
        )
    elif data == "hellaswag":
        data_module = HellaSwagDataModule(
            tokenizer=tokenizer,
            split=split,
            **kwargs
        )
    elif data == "winogrande":
        data_module = WinoGrandeDataModule(
            tokenizer=tokenizer,
            split=split,
            **kwargs
        )
    elif data == "openbookqa":
        data_module = OpenBookQADataModule(
            tokenizer=tokenizer,
            split=split,
            **kwargs
        )
    elif data.startswith("cifar"):
        cifar, *subset_size = data.split("_")

        if len(subset_size) > 0:
            subset_size = int(subset_size[0])

            if split in ("val", "test"):
                subset_size = int(0.1 * subset_size)
        else:
            subset_size = None

        data_module = CifarDataModule(
            tokenizer=tokenizer,
            split=split,
            subset_size=subset_size,
            has_test_split=cifar.endswith("-tvt"),
            **kwargs
        )
    elif data.startswith("mnist"):
        mnist, *subset_size = data.split("_")

        if len(subset_size) > 0:
            subset_size = int(subset_size[0])

            if split in ("val", "test"):
                subset_size = int(0.1 * subset_size)
        else:
            subset_size = None

        data_module = MnistDataModule(
            tokenizer=tokenizer,
            split=split,
            subset_size=subset_size,
            **kwargs
        )
    else:
        raise Exception(data)
    
    if not return_module:
        data_module = data_module.dataset

    return data_module
