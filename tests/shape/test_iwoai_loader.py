from __future__ import annotations

from cartidt.frontend.sources.iwoai_split import IWOAIPartition


def test_iwoai_partition_dataclass_fields_present() -> None:
    fields = IWOAIPartition.__dataclass_fields__
    assert {"seg_train_zib_only", "seg_iwoai_train", "seg_iwoai_val", "seg_iwoai_test"} <= set(fields.keys())
