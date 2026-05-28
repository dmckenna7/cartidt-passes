from __future__ import annotations

from cartidt.frontend.catalog import LoaderBundle


def test_loader_bundle_has_three_loaders() -> None:
    assert {"train", "val", "test"} == set(LoaderBundle.__dataclass_fields__)
