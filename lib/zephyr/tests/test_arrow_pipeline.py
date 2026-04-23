# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ArrowBatchPipeline planning and runtime path."""


import pyarrow as pa
import pyarrow.parquet as pq

from zephyr import Dataset, compute_plan
from zephyr.dataset import FilterOp, LoadFileOp, MapOp, SelectOp, TakePerShardOp
from zephyr.expr import col
from zephyr.plan import (
    ArrowBatchPipeline,
    Map,
    _can_fuse_to_arrow,
    _compose_arrow_pipeline,
    _run_arrow_batch_pipeline,
)

# ---------------------------------------------------------------------------
# _can_fuse_to_arrow
# ---------------------------------------------------------------------------


def test_can_fuse_to_arrow_simple():
    ops = [
        LoadFileOp(format="parquet"),
        FilterOp(col("score") > 0.5),
    ]
    assert _can_fuse_to_arrow(ops) is True


def test_can_fuse_to_arrow_rejects_lambda_filter():
    ops = [
        LoadFileOp(format="parquet"),
        FilterOp(predicate=lambda r: r["score"] > 0.5),
    ]
    assert _can_fuse_to_arrow(ops) is False


def test_can_fuse_to_arrow_rejects_jsonl():
    ops = [
        LoadFileOp(format="jsonl"),
        FilterOp(col("score") > 0.5),
    ]
    assert _can_fuse_to_arrow(ops) is False


def test_can_fuse_to_arrow_rejects_map():
    ops = [
        LoadFileOp(format="parquet"),
        MapOp(fn=lambda r: r),
    ]
    assert _can_fuse_to_arrow(ops) is False


def test_can_fuse_to_arrow_accepts_select_take():
    ops = [
        LoadFileOp(format="parquet"),
        FilterOp(col("score") > 0.5),
        SelectOp(("id", "score")),
        TakePerShardOp(10),
    ]
    assert _can_fuse_to_arrow(ops) is True


def test_can_fuse_to_arrow_no_loadfile():
    ops = [
        FilterOp(col("score") > 0.5),
    ]
    assert _can_fuse_to_arrow(ops) is False


# ---------------------------------------------------------------------------
# _compose_arrow_pipeline
# ---------------------------------------------------------------------------


def test_compose_arrow_pipeline_combines_filters():
    ops = [
        LoadFileOp(format="parquet"),
        FilterOp(col("score") > 0.5),
        FilterOp(col("active") == True),  # noqa: E712
        SelectOp(("id", "score")),
        TakePerShardOp(10),
    ]
    arrow = _compose_arrow_pipeline(ops)
    assert isinstance(arrow, ArrowBatchPipeline)
    assert arrow.projection_columns == ("id", "score")
    assert arrow.take_limit == 10
    assert arrow.filter_expr is not None


def test_compose_arrow_pipeline_no_filter():
    ops = [
        LoadFileOp(format="parquet"),
        SelectOp(("id",)),
    ]
    arrow = _compose_arrow_pipeline(ops)
    assert arrow.filter_expr is None
    assert arrow.projection_columns == ("id",)


# ---------------------------------------------------------------------------
# Planner integration - compute_plan emits ArrowBatchPipeline
# ---------------------------------------------------------------------------


def test_plan_emits_arrow_pipeline_for_parquet_filter(tmp_path):
    # Create a tiny parquet file
    table = pa.table({"score": [0.1, 0.6, 0.8], "id": [1, 2, 3]})
    pq.write_table(table, str(tmp_path / "data.parquet"))

    ds = Dataset.from_files(str(tmp_path / "*.parquet")).load_parquet().filter(col("score") > 0.5).select("id", "score")
    plan = compute_plan(ds)

    assert len(plan.stages) == 1
    assert len(plan.stages[0].operations) == 1
    assert isinstance(plan.stages[0].operations[0], ArrowBatchPipeline)


def test_plan_emits_map_for_lambda_filter(tmp_path):
    table = pa.table({"score": [0.1, 0.6, 0.8], "id": [1, 2, 3]})
    pq.write_table(table, str(tmp_path / "data.parquet"))

    ds = Dataset.from_files(str(tmp_path / "*.parquet")).load_parquet().filter(lambda r: r["score"] > 0.5)
    plan = compute_plan(ds)

    assert len(plan.stages) == 1
    assert isinstance(plan.stages[0].operations[0], Map)


def test_plan_emits_map_for_jsonl(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"score": 0.6}\n{"score": 0.1}\n')

    ds = Dataset.from_files(str(tmp_path / "*.jsonl")).load_jsonl().filter(col("score") > 0.5)
    plan = compute_plan(ds)

    assert len(plan.stages) == 1
    assert isinstance(plan.stages[0].operations[0], Map)


# ---------------------------------------------------------------------------
# _run_arrow_batch_pipeline execution
# ---------------------------------------------------------------------------


def test_run_arrow_batch_pipeline_filter_and_select(tmp_path):
    table = pa.table({"score": [0.1, 0.6, 0.8], "id": [1, 2, 3]})
    pq.write_table(table, str(tmp_path / "data.parquet"))

    pipeline = ArrowBatchPipeline(
        filter_expr=(col("score") > 0.5),
        projection_columns=("id",),
        take_limit=None,
    )
    source = [{"path": str(tmp_path / "data.parquet"), "format": "parquet"}]
    result = list(_run_arrow_batch_pipeline(iter(source), pipeline))

    assert result == [{"id": 2}, {"id": 3}]


def test_run_arrow_batch_pipeline_take_limit(tmp_path):
    table = pa.table({"score": [0.1, 0.6, 0.8, 0.9], "id": [1, 2, 3, 4]})
    pq.write_table(table, str(tmp_path / "data.parquet"))

    pipeline = ArrowBatchPipeline(
        filter_expr=(col("score") > 0.0),
        projection_columns=None,
        take_limit=2,
    )
    source = [{"path": str(tmp_path / "data.parquet"), "format": "parquet"}]
    result = list(_run_arrow_batch_pipeline(iter(source), pipeline))
    assert len(result) == 2


def test_run_arrow_batch_pipeline_no_match(tmp_path):
    table = pa.table({"score": [0.1, 0.2], "id": [1, 2]})
    pq.write_table(table, str(tmp_path / "data.parquet"))

    pipeline = ArrowBatchPipeline(
        filter_expr=(col("score") > 0.5),
    )
    source = [{"path": str(tmp_path / "data.parquet"), "format": "parquet"}]
    result = list(_run_arrow_batch_pipeline(iter(source), pipeline))
    assert result == []
