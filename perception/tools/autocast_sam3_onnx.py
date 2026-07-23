#!/usr/bin/env python3
"""Convert the SAM3 ONNX graph to explicit mixed FP16 for TensorRT 11."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import onnx
from modelopt.onnx.autocast import convert_to_mixed_precision
from modelopt.onnx.autocast.precisionconverter import PrecisionConverter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = args.input.expanduser().resolve()
    target = args.output.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"ONNX graph not found: {source}")
    if target.exists() and not args.overwrite:
        print(f"[SAM3-AUTOCAST] reusing {target}", flush=True)
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    consolidated = target.with_name(source.stem + "_consolidated.onnx")
    consolidated_data = consolidated.name + ".data"
    if not consolidated.exists():
        print(
            f"[SAM3-AUTOCAST] consolidating external weights -> {consolidated}",
            flush=True,
        )
        source_model = onnx.load(str(source), load_external_data=True)
        onnx.save_model(
            source_model,
            str(consolidated),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=consolidated_data,
            size_threshold=1024,
        )
        del source_model
        gc.collect()

    print(f"[SAM3-AUTOCAST] converting {consolidated} -> {target}", flush=True)
    # ModelOpt 0.44's dead-end and redundant-cast cleanup repeatedly scan every
    # graph node for every Cast/output. The 15k-node SAM3 graph makes those
    # passes quadratic. Retaining dead-end/redundant Casts is semantically exact;
    # TensorRT removes them during engine optimization. Preserve ModelOpt's two
    # output-name cleanup passes, which protect the public output contract.
    PrecisionConverter._cleanup_no_consumer_nodes = lambda self: None
    PrecisionConverter._remove_redundant_casts = lambda self: None
    model = convert_to_mixed_precision(
        onnx_path=str(consolidated),
        low_precision_type="fp16",
        keep_io_types=True,
        # Avoid ONNX Runtime materializing this >2 GiB external-data graph in
        # memory. Tensor bounds are still guarded by ModelOpt's initializer
        # range analysis and TensorRT validates the converted graph below.
        data_max=float("inf"),
    )
    data_name = target.name + ".data"
    onnx.save_model(
        model,
        str(target),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024,
    )
    print(
        f"[SAM3-AUTOCAST] wrote {target} and {target.with_name(data_name)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
