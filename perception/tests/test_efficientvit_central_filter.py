import numpy as np

from seg_web_node.efficientvit_backend import filter_and_merge_central_masks


def test_central_filter_rejects_lower_dominated_and_merges_nested_masks():
    upper = np.zeros((100, 100), dtype=bool)
    upper[20:50, 35:65] = True
    nested = np.zeros((100, 100), dtype=bool)
    nested[25:45, 40:60] = True
    floor = np.zeros((100, 100), dtype=bool)
    floor[65:100, :] = True

    output = filter_and_merge_central_masks(
        [(upper, 0.91), (nested, 0.88), (floor, 0.99)],
        lower_region_start_fraction=0.60,
        lower_region_mask_fraction=0.80,
        merge_overlap_min_fraction=0.50,
        max_masks=12,
    )

    assert len(output) == 1
    mask, score = output[0]
    assert score == 0.91
    assert np.array_equal(mask, upper)
