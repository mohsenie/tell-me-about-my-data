"""Tuning constants for the profiling/catalog logic (package-internal)."""

GAP_FACTOR = 5        # gap > GAP_FACTOR * median_dt => counts as "intermittent"
MAD_SIGMA = 3         # robust envelope width in MAD-sigma units
MAD_SCALE = 1.4826    # MAD -> stddev-equivalent for normally distributed data
