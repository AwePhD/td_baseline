# Scripts

The scripts are divided in two kinds: generate the outputs for various evaluations and those evaluations. The formers have a `generate_` prefix and the former have a `evaluate_` ones. The generation should be all run before the evaluations, type `python scripts/generale_all.py`.

The evaluations are split in two sets of scripts: the TDReID evaluation and the experiments. The TDReID evaluation are the TReID, DReID, text-only and text+frame evaluations. The experiments are variant of base evaluation in order to assess some property of this baseline:

- On multi-task
    * [`TReID with crops annotations`](./evaluate_treid_annotations.py) studies the difference of varying dimensions input againts fixed size crops.
    * Comparison TReID and text-only
        + [`text-only with only TP`](./evaluate_textonly_tponly.py) only good (automatic) candidate for the text-only task.
        + [`TReID with only TP (from automatic)`](./evaluate_treid_tponly.py) reduces gallery to the TPs from automatic detection.
