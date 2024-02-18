# Prox-ROT

Code for Structured Transforms Across Spaces using Optimal Transport, AISTATS 2024 (Othmane Sebbouh, Marco Cuturi, Gabriel Peyré)

It is possible to either run the experiments from scratch, or see the figures in the ``precomputed_figures`` folder.

## Running the experiments from scratch

- For the experiments generating figure 1 and figure 2, run the following commands:
```
python train.py -e spatial -ep runs
python moscot_lr.py
python moscot_lr_fast.py
python moscot_sinkhorn.py
```

## Generating the plots
Each plot can be run with the corresponding jupyter notebook ```visualize_#.ipynb``` notebook once the experiments are run.
