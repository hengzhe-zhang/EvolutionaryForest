# RAG-SR: Retrieval-Augmented Generation for Neural Symbolic Regression

This repository contains the code for the RAG-SR model, a retrieval-augmented generation method for neural symbolic
regression.

## SRBench Experiment

To run the black-box experiment on the SRBench dataset:

1. Move the `methods/RAG_SR.py` file into the `experiment/methods` directory within SRBench.
2. Run the following command:
   ```bash
   python3 analyze.py datasets -n_jobs 10 -ml RAG_SR -n_trials 10 -results ./results --local -time_limit 48:00 -job_limit 100000
   ```

# Semantic Library Experiment

To explore the semantic library experiment, refer to the `example/train_semantic_library.py` script.
