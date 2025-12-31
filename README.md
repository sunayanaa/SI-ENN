Signal-Informed Evidential Neural Networks (SI-ENN)

This repository contains the official implementation of the Adaptive Signal-Informed Evidential Neural Network (SI-ENN). This framework integrates domain-specific signal priors (spectral entropy of the LP residual) with Evidential Deep Learning (EDL) to quantify epistemic uncertainty and enhance the trustworthiness of audio spoofing detection.

Repository Structure

The scripts are organized into four phases: Setup, Core Training, Advanced Validation, and Visualization.
1. Data Preparation & Methodology

    00_setup.py: Initializes the project environment and maps the ASVspoof 2019 Logical Access (LA) dataset.
    01_verify_protocols.py: Diagnostic utility to verify dataset integrity and label parsing.
    01_feature_extraction.py: Isolates the excitation signal using Linear Prediction (LP) analysis (p=16) and computes normalized spectral entropy H(r).
    02_si_enn_model.py: Defines the SI-ENN architecture (RawNet2 backbone) with the Adaptive Modulation Layer that learns the optimal entropy priors (μ, σ) during training.
    03_losses_and_metrics.py: Implements the Type II Maximum Likelihood loss and KL-divergence regularization for evidence quantification.

2. Core Training & Reproduction

    04_train_master.py: The main entry point for training the standard model.
    05_reproduce_fig1.py: Reproduces Figure 1 (Violin Plot), providing the empirical justification for using spectral entropy as a signal prior.
    06_evaluate_and_plot.py: Computes standard performance metrics (EER, ECE) on the evaluation set.
    06_reproduce_fig2.py: Reproduces the Risk-Coverage Curve for known attacks, establishing baseline trustworthiness.

3. Advanced Validation (Ablation & Robustness)

    experiment_01_ablation_components.py: (Table 5) A rigorous comparison of variants (Baseline vs. Static vs. Adaptive) to prove component efficacy.
    experiment_02_unseen_robustness.py: (Table 6) Proves generalization by training on known attacks (A01-A03) and testing on Unseen Attacks (A04-A06).

4. Publication Figures

    16_plot_unseen_robustness.py: Generates Figure 3 (Risk-Coverage on Unseen Attacks). Visualizes the "safety margin" gained by rejecting uncertain samples.
    17_tsne_visualization.py: Generates Figure 4 (t-SNE Embeddings). A qualitative visualization showing how the SI-ENN separates unseen attacks from bonafide speech based on epistemic uncertainty.
    18_plot_modulation_curve.py: Generates Figure 2 (Modulation Function). Visualizes the Gaussian prior g_b(H) used to modulate evidence based on entropy.

Execution Guide

To replicate the full set of results for the paper, execute the scripts in the following order:
Phase 1: Setup

    Run 00_setup.py to organize data.
    Run 01_verify_protocols.py to ensure data integrity.

Phase 2: Core Experiments

    Run 04_train_master.py to train the primary model.
    Run 05_reproduce_fig1.py to generate the data distribution figure.

Phase 3: Validation Experiments (TIFS Requirements)

    Run experiment_01_ablation_components.py to generate data for the Ablation Study.
    Run experiment_02_unseen_robustness.py to generate data for the Generalization Study.

Phase 4: Final Visuals

    Run 18_plot_modulation_curve.py to create Figure 2 (Methodology).
    Run 16_plot_unseen_robustness.py to create Figure 3 (Risk Results).
    Run 17_tsne_visualization.py to create Figure 4 (Qualitative Proof).

Key Results

    Component Analysis: Adaptive parameters outperform static priors, confirming that learning the entropy distribution is superior to hard-coding it.
    Unseen Robustness: On attacks never seen during training (A04-A06), the SI-ENN successfully flags them as uncertain, reducing operational risk compared to a standard Softmax baseline.

