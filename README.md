<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SI-ENN Project Documentation</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #24292e;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3 {
            border-bottom: 1px solid #eaecef;
            padding-bottom: 0.3em;
            margin-top: 1.5em;
        }
        h1 { font-size: 2em; border-bottom: none; }
        code {
            background-color: rgba(27,31,35,0.05);
            border-radius: 3px;
            padding: 0.2em 0.4em;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            font-size: 85%;
        }
        .folder-list {
            background: #f6f8fa;
            padding: 20px;
            border-radius: 6px;
            border: 1px solid #d1d5da;
        }
        ul, ol { padding-left: 2em; }
        li { margin-bottom: 0.5em; }
        .math { font-style: italic; font-family: "Times New Roman", serif; }
        .note {
            padding: 15px;
            margin-bottom: 20px;
            border: 1px solid #cce5ff;
            border-radius: 4px;
            color: #004085;
            background-color: #cce5ff;
        }
    </style>
</head>
<body>

    <h1>Signal-Informed Evidential Neural Networks (SI-ENN)</h1>
    <p>
        This repository contains the official implementation of the <strong>Adaptive Signal-Informed Evidential Neural Network (SI-ENN)</strong>. 
        This framework integrates domain-specific signal priors (spectral entropy of the LP residual) with Evidential Deep Learning (EDL) 
        to quantify epistemic uncertainty and enhance the trustworthiness of audio spoofing detection.
    </p>

    <h2>📂 Repository Structure</h2>
    <p>The scripts are organized into four phases: <strong>Setup</strong>, <strong>Core Training</strong>, <strong>Advanced Validation</strong>, and <strong>Visualization</strong>.</p>

    <div class="folder-list">
        <h3>1. Data Preparation & Methodology</h3>
        <ul>
            <li><code>00_setup.py</code>: Initializes the project environment and maps the <strong>ASVspoof 2019 Logical Access (LA)</strong> dataset.</li>
            <li><code>01_verify_protocols.py</code>: Diagnostic utility to verify dataset integrity and label parsing.</li>
            <li><code>01_feature_extraction.py</code>: Isolates the excitation signal using <strong>Linear Prediction (LP) analysis</strong> (<span class="math">p=16</span>) and computes normalized spectral entropy <span class="math">H(r)</span>.</li>
            <li><code>02_si_enn_model.py</code>: Defines the <strong>SI-ENN architecture</strong> (RawNet2 backbone) with the <strong>Adaptive Modulation Layer</strong> that learns the optimal entropy priors (<span class="math">&mu;, &sigma;</span>) during training.</li>
            <li><code>03_losses_and_metrics.py</code>: Implements the <strong>Type II Maximum Likelihood loss</strong> and <strong>KL-divergence regularization</strong> for evidence quantification.</li>
        </ul>

        <h3>2. Core Training & Reproduction</h3>
        <ul>
            <li><code>04_train_master.py</code>: The main entry point for training the standard model.</li>
            <li><code>05_reproduce_fig1.py</code>: Reproduces <strong>Figure 1</strong> (Violin Plot), providing the empirical justification for using spectral entropy as a signal prior.</li>
            <li><code>06_evaluate_and_plot.py</code>: Computes standard performance metrics (<strong>EER</strong>, <strong>ECE</strong>) on the evaluation set.</li>
            <li><code>06_reproduce_fig2.py</code>: Reproduces the <strong>Risk-Coverage Curve</strong> for known attacks, establishing baseline trustworthiness.</li>
        </ul>

        <h3>3. Advanced Validation (Ablation & Robustness)</h3>
        <ul>
            <li><code>experiment_01_ablation_components.py</code>: <strong>(Table 5)</strong> A rigorous comparison of variants (Baseline vs. Static vs. Adaptive) to prove component efficacy.</li>
            <li><code>experiment_02_unseen_robustness.py</code>: <strong>(Table 6)</strong> Proves generalization by training on known attacks (A01-A03) and testing on <strong>Unseen Attacks (A04-A06)</strong>.</li>
        </ul>

        <h3>4. Publication Figures</h3>
        <ul>
            <li><code>16_plot_unseen_robustness.py</code>: Generates <strong>Figure 3</strong> (Risk-Coverage on Unseen Attacks). Visualizes the "safety margin" gained by rejecting uncertain samples.</li>
            <li><code>17_tsne_visualization.py</code>: Generates <strong>Figure 4</strong> (t-SNE Embeddings). A qualitative visualization showing how the SI-ENN separates unseen attacks from bonafide speech based on epistemic uncertainty.</li>
            <li><code>18_plot_modulation_curve.py</code>: Generates <strong>Figure 2</strong> (Modulation Function). Visualizes the Gaussian prior <span class="math">g_b(H)</span> used to modulate evidence based on entropy.</li>
        </ul>
    </div>

    <h2>🚀 Execution Guide</h2>
    <p>To replicate the full set of results for the paper, execute the scripts in the following order:</p>

    <h3>Phase 1: Setup</h3>
    <ol>
        <li>Run <code>00_setup.py</code> to organize data.</li>
        <li>Run <code>01_verify_protocols.py</code> to ensure data integrity.</li>
    </ol>

    <h3>Phase 2: Core Experiments</h3>
    <ol start="3">
        <li>Run <code>04_train_master.py</code> to train the primary model.</li>
        <li>Run <code>05_reproduce_fig1.py</code> to generate the data distribution figure.</li>
    </ol>

    <h3>Phase 3: Validation Experiments (TIFS Requirements)</h3>
    <ol start="5">
        <li>Run <code>experiment_01_ablation_components.py</code> to generate data for the <strong>Ablation Study</strong>.</li>
        <li>Run <code>experiment_02_unseen_robustness.py</code> to generate data for the <strong>Generalization Study</strong>.</li>
    </ol>

    <h3>Phase 4: Final Visuals</h3>
    <ol start="7">
        <li>Run <code>18_plot_modulation_curve.py</code> to create <strong>Figure 2</strong> (Methodology).</li>
        <li>Run <code>16_plot_unseen_robustness.py</code> to create <strong>Figure 3</strong> (Risk Results).</li>
        <li>Run <code>17_tsne_visualization.py</code> to create <strong>Figure 4</strong> (Qualitative Proof).</li>
    </ol>

    <h2>📊 Key Results</h2>
    <ul>
        <li><strong>Component Analysis:</strong> Adaptive parameters outperform static priors, confirming that learning the entropy distribution is superior to hard-coding it.</li>
        <li><strong>Unseen Robustness:</strong> On attacks never seen during training (A04-A06), the SI-ENN successfully flags them as uncertain, reducing operational risk compared to a standard Softmax baseline.</li>
    </ul>

</body>
</html>
