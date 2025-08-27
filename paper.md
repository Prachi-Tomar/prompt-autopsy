---
title: 'Prompt Autopsy: A Forensic Analysis Tool for Large Language Model Outputs'
tags:
  - Python
  - Machine Learning
  - Natural Language Processing
  - Explainability
  - Evaluation
authors:
  - name: Prachi Tomar
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 2025-08-27
bibliography: paper.bib
---

# Summary

Prompt Autopsy is an open-source forensic analysis tool that compares responses from large language models (LLMs) to the same prompt and explains why they diverge. Unlike traditional evaluation frameworks that assign scores or benchmarks, Prompt Autopsy provides token-level diagnostics, log-probability analysis, and embedding-based similarity metrics to uncover subtle behavioral shifts between models or runs. This enables researchers, practitioners, and policy experts to gain deeper insight into LLM behavior, robustness, and reliability.

# Statement of Need

The rapid proliferation of generative AI systems has made evaluation and debugging increasingly critical. Existing tools (e.g., HELM [@liang2022helm], Ragas [@es2023ragas], and GEMMA [@gehrmann2023gemma]) primarily focus on performance benchmarks or end-task accuracy. However, these approaches often obscure *why* models behave differently under seemingly identical conditions. Prompt Autopsy addresses this gap by introducing a "forensic" perspective: analyzing divergence at the level of tokens, probabilities, and embeddings. This helps users in AI safety, prompt engineering, and scientific reproducibility by making invisible differences visible and interpretable.

# State of the Field

LLM evaluation frameworks typically emphasize aggregate accuracy, coverage, or fairness. For example, HELM provides multidimensional benchmarks [@liang2022helm], while Ragas offers automated QA metrics [@es2023ragas]. Other methods like BERTScore [@zhang2020bertscore] and the Universal Sentence Encoder [@cer2018use] provide embedding-based measures of similarity. Prompt Autopsy complements these by focusing not on *how well* a model performed, but on *why responses differ*. This unique forensic lens makes it particularly useful for AI auditing, research reproducibility, and model comparison studies.

# Functionality

Prompt Autopsy includes:
- **Token-level diffs with logprobs**: Visualizing divergence in probability space across models or runs.
- **Embedding-based similarity**: Heatmaps and similarity scores quantify semantic drift using techniques like BERTScore [@zhang2020bertscore].
- **Risk drift analysis**: Automatic detection of hallucination likelihood and system prompt/temperature influence.
- **Side-by-side UI**: A Streamlit frontend for interactive exploration and reporting.
- **Artifacts for reproducibility**: CSV/HTML reports summarizing differences for academic and industry use.

The backend is implemented in Python with FastAPI, while the frontend uses Streamlit. Docker support is included for easy deployment.

# Example

A typical workflow involves comparing two models (e.g., GPT-4o vs Claude 3.5) on the same prompt. The tool produces:
- Side-by-side outputs with token-level diffs
- Probability charts illustrating divergence
- Embedding similarity visualizations
- An "autopsy report" summarizing hallucination risk and divergence causes

Example usage and screenshots are available in the repository's `examples/` directory and `README.md`.

# Acknowledgements

This project builds upon open-source contributions in the LLM evaluation ecosystem and was inspired by ongoing research into AI robustness, transparency, and explainability.

# References