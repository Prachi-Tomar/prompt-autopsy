import streamlit as st
import requests
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import os
from utils import get_backend_host

st.set_page_config(page_title="Prompt Autopsy", layout="wide")
st.title("Prompt Autopsy")

# Check for mock mode
MOCK_MODE = os.getenv("MOCK_MODE", "0")
if MOCK_MODE == "1":
    st.info("Mock Mode is ON. Data shown below is synthetic.")

st.markdown("""
<style>
.pa-eq { opacity: 0.85; }
.pa-del { background: rgba(255,0,0,0.15); text-decoration: line-through; padding: 2px; border-radius: 3px; }
.pa-ins { background: rgba(0,128,0,0.15); padding: 2px; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
.pa-tip { position: relative; display:inline-block; cursor: help; border-bottom: 1px dotted #888; }
.pa-tip:hover .pa-tiptext { visibility: visible; opacity: 1; }
.pa-tip .pa-tiptext {
  visibility: hidden; opacity: 0; transition: opacity .2s;
  position: absolute; z-index: 10; width: 260px;
  background: #111; color: #fff; padding: 8px 10px; border-radius: 6px;
  bottom: 125%; left: 50%; transform: translateX(-50%);
}
</style>
""", unsafe_allow_html=True)

# Tabs
tab_main, tab_exp = st.tabs(["Main", "Experiments"])

with st.sidebar:
    st.subheader("Settings")
    prompt = st.text_area(
        "Prompt",
        "Explain quantum computing in simple terms.",
        help="The actual question or task to send to the models."
    )
    models = st.multiselect(
        "Models",
        [
            "gpt-5", "gpt-5-mini",
            "gpt-4o", "gpt-4o-mini",
            "claude-3.5-sonnet-2024-10-22",
            "claude-3.5-sonnet-2024-06-20",
            "claude-3.5-haiku",
            "claude-3-haiku"
        ],
        default=["gpt-4o", "claude-3.5-sonnet-2024-10-22"],
        help="Choose one or more LLMs to compare. The same prompt will be sent to each."
    )
    temperature = st.slider(
        "Temperature",
        0.0, 1.0, 0.2, 0.1,
        help="Higher = more random output. Lower = more deterministic."
    )
    system_prompt = st.text_area(
        "System prompt (optional)",
        "",
        help="Optional background instruction for the model, e.g. 'You are an expert financial advisor.'"
    )
    run = st.button("Run Comparison")

# Initialize data to None
data = None

if run:
    payload = {
        "prompt": prompt,
        "models": models,
        "temperature": temperature,
        "system_prompt": system_prompt or None
    }
    try:
        resp = requests.post(f"{get_backend_host()}/compare", json=payload, timeout=120)
        data = resp.json()
    except Exception as e:
        st.error(f"Request failed: {e}")
        st.stop()

# Main tab content
with tab_main:
    # Check if we have comparison data
    if not run or data is None or "results" not in data:
        st.info("No comparison results yet. Enter your prompt and settings in the sidebar, then click **Run Comparison**.")
    else:
        # Determine number of columns based on number of models
        num_models = len(data["results"])
        if num_models == 1:
            num_cols = 1
        elif num_models == 2:
            num_cols = 2
        else:
            num_cols = 3
        
        # Create columns for model cards
        cols = st.columns(num_cols)
        
        # Display each model's results in a card
        for i, r in enumerate(data["results"]):
            col_idx = i % num_cols
            with cols[col_idx]:
                # Card container with consistent styling
                st.markdown(f"""
                <div style="
                    border: 1px solid #e0e0e0;
                    border-radius: 8px;
                    padding: 16px;
                    margin-bottom: 20px;
                    background-color: #ffffff;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    height: 100%;
                    display: flex;
                    flex-direction: column;
                ">
                <div style="flex-grow: 1;">
                """, unsafe_allow_html=True)
                
                st.markdown(f"**{r['model']}**")
                st.code(r["output_text"])
                # Highlight provider errors
                if "ERROR:" in r["output_text"].upper():
                    st.warning(f"Provider error for {r['model']}. See output above for details.")
                st.markdown(f"**Hallucination risk:** {r['hallucination_risk']:.1f} / 100")
                if r.get("hallucination_reasons"):
                    st.markdown("Reasons:")
                    for reason in r["hallucination_reasons"]:
                        st.markdown(f"- {reason}")
                
                # Display token usage and cost meta line
                meta = []
                if r.get("prompt_tokens") is not None:
                    meta.append(f"prompt={r['prompt_tokens']}")
                if r.get("completion_tokens") is not None:
                    meta.append(f"completion={r['completion_tokens']}")
                if r.get("total_tokens") is not None:
                    meta.append(f"total={r['total_tokens']}")
                if r.get("cost_usd") is not None:
                    meta.append(f"~${r['cost_usd']:.6f}")
                if meta:
                    st.caption(" • ".join(meta))
                
                # Display fact-check information
                fc = r.get("factcheck")
                if fc:
                    st.markdown("<span class='pa-tip'>Fact-check<span class='pa-tiptext'>Quick Wikipedia-based check of factual claims. Not authoritative; for reference only.</span></span>", unsafe_allow_html=True)
                    st.markdown(f"**Fact-check:** ✅ {fc['supported']} supported • ⚠️ {fc['ambiguous']} ambiguous • ❌ {fc['unsupported']} unsupported")
                    with st.expander("Claims & evidence"):
                        for item in fc.get("claims", [])[:10]:
                            st.markdown(f"- **{item['verdict']}** — {item['claim']}" + (f"  _(source: {item['evidence']})_" if item.get('evidence') else ""))
                
                # Display logprobs chart if available
                if r.get("logprobs") is not None:
                    logprobs = r["logprobs"]
                    tokens = r.get("tokens", [])
                    
                    # Create x-axis values (token indices)
                    x_values = list(range(len(logprobs)))
                    
                    # Create hover text with token and value information
                    hover_text = []
                    for i, (logprob, token) in enumerate(zip(logprobs, tokens if tokens else [None]*len(logprobs))):
                        token_text = token if token is not None else f"Token {i}"
                        hover_text.append(f"Token: {token_text}<br>Index: {i}<br>LogProb: {logprob:.3f}")
                    
                    # Create the figure
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=x_values,
                        y=logprobs,
                        hovertext=hover_text,
                        hoverinfo="text"
                    ))
                    
                    # Update layout with proper labels and title
                    fig.update_layout(
                        title="Token log probabilities",
                        xaxis_title="Token index in output",
                        yaxis_title="Log probability (natural log)",
                        height=320,
                        margin=dict(l=50, r=20, t=40, b=50),
                        hovermode='x unified'  # Better hover experience
                    )
                    
                    # Optional: Add token annotations for all tokens
                    annotations = []
                    for i in range(len(tokens)):
                        if i < len(tokens) and tokens[i] is not None:
                            # Truncate long token strings for better display
                            token_label = tokens[i][:10] + "..." if len(tokens[i]) > 10 else tokens[i]
                            annotations.append(dict(
                                x=i,
                                y=logprobs[i],
                                text=token_label,
                                showarrow=True,
                                arrowhead=2,
                                arrowsize=1,
                                arrowwidth=1,
                                arrowcolor="#636363",
                                ax=0,
                                ay=-40,
                                yshift=20,
                                font=dict(size=10),
                                bgcolor="rgba(0,0,0,0.7)",
                                bordercolor="#636363",
                                borderwidth=1
                            ))
                    
                    if annotations:
                        fig.update_layout(annotations=annotations)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("💡 Tip: You can zoom in by clicking and dragging on the chart to select a specific region. Hover over bars to see token details.")
                
                st.markdown("</div></div>", unsafe_allow_html=True)

        st.divider()
        st.subheader("Embedding similarity")
        labels = [r["model"] for r in data["results"]]
        mat = np.array([[data["embedding_similarity"][a][b] for b in labels] for a in labels])
        heat = go.Figure(data=go.Heatmap(z=mat, x=labels, y=labels))
        heat.update_layout(
            height=320,
            margin=dict(l=50, r=20, t=40, b=50)
        )
        st.plotly_chart(heat, use_container_width=True)

        st.divider()
        st.markdown("<span class='pa-tip'>Autopsy summary<span class='pa-tiptext'>High-level narrative of how models differed, including risk, similarity, and key differences.</span></span>", unsafe_allow_html=True)
        # Main tab glossary
        with st.expander("What do these metrics mean? (Glossary)"):
            st.markdown("""
- **Hallucination risk** — Heuristic score estimating likelihood of unsupported claims.
- **Similarity (cosine)** — 1.0 = same meaning, 0.0 = unrelated.
- **Logprob differences** — Token-by-token confidence gap between models.
- **Semantic divergence** — Estimates if differences are style, omissions, contradictions, etc.
- **Fact-check** — Quick Wikipedia-based verification of factual claims.
            """)
        summ = data.get("summaries", {})
        rec = summ.get("top_model")
        if rec:
            st.markdown(
                f"<div style='display:inline-block;padding:6px 10px;border-radius:8px;background:#eef6ff;border:1px solid #cfe3ff;font-weight:600;color:#000;'>"
                f"✅ Recommended model: {rec}"
                f"</div>",
                unsafe_allow_html=True
            )
        if isinstance(summ, dict) and summ:
            if "summary" in summ:
                st.write(summ["summary"])
            if "highlights" in summ and isinstance(summ["highlights"], list):
                for h in summ["highlights"]:
                    st.markdown(f"- {h}")
            # Display influence sentence if available
            if summ.get("influence_sentence"):
                st.info(summ["influence_sentence"])
            # Optional badges
            meta_cols = st.columns(3)
            hm = summ.get("highest_risk", {})
            if hm:
                meta_cols[0].metric(
                    "Highest risk",
                    f"{hm.get('model','?')}",
                    f"{hm.get('score',0):.1f}",
                    help="Model with the highest estimated hallucination risk score."
                )
            cp = summ.get("closest_pair", {})
            if cp:
                meta_cols[1].metric(
                    "Closest pair (cos)",
                    f"{cp.get('a','?')} vs {cp.get('b','?')}",
                    f"{cp.get('cosine',0):.3f}",
                    help="Pair of models whose responses are most semantically similar (cosine similarity)."
                )
            fp = summ.get("farthest_pair", {})
            if fp:
                meta_cols[2].metric(
                    "Most divergent (cos)",
                    f"{fp.get('a','?')} vs {fp.get('b','?')}",
                    f"{fp.get('cosine',0):.3f}",
                    help="Pair of models whose responses differ most in meaning."
                )
        else:
            st.info("No summary available yet.")

        st.subheader("Hallucination risk by model")
        risk_fig = go.Figure(go.Bar(x=[res['hallucination_risk'] for res in data['results']],
                                   y=[res['model'] for res in data['results']],
                                   orientation='h'))
        risk_fig.update_layout(
            height=320,
            margin=dict(l=50, r=20, t=40, b=50)
        )
        st.plotly_chart(risk_fig, use_container_width=True)

        st.divider()
        # Token-level diff section
        if len(data["results"]) > 1 and data.get("token_diffs"):
            st.subheader("Token-level diff")
            pair_keys = sorted(data["token_diffs"].keys())
            if pair_keys:
                selected_pair = st.selectbox("Select model pair to compare:", pair_keys)
                if selected_pair in data["token_diffs"]:
                    st.markdown(data["token_diffs"][selected_pair]["html"], unsafe_allow_html=True)
                    with st.expander("Unified diff (text)"):
                        st.code(data["token_diffs"][selected_pair]["unified"], language="diff")
            else:
                st.info("No token diffs available for the selected models.")
        elif len(data["results"]) > 1:
            st.info("Token diffs are not available for the selected models.")
        elif len(data["results"]) == 1:
            st.info("Token diffs are only available when comparing multiple models. Select at least two models to see diffs.")

        if data.get("logprob_diffs"):
            st.markdown("<span class='pa-tip'>Logprob differences between models<span class='pa-tiptext'>How confident each model was in generating each token. Large differences may indicate uncertainty or stylistic variation.</span></span>", unsafe_allow_html=True)
            for pair, stats in data["logprob_diffs"].items():
                st.markdown(f"**{pair}** — mean |Δlogprob|: {stats['mean_abs_diff']:.3f}, max: {stats['max_abs_diff']:.3f}, tokens compared: {int(stats['n_compared'])}")

        st.divider()
        # Semantic divergence (heuristics)
        sd = data.get("semantic_diffs", {})
        if sd:
            st.markdown("<span class='pa-tip'>Semantic divergence (heuristics)<span class='pa-tiptext'>Estimates the nature of differences between responses, e.g., omissions, contradictions, or rewording.</span></span>", unsafe_allow_html=True)
            for pair, scores in sd.items():
                # Show top 2 labels by score
                top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:2]
                tags = " • ".join([f"{k}: {v:.2f}" for k,v in top])
                st.markdown(f"**{pair}** — {tags}")

st.divider()
# Display aligned token data as tables
if data is not None:
    apairs = data.get("aligned_logprobs", {})
    if apairs:
        for pair, rows in apairs.items():
            st.markdown(f"**Token logprob diff for {pair}**")
            st.dataframe(
                [{"A_token": a, "A_lp": la, "B_token": b, "B_lp": lb, "Δlp": diff}
                 for (a, la, b, lb, diff) in rows[:50]],  # limit to first 50 rows
                use_container_width=True
            )
        # Autopsy Report section
        def make_markdown_report(data, original_prompt, system_prompt):
            import datetime
            from utils import make_ascii_table
            
            # Header
            report = "# Prompt Autopsy Report\n\n"
            report += f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # Prompt information
            report += f"**Prompt:**\n\n```\n{original_prompt}\n```\n\n"
            if system_prompt:
                report += f"**System Prompt:**\n\n```\n{system_prompt}\n```\n\n"
            
            # Summary information
            summ = data.get("summaries", {}) if isinstance(data, dict) else {}
            hl = summ.get("highlights", []) if isinstance(summ, dict) else []
            
            report += "## Autopsy Summary\n"
            report += f"{summ.get('summary', 'No summary available.')}\n\n"
            
            report += "### Highlights\n"
            report += f"{chr(10).join(['- ' + str(x) for x in hl]) if hl else '_None_'}\n\n"
            
            # Optional metrics
            report += f"- **Recommended model:** {summ.get('top_model', 'N/A')}\n"
            report += f"- **Highest risk:** { (summ.get('highest_risk') or {}).get('model', 'N/A') } ({ (summ.get('highest_risk') or {}).get('score', 'N/A') })\n\n"
            
            # Models compared
            models = [r['model'] for r in data['results']]
            report += "**Models compared:**\n\n"
            for model in models:
                report += f"- {model}\n"
            report += "\n"
            
            # Hallucination risk table
            report += "| Model | Risk/100 | Reasons |\n"
            report += "|-------|----------|---------|\n"
            for r in data['results']:
                risk = r.get('hallucination_risk', 'N/A')
                reasons = "; ".join(r.get('hallucination_reasons', [])) if r.get('hallucination_reasons') else "N/A"
                report += f"| {r['model']} | {risk} | {reasons} |\n"
            report += "\n"
            
            # Embedding similarity matrix
            report += "**Embedding similarity:**\n\n"
            labels = [r['model'] for r in data['results']]
            matrix_data = []
            for a in labels:
                row = [f"{data['embedding_similarity'][a][b]:.3f}" for b in labels]
                matrix_data.append(row)
            report += make_ascii_table([labels] + matrix_data, [""] + labels)
            report += "\n\n"
            
            # Per-model outputs
            for r in data['results']:
                report += f"## {r['model']}\n\n"
                report += f"```\n{r['output_text']}\n```\n\n"
                risk = r.get('hallucination_risk', 'N/A')
                report += f"- Risk: {risk}/100\n"
                if r.get('hallucination_reasons'):
                    report += "- Reasons:\n"
                    for reason in r['hallucination_reasons']:
                        report += f"  - {reason}\n"
                report += "\n"
             
            # Semantic divergence (heuristics)
            sd = data.get("semantic_diffs", {})
            if sd:
                report += "\n## Semantic divergence (heuristics)\n"
                for pair, scores in sd.items():
                    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
                    line = ", ".join([f"{k}: {v:.2f}" for k,v in ordered])
                    report += f"- **{pair}** — {line}\n"
             
            # Fact-check summary
            if data.get("factcheck_summary"):
                s = data["factcheck_summary"]
                report += "\n## Fact-check Summary\n"
                report += f"Supported: {s['supported']}, Ambiguous: {s['ambiguous']}, Unsupported: {s['unsupported']}\n"
            
            # Per-model fact-check details
            for r in data.get("results", []):
                fc = r.get("factcheck")
                if not fc: continue
                report += f"\n### Fact-check — {r['model']}\n"
                report += f"Supported: {fc['supported']}, Ambiguous: {fc['ambiguous']}, Unsupported: {fc['unsupported']}\n"
                for item in fc.get("claims", [])[:10]:
                    src = f" (source: {item['evidence']})" if item.get("evidence") else ""
                    report += f"- **{item['verdict']}** — {item['claim']}{src}\n"
            
            # Pairwise diffs
            if data.get('token_diffs'):
                for pair_key, diff_data in data['token_diffs'].items():
                    model_a, model_b = pair_key.split('||')
                    report += f"### {model_a} vs {model_b}\n\n"
                    report += f"```diff\n{diff_data['unified']}\n```\n\n"
            
            return report
        
        st.divider()
        # Download report button
        md = make_markdown_report(data, prompt, system_prompt)
        st.download_button(
            "Download Autopsy Report (.md)",
            data=md,
            file_name="prompt_autopsy_report.md",
            mime="text/markdown"
        )
        
        # Display total estimated cost
        total_cost = sum([r.get("cost_usd") or 0.0 for r in data.get("results", [])])
        st.markdown(f"**Estimated total cost:** ~${total_cost:.6f}")
        
        # Display aggregate fact-check summary
        if data.get("factcheck_summary"):
            s = data["factcheck_summary"]
            st.markdown(f"**Fact-check (aggregate):** ✅ {s['supported']} • ⚠️ {s['ambiguous']} • ❌ {s['unsupported']}")

with tab_exp:
    st.subheader("Experiments")
    with st.expander("What do these settings mean? (Glossary)"):
        st.markdown("""
- **Model** — The LLM you're testing (OpenAI/Anthropic etc.).
- **Prompt** — The question or instruction sent to the models.
- **System prompt** — Hidden preface that sets role/tone (e.g., "You are a concise teacher.").
- **Temperature** — Randomness. Low = consistent; high = creative (and sometimes riskier).
- **Seed** — Controls the random path. If supported, same seed + same settings ≈ same output.
- **Hallucination risk** — Heuristic score estimating likelihood of unsupported claims.
- **Stability** — How similar a run is to that model's average answer (cosine similarity).
- **Similarity (cosine)** — 1.0 = same meaning, 0.0 = unrelated.
- **Parameter influence** — How much changing a setting (temp/system prompt) shifts meaning and risk.
        """)
    e_prompt = st.text_area(
        "Prompt",
        "Explain quantum computing in simple terms.",
        key="exp_prompt",
        help="The question/task sent to every run in the grid."
    )
    e_models = st.multiselect(
        "Models",
        ["gpt-5","gpt-5-mini","gpt-4o","gpt-4o-mini","claude-3.5-sonnet-2024-10-22","claude-3.5-sonnet-2024-06-20","claude-3.5-haiku","claude-3-haiku"],
        default=["gpt-4o", "claude-3.5-sonnet-2024-10-22"],
        key="exp_models",
        help="Pick the models to test. Each model is run with every temperature/system prompt/seed you specify."
    )
    temps_str = st.text_input(
        "Temperatures (comma separated)",
        "0.2,0.7",
        key="exp_temps",
        help="Controls randomness: lower = more deterministic; higher = more creative. We'll test each value."
    )
    sys_multi = st.text_area(
        "System prompts (one per line; blank allowed)",
        "",
        key="exp_sys",
        help="Optional hidden instruction that sets role/style. Put each variant on its own line. Leave empty to use the model default."
    )
    seeds_str = st.text_input(
        "Seeds (optional, comma separated)",
        "",
        key="exp_seeds",
        help="If supported, using the same seed makes runs repeatable at the same temperature. Leave blank to ignore."
    )

    st.button(
        "Run Experiments",
        help="Runs the full grid: models × temperatures × system prompts × seeds."
    )
    
    if "gpt-5" in e_models:
        st.caption("Note: Some GPT-5 variants force default temperature. If set, we'll ignore custom temperature for that model.")
        
    temps = [float(x.strip()) for x in temps_str.split(",") if x.strip()]
    sys_list = [s if s.strip() else "" for s in sys_multi.splitlines()] or [""]
    seeds = [int(x.strip()) for x in seeds_str.split(",") if x.strip()] if seeds_str.strip() else []
    
    if st.button("Run Experiments"):
        payload = {"prompt": e_prompt, "models": e_models, "temperatures": temps, "system_prompts": sys_list, "seeds": seeds}
        try:
            resp = requests.post(f"{get_backend_host()}/experiment", json=payload, timeout=600)
            if resp.status_code != 200:
                st.error(f"Experiment request failed with status code {resp.status_code}")
                try:
                    error_detail = resp.json()
                    st.error(f"Error details: {error_detail}")
                except:
                    st.error(f"Response text: {resp.text}")
                st.stop()
            exp = resp.json()
        except Exception as e:
            st.error(f"Experiment request failed: {e}")
            st.stop()

        # Table
        rows = []
        for r in exp["runs"]:
            rows.append({
                "model": r["model"],
                "temperature": r["temperature"],
                "system_prompt": r["system_prompt"],
                "seed": r["seed"],
                "latency_ms": round(r["latency_ms"], 1),
                "risk": round(r["hallucination_risk"], 2),
                "logprob_avg": r.get("logprob_avg"),
                "logprob_std": r.get("logprob_std"),
                "logprob_frac_low": r.get("logprob_frac_low"),
            })
        df = pd.DataFrame(rows)
        st.subheader("Results table", help="Each row is one run with its settings, risk, and latency.")
        st.dataframe(df, use_container_width=True)

        # Risk by temperature chart (grouped bars per model)
        st.subheader("Risk by temperature", help="Average hallucination risk per model at each temperature. Higher bars suggest risk increases with randomness.")
        by_temp = exp["drift"]["by_temperature"]
        # Build arrays
        fig_rt = go.Figure()
        for m, temps_map in by_temp.items():
            x = list(temps_map.keys())
            y = [temps_map[t]["mean_risk"] for t in x]
            fig_rt.add_bar(name=m, x=[str(v) for v in x], y=y)
        fig_rt.update_layout(barmode='group', height=350)
        st.plotly_chart(fig_rt, use_container_width=True)

        # Stability line chart per model
        st.subheader("Stability (cosine to centroid)", help="How close each run is to that model's 'typical' answer. Higher = more consistent.")
        stab = exp["drift"]["stability"]
        fig_stab = go.Figure()
        for m, vals in stab.items():
            fig_stab.add_trace(go.Scatter(x=list(range(1, len(vals)+1)), y=vals, mode="lines+markers", name=m))
        fig_stab.update_layout(height=350, yaxis=dict(range=[0,1]))
        st.plotly_chart(fig_stab, use_container_width=True)

        # Risk vs Stability scatter
        st.subheader("Risk vs stability", help="Tradeoff view: faster isn't always safer. Points to the right are riskier; higher is slower.")
        xs, ys, names = [], [], []
        for r in exp["runs"]:
            xs.append(r["hallucination_risk"])
            # find the stability value by recomputing cosine to centroid quickly:
            # Not available directly per-run in response, so approximate by model mean in drift.stability if needed
            # Simpler: skip exact per-run stability here; use risk vs latency as an alternative
        fig_rs = go.Figure()
        fig_rs.add_trace(go.Scatter(x=[rr["hallucination_risk"] for rr in exp["runs"]],
                                    y=[rr["latency_ms"] for rr in exp["runs"]],
                                    mode="markers",
                                    text=[f"{rr['model']} | T={rr['temperature']} | S={rr['system_prompt'][:20]}{'...' if len(rr['system_prompt']) > 20 else ''}" for rr in exp["runs"]]))
        fig_rs.update_layout(xaxis_title="Risk", yaxis_title="Latency (ms)", height=350)
        st.plotly_chart(fig_rs, use_container_width=True)
# Parameter influence (auto-detected)
        pi = exp.get("drift", {}).get("parameter_influence")
        if pi:
            st.subheader("Parameter influence (auto-detected)", help="How much changing temperature or system prompt shifts meaning (similarity) and risk compared to a baseline.")
            for param, models_map in pi.items():
                st.markdown(f"**{param.capitalize()}**")
                for model, vals in models_map.items():
                    st.markdown(f"- *{model}*")
                    for v, metrics in vals.items():
                        st.markdown(f"  - {param} = `{v}` → similarity to baseline: {metrics['similarity_to_baseline']:.3f}, risk change: {metrics['risk_change']:+.2f}")

        # CSV download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download runs as CSV", data=csv, file_name="prompt_autopsy_experiments.csv", mime="text/csv")