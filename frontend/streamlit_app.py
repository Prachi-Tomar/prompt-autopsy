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

# Tabs
tab_main, tab_exp = st.tabs(["Main", "Experiments"])

with st.sidebar:
    st.subheader("Settings")
    prompt = st.text_area("Prompt", "Explain quantum computing in simple terms.")
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
        default=["gpt-4o", "claude-3.5-sonnet-2024-10-22"]
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    system_prompt = st.text_area("System prompt (optional)", "")
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
        cols = st.columns(len(data["results"]))
        for c, r in zip(cols, data["results"]):
            with c:
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
                
                if r.get("logprobs") is not None:
                    fig = go.Figure()
                    fig.add_bar(x=list(range(len(r["logprobs"]))), y=r["logprobs"])
                    fig.update_layout(title="Logprobs (placeholder)")
                    st.plotly_chart(fig, use_container_width=True)

        st.subheader("Embedding similarity")
        labels = [r["model"] for r in data["results"]]
        mat = np.array([[data["embedding_similarity"][a][b] for b in labels] for a in labels])
        heat = go.Figure(data=go.Heatmap(z=mat, x=labels, y=labels))
        heat.update_layout(height=400)
        st.plotly_chart(heat, use_container_width=True)

        st.subheader("Autopsy summary")
        summ = data.get("summaries", {})
        rec = summ.get("top_model")
        if rec:
            st.markdown(
                f"<div style='display:inline-block;padding:6px 10px;border-radius:8px;background:#eef6ff;border:1px solid #cfe3ff;font-weight:600;'>"
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
                meta_cols[0].metric("Highest risk", f"{hm.get('model','?')}", f"{hm.get('score',0):.1f}")
            cp = summ.get("closest_pair", {})
            if cp:
                meta_cols[1].metric("Closest pair (cos)", f"{cp.get('a','?')} vs {cp.get('b','?')}", f"{cp.get('cosine',0):.3f}")
            fp = summ.get("farthest_pair", {})
            if fp:
                meta_cols[2].metric("Most divergent (cos)", f"{fp.get('a','?')} vs {fp.get('b','?')}", f"{fp.get('cosine',0):.3f}")
        else:
            st.info("No summary available yet.")

        st.subheader("Hallucination risk by model")
        risk_fig = go.Figure(go.Bar(x=[res['hallucination_risk'] for res in data['results']],
                                   y=[res['model'] for res in data['results']],
                                   orientation='h'))
        risk_fig.update_layout(height=300, margin=dict(l=80, r=20, t=20, b=20))
        st.plotly_chart(risk_fig, use_container_width=True)

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
            st.subheader("Logprob differences between models")
            for pair, stats in data["logprob_diffs"].items():
                st.markdown(f"**{pair}** — mean |Δlogprob|: {stats['mean_abs_diff']:.3f}, max: {stats['max_abs_diff']:.3f}, tokens compared: {int(stats['n_compared'])}")

        # Semantic divergence (heuristics)
        sd = data.get("semantic_diffs", {})
        if sd:
            st.subheader("Semantic divergence (heuristics)")
            for pair, scores in sd.items():
                # Show top 2 labels by score
                top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:2]
                tags = " • ".join([f"{k}: {v:.2f}" for k,v in top])
                st.markdown(f"**{pair}** — {tags}")

# Display aligned token data as tables
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
             
            # Pairwise diffs
            if data.get('token_diffs'):
                for pair_key, diff_data in data['token_diffs'].items():
                    model_a, model_b = pair_key.split('||')
                    report += f"### {model_a} vs {model_b}\n\n"
                    report += f"```diff\n{diff_data['unified']}\n```\n\n"
            
            return report
        
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

with tab_exp:
    st.subheader("Experiments")
    e_prompt = st.text_area("Prompt", "Explain quantum computing in simple terms.", key="exp_prompt")
    e_models = st.multiselect(
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
        key="exp_models"
    )
    temps_str = st.text_input("Temperatures (comma separated)", "0.2,0.7", key="exp_temps")
    sys_multi = st.text_area("System prompts (one per line; blank allowed)", "", key="exp_sys")
    seeds_str = st.text_input("Seeds (optional, comma separated)", "", key="exp_seeds")

    if st.button("Run Experiments"):
        temps = [float(x.strip()) for x in temps_str.split(",") if x.strip()]
        sys_list = [s if s.strip() else "" for s in sys_multi.splitlines()] or [""]
        seeds = [int(x.strip()) for x in seeds_str.split(",") if x.strip()] if seeds_str.strip() else []
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
        st.dataframe(df, use_container_width=True)

        # Risk by temperature chart (grouped bars per model)
        st.subheader("Risk by temperature")
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
        st.subheader("Stability (cosine to centroid)")
        stab = exp["drift"]["stability"]
        fig_stab = go.Figure()
        for m, vals in stab.items():
            fig_stab.add_trace(go.Scatter(x=list(range(1, len(vals)+1)), y=vals, mode="lines+markers", name=m))
        fig_stab.update_layout(height=350, yaxis=dict(range=[0,1]))
        st.plotly_chart(fig_stab, use_container_width=True)

        # Risk vs Stability scatter
        st.subheader("Risk vs stability")
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
                                    text=[f"{rr['model']} | T={rr['temperature']} | S={rr['system_prompt']}" for rr in exp["runs"]]))
        fig_rs.update_layout(xaxis_title="Risk", yaxis_title="Latency (ms)", height=350)
        st.plotly_chart(fig_rs, use_container_width=True)
# Parameter influence (auto-detected)
        pi = exp.get("drift", {}).get("parameter_influence")
        if pi:
            st.subheader("Parameter influence (auto-detected)")
            for param, models_map in pi.items():
                st.markdown(f"**{param.capitalize()}**")
                for model, vals in models_map.items():
                    st.markdown(f"- *{model}*")
                    for v, metrics in vals.items():
                        st.markdown(f"  - {param} = `{v}` → similarity to baseline: {metrics['similarity_to_baseline']:.3f}, risk change: {metrics['risk_change']:+.2f}")

        # CSV download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download runs as CSV", data=csv, file_name="prompt_autopsy_experiments.csv", mime="text/csv")