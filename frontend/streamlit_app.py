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
    models = st.multiselect("Models", ["gpt-4", "claude-3-opus", "gpt-4o"], default=["gpt-4", "claude-3-opus"])
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
                st.markdown(f"**Hallucination risk:** {r['hallucination_risk']:.1f} / 100")
                if r.get("hallucination_reasons"):
                    st.markdown("Reasons:")
                    for reason in r["hallucination_reasons"]:
                        st.markdown(f"- {reason}")
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
        st.write(data["summaries"])

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

with tab_exp:
    st.subheader("Experiments")
    e_prompt = st.text_area("Prompt", "Explain quantum computing in simple terms.", key="exp_prompt")
    e_models = st.multiselect("Models", ["gpt-4","gpt-4o","claude-3-opus"], default=["gpt-4","claude-3-opus"], key="exp_models")
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

        # CSV download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download runs as CSV", data=csv, file_name="prompt_autopsy_experiments.csv", mime="text/csv")