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

# Call /health once on load and cache the result
@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_health_status():
    try:
        resp = requests.get(f"{get_backend_host()}/health", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None

health_data = get_health_status()

# Add CSS for card layout
st.markdown("""
<style>
.pa-eq { opacity: 0.85; }
.pa-del { background: rgba(255,0,0,0.15); text-decoration: line-through; padding: 2px; border-radius: 3px; }
.pa-ins { background: rgba(0,128,0,0.15); padding: 2px; border-radius: 3px; }
.pa-card {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 20px;
    background-color: #ffffff;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    height: 100%;
    display: flex;
    flex-direction: column;
}
.pa-card-content {
    flex-grow: 1;
}
.pa-card-meta {
    font-size: 0.85em;
    color: #666;
    margin-top: 8px;
}
.pa-card-factcheck {
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid #eee;
}
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
            "claude-3-haiku",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-2.5-pro",
            "gemini-2.5-flash"
        ],
        default=["gpt-4o", "claude-3.5-sonnet-2024-10-22"],
        help="Choose one or more LLMs to compare. The same prompt will be sent to each."
    )
    
    # Check if any selected model is a gemini model and vertex is not ready
    if health_data and not health_data.get("vertex_ready", False):
        gemini_models_selected = any(model.startswith("gemini") for model in models)
        if gemini_models_selected:
            st.warning("Gemini requires Vertex AI config; see README and .env")
    
    st.caption("Gemini logprobs require Vertex AI SDK; OpenAI-compatible API does not support them.")
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
        if resp.status_code != 200:
            snippet = resp.text[:200] + "..." if len(resp.text) > 200 else resp.text
            st.error(f"Compare request failed with status {resp.status_code}: {snippet}")
            st.stop()
        data = resp.json()
    except Exception as e:
        st.error(f"Request failed: {e}")
        st.stop()

def render_model_card(result_dict):
    """Render a model result in a consistent card format."""
    # Card container with consistent styling
    st.markdown('<div class="pa-card"><div class="pa-card-content">', unsafe_allow_html=True)
    
    # Model name as header
    st.markdown(f"**{result_dict['model']}**")
    
    # Model output in code block
    st.code(result_dict["output_text"], wrap_lines=True)
    
    # Highlight provider errors
    if "ERROR:" in result_dict["output_text"].upper():
        st.warning(f"Provider error for {result_dict['model']}. See output above for details.")
    
    # Hallucination risk
    st.markdown(f"**Hallucination risk:** {result_dict['hallucination_risk']:.1f} / 100")
    if result_dict.get("hallucination_reasons"):
        st.markdown("Reasons:")
        # Show top 3 reasons
        for reason in result_dict["hallucination_reasons"][:3]:
            st.markdown(f"- {reason}")
    
    # Display token usage and cost meta line
    meta = []
    if result_dict.get("prompt_tokens") is not None:
        meta.append(f"prompt={result_dict['prompt_tokens']}")
    if result_dict.get("completion_tokens") is not None:
        meta.append(f"completion={result_dict['completion_tokens']}")
    if result_dict.get("total_tokens") is not None:
        meta.append(f"total={result_dict['total_tokens']}")
    if result_dict.get("cost_usd") is not None:
        meta.append(f"~${result_dict['cost_usd']:.6f}")
    if meta:
        st.markdown('<div class="pa-card-meta">' + " • ".join(meta) + '</div>', unsafe_allow_html=True)
    
    # Display fact-check information
    fc = result_dict.get("factcheck")
    if fc:
        st.markdown('<div class="pa-card-factcheck">', unsafe_allow_html=True)
        st.markdown("<span class='pa-tip'>Fact-check<span class='pa-tiptext'>Quick Wikipedia-based check of factual claims. Not authoritative; for reference only.</span></span>", unsafe_allow_html=True)
        st.markdown(f"**Fact-check:** ✅ {fc['supported']} supported • ⚠️ {fc['ambiguous']} ambiguous • ❌ {fc['unsupported']} unsupported")
        with st.expander("Claims & evidence"):
            for item in fc.get("claims", [])[:10]:
                st.markdown(f"- **{item['verdict']}** — {item['claim']}" + (f"  _(source: {item['evidence']})_" if item.get('evidence') else ""))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display logprobs chart if available
    if result_dict.get("logprobs") is not None:
        # Ensure tokens is properly handled even if None or missing
        tokens = result_dict.get("tokens")
        if tokens is None:
            # Create a list of None values with same length as logprobs for proper hover text
            tokens = [None] * len(result_dict["logprobs"])
        st.markdown("**Token log probabilities chart:**")
        render_logprob_chart(result_dict['model'], tokens, result_dict["logprobs"])
    elif result_dict.get("logprobs") is None and result_dict.get("tokens") is not None:
        # If we have tokens but no logprobs, show a caption
        st.caption(f"{result_dict['model']}: logprobs unavailable for this model/provider")
    
    st.markdown("</div></div>", unsafe_allow_html=True)

def render_logprob_chart(model_label, tokens, logprobs):
    """Render a Plotly bar chart for token log probabilities."""
    # If logprobs is missing/empty, show a caption
    if not logprobs or all(logprob == 0 for logprob in logprobs):
        st.caption(f"{model_label}: logprobs unavailable")
        return
    
    # Create x-axis values (token indices)
    x_values = list(range(len(logprobs)))
    
    # Create hover text with token and value information
    hover_text = []
    for i, (logprob, token) in enumerate(zip(logprobs, tokens if tokens else [None]*len(logprobs))):
        token_text = token if token is not None else f"Token {i}"
        # Escape HTML characters in token text for hover
        token_text_escaped = token_text.replace("&", "&").replace("<", "<").replace(">", ">").replace('"', '"')
        hover_text.append(f"Index: {i}<br>Token: {token_text_escaped}<br>LogProb: {logprob:.3f}")
    
    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_values,
        y=logprobs,
        text=[f"{logprob:.2f}" for logprob in logprobs],  # Add text annotations to bars
        textposition="auto",  # Position text automatically
        hovertext=hover_text,
        hoverinfo="text+x+y+name"  # Include custom text, axis info, and trace name
    ))
    
    # Update layout with proper labels and title
    fig.update_layout(
        title=f"Token log probabilities for {model_label}",
        xaxis_title="Token index in output",
        yaxis_title="Log probability (natural log)",
        height=320,
        margin=dict(l=50, r=20, t=40, b=50),
        hovermode='x unified'  # Better hover experience
    )
    
    # Add a note about chart interaction
    st.caption("Hover over bars to see token details. Click and drag to zoom. Double-click to reset zoom.")
    
    st.plotly_chart(fig, use_container_width=True)

# Main tab content
with tab_main:
    # Show Vertex AI status badge
    if health_data:
        if health_data.get("vertex_ready", False):
            location = health_data.get("location", "unknown")
            st.caption(f"Live • Vertex AI • {location}")
        else:
            st.caption("Vertex AI not configured")
    
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
                render_model_card(r)
        
        # Add combined overlay chart if 2 or more models have logprobs
        models_with_logprobs = [r for r in data["results"] if r.get("logprobs") and len(r["logprobs"]) > 0]
        if len(models_with_logprobs) >= 2:
            st.divider()
            st.subheader("Combined log probability comparison")
            st.caption("Note: Token indices may not align perfectly between models due to different tokenizations.")
            
            # Create combined figure
            fig_combined = go.Figure()
            
            # Add a trace for each model
            for result in models_with_logprobs:
                model_label = result["model"]
                logprobs = result["logprobs"]
                tokens = result.get("tokens", [None]*len(logprobs))
                
                # Create x-axis values (token indices)
                x_values = list(range(len(logprobs)))
                
                # Create hover text
                hover_text = []
                for i, (logprob, token) in enumerate(zip(logprobs, tokens)):
                    token_text = token if token is not None else f"Token {i}"
                    token_text_escaped = token_text.replace("&", "&").replace("<", "<").replace(">", ">").replace('"', '"')
                    hover_text.append(f"Model: {model_label}<br>Index: {i}<br>Token: {token_text_escaped}<br>LogProb: {logprob:.3f}")
                
                # Add trace for this model
                fig_combined.add_trace(go.Scatter(
                    x=x_values,
                    y=logprobs,
                    mode='lines+markers+text',  # Add text mode
                    name=model_label,
                    text=[f"{logprob:.2f}" for logprob in logprobs],  # Add text annotations
                    textposition="top center",  # Position text at top center
                    hovertext=hover_text,
                    hoverinfo="text+x+y+name"  # Include custom text, axis info, and trace name
                ))
            
            # Update layout
            fig_combined.update_layout(
                title="Token log probabilities comparison",
                xaxis_title="Token index in output",
                yaxis_title="Log probability (natural log)",
                height=320,
                margin=dict(l=50, r=20, t=40, b=50),
                hovermode='x unified'
            )
            
            # Add a note about chart interaction
            st.caption("Hover over lines to see token details. Click and drag to zoom. Double-click to reset zoom.")
            
            st.plotly_chart(fig_combined, use_container_width=True)
        elif len(models_with_logprobs) == 1:
            # Explain why combined chart isn't shown
            st.divider()
            st.subheader("Combined log probability comparison")
            st.info("Combined chart not shown because only one model returned log probabilities. "
                    "Some models (like Anthropic's Claude) don't currently support token-level log probabilities.")
        elif len(models_with_logprobs) == 0:
            # Explain why combined chart isn't shown
            st.divider()
            st.subheader("Combined log probability comparison")
            st.info("Combined chart not shown because none of the selected models returned log probabilities. "
                    "Some models (like Anthropic's Claude) don't currently support token-level log probabilities.")

        st.divider()
        st.subheader("Embedding similarity")
        if data.get("embedding_similarity") and data["results"]:
            labels = [r["model"] for r in data["results"]]
            mat = np.array([[data["embedding_similarity"][a][b] for b in labels] for a in labels])
            heat = go.Figure(data=go.Heatmap(z=mat, x=labels, y=labels))
            heat.update_layout(
                height=320,
                margin=dict(l=50, r=20, t=40, b=50)
            )
            st.plotly_chart(heat, use_container_width=True)
        else:
            st.caption("Embedding similarity data is not available.")

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
        st.caption("Estimated hallucination risk score for each model. Higher scores indicate a greater likelihood of unsupported claims.")
        if data.get("results"):
            risk_fig = go.Figure(go.Bar(x=[res['hallucination_risk'] for res in data['results']],
                                       y=[res['model'] for res in data['results']],
                                       orientation='h'))
            risk_fig.update_layout(
                height=320,
                margin=dict(l=50, r=20, t=40, b=50)
            )
            st.plotly_chart(risk_fig, use_container_width=True)
        else:
            st.caption("Hallucination risk data is not available.")

        st.divider()
        # Token-level diff section
        if len(data["results"]) > 1 and data.get("token_diffs"):
            st.subheader("Token-level diff")
            st.caption("Visual comparison of token-level differences between model outputs. Red strikethrough shows deletions, green highlights show insertions.")
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
            st.caption("How confident each model was in generating each token. Large differences may indicate uncertainty or stylistic variation.")
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
    else:
        st.caption("Aligned logprob data is not available.")
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
        ["gpt-5","gpt-5-mini","gpt-4o","gpt-4o-mini","claude-3.5-sonnet-2024-10-22","claude-3.5-sonnet-2024-06-20","claude-3.5-haiku","claude-3-haiku","gemini-1.5-pro","gemini-1.5-flash","gemini-2.5-pro","gemini-2.5-flash"],
        default=["gpt-4o", "claude-3.5-sonnet-2024-10-22"],
        key="exp_models",
        help="Pick the models to test. Each model is run with every temperature/system prompt/seed you specify."
    )
    st.caption("Gemini logprobs require Vertex AI SDK; OpenAI-compatible API does not support them.")
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