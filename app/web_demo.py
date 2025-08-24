import os, sys, json, asyncio, time
from pathlib import Path
import streamlit as st
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from web_backend import (
    prepare_agents_for_app, run_case_for_app,
    DEFAULT_FUSION_PARAMS, DEFAULT_REVIEWER_WEIGHTS
)
from core.util import ResourcePool

# ---------- Initialize heavy resources once ----------
ResourcePool.initialize(
    case_out_dir='case_out/history_case_rag',
    st_model_name="intfloat/e5-large-v2",
    umls_sqlite_path='umls_out/mini_kit.sqlite'
)

# ---------- Presets ----------
def load_presets() -> dict:
    """
    Load JSON files from app/presets/*.json.
    If folder is missing, return a small built-in catalog.
    JSON schema:
    {
      "symptoms": "...", "history": "",
      "topk": 3,
      "fusion": {"p_lock": 0.4, "margin": 0.15},
      "name": "Drug reaction (demo)"
    }
    """
    base = Path(__file__).parent / "presets"
    out = {}
    if base.exists():
        for p in sorted(base.glob("*.json")):
            try:
                obj = json.loads(p.read_text("utf-8"))
                out[obj.get("name", p.stem)] = obj
            except Exception:
                continue
    if not out:
        out = {
            "peptic ulcer disease": {
                "name": "peptic ulcer disease",
                "symptoms": "I have a burning sensation in my stomach that comes and goes. It's worse when I eat and when I lie down. I also have heartburn and indigestion.",
                "history": "",
                "topk": 2,
                "fusion": {"p_lock": 0.4, "margin": 0.15}
            },
            "fungal infection": {
                "name": "fungal infection",
                "symptoms": "I've been scratching myself a lot lately, and my skin is covered in rashy places. I also have a few pimples that are pretty firm, and there are some spots on my body that are a different shade of brown than the rest of my skin.",
                "history": "",
                "topk": 2,
                "fusion": {"p_lock": 0.6, "margin": 0.15}
            },
            "arthritis": {
                "name": "arthritis",
                "symptoms": "My muscles are weak and my neck is tight. I have swollen joints and it is hard to move around without getting stiff. It is also hard to walk.",
                "history": "",
                "topk": 3,
                "fusion": {"p_lock": 0.5, "margin": 0.2}
            }
        }
    return out

PRESETS = load_presets()

# Page config & CSS
st.set_page_config(page_title="Multi-Agent Clinical Demo", layout="wide")

CSS = """
<style>
.small-muted { color: #6a6f7a; font-size: 12px; }
.badge { display:inline-block; padding:2px 8px; border-radius: 999px; font-size:12px; margin-right:6px; }
.badge-blue { background:#e7f0ff; color:#174ea6; }
.badge-green{ background:#e6f4ea; color:#137333; }
.badge-gray { background:#edeff1; color:#3c4043; }
.badge-red  { background:#fce8e6; color:#b80600; }
.ref-key { background:#fff7e6; padding:2px 6px; border-radius:6px; }
.kpi { font-weight:600; font-size:24px; }
.card { border: 1px solid #e9ecef; border-radius:12px; padding:14px 18px; margin-bottom:16px; }
.q { color:#174ea6; }
.a { color:#137333; }
hr.light { border:none; height:1px; background:#ececec; margin:8px 0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------- Session agents ----------
if "agents" not in st.session_state:
    st.session_state.agents = asyncio.run(prepare_agents_for_app())

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("Controls")
    preset_names = ["(custom)"] + list(PRESETS.keys())
    choice = st.selectbox("Demo preset", preset_names, index=0)
    if st.button("Load preset"):
        if choice != "(custom)":
            preset = PRESETS[choice]
            st.session_state["sym_text"] = preset["symptoms"]
            st.session_state["hist_text"] = preset.get("history", "")
            st.session_state["topk"] = preset.get("topk", 3)
            st.session_state["p_lock"] = preset.get("fusion", {}).get("p_lock", 0.4)
            st.session_state["margin"] = preset.get("fusion", {}).get("margin", 0.15)

    sym = st.text_area("Patient symptoms", st.session_state.get("sym_text", ""), height=110)
    hist = st.text_area("History (optional)", st.session_state.get("hist_text", ""), height=70)

    topk = st.slider("Top-K from General Doctor", 1, 5, st.session_state.get("topk", 3))
    st.markdown("Reviewer gate")
    p_lock = st.slider("Reviewer lock (p_lock)", 0.0, 1.0, float(st.session_state.get("p_lock", 0.4)), 0.05)
    margin = st.slider("Reviewer margin", 0.0, 0.5, float(st.session_state.get("margin", 0.15)), 0.01)

    st.markdown("Fusion weights")
    alpha = st.slider("α - Doctor", 0.0, 1.0, 0.80, 0.05)
    beta  = st.slider("β - Reviewer", 0.0, 1.0, 0.15, 0.05)
    gamma = st.slider("γ - Expert", 0.0, 1.0, 0.05, 0.05)

    run_btn = st.button("▶ Run", use_container_width=True)
    clear_btn = st.button("🧹 Clear", use_container_width=True)

if clear_btn:
    for k in ("sym_text","hist_text"):
        st.session_state.pop(k, None)
    st.experimental_rerun()

# ---------- Header ----------
st.title("Audit-ready Multi-Agent Clinical Diagnosis")
st.caption("Doctor-First fusion + Early-stopped Critic-Expert dialogue with conditional Reviewer gating. All artifacts generated live from your inputs.")

# ---------- Helpers ----------
def highlight_reference(ref: str, symptoms_text: str) -> str:
    """
    Very light-weight highlighter: bold keys, wrap lists with .ref-key.
    Highlight symptom words if they appear in 'matched=[...]' or 'typical=[...]'.
    """
    if not ref:
        return ""
    txt = ref
    # bold tags
    for key in ["[History RAG]", "[BaseKnowledge RAG]", "disease=", "matched=", "typical=", "missing_keys=", "def="]:
        txt = txt.replace(key, f"**{key}**")
    # wrap bracketed lists
    txt = txt.replace("[", "<span class='ref-key'>[")
    txt = txt.replace("]", "]</span>")
    return txt

def show_kpis(summary: dict, gate: dict):
    c1, c2, c3, c4 = st.columns([1,1,2,2])
    c1.metric("Latency (ms)", summary.get("latency_ms", "-"))
    c2.metric("LLM calls", summary.get("llm_calls", "-"))
    gate_badge = "Skipped" if not gate.get("need_review") else "Triggered"
    gate_color = "badge-green" if gate_badge == "Skipped" else "badge-blue"
    c3.markdown(f"<div class='kpi'><span class='badge {gate_color}'>{gate_badge}</span> <span class='small-muted'>{gate.get('reason','')}</span></div>", unsafe_allow_html=True)
    top1 = summary.get("top1") or "-"
    c4.markdown(f"<div class='kpi'>Top-1: {top1}</div>", unsafe_allow_html=True)

def show_candidate_card(idx: int, cand: dict, reviewers: dict, aggregate: List[dict]):
    st.markdown(f"### {idx}. {cand['diagnosis']}")
    with st.container():
        c1, c2, c3 = st.columns(3)
        c1.metric("Doctor Conf", f"{cand.get('doctor_confidence',0.0):.2f}", help="From General Doctor")
        c2.metric("Expert Conf", f"{cand.get('expert_confidence_last_round',0.0):.2f}", help="From last Expert round")
        # pick review score if exists
        review_score = 0.0
        for a in aggregate:
            if a["diagnosis"] == cand["diagnosis"]:
                review_score = a["final_score"]
                break
        c3.metric("Review Score", f"{review_score:.2f}", help="Weighted Accuracy/Coverage/Interpretability/Specificity")

        # Critic–Expert Q/A
        st.subheader("Critic–Expert Dialogue")
        for r_i, r in enumerate(cand.get("rounds", []), start=1):
            st.markdown(f"**Round {r_i}**")
            colq, cola = st.columns(2)

            # Critic
            cq = r.get("critic", {})
            with colq:
                st.markdown("<span class='badge badge-blue'>Critic</span>", unsafe_allow_html=True)
                st.write(f"Diagnose: **{', '.join(cq.get('diagnose',[]))}**")
                for q in cq.get("critique", []):
                    st.markdown(f"- <span class='q'>{q}</span>", unsafe_allow_html=True)
                # review tags from previous round
                if cq.get("review"):
                    tags = " ".join([f"<span class='badge badge-gray'>{t}</span>" for t in cq["review"]])
                    st.markdown(tags, unsafe_allow_html=True)

            # Expert
            if "expert" in r:
                ex = r["expert"]
                with cola:
                    st.markdown("<span class='badge badge-green'>Expert</span>", unsafe_allow_html=True)
                    st.write(f"Diagnose: **{', '.join(ex.get('diagnose',[]))}**")
                    st.caption(f"Confidence: {ex.get('confidence',[0.0])[0]:.2f}")
                    for a in ex.get("response", []):
                        st.markdown(f"- <span class='a'>{a}</span>", unsafe_allow_html=True)
                    if ex.get("reference"):
                        with st.expander("Expert References"):
                            for ref in ex["reference"]:
                                st.markdown(highlight_reference(ref, ""), unsafe_allow_html=True)
            st.markdown("<hr class='light'/>", unsafe_allow_html=True)

# ---------- Main run ----------
if run_btn:
    # Fuse params
    fusion = DEFAULT_FUSION_PARAMS.copy()
    fusion.update({"alpha": float(alpha), "beta": float(beta), "gamma": float(gamma),
                   "p_lock": float(p_lock), "margin": float(margin)})
    try:
        with st.spinner("Running agents..."):
            out = asyncio.run(run_case_for_app(
                agents=st.session_state.agents,
                symptoms_text=sym.strip(),
                patient_history=hist.strip(),
                topk_from_doctor=int(topk),
                reviewer_weights=DEFAULT_REVIEWER_WEIGHTS,
                fusion_params=fusion,
                truth=None
            ))
        # ---------- Summary bar ----------
        show_kpis(
            {**out.get("summary", {}), "llm_calls": out.get("llm_calls")},
            out.get("gate", {})
        )

        # ---------- Section: Inputs & Weights ----------
        st.markdown("## Inputs")
        st.text_area("Symptoms", out["input"]["symptoms_text"], height=80, disabled=True)

        # ---------- Section: General Doctor ----------
        st.markdown("## General Doctor — Candidates")
        # Render per candidate card
        for i, cand in enumerate(out.get("candidates", []), start=1):
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            show_candidate_card(i, cand, out.get("reviewers", {}), out.get("review_aggregate", []))
            st.markdown("</div>", unsafe_allow_html=True)

        # ---------- Section: Reviewers ----------
        st.markdown("## Reviewer")
        gate = out.get("gate", {})
        gate_badge = "Skipped" if not gate.get("need_review") else "Triggered"
        gate_color = "badge-green" if gate_badge == "Skipped" else "badge-blue"
        st.markdown(f"<span class='badge {gate_color}'>{gate_badge}</span> {gate.get('reason','')}", unsafe_allow_html=True)

        if gate.get("need_review"):
            cols = st.columns(4)
            rvw = out.get("reviewers", {})
            for i, k in enumerate(["accuracy", "coverage", "interpretability", "specificity"]):
                with cols[i]:
                    blk = rvw.get(k) or {}
                    st.write(k.capitalize())
                    st.caption(f"Scores: {', '.join([f'{s:.2f}' for s in blk.get('scores',[])])}" if blk else "-")

            st.write("**Aggregated Review Score**")
            for a in out.get("review_aggregate", []):
                st.write(f"- {a['diagnosis']}: {a['final_score']:.3f}")

        # ---------- Section: Final Ranking ----------
        st.markdown("## Final Ranking")
        for r in out.get("ranking", []):
            st.markdown(
                f"**{r['final_rank']}. {r['diagnosis']}**  "
                f"<span class='small-muted'>Doctor {r['doctor_conf']:.2f} • Expert {r['expert_conf']:.2f} • Review {r['review_score']:.2f}</span>",
                unsafe_allow_html=True
            )

        # ---------- Export JSON ----------
        st.download_button(
            "Download this case JSON",
            data=json.dumps(out, ensure_ascii=False, indent=2),
            file_name="case_result.json",
            mime="application/json",
            use_container_width=True
        )

        # Raw artifacts (debug)
        with st.expander("General Doctor (raw)"):
            st.json(out.get("general_doctor", {}))
        with st.expander("Timing & Logs"):
            st.json({"timing": out.get("timing"), "llm_calls": out.get("llm_calls")})

    except Exception as e:
        st.error(str(e))
else:
    st.info("Configure inputs on the sidebar, then click **Run**. You can also load a preset.")
