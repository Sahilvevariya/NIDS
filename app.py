"""Network Intrusion Detection Dashboard.

Dataset-based replay prototype demonstrating a trained two-stage ML
classification pipeline (Stage 1: BENIGN vs ATTACK; Stage 2: attack
category) on CIC-IDS2017-style flow records.
"""
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    APP_TITLE,
    COLORS,
)
from backend.model_loader import load_all_artifacts, ModelLoadError
from backend.data_loader import load_demo_sample, get_attack_categories
from backend.detector import get_detector
from backend.evaluation import compute_session_metrics
from backend.metrics import (
    compute_kpis,
    get_recent_predictions,
    get_attack_distribution,
    get_prediction_timeline,
    get_alerts,
    filter_alerts_by_category,
)
from backend.replay_engine import (
    init_replay_state,
    get_replay_engine,
    get_replay_results,
    add_replay_results,
    clear_replay_results,
    update_replay_config,
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
:root{
  --bg:#F8FAFC;
  --surface:#FFFFFF;
  --border:#E2E8F0;
  --text:#0F172A;
  --muted:#64748B;
  --light:#94A3B8;
  --primary:#1D4ED8;
  --primary-hover:#1E40AF;
  --benign:#15803D;
  --attack:#DC2626;
  --warning:#D97706;
  --info:#0EA5E9;
  --shadow:0 1px 3px 0 rgb(0 0 0 / 0.08), 0 1px 2px -1px rgb(0 0 0 / 0.04);
}
html, body, .stApp, [data-testid="stAppViewContainer"]{
  background:var(--bg) !important;
  color:var(--text) !important;
}
.block-container{
  padding-top:1.4rem;
  padding-bottom:3rem;
  max-width:1240px;
  padding-left:1rem;
  padding-right:1rem;
}
#MainMenu, footer, header{visibility:hidden;}
section[data-testid="stSidebar"]{
  background:var(--surface) !important;
  border-right:1px solid var(--border) !important;
  width:290px;
}
section[data-testid="stSidebar"] .block-container{
  padding-top:1.5rem;
  padding-left:0.9rem;
  padding-right:0.9rem;
}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3{
  color:var(--text);
  font-size:0.95rem;
  font-weight:700;
  text-transform:uppercase;
  letter-spacing:0.08em;
  margin:0 0 1rem 0;
  padding-bottom:0.5rem;
  border-bottom:1px solid var(--border);
}
section[data-testid="stSidebar"] [data-baseweb="radio"]{
  background:transparent !important;
  border:1px solid transparent !important;
  padding:0.5rem 0.85rem !important;
  margin:0 0 0.35rem 0 !important;
  border-radius:8px !important;
  cursor:pointer;
}
section[data-testid="stSidebar"] [data-baseweb="radio"][aria-checked="true"]{
  background:#EFF6FF !important;
  border-color:#BFDBFE !important;
}
section[data-testid="stSidebar"] label{margin:0 !important;}
section[data-testid="stSidebar"] label span{
  color:var(--text) !important;
  font-weight:500;
  font-size:0.92rem;
}
section[data-testid="stSidebar"] [aria-checked="true"] label span{
  color:var(--primary) !important;
  font-weight:600;
}
h1{font-size:1.7rem;font-weight:700;letter-spacing:-0.02em;color:var(--text);margin:0 0 0.25rem 0;}
h2{font-size:1.2rem;font-weight:600;color:var(--text);margin:1.4rem 0 0.7rem 0;padding-bottom:0.5rem;border-bottom:1px solid var(--border);}
h3{font-size:1rem;font-weight:600;color:var(--text);margin:1rem 0 0.5rem 0;}
p, .stCaption, caption{color:var(--muted);font-size:0.88rem;}
.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:0.75rem;margin:1rem 0;}
.kpi-card{
  background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:0.85rem 1rem;box-shadow:var(--shadow);display:flex;flex-direction:column;gap:0.3rem;
}
.kpi-card .label{font-size:0.78rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:0.04em;}
.kpi-card .value{font-size:1.75rem;font-weight:700;color:var(--text);line-height:1.15;}
.kpi-card .sub{font-size:0.78rem;color:var(--light);}
.kpi-card.accent-blue{border-top:3px solid var(--primary);}
.kpi-card.accent-green{border-top:3px solid var(--benign);}
.kpi-card.accent-red{border-top:3px solid var(--attack);}
.kpi-card.accent-amber{border-top:3px solid var(--warning);}
.badge{
  display:inline-flex;align-items:center;padding:0.18rem 0.6rem;border-radius:6px;
  font-size:0.78rem;font-weight:600;letter-spacing:0.02em;
}
.badge-ok{background:#ECFDF5;color:var(--benign);border:1px solid #BBF7D0;}
.badge-warn{background:#FFFBEB;color:var(--warning);border:1px solid #FDE68A;}
.badge-error{background:#FEF2F2;color:var(--attack);border:1px solid #FECACA;}
.badge-info{background:#EFF6FF;color:#0C4A6E;border:1px solid #BFDBFE;}
.badge-plain{background:#F1F5F9;color:var(--text);border:1px solid var(--border);}
.stButton>button{
  background:var(--primary);color:white;border:none;border-radius:8px;
  padding:0.5rem 1.1rem;font-weight:600;font-size:0.88rem;
}
.stButton>button:hover{background:var(--primary-hover);}
.stButton>button[kind="secondary"]{
  background:var(--surface);color:var(--text);border:1px solid var(--border);
}
.stButton>button[kind="secondary"]:hover{background:#F1F5F9;}
.stDataFrame{border:1px solid var(--border);border-radius:8px;overflow:hidden;}
.section-card{
  background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:1.1rem;box-shadow:var(--shadow);margin:1rem 0;
}
.empty-state{
  text-align:center;padding:3rem 1rem;color:var(--muted);
}
.empty-state h3{color:var(--text);margin:0 0 0.5rem 0;}
.model-info{
  background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:1rem;box-shadow:var(--shadow);
}
.model-info .row{display:flex;justify-content:space-between;padding:0.35rem 0;border-bottom:1px solid #F1F5F9;}
.model-info .row:last-child{border:none;}
.model-info .k{color:var(--muted);font-size:0.82rem;font-weight:600;}
.model-info .v{color:var(--text);font-size:0.85rem;}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading model artifacts...")
def load_models():
    try:
        stage1, stage2 = load_all_artifacts()
        return stage1, stage2, None
    except ModelLoadError as e:
        return None, None, str(e)
    except Exception as e:
        return None, None, f"Unexpected error: {e}"


@st.cache_data(show_spinner="Loading demo data...")
def load_data():
    try:
        df = load_demo_sample()
        return df, None
    except Exception as e:
        return None, str(e)


def init_session_state():
    if "mode" not in st.session_state:
        st.session_state.mode = "Demonstration"
    if "detector_loaded" not in st.session_state:
        st.session_state.detector_loaded = False
    if "traffic_result" not in st.session_state:
        st.session_state.traffic_result = None
    if "traffic_flow_index" not in st.session_state:
        st.session_state.traffic_flow_index = 0


def kpi_card(label, value, sub="", accent="blue"):
    return (
        f'<div class="kpi-card accent-{accent}">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="sub">{sub}</div>'
        "</div>"
    )


def verdict_badge(verdict):
    mapping = {
        "CORRECT_DETECTION": ("Correct", "badge-ok"),
        "FALSE_POSITIVE": ("False Positive", "badge-warn"),
        "MISSED_ATTACK": ("Missed Detection", "badge-error"),
        "WRONG_CATEGORY": ("Wrong Category", "badge-error"),
        "ATTACK_DETECTED": ("Attack Detected", "badge-info"),
    }
    text, cls = mapping.get(verdict, ("Detection", "badge-info"))
    return f'<span class="badge {cls}">{text}</span>'


def status_badge(label, ok):
    cls = "badge-ok" if ok else "badge-error"
    text = "Loaded" if ok else "Not loaded"
    return f'<span class="badge {cls}">{label}: {text}</span>'


def chart_layout(title, height=360):
    return dict(
        template="plotly_white",
        font=dict(family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif", size=11, color="#0F172A"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        title=dict(text=title, x=0, xanchor="left", font=dict(size=13, color="#0F172A")),
        margin=dict(l=60, r=20, t=40, b=40),
        height=height,
        showlegend=True,
        legend=dict(font=dict(size=11), bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor="#F1F5F9", linecolor="#E2E8F0", tickfont=dict(size=11), title_font=dict(size=12, color="#475569")),
        yaxis=dict(showgrid=True, gridcolor="#F1F5F9", linecolor="#E2E8F0", tickfont=dict(size=11), title_font=dict(size=12, color="#475569")),
    )

NAV_PAGES = [
    "Overview Dashboard",
    "Traffic Analysis",
    "Detection Simulator",
    "Attack Alerts",
    "Model Performance",
]


def render_sidebar(stage1_artifacts, stage2_artifacts, metadata):
    with st.sidebar:
        st.markdown(
            '<div style="font-size:1.4rem;font-weight:700;letter-spacing:-0.02em;margin-bottom:0.2rem;">NIDS</div>'
            '<div style="font-size:0.8rem;color:var(--muted);margin-bottom:1.2rem;">Network intrusion detection</div>',
            unsafe_allow_html=True,
        )
        page = st.radio("Navigation", NAV_PAGES, key="nav_page", label_visibility="collapsed")
        st.markdown("---")
        st.session_state.mode = st.radio(
            "Mode",
            ["Demonstration", "Evaluation"],
            index=0 if st.session_state.mode == "Demonstration" else 1,
            help="Demonstration: predictions only. Evaluation: compare with ground truth.",
        )
        st.markdown("---")
        s1_ok = stage1_artifacts is not None
        s2_ok = stage2_artifacts is not None
        st.markdown(status_badge("Stage 1", s1_ok), unsafe_allow_html=True)
        st.markdown(status_badge("Stage 2", s2_ok), unsafe_allow_html=True)
        if metadata:
            trained = metadata.get("sklearn_version", "unknown")
            import sklearn
            current = sklearn.__version__
            if trained != current:
                st.markdown(
                    f'<span class="badge badge-warn">sklearn {trained} &#8594; {current}</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f'<span class="badge badge-ok">sklearn {current}</span>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.78rem;color:var(--light);line-height:1.4;">Dataset-based replay prototype. Not a live NIDS.</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Model Information", expanded=False):
            st.markdown('<div class="model-info">', unsafe_allow_html=True)
            rows = [
                ("Stage 1 algorithm", "Random Forest (binary)"),
                ("Stage 1 preprocessing", "StandardScaler + PCA"),
                ("PCA components", str(metadata.get("pca_components", "—")) if metadata else "—"),
                ("PCA explained variance", f"{float(metadata.get('explained_variance', 0) or 0) * 100:.2f}%" if metadata else "—"),
                ("Stage 1 input features", "79 flow attributes"),
                ("Stage 2 algorithm", "Random Forest (multiclass)"),
                ("Stage 2 preprocessing", "StandardScaler"),
                ("Attack classes", "14 categories"),
                ("Training dataset", "CIC-IDS2017"),
                ("Model artifacts loaded", "Yes" if (s1_ok and s2_ok) else "No"),
            ]
            for k, v in rows:
                st.markdown(f'<div class="row"><span class="k">{k}</span><span class="v">{v}</span></div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    return page


def render_overview_page(results):
    st.header("Overview Dashboard")
    if not results:
        st.markdown(
            '<div class="empty-state">'
            "<h3>No flows processed yet</h3>"
            "<p>Start the Detection Simulator to replay traffic and generate "
            "predictions. Results will populate this dashboard.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return
    kpis = compute_kpis(results)
    total = kpis["total_flows"]
    benign = kpis["benign_count"]
    attack = kpis["attack_count"]
    rate = kpis["attack_rate"]
    cards = (
        kpi_card("Total flows", f"{total:,}", "Flows processed this session", "blue")
        + kpi_card("Benign", f"{benign:,}", f"{100 - rate:.1f}% of traffic", "green")
        + kpi_card("Attacks", f"{attack:,}", f"{rate:.1f}% of traffic", "red")
        + kpi_card("Attack types", f"{kpis['unique_attack_types']}", "Unique categories", "amber")
    )
    st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 1])
    with col_left:
        timeline = get_prediction_timeline(results)
        if timeline:
            df_tl = pd.DataFrame(timeline)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_tl["flow_index"], y=df_tl["cumulative_attacks"],
                mode="lines", name="Cumulative attacks",
                line=dict(color=COLORS["attack"], width=2),
            ))
            fig.add_trace(go.Scatter(
                x=df_tl["flow_index"], y=df_tl["cumulative_total"] - df_tl["cumulative_attacks"],
                mode="lines", name="Cumulative benign",
                line=dict(color=COLORS["benign"], width=2),
            ))
            fig.update_layout(**chart_layout("Prediction timeline", height=360))
            fig.update_xaxes(title_text="Flow index")
            fig.update_yaxes(title_text="Cumulative count")
            st.plotly_chart(fig, use_container_width=True)
    with col_right:
        dist = get_attack_distribution(results)
        if dist:
            df_dist = pd.DataFrame(list(dist.items()), columns=["Attack type", "Count"])
            fig = go.Figure(
                go.Bar(x=df_dist["Count"], y=df_dist["Attack type"], orientation="h", marker_color=COLORS["attack"])
            )
            fig.update_layout(**chart_layout("Attack type distribution", height=360))
            fig.update_xaxes(title_text="Count")
            fig.update_yaxes(title_text="", automargin=True, categoryorder="total ascending")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent predictions")
    recent = get_recent_predictions(results, n=20)
    if recent:
        df_recent = pd.DataFrame(recent)
        display_cols = ["flow_index", "stage1_label", "stage2_label", "final_label"]
        if st.session_state.mode == "Evaluation" and "ground_truth" in df_recent.columns:
            display_cols += ["ground_truth", "correct"]
        df_display = df_recent[display_cols].copy().iloc[::-1]
        st.dataframe(df_display, use_container_width=True, hide_index=True)


def render_traffic_analysis_page(df, results):
    st.header("Traffic Analysis")
    if df is None or len(df) == 0:
        st.error("No data loaded.")
        return

    col_sel, col_act = st.columns([1.7, 1])
    with col_sel:
        default_idx = st.session_state.traffic_flow_index if st.session_state.traffic_flow_index < len(df) else 0
        flow_idx = st.number_input("Flow index", min_value=0, max_value=len(df) - 1, value=int(default_idx), step=1)
    with col_act:
        st.write("")
        analyze = st.button("Analyze flow", type="primary", use_container_width=True)

    if analyze:
        st.session_state.traffic_flow_index = int(flow_idx)
        row = df.loc[int(flow_idx)]
        gt = row.get("Label") if "Label" in df.columns else None
        detector = get_detector()
        detector.load()
        batch = pd.DataFrame([row])
        gt_series = pd.Series([gt], index=[row.name]) if gt is not None else None
        preds = detector.predict_batch(batch, gt_series)
        st.session_state.traffic_result = preds[0]

    result = st.session_state.traffic_result
    if result is None:
        st.markdown(
            '<div class="empty-state">'
            "<h3>Select a flow to analyze</h3>"
            "<p>Pick a flow index from the dataset and run inference to inspect "
            "how the two-stage pipeline classified it.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    flow_index = result.get("flow_index", st.session_state.traffic_flow_index)
    row = df.loc[flow_index]
    s1_label = result.get("stage1_label", "UNKNOWN")
    s2_label = result.get("stage2_label")
    final_label = result.get("final_label", "UNKNOWN")

    col1, col2, col3 = st.columns(3)
    with col1:
        cls = "badge-ok" if s1_label == "BENIGN" else "badge-error"
        st.markdown(
            '<div class="section-card" style="text-align:center">'
            '<div style="font-size:0.8rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;">Stage 1 - Binary</div>'
            f'<span class="badge {cls}" style="font-size:1.3rem;">{s1_label}</span>'
            f'<div style="margin-top:0.4rem;font-size:0.85rem;">Confidence: {result.get("stage1_score", 0):.1%}</div>'
            "</div>",
            unsafe_allow_html=True,
        )
    with col2:
        if s2_label:
            st.markdown(
                '<div class="section-card" style="text-align:center">'
                '<div style="font-size:0.8rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;">Stage 2 - Attack class</div>'
                f'<span class="badge badge-error" style="font-size:1.3rem;">{s2_label}</span>'
                f'<div style="margin-top:0.4rem;font-size:0.85rem;">Confidence: {result.get("stage2_score", 0):.1%}</div>'
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="section-card" style="text-align:center">'
                '<div style="font-size:0.8rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;">Stage 2 - Attack class</div>'
                '<span class="badge badge-plain" style="font-size:1.3rem;">N/A</span>'
                '<div style="margin-top:0.4rem;font-size:0.85rem;color:var(--muted);">Skipped - benign flow</div>'
                "</div>",
                unsafe_allow_html=True,
            )
    with col3:
        fcls = "badge-ok" if final_label == "BENIGN" else "badge-error"
        st.markdown(
            '<div class="section-card" style="text-align:center">'
            '<div style="font-size:0.8rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;">Final classification</div>'
            f'<span class="badge {fcls}" style="font-size:1.3rem;">{final_label}</span>'
            f'<div style="margin-top:0.4rem;font-size:0.8rem;color:var(--muted);">Flow #{flow_index}</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    st.subheader("Key flow features")
    key_features = [
        "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets",
        "Flow Bytes/s", "Flow Packets/s", "Packet Length Mean",
        "Packet Length Std", "Protocol", "Source Port", "Destination Port",
    ]
    feature_data = []
    for feat in key_features:
        if feat in row.index:
            val = row[feat]
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                val = f"{val:,.2f}" if isinstance(val, float) else f"{val:,}"
            feature_data.append({"Feature": feat, "Value": val})
    st.dataframe(pd.DataFrame(feature_data), use_container_width=True, hide_index=True)

    with st.expander("Show all features"):
        all_features = pd.DataFrame({"Feature": row.index, "Value": row.values})
        st.dataframe(all_features, use_container_width=True, hide_index=True)

    if st.session_state.mode == "Evaluation" and result.get("ground_truth"):
        st.markdown("---")
        st.subheader("Ground truth comparison")
        gt = result["ground_truth"]
        correct = result.get("correct")
        col_gt1, col_gt2 = st.columns(2)
        with col_gt1:
            st.metric("Ground truth", gt)
        with col_gt2:
            st.metric("Prediction", final_label)
        if correct:
            st.markdown('<div class="badge badge-ok" style="font-size:1rem;padding:0.6rem 1rem;">Correct classification</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="badge badge-error" style="font-size:1rem;padding:0.6rem 1rem;">Incorrect classification</div>', unsafe_allow_html=True)

def render_simulator_page(df):
    st.header("Detection Simulator")
    if df is None or len(df) == 0:
        st.error("No data loaded.")
        return

    engine = init_replay_state(df)
    st.markdown(
        '<div class="section-card">'
        "<p style=\"margin:0 0 0.5rem 0;font-size:0.82rem;color:var(--muted);\">"
        "Dataset replay - flows are drawn from the bundled CIC-IDS2017-style "
        "demo sample and scored by the trained two-stage model. This is not live "
        "network monitoring.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Replay configuration")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        categories = ["Random", "Benign only", "Attack only"] + get_attack_categories(df)
        mode_label = st.selectbox("Sample mode", categories, index=0, key="replay_mode")
    with c2:
        batch_size = st.slider("Batch size", 10, 500, int(engine.batch_size), step=10, key="replay_batch_size")
    with c3:
        interval = st.slider("Interval (seconds)", 2, 5, int(engine.interval), step=1, key="replay_interval")
    with c4:
        random_seed = st.number_input("Random seed", 0, 999999, int(engine.random_state), step=1, key="replay_seed")

    mode_map = {"Random": "random", "Benign only": "benign", "Attack only": "attack"}
    mode = mode_map.get(mode_label, mode_label)
    config = {"mode": mode, "batch_size": batch_size, "interval": interval, "random_state": int(random_seed)}
    if mode not in ("random", "benign", "attack"):
        config["attack_category"] = mode_label
    update_replay_config(config)

    st.markdown("---")
    engine = get_replay_engine()
    progress = engine.get_progress()
    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1:
        if st.button("Start / Resume", type="primary", disabled=engine.running and not engine.paused, use_container_width=True):
            engine.running = True
            engine.paused = False
            st.rerun()
    with cc2:
        if st.button("Pause", disabled=not engine.running or engine.paused, use_container_width=True):
            engine.paused = True
            st.rerun()
    with cc3:
        if st.button("Reset", use_container_width=True):
            engine.reset()
            clear_replay_results()
            st.rerun()
    with cc4:
        if st.button("Single step", disabled=engine.running and not engine.paused, use_container_width=True):
            detector = get_detector()
            if not st.session_state.detector_loaded:
                detector.load()
                st.session_state.detector_loaded = True
            batch = engine.get_next_batch()
            if batch is not None:
                gt = batch["Label"] if "Label" in batch.columns else None
                preds = detector.predict_batch(batch, gt)
                add_replay_results(preds)
                st.rerun()
            else:
                st.warning("Replay exhausted")

    st.markdown("---")
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        st.metric("Progress", f"{progress['position']}/{progress['total']} ({progress['percent']:.1f}%)")
    with pc2:
        st.metric("Total processed", progress["processed"])
    with pc3:
        st.metric("Remaining", progress["remaining"])
    st.progress(progress["percent"] / 100)

    results = get_replay_results()
    if results:
        st.subheader("Live predictions")
        recent = get_recent_predictions(results, n=30)
        df_recent = pd.DataFrame(recent)
        df_recent = df_recent.iloc[::-1]
        display_cols = ["flow_index", "stage1_label", "stage2_label", "final_label"]
        if st.session_state.mode == "Evaluation" and "ground_truth" in df_recent.columns:
            display_cols += ["ground_truth", "correct"]
        st.dataframe(df_recent[display_cols], use_container_width=True, hide_index=True)

    if engine.running and not engine.paused and not engine.is_exhausted():
        detector = get_detector()
        if not st.session_state.detector_loaded:
            detector.load()
            st.session_state.detector_loaded = True
        batch = engine.get_next_batch()
        if batch is not None:
            gt = batch["Label"] if "Label" in batch.columns else None
            preds = detector.predict_batch(batch, gt)
            add_replay_results(preds)
            time.sleep(engine.interval)
            st.rerun()
        else:
            engine.running = False
            st.success("Replay completed")
            st.rerun()


def render_alerts_page(results):
    st.header("Attack Alerts")
    if not results:
        st.markdown(
            '<div class="empty-state">'
            "<h3>No alerts yet</h3>"
            "<p>Run the Detection Simulator to generate predictions. Alerts for "
            "flows classified as attacks will appear here.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return
    evaluation = st.session_state.mode == "Evaluation"
    alerts = get_alerts(results, evaluation_mode=evaluation)
    if not alerts:
        st.markdown(
            '<div class="empty-state">'
            "<h3>No attacks detected</h3>"
            "<p>No flows were classified as attacks in the current session.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    categories = ["All"] + sorted(set(a["predicted_attack"] for a in alerts))
    selected_cat = st.selectbox("Filter by attack category", categories, key="alert_category")
    filtered = filter_alerts_by_category(alerts, selected_cat)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total alerts", len(filtered))
    with c2:
        if evaluation:
            correct = sum(1 for a in filtered if a.get("verdict") == "CORRECT_DETECTION")
            st.metric("Correct detections", correct)
    with c3:
        if evaluation:
            false_pos = sum(1 for a in filtered if a.get("verdict") == "FALSE_POSITIVE")
            st.metric("False positives", false_pos)

    st.subheader(f"Alerts ({len(filtered)})")
    for alert in filtered:
        verdict = alert.get("verdict", "ATTACK_DETECTED")
        verdict_display = verdict.replace("_", " ").title()
        with st.expander(f"Flow {alert['flow_index']} - {alert['predicted_attack']} [{verdict_display}]"):
            ca1, ca2 = st.columns(2)
            with ca1:
                st.markdown(f"**Stage 1:** {alert.get('stage1_prediction')}")
                st.markdown(f"**Stage 2:** {alert.get('stage2_prediction')}")
            with ca2:
                if evaluation and "ground_truth" in alert:
                    st.markdown(f"**Ground truth:** {alert['ground_truth']}")
                    st.markdown(f"**Verdict:** {verdict_badge(verdict)}", unsafe_allow_html=True)

    if st.button("Export alerts to CSV"):
        df_alerts = pd.DataFrame(filtered)
        csv = df_alerts.to_csv(index=False)
        st.download_button("Download CSV", csv, "attack_alerts.csv", "text/csv")


def render_performance_page(results):
    st.header("Model Performance")
    if not results:
        st.markdown(
            '<div class="empty-state">'
            "<h3>No results to evaluate</h3>"
            "<p>Run the Detection Simulator in Evaluation mode to compute "
            "metrics from actual predictions and ground truth.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return
    if st.session_state.mode != "Evaluation":
        st.markdown(
            '<div class="section-card" style="padding:1.5rem;text-align:center;">'
            "<p style=\"color:var(--muted);margin:0;\">Switch to <b>Evaluation</b> mode in the sidebar to see performance metrics.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    metrics = compute_session_metrics(results)
    metadata = (st.session_state.get("stage1_artifacts") or {}).get("metadata", {})

    st.subheader("Current session - evaluation results")
    overall = metrics.get("overall", {})
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Overall accuracy", f"{overall.get('accuracy', 0):.2%}")
    with c2:
        st.metric("Correct", overall.get("correct_predictions", 0))
    with c3:
        st.metric("False positives", overall.get("false_positives", 0))
    with c4:
        st.metric("False negatives", overall.get("false_negatives", 0))

    st.markdown("---")
    st.subheader("Stage 1 - Binary detection (BENIGN vs ATTACK)")
    stage1 = metrics.get("stage1", {})
    if stage1:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Accuracy", f"{stage1.get('accuracy', 0):.2%}")
        with c2:
            st.metric("Precision", f"{stage1.get('precision', 0):.2%}")
        with c3:
            st.metric("Recall", f"{stage1.get('recall', 0):.2%}")
        with c4:
            st.metric("F1", f"{stage1.get('f1', 0):.2%}")
        cm = stage1.get("confusion_matrix", [])
        labels = stage1.get("labels", ["BENIGN", "ATTACK"])
        if cm:
            st.markdown("**Confusion matrix**")
            st.dataframe(pd.DataFrame(cm, index=labels, columns=labels), use_container_width=True)

    st.markdown("---")
    st.subheader("Stage 2 - Attack classification (14 classes)")
    stage2 = metrics.get("stage2", {})
    if stage2:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Accuracy", f"{stage2.get('accuracy', 0):.2%}")
        with c2:
            st.metric("Weighted F1", f"{stage2.get('f1_weighted', 0):.2%}")
        with c3:
            st.metric("Macro F1", f"{stage2.get('f1_macro', 0):.2%}")
        with c4:
            st.metric("Weighted precision", f"{stage2.get('precision_weighted', 0):.2%}")
        per_class = stage2.get("per_class", {})
        if per_class:
            st.markdown("**Per-class performance**")
            rows = []
            for label, m in per_class.items():
                rows.append({"Class": label, "Precision": f"{m['precision']:.4f}", "Recall": f"{m['recall']:.4f}", "F1": f"{m['f1']:.4f}", "Support": m["support"]})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        cm = stage2.get("confusion_matrix", [])
        labels2 = stage2.get("labels", [])
        if cm and labels2:
            st.markdown("**Confusion matrices**")
            st.dataframe(pd.DataFrame(cm, index=labels2, columns=labels2), use_container_width=True)
        support = stage2.get("support", {})
        rare = [c for c, s in support.items() if s < 10]
        if rare:
            st.markdown(
                f'<div class="badge badge-warn" style="padding:0.5rem 0.8rem;display:inline-block;">'
                f"Rare classes with low support (high variance): {', '.join(rare)}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.subheader("Training-time results (reference)")
    st.caption("Values below are reported training-time results from the model metadata, not computed from this demo session.")
    tcol1, tcol2, tcol3 = st.columns(3)
    with tcol1:
        if metadata:
            st.metric("Stage 1 test accuracy", f"{float(metadata.get('test_accuracy', 0) or 0):.2%}")
            st.metric("Stage 1 train accuracy", f"{float(metadata.get('train_accuracy', 0) or 0):.2%}")
    with tcol2:
        s2 = metadata.get("stage2", {}) if metadata else {}
        if s2:
            st.metric("Stage 2 test accuracy", f"{float(s2.get('test_accuracy', 0) or 0):.2%}")
            st.metric("Stage 2 weighted F1", f"{float(s2.get('weighted_f1', 0) or 0):.2%}")
    with tcol3:
        st.metric("Flows evaluated this session", metrics.get("total_flows", 0))
        st.metric("With ground truth", metrics.get("flows_with_ground_truth", 0))

    st.markdown("---")
    st.caption("Session metrics are computed from actual predictions and ground truth in the current demo session. Training-time metrics may differ due to different test splits and sampling.")


def main():
    init_session_state()
    stage1_artifacts, stage2_artifacts, load_error = load_models()
    df, data_error = load_data()
    metadata = (stage1_artifacts or {}).get("metadata", {})

    if load_error:
        st.error(f"Model loading error: {load_error}")
        st.info("Run the training scripts (train_stage1.py, train_stage2.py) first.")
        return
    if data_error:
        st.error(f"Data loading error: {data_error}")
        st.info("Run generate_synthetic_data.py first.")
        return

    st.session_state.stage1_artifacts = stage1_artifacts
    st.session_state.stage2_artifacts = stage2_artifacts

    detector = get_detector()
    if not st.session_state.detector_loaded:
        with st.spinner("Initializing detector..."):
            detector.load()
            st.session_state.detector_loaded = True

    results = get_replay_results()
    page = render_sidebar(stage1_artifacts, stage2_artifacts, metadata)

    if page == "Overview Dashboard":
        render_overview_page(results)
    elif page == "Traffic Analysis":
        render_traffic_analysis_page(df, results)
    elif page == "Detection Simulator":
        render_simulator_page(df)
    elif page == "Attack Alerts":
        render_alerts_page(results)
    elif page == "Model Performance":
        render_performance_page(results)


if __name__ == "__main__":
    main()