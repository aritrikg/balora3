"""Streamlit entry point for the BALoRA research demonstration."""
import altair as alt
import pandas as pd
import streamlit as st

from cloud_upload import upload_page

from study_data import (
    COLORS, DATA, METHODS, METRICS, PAIR, PROMPT, aggregate, case_view,
    organ_label, read_case, read_json, read_table,
)

st.set_page_config(page_title="BALoRA | Abdominal CT study", page_icon="🔬", layout="wide")
st.html("""
<style>
  [data-testid="stMainBlockContainer"] {max-width: 73rem; padding-top: 2.1rem; padding-bottom: 3rem;}
  .study-header {background:#142743;color:#fff;border-radius:1.05rem;padding:1.65rem 2rem;margin-bottom:1.05rem;}
  .study-header h1 {color:#fff;font-size:clamp(1.9rem, 4vw, 2.75rem);line-height:1.12;margin:0;padding:0;}
  .study-header p {color:#d7e3f4;font-size:1rem;margin:.7rem 0 0;}
  .study-header .scope {color:#a5e4d9;font-size:.8rem;font-weight:650;letter-spacing:.025em;margin-top:1rem;}
  [data-testid="stMetric"] {background:#fff;border:1px solid #dbe4ef;padding:1rem 1rem .8rem;border-radius:.8rem;}
  [data-testid="stImage"] img {border-radius:.55rem;}
  @media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] {padding-top:1rem;}
    .study-header {padding:1.25rem;}
  }
</style>
<div class="study-header">
  <h1>BALoRA · Abdominal CT segmentation</h1>
  <p>Compare frozen SAM2 with deterministic and Gaussian decoder LoRA.</p>
  <p class="scope">BTCV Fold 1 &nbsp;·&nbsp; Seed 42 &nbsp;·&nbsp; Precomputed research results</p>
</div>
""")


@st.cache_data(show_spinner=False)
def table(name):
    return read_table(name)


@st.cache_data(show_spinner=False)
def report(name):
    return read_json(name)


@st.cache_data(show_spinner=False, max_entries=3)
def example(stem):
    return read_case(stem)


def method_column(frame):
    return frame.assign(Method=frame.method.map(METHODS))


def csv_button(label, frame, filename):
    st.download_button(label, frame.to_csv(index=False).encode("utf-8"), filename, "text/csv",
                       on_click="ignore")


def method_color():
    return alt.Color("Method:N", scale=alt.Scale(domain=list(METHODS.values()), range=COLORS),
                     legend=alt.Legend(title=None, orient="bottom"))


def delta_interval_viz(by_volume, value, label):
    points = alt.Chart(by_volume).mark_circle(size=85, color="#2e58c5").encode(
        x=alt.X(f"{value}:Q", title=label, scale=alt.Scale(zero=True)),
        y=alt.Y("volume:N", title=None),
        tooltip=[alt.Tooltip("volume:N", title="Volume"), alt.Tooltip(f"{value}:Q", title=label, format="+.4f")],
    )
    zero = alt.Chart(pd.DataFrame({value: [0]})).mark_rule(color="#627490", strokeDash=[4, 3]).encode(
        x=alt.X(f"{value}:Q", title=label)
    )
    st.altair_chart((zero + points).properties(height=235), width="stretch")


def benchmark():
    raw = table("per_prompt.csv.gz")
    summary = report("analysis_summary.json")
    st.header("Benchmark results")
    for box, (key, label) in zip(st.columns(3), METHODS.items()):
        box.metric(label, f"{summary['dice_pair_weighted'][key]:.4f}", help="Mean Dice over 78 volume–organ pairs.")
    st.caption(f"Full test split · {raw.volume.nunique()} volumes · {raw.organ.nunique()} organs · "
               f"{raw.key.nunique():,} prompts per method. Prompts were sampled inside reference masks.")

    st.subheader("Explore metrics")
    options = st.columns(3)
    metric = options[0].selectbox("Metric", list(METRICS), key="metric")
    weighting = options[1].selectbox("Weighting", [PAIR, PROMPT], key="weighting")
    group = options[2].selectbox("Break down by", ["Organ", "Volume"], key="breakdown").lower()
    filters = st.columns(2)
    organs = filters[0].multiselect("Filter organs", sorted(raw.organ.unique()), format_func=organ_label)
    volumes = filters[1].multiselect("Filter volumes", sorted(raw.volume.unique()))
    filtered = raw
    if organs:
        filtered = filtered[filtered.organ.isin(organs)]
    if volumes:
        filtered = filtered[filtered.volume.isin(volumes)]
    if filtered.empty:
        st.info("No archived predictions match these filters. Clear an organ or volume filter.")
        return

    field, hint = METRICS[metric]
    selected = aggregate(filtered, ["method", group], field, weighting)
    selected = method_column(selected).assign(Group=selected[group].map(organ_label) if group == "organ" else selected[group])
    fallback = alt.Scale(domain=[0, 1]) if field in {"dice", "iou"} else alt.Scale(zero=True)
    chart = alt.Chart(selected).mark_point(filled=True, size=94).encode(
        x=alt.X(f"{field}:Q", title=metric, scale=fallback),
        y=alt.Y("Group:N", title=None, sort="ascending"),
        color=method_color(),
        shape=alt.Shape("Method:N", scale=alt.Scale(domain=list(METHODS.values()), range=["circle", "square", "diamond"]),
                        legend=None),
        tooltip=[alt.Tooltip("Group:N", title=group.title()), "Method:N",
                 alt.Tooltip(f"{field}:Q", title=metric, format=".5f")],
    ).properties(height=max(245, 29 * selected.Group.nunique()))
    st.altair_chart(chart, width="stretch")
    aggregation = "Slice scores are averaged within each volume–organ pair first." if weighting == PAIR else "Every prompt has equal weight."
    st.caption(f"{filtered.key.nunique():,} matched prompts in this view. {aggregation} {hint}")

    totals = method_column(aggregate(filtered, ["method"], field, weighting))
    totals = totals.set_index("Method").reindex(METHODS.values()).reset_index()
    st.dataframe(totals[["Method", field]].rename(columns={field: metric}), hide_index=True, width="stretch",
                 column_config={metric: st.column_config.NumberColumn(format="%.5f")})
    csv_button("Download filtered prediction rows", filtered, "balora_filtered_predictions.csv")

    with st.expander("Paired Dice effect on the full test split"):
        effect = summary["paired_dice"]
        prompt = summary["dice_prompt_weighted"]
        st.write(f"Deterministic minus Gaussian was **{effect['mean_pp']:+.3f} percentage points** "
                 f"with equal volume–organ weighting and "
                 f"**{100 * (prompt['deterministic'] - prompt['balora']):+.3f} points** with equal prompt weighting.")
        low, high = effect["t_interval_pp"]
        st.caption(f"The descriptive 95% t interval across six test volumes is [{low:+.3f}, {high:+.3f}] points. "
                   "The relative overlap of the two adapted models remains inconclusive.")
    with st.expander("Original model-call timing"):
        time = summary["timing"]
        frame = pd.DataFrame([{
            "Method": label, "Encoder (ms/slice)": time[key]["mean_encoder_ms"],
            "Decoder (ms/prompt)": time[key]["mean_decoder_ms_per_prompt"],
            "Recorded total (s/slice)": time[key]["mean_compute_s"],
        } for key, label in METHODS.items()])
        st.dataframe(frame, hide_index=True, width="stretch")
        st.caption("Original T4 model-call measurements. Totals exclude data loading, metric calculation, and export.")


def cases():
    st.header("CT case explorer")
    st.caption("Archive samples selected for qualitative inspection. The same reference mask and prompt are shown for all methods.")
    index_file = DATA / "case_index.json"
    available = [x for x in report(index_file.name) if (DATA / "examples" / f"{x['stem']}.npz").is_file()] if index_file.is_file() else []
    if not available:
        st.info("This deployment includes aggregate results. Add the saved example assets to enable mask comparisons.")
        return
    pick1, pick2 = st.columns([1, 2])
    organ = pick1.selectbox("Organ", sorted({c["organ"] for c in available}),
                            format_func=organ_label, key="case_organ")
    by_key = {c["key"]: c for c in available if c["organ"] == organ}
    key = pick2.selectbox("Archived case", list(by_key), format_func=lambda k: (
        f"{by_key[k]['volume']} · slice {by_key[k]['slice']} · Gaussian Dice {by_key[k]['metrics']['balora']['dice']:.3f}"
    ), key="case_key")
    c = by_key[key]
    view = st.radio("Overlay", ["Contours", "Mask fills", "Mask errors"], horizontal=True, key="overlay")
    controls = st.columns(3)
    ref = controls[0].toggle("Reference outline", value=True, key="reference_outline", disabled=view == "Mask errors")
    point = controls[1].toggle("Prompt marker", value=True, key="prompt_marker")
    zoom = controls[2].toggle("Zoom to mask region", value=True, key="zoom")
    arrays = example(c["stem"])
    selected_methods = st.multiselect("Compare methods", list(METHODS), default=list(METHODS),
                                      format_func=METHODS.__getitem__, key="case_methods")
    order = ["reference", *selected_methods]
    if view == "Mask errors":
        st.caption("Teal = overlap · Coral = false positive · Blue = false negative · Pale yellow cross = prompt.")
    else:
        st.caption("Mint = reference outline · Amber = predicted mask · Pale yellow cross = prompt.")
    st.caption(f"{organ_label(organ)} · {c['volume']} · slice {c['slice']} · positive point ({c['x']}, {c['y']})")
    for col, method in zip(st.columns(len(order)), order):
        with col:
            st.markdown(f"**{METHODS.get(method, 'Reference')}**")
            st.image(case_view(arrays, c["x"], c["y"], method, view, ref, point, zoom), width="stretch")
            if method in METHODS:
                m = c["metrics"][method]
                st.caption(f"Dice {m['dice']:.4f} · IoU {m['iou']:.4f}")
    if st.toggle("Show Gaussian decoder mutual information", key="show_mi"):
        st.image(case_view(arrays, c["x"], c["y"], "mi", "Contours", ref, point, zoom), width=520)
        st.caption("Archived pixel-level MI in nats. Values below 0.005 are hidden and colors saturate at 0.15 nats. "
                   "The map shows variation across sampled predictions.")


VARIANTS = {
    "frozen_single": "Frozen single output",
    "frozen_multimask_iou": "Frozen multimask · predicted IoU",
    "frozen_multimask_oracle": "Frozen multimask · reference-selected",
    "frozen_single_or_multimask_oracle": "Frozen single or multimask · reference-selected",
    "deterministic_single": "Deterministic single output",
    "decoder_mc_dropout": "Deterministic MC dropout",
    "gaussian_learned": "Gaussian learned SD",
    "gaussian_fixed_sd": "Gaussian fixed SD",
}


def replay():
    st.header("Inference replay")
    st.caption("Four replay arms evaluated the same 2,466 prompts on 460 slices across six test volumes.")
    section = st.radio("Replay view", ("Overlap", "Failure ranking", "Mask fallback", "Encoder timing"),
                       horizontal=True, key="replay_section")
    if section == "Overlap":
        variants = table("variant_summary.csv").set_index("variant").reindex(VARIANTS).reset_index()
        variants["variant"] = variants.variant.map(VARIANTS)
        view = variants.rename(columns={"variant": "Variant", "pair_weighted_dice": "Pair-weighted Dice",
                                          "prompt_weighted_dice": "Prompt-weighted Dice", "failures": "Dice < 0.5"})
        st.dataframe(view, hide_index=True, width="stretch", column_config={
            field: st.column_config.NumberColumn(format="%.5f") for field in ("Pair-weighted Dice", "Prompt-weighted Dice")
        })
        st.caption("Reference-selected variants use the target mask to choose a candidate. Their Dice gives a ceiling for the candidates evaluated.")
        st.write("Fixing Gaussian decoder SD at 0.05 changed pair-weighted Dice from **0.7982 to 0.7985** in this replay. "
                 "The same fitted checkpoint was used with rank selection enabled.")
        csv_button("Download replay overlap table", view, "balora_replay_overlap.csv")
        with st.expander("Paired Dice differences by volume"):
            paired = table("paired_dice.csv")
            labels = paired.comparison.str.split("_minus_", n=1, expand=True)
            paired = paired.assign(
                comparison=labels[0].map(VARIANTS) + " minus " + labels[1].map(VARIANTS),
                dice_points=100 * paired.difference,
            )
            matched = st.selectbox("Replay Dice comparison", paired.comparison.unique())
            delta_interval_viz(paired[paired.comparison == matched], "dice_points", "Dice change (percentage points)")
    elif section == "Failure ranking":
        st.caption("Failure means Dice below 0.5. AUROC is calculated within each volume and then averaged across the six volumes.")
        compare = st.selectbox("Failure-ranking comparison", (
            "Dropout MI added to organ, area and entropy",
            "Dropout MI alone versus entropy alone",
            "Gaussian MI added to organ, area and entropy",
            "Fixed-SD MI versus learned-SD MI",
        ), key="replay_comparison")
        if compare == "Dropout MI added to organ, area and entropy":
            df = table("dropout_increment_by_volume.csv")
            baseline, extended = "organ_area_entropy_auroc", "organ_area_entropy_dropout_mi_auroc"
            st.caption("Outcome labels are fixed to the deterministic single-output masks. The risk model holds out each volume in turn.")
        elif compare == "Gaussian MI added to organ, area and entropy":
            df = table("failure_auroc_by_volume.csv").copy()
            counts = table("failure_counts_by_volume.csv")
            df = df.merge(counts.loc[counts.method.eq("balora"), ["volume", "failures"]],
                          on="volume", validate="one_to_one")
            baseline, extended = "organ_area_entropy", "organ_area_entropy_mi"
            df["increment"] = df[extended] - df[baseline]
            st.caption("Outcome labels are fixed to the Gaussian learned-SD masks. This comparison uses the archived Gaussian risk-model results.")
        else:
            alt_key = "decoder_mc_dropout" if compare.startswith("Dropout") else "gaussian_fixed_sd"
            df = table("fixed_outcome_auroc.csv").query("alternative == @alt_key").copy()
            baseline, extended = "reference_auroc", "alternative_auroc"
            df["increment"] = df[extended] - df[baseline]
            labels = "deterministic single-output masks" if alt_key == "decoder_mc_dropout" else "Gaussian learned-SD masks"
            st.caption(f"Outcome labels are fixed to the {labels}. Both uncertainty scores use those labels.")
        avg = df.increment.mean()
        # The two-sided 95% t critical value for the six observed volumes.
        half = 2.5705818366147395 * df.increment.sem()
        cols = st.columns(3)
        cols[0].metric("Baseline AUROC", f"{df[baseline].mean():.4f}")
        cols[1].metric("Replay AUROC", f"{df[extended].mean():.4f}")
        cols[2].metric("Paired change", f"{avg:+.4f}")
        st.caption(f"{df.failures.sum()} fixed failures. Descriptive 95% t interval [{avg-half:+.4f}, {avg+half:+.4f}].")
        delta_interval_viz(df, "increment", "AUROC change")
        csv_button("Download volume-level AUROC", df, "balora_replay_auroc.csv")
    elif section == "Mask fallback":
        raw = table("fallback_counts.csv")
        volume = st.selectbox("Volume", ["All volumes", *sorted(raw.volume.unique())], key="fallback_volume")
        if volume != "All volumes":
            raw = raw[raw.volume == volume]
        fields = ["substitutions", "single_output_calls", "targets_with_any_substitution", "targets"]
        counts = raw.groupby(["arm", "phase"], as_index=False)[fields].sum()
        counts["call_rate"] = counts.substitutions / counts.single_output_calls
        counts["target_rate"] = counts.targets_with_any_substitution / counts.targets
        counts["arm"] = counts.arm.map({
            "frozen": "Frozen SAM2", "deterministic": "Deterministic LoRA",
            "gaussian_learned": "Gaussian learned SD", "gaussian_fixed_sd": "Gaussian fixed SD",
        })
        frame = counts.rename(columns={"arm": "Method", "phase": "Phase", "substitutions": "Substitutions",
                                      "single_output_calls": "Calls", "targets_with_any_substitution": "Targets with fallback",
                                      "targets": "Targets", "call_rate": "Call rate", "target_rate": "Target rate"})
        defs = {f: st.column_config.NumberColumn(format="percent") for f in ["Call rate", "Target rate"]}
        st.dataframe(frame, width="stretch", hide_index=True, column_config=defs)
        st.caption("Call rate counts substituted decoder calls. Target rate counts prompts with at least one substitution.")
        csv_button("Download fallback counts", frame, "balora_replay_fallback.csv")
    else:
        df = table("encoder_timing.csv")
        for col, label, field in zip(st.columns(2), list(METHODS.values())[1:], ["deterministic", "balora"]):
            col.metric(label, f"{df[field].mean():.2f} ms")
        st.caption("Mean of 12 slice-level medians on Tesla T4. Three timed repeats per slice followed warm-up.")
        st.dataframe(df.rename(columns={"deterministic": "Deterministic (ms)", "balora": "Gaussian (ms)",
                                        "difference_ms": "Deterministic − Gaussian (ms)"}),
                     hide_index=True, width="stretch")
        csv_button("Download timing measurements", df, "balora_replay_timing.csv")


def protocol():
    st.header("Study protocol")
    st.write("The study tests how two low-rank adaptations affect point-prompted abdominal CT segmentation on one BTCV split.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Training volumes", "19")
    c2.metric("Validation volumes", "5")
    c3.metric("Test volumes", "6")
    st.write("Each target is a separate binary mask on a 1024 × 1024 slice. A positive point was sampled inside the reference "
             "organ mask and reused across methods. The decoder averaged four final draws for Gaussian LoRA.")
    st.image(str(DATA / "method.png"), width=820)
    st.subheader("Validation and checkpoint choice")
    history = report("validation_histories.json")
    frame = pd.concat([pd.DataFrame(h).assign(method=m) for m, h in history.items()], ignore_index=True)
    frame = method_column(frame)
    history_plot = alt.Chart(frame).mark_line(point=True).encode(
        x=alt.X("epoch:O", title="Completed epoch"),
        y=alt.Y("val_dice:Q", title="Validation Dice", scale=alt.Scale(zero=False)),
        color=method_color(),
        tooltip=["Method:N", alt.Tooltip("epoch:O", title="Epoch"), alt.Tooltip("val_dice:Q", format=".4f")],
    )
    st.altair_chart(history_plot.properties(height=280), width="stretch")
    st.caption("Both fits ran 13 epochs. Both validation-selected checkpoints were epoch 7 in the completed F1 matched archive.")
    st.markdown("**Scope.** The study used one split and one training seed, with reference-assisted prompts. "
                "The case explorer displays saved outputs. Use Upload CT for newly uploaded images.")
    st.markdown("Aritrik Ghosh and Sourav Saha · Swami Vivekananda University, Kolkata, India")
    st.markdown("[BTCV challenge data](https://www.synapse.org/Synapse:syn3193805/wiki/89480) · "
                "[SAM2 source](https://github.com/facebookresearch/sam2)")


PAGES = {
    "Benchmark": benchmark,
    "CT cases": cases,
    "Upload CT": upload_page,
    "Inference replay": replay,
    "Study protocol": protocol,
}
selected_page = st.radio("Research dashboard section", list(PAGES), horizontal=True,
                         key="page", label_visibility="collapsed")
PAGES[selected_page]()
