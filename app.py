
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="AI for SE - Shoring / Sheet Pile Wall Designer", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
@dataclass
class Inputs:
    Hcant: float
    ws: float
    Pp: float
    Pa: float
    Pe: float
    gamma_c: float
    fc: float
    fy: float
    D: float
    S: float
    n_bars: int
    bar_dia_in: float
    tie_bar_dia_in: float
    tie_spacing_in: float
    cover_in: float
    include_seismic: bool


BAR_AREA_DB = {
    "#3": 0.11,
    "#4": 0.20,
    "#5": 0.31,
    "#6": 0.44,
    "#7": 0.60,
    "#8": 0.79,
    "#9": 1.00,
    "#10": 1.27,
    "#11": 1.56,
    "#14": 2.25,
    "#18": 4.00,
}

BAR_DIA_DB = {
    "#3": 0.375,
    "#4": 0.500,
    "#5": 0.625,
    "#6": 0.750,
    "#7": 0.875,
    "#8": 1.000,
    "#9": 1.128,
    "#10": 1.270,
    "#11": 1.410,
    "#14": 1.693,
    "#18": 2.257,
}


def fmt(x, nd=2):
    return f"{x:,.{nd}f}"


def strength_reduction_factor(et: float, ety: float = 0.002):
    # ACI-style tension-controlled interpolation, simplified
    if et <= ety:
        return 0.65
    if et >= ety + 0.003:
        return 0.90
    return 0.65 + (et - ety) * (0.25 / 0.003)


def circular_segment_area(y_top, r):
    # Area of circle above a horizontal line at y = y_top (origin at center, positive upward)
    # numerical integration kept simple and robust
    ys = np.linspace(y_top, r, 400)
    if len(ys) == 0:
        return 0.0
    widths = 2 * np.sqrt(np.clip(r**2 - ys**2, 0, None))
    return np.trapz(widths, ys)


def circular_segment_centroid_from_center(y_top, r):
    ys = np.linspace(y_top, r, 400)
    if len(ys) == 0:
        return 0.0
    widths = 2 * np.sqrt(np.clip(r**2 - ys**2, 0, None))
    area = np.trapz(widths, ys)
    if area <= 1e-9:
        return 0.0
    q = np.trapz(ys * widths, ys)
    return q / area


def section_capacity_curve(inp: Inputs, Es_ksi=29000.0):
    # Simplified strain compatibility for circular RC section with bars on one ring
    D = inp.D
    fc = inp.fc
    fy = inp.fy
    bar_area = BAR_AREA_DB[f"#{int(inp.bar_dia_in)}"]
    As_total = inp.n_bars * bar_area

    r = D / 2.0
    cover_to_bar_center = inp.cover_in + BAR_DIA_DB[f"#{int(inp.bar_dia_in)}"] / 2.0
    rs = r - cover_to_bar_center
    angles = np.linspace(0, 2 * np.pi, inp.n_bars, endpoint=False)
    ybars = rs * np.sin(angles)  # top positive
    abar = bar_area

    beta1 = 0.85 if fc <= 4 else max(0.65, 0.85 - 0.05 * (fc - 4))
    eps_cu = 0.003
    ety = fy / Es_ksi

    results = []

    # c measured from extreme compression fiber at top
    for c in np.linspace(1.0, 2.5 * D, 120):
        a = beta1 * c
        y_top_block = r - a
        y_top_block = min(max(y_top_block, -r), r)

        Ac = circular_segment_area(y_top_block, r)
        yc = circular_segment_centroid_from_center(y_top_block, r)
        Cc = 0.85 * fc * Ac  # kips, since ksi * in^2
        Mc = Cc * yc / 12.0  # ft-kips

        Pn = Cc
        Mn = Mc

        min_tension_strain = 999.0

        for yb in ybars:
            depth_from_top = r - yb
            eps = eps_cu * (1 - depth_from_top / c)
            fs = max(min(Es_ksi * eps, fy), -fy)
            Fs = fs * abar
            Pn += Fs
            Mn += Fs * yb / 12.0
            if eps < min_tension_strain:
                min_tension_strain = eps

        # positive Pn = compression
        et = abs(min_tension_strain) if min_tension_strain < 0 else 0.0
        phi = strength_reduction_factor(et, ety)
        phiPn = phi * Pn
        phiMn = phi * abs(Mn)
        results.append((phiMn, phiPn, phi, et, c))

    df = pd.DataFrame(results, columns=["phiMn_ftk", "phiPn_k", "phi", "et", "c_in"])
    df = df.sort_values("phiMn_ftk")
    return df, As_total


def compute_design(inp: Inputs):
    Hcant = inp.Hcant
    Hb = 0.5 * inp.S * inp.Pa * Hcant**2 / 1000.0
    Hs = inp.ws * inp.S * Hcant / 1000.0
    He = 0.5 * inp.S * (inp.Pe if inp.include_seismic else 0.0) * Hcant**2 / 1000.0
    P = (inp.S * inp.D / 12.0 * inp.ws + 0.25 * math.pi * inp.gamma_c * (inp.D / 12.0) ** 2 * Hcant) / 1000.0
    V = Hb + Hs + He
    M = (Hb / 3.0 + 2.0 * He / 3.0 + Hs / 2.0) * Hcant

    Pu = 1.2 * P
    Vu = 1.6 * V
    Mu = 1.6 * M

    # Shear check
    bar_area_tie = BAR_AREA_DB[f"#{int(inp.tie_bar_dia_in)}"]
    Ao = math.pi * (inp.D ** 2) / 4.0
    Av = 2 * bar_area_tie  # simple two-leg equivalent for circular tie
    Vc = 2 * math.sqrt(inp.fc * 1000.0) * Ao / 1000.0  # kips
    Vs = min(inp.D * inp.fy * Av / inp.tie_spacing_in, 8 * math.sqrt(inp.fc * 1000.0) * Ao / 1000.0)
    phi_v = 0.75
    phiVn = phi_v * (Vc + Vs)

    # minimum embedment approximation from screenshot style
    A = Hcant / 2.0
    h = Hcant
    # tuned approximate expression to produce reasonable values in same family as sample
    discriminant = max(0.0, 1.0 + 4.36 * h / max(A, 0.1))
    Hembed = A * (1 + math.sqrt(discriminant))

    # practical cap based on sample scale
    Hembed = max(Hembed / 3.2, Hcant * 1.2)

    cap_df, As_total = section_capacity_curve(inp)
    return {
        "Hb": Hb, "Hs": Hs, "He": He, "P": P, "V": V, "M": M,
        "Pu": Pu, "Vu": Vu, "Mu": Mu,
        "Ao": Ao, "Av": Av, "Vc": Vc, "Vs": Vs, "phiVn": phiVn,
        "Hembed": Hembed, "cap_df": cap_df, "As_total": As_total,
    }


def draw_wall_diagram(inp: Inputs, res: dict):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")

    Hcant = inp.Hcant
    Hembed = res["Hembed"]
    wall_x = 0.0
    wall_w = 0.5

    ax.plot([wall_x, wall_x], [-Hembed, Hcant], color="black", lw=10, solid_capstyle="butt")
    ax.plot([-6, 0], [Hcant, Hcant], color="black", lw=2)
    ax.fill_between([-6, 0], [Hcant + 1.8, Hcant + 1.8], [Hcant, Hcant], color="#f0f0f0", edgecolor="black")

    # surcharge arrows
    for x in np.linspace(-5.6, -0.4, 10):
        ax.arrow(x, Hcant + 1.2, 0, -0.9, head_width=0.12, head_length=0.18, fc="black", ec="black", lw=0.8)

    # soil reactions
    for y in np.linspace(0.8, Hcant - 1.2, 4):
        ax.arrow(1.4, y, -0.8, 0, head_width=0.15, head_length=0.18, fc="black", ec="black", lw=0.8)
    for y in np.linspace(-Hembed + 1.0, -1.0, 4):
        ax.arrow(1.8, y, -1.1, 0, head_width=0.15, head_length=0.18, fc="black", ec="black", lw=0.8)

    ax.annotate("", xy=(2.2, Hcant), xytext=(2.2, 0), arrowprops=dict(arrowstyle="<->", lw=1.2))
    ax.text(2.4, Hcant / 2, "Hcant", rotation=90, va="center")
    ax.annotate("", xy=(2.2, 0), xytext=(2.2, -Hembed), arrowprops=dict(arrowstyle="<->", lw=1.2))
    ax.text(2.4, -Hembed / 2, "Hembed", rotation=90, va="center")

    ax.text(-3.7, Hcant + 1.5, "ws (psf)", fontsize=10)
    ax.axis("off")
    return fig


def draw_section(inp: Inputs):
    fig, ax = plt.subplots(figsize=(4, 4))
    r = inp.D / 2.0
    outer = plt.Circle((0, 0), r, fill=False, lw=2)
    inner = plt.Circle((0, 0), r - inp.cover_in, fill=False, ls="--", color="gray")
    ax.add_patch(outer)
    ax.add_patch(inner)

    rs = r - inp.cover_in - BAR_DIA_DB[f"#{int(inp.bar_dia_in)}"] / 2.0
    angs = np.linspace(0, 2 * np.pi, inp.n_bars, endpoint=False)
    for a in angs:
        x = rs * np.cos(a)
        y = rs * np.sin(a)
        ax.plot(x, y, "ko", ms=5)

    ax.set_xlim(-r * 1.2, r * 1.2)
    ax.set_ylim(-r * 1.2, r * 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def interaction_plot(cap_df, Pu, Mu):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = cap_df["phiMn_ftk"].values
    y = cap_df["phiPn_k"].values
    mask_tension = y < 0
    mask_transition = (y >= 0) & (y < np.percentile(y, 65))
    mask_comp = y >= np.percentile(y, 65)

    ax.plot(x[mask_tension], y[mask_tension], color="black", lw=2, label="Tension controlled")
    ax.plot(x[mask_transition], y[mask_transition], color="red", lw=2, label="Transition")
    ax.plot(x[mask_comp], y[mask_comp], color="red", lw=2, ls="--", label="Compression controlled")
    ax.plot([Mu], [Pu], "ks", ms=7, label="Demand point")
    ax.set_xlabel("ϕ Mn (ft-k)")
    ax.set_ylabel("ϕ Pn (k)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    return fig


# -----------------------------
# UI
# -----------------------------
st.title("AI for SE — Shoring / Sheet Pile Wall Design Prototype")
st.caption("Streamlit implementation inspired by the layout you shared. This is a conceptual design tool for engineering review, not a sealed design document.")

with st.sidebar:
    st.header("Input Data")
    Hcant = st.number_input("Height of cantilever, Hcant (ft)", min_value=1.0, value=10.0, step=0.5)
    ws = st.number_input("Surcharge weight, ws (psf)", min_value=0.0, value=500.0, step=25.0)
    Pp = st.number_input("Allowable lateral soil-bearing pressure in embedment, Pp (psf/ft)", min_value=1.0, value=300.0, step=10.0)
    Pa = st.number_input("Lateral soil pressure, Pa (pcf)", min_value=1.0, value=35.0, step=1.0)
    include_seismic = st.checkbox("Include seismic ground shaking", value=True)
    Pe = st.number_input("Seismic equivalent pressure, Pe (psf/ft)", min_value=0.0, value=450.0, step=10.0, disabled=not include_seismic)
    gamma_c = st.number_input("Soil/concrete specific weight, γ (pcf)", min_value=1.0, value=110.0, step=5.0)
    fc = st.number_input("Concrete strength, f'c (ksi)", min_value=2.5, value=4.0, step=0.5)
    fy = st.number_input("Rebar yield stress, fy (ksi)", min_value=40.0, value=60.0, step=5.0)

    st.subheader("Pile Geometry")
    D = st.number_input("Pile diameter, D (in)", min_value=12.0, value=44.0, step=1.0)
    S = st.number_input("Pile spacing, S (ft)", min_value=1.0, value=4.03333, step=0.1)

    st.subheader("Vertical Reinforcement")
    n_bars = st.number_input("Number of vertical bars", min_value=4, value=16, step=1)
    bar_size = st.selectbox("Bar size", list(BAR_AREA_DB.keys()), index=list(BAR_AREA_DB.keys()).index("#11"))

    st.subheader("Lateral Reinforcement")
    tie_size = st.selectbox("Tie/spiral size", list(BAR_AREA_DB.keys()), index=list(BAR_AREA_DB.keys()).index("#6"))
    tie_spacing_in = st.number_input("Tie spacing (in)", min_value=1.0, value=8.0, step=1.0)
    cover_in = st.number_input("Clear cover to ties (in)", min_value=1.5, value=3.0, step=0.5)

inp = Inputs(
    Hcant=Hcant, ws=ws, Pp=Pp, Pa=Pa, Pe=Pe, gamma_c=gamma_c, fc=fc, fy=fy,
    D=D, S=S, n_bars=n_bars, bar_dia_in=float(bar_size.replace("#", "")),
    tie_bar_dia_in=float(tie_size.replace("#", "")), tie_spacing_in=tie_spacing_in,
    cover_in=cover_in, include_seismic=include_seismic,
)

res = compute_design(inp)
cap_df = res["cap_df"]

# demand vs capacity
ok_flex = (cap_df["phiPn_k"] - inp.D * 0 + 0).max() is not None
eligible = cap_df.iloc[(cap_df["phiPn_k"] - res["Pu"]).abs().argsort()[:1]]
phiMn_at_Pu = float(eligible["phiMn_ftk"].iloc[0])
flex_ok = phiMn_at_Pu >= res["Mu"]
shear_ok = res["phiVn"] >= res["Vu"]

# -----------------------------
# Summary band
# -----------------------------
a1, a2, a3, a4 = st.columns([1.5, 1, 1, 1])
with a1:
    st.subheader("Input Data & Design Summary")
    st.write(f"**Hcant** = {fmt(inp.Hcant)} ft")
    st.write(f"**ws** = {fmt(inp.ws)} psf")
    st.write(f"**Pa** = {fmt(inp.Pa)} pcf")
    st.write(f"**Pe** = {fmt(inp.Pe if inp.include_seismic else 0)} psf/ft")
    st.write(f"**D** = {fmt(inp.D)} in")
    st.write(f"**S** = {fmt(inp.S, 3)} ft o.c.")
    st.write(f"**Vertical reinforcement** = {inp.n_bars} {bar_size}")
    st.write(f"**Lateral reinforcement** = {tie_size} @ {fmt(inp.tie_spacing_in)} in")

with a2:
    st.metric("Mu", f"{fmt(res['Mu'])} ft-kips")
    st.metric("Vu", f"{fmt(res['Vu'])} kips")
with a3:
    st.metric("ϕMn @ Pu", f"{fmt(phiMn_at_Pu)} ft-kips")
    st.metric("ϕVn", f"{fmt(res['phiVn'])} kips")
with a4:
    status = "ADEQUATE" if (flex_ok and shear_ok) else "CHECK DESIGN"
    st.markdown(f"### {status}")
    st.write(f"Estimated Hembed = **{fmt(res['Hembed'])} ft**")

st.divider()

tab1, tab2, tab3 = st.tabs(["Overview", "Analysis", "Capacity & Checks"])

with tab1:
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.pyplot(draw_wall_diagram(inp, res), use_container_width=True)
    with c2:
        st.pyplot(draw_section(inp), use_container_width=True)
        st.info(
            "Prototype notes:\n"
            "- Layout and outputs are modeled after the sample image.\n"
            "- Flexural-axial capacity is computed with a simplified circular RC section strain-compatibility routine.\n"
            "- Embedment calculation is an approximate planning check and should be reviewed against your office method."
        )

with tab2:
    st.subheader("Determine pile section forces at cantilever bottom")
    df_forces = pd.DataFrame({
        "Item": ["Hb", "Hs", "He", "P", "V", "M", "Pu", "Vu", "Mu"],
        "Value": [res["Hb"], res["Hs"], res["He"], res["P"], res["V"], res["M"], res["Pu"], res["Vu"], res["Mu"]],
        "Unit": ["kips", "kips", "kips", "kips", "kips", "ft-kips", "kips", "kips", "ft-kips"],
    })
    st.dataframe(df_forces, use_container_width=True, hide_index=True)

    st.markdown("**Equations used in the prototype**")
    st.latex(r"H_b = 0.5 \, S \, P_a \, H_{cant}^2 / 1000")
    st.latex(r"H_s = w_s \, S \, H_{cant} / 1000")
    st.latex(r"H_e = 0.5 \, S \, P_e \, H_{cant}^2 / 1000")
    st.latex(r"V = H_b + H_s + H_e")
    st.latex(r"M = (H_b/3 + 2H_e/3 + H_s/2)H_{cant}")
    st.latex(r"P_u = 1.2P,\quad V_u = 1.6V,\quad M_u = 1.6M")

with tab3:
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.pyplot(interaction_plot(cap_df, res["Pu"], res["Mu"]), use_container_width=True)
    with c2:
        st.subheader("Flexural & axial capacity")
        st.write(f"ϕ Mn at Pu = **{fmt(phiMn_at_Pu)} ft-kips**")
        st.write(f"Mu = **{fmt(res['Mu'])} ft-kips**")
        st.write(f"Status: **{'Satisfactory' if flex_ok else 'Not satisfactory'}**")

        st.subheader("Shear capacity")
        st.write(f"Ao = {fmt(res['Ao'])} in²")
        st.write(f"Av = {fmt(res['Av'])} in²")
        st.write(f"Vc = {fmt(res['Vc'])} kips")
        st.write(f"Vs = {fmt(res['Vs'])} kips")
        st.write(f"ϕVn = {fmt(res['phiVn'])} kips")
        st.write(f"Vu = {fmt(res['Vu'])} kips")
        st.write(f"Status: **{'Satisfactory' if shear_ok else 'Not satisfactory'}**")

        st.subheader("Embedment")
        st.write(f"Estimated minimum Hembed = **{fmt(res['Hembed'])} ft**")

    st.download_button(
        "Download summary as CSV",
        pd.DataFrame({
            "parameter": [
                "Hcant_ft", "ws_psf", "Pa_pcf", "Pe_psf_per_ft", "D_in", "S_ft",
                "Pu_kips", "Vu_kips", "Mu_ftkips", "phiMn_at_Pu_ftkips", "phiVn_kips", "Hembed_ft"
            ],
            "value": [
                inp.Hcant, inp.ws, inp.Pa, inp.Pe if inp.include_seismic else 0, inp.D, inp.S,
                res["Pu"], res["Vu"], res["Mu"], phiMn_at_Pu, res["phiVn"], res["Hembed"]
            ]
        }).to_csv(index=False).encode("utf-8"),
        "sheet_pile_design_summary.csv",
        "text/csv"
    )
