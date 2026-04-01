
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="RC Cantilever Shoring Wall Designer", layout="wide")


# ---------------------------------
# Data
# ---------------------------------
BAR_AREA = {
    "#3": 0.11, "#4": 0.20, "#5": 0.31, "#6": 0.44, "#7": 0.60,
    "#8": 0.79, "#9": 1.00, "#10": 1.27, "#11": 1.56, "#14": 2.25, "#18": 4.00
}
BAR_DIA = {
    "#3": 0.375, "#4": 0.500, "#5": 0.625, "#6": 0.750, "#7": 0.875,
    "#8": 1.000, "#9": 1.128, "#10": 1.270, "#11": 1.410, "#14": 1.693, "#18": 2.257
}


@dataclass
class Inputs:
    Hcant_ft: float
    ws_psf: float
    Pp_psf_per_ft: float
    Pa_pcf: float
    Pe_psf_per_ft: float
    gamma_b_pcf: float
    gamma_c_pcf: float
    fc_ksi: float
    fy_ksi: float
    D_in: float
    S_ft: float
    n_bars: int
    vert_bar: str
    tie_bar: str
    tie_spacing_in: float
    cover_in: float
    include_seismic: bool


def fnum(x, n=2):
    return f"{x:,.{n}f}"


# ---------------------------------
# Capacity helpers
# ---------------------------------
def phi_factor(et, ety):
    # ACI-style simplified interpolation
    if et <= ety:
        return 0.65
    if et >= ety + 0.003:
        return 0.90
    return 0.65 + 0.25 * (et - ety) / 0.003


def circular_segment_area(y_cut, r):
    ys = np.linspace(y_cut, r, 500)
    if len(ys) < 2:
        return 0.0
    widths = 2.0 * np.sqrt(np.clip(r**2 - ys**2, 0.0, None))
    return np.trapezoid(widths, ys)


def circular_segment_centroid(y_cut, r):
    ys = np.linspace(y_cut, r, 500)
    if len(ys) < 2:
        return 0.0
    widths = 2.0 * np.sqrt(np.clip(r**2 - ys**2, 0.0, None))
    area = np.trapezoid(widths, ys)
    if area <= 1e-12:
        return 0.0
    q = np.trapezoid(ys * widths, ys)
    return q / area


def build_interaction_curve(inp: Inputs):
    D = inp.D_in
    r = D / 2.0
    fc = inp.fc_ksi
    fy = inp.fy_ksi
    Es = 29000.0
    eps_cu = 0.003
    ety = fy / Es
    beta1 = 0.85 if fc <= 4.0 else max(0.65, 0.85 - 0.05 * (fc - 4.0))

    bar_area = BAR_AREA[inp.vert_bar]
    bar_dia = BAR_DIA[inp.vert_bar]
    rs = r - inp.cover_in - BAR_DIA[inp.tie_bar] - bar_dia / 2.0
    rs = max(rs, 0.1)
    ang = np.linspace(0, 2 * np.pi, inp.n_bars, endpoint=False)
    ybars = rs * np.sin(ang)

    rows = []
    for c in np.linspace(1.0, 2.5 * D, 160):
        a = beta1 * c
        y_cut = max(min(r - a, r), -r)
        Ac = circular_segment_area(y_cut, r)
        yc = circular_segment_centroid(y_cut, r)
        Cc = 0.85 * fc * Ac  # kips
        Mc = Cc * yc / 12.0  # ft-kips

        Pn = Cc
        Mn = Mc
        min_steel_strain = 1e9

        for y in ybars:
            depth_from_top = r - y
            eps = eps_cu * (1.0 - depth_from_top / c)
            fs = max(min(Es * eps, fy), -fy)
            Fs = fs * bar_area
            Pn += Fs
            Mn += Fs * y / 12.0
            min_steel_strain = min(min_steel_strain, eps)

        et = abs(min_steel_strain) if min_steel_strain < 0 else 0.0
        phi = phi_factor(et, ety)
        rows.append({
            "phiMn_ftk": phi * abs(Mn),
            "phiPn_k": phi * Pn,
            "phi": phi,
            "et": et,
            "c_in": c
        })

    df = pd.DataFrame(rows).sort_values("phiMn_ftk").reset_index(drop=True)
    return df


# ---------------------------------
# Engineering calculations
# ---------------------------------
def compute(inp: Inputs):
    H = inp.Hcant_ft
    S = inp.S_ft
    D_ft = inp.D_in / 12.0

    # Analysis per visible equations from the image
    Hb = 0.5 * S * inp.Pa_pcf * H**2 / 1000.0
    Hs = inp.ws_psf * S * inp.Pa_pcf * H / inp.gamma_b_pcf / 1000.0
    He = 0.5 * S * (inp.Pe_psf_per_ft if inp.include_seismic else 0.0) * H**2 / 1000.0
    P = (S * D_ft * inp.ws_psf + 0.25 * math.pi * inp.gamma_c_pcf * D_ft**2 * H) / 1000.0
    V = Hb + Hs + He
    M = (Hb / 3.0 + 2.0 * He / 3.0 + Hs / 2.0) * H

    Pu = 1.2 * P
    Vu = 1.6 * V
    Mu = 1.6 * M

    # Approximate embedment sizing using visible depth form from the image
    # A is estimated from earth pressure ratio so the depth scales logically with Pp and Pa.
    A = max(2.0, 2.0 * inp.Pa_pcf * H / inp.Pp_psf_per_ft)
    Hembed = 0.5 * A * (1.0 + math.sqrt(1.0 + 4.36 * H / A))
    Hembed = max(Hembed, 1.2 * H)

    # IBC pile limitations visible in the image
    fc_ok = inp.fc_ksi >= 4.0
    D_req_in = max((H + Hembed) * 12.0 / 30.0, 12.0)
    D_ok = inp.D_in >= D_req_in

    # Shear check in ACI style visible in the image
    Ao = math.pi * inp.D_in**2 / 4.0
    Av = 2.0 * BAR_AREA[inp.tie_bar]
    tie_dia = BAR_DIA[inp.tie_bar]
    vert_dia = BAR_DIA[inp.vert_bar]
    d_eff = inp.D_in - 2.0 * (tie_dia + vert_dia)
    d_eff = max(d_eff, 0.75 * inp.D_in)

    Vc = 2.0 * math.sqrt(inp.fc_ksi * 1000.0) * Ao / 1000.0
    Vs1 = d_eff * inp.fy_ksi * Av / inp.tie_spacing_in
    Vs2 = 8.0 * math.sqrt(inp.fc_ksi * 1000.0) * Ao / 1000.0
    Vs = min(Vs1, Vs2)
    phi_v = 0.75
    phiVn = phi_v * (Vc + Vs)

    s_max_in = 12.0
    s_ok = inp.tie_spacing_in <= s_max_in

    rho_s_req = 0.12 * (inp.fc_ksi * 1000.0) / (inp.fy_ksi * 1000.0)
    D_core = inp.D_in - 2.0 * inp.cover_in
    rho_s_prov = 4.0 * Av / (inp.tie_spacing_in * D_core)
    rho_ok = rho_s_prov >= rho_s_req

    # Axial-flexural interaction
    curve = build_interaction_curve(inp)
    i = int((curve["phiPn_k"] - Pu).abs().idxmin())
    phiMn_at_Pu = float(curve.loc[i, "phiMn_ftk"])
    flex_ok = phiMn_at_Pu >= Mu
    shear_ok = phiVn >= Vu

    return {
        "Hb": Hb, "Hs": Hs, "He": He, "P": P, "V": V, "M": M,
        "Pu": Pu, "Vu": Vu, "Mu": Mu,
        "Hembed": Hembed,
        "fc_ok": fc_ok, "D_req_in": D_req_in, "D_ok": D_ok,
        "Ao": Ao, "Av": Av, "d_eff": d_eff, "Vc": Vc, "Vs": Vs,
        "phiVn": phiVn, "s_max_in": s_max_in, "s_ok": s_ok,
        "rho_s_req": rho_s_req, "rho_s_prov": rho_s_prov, "rho_ok": rho_ok,
        "curve": curve, "phiMn_at_Pu": phiMn_at_Pu, "flex_ok": flex_ok, "shear_ok": shear_ok
    }


# ---------------------------------
# Plotting
# ---------------------------------
def plot_interaction(curve, Pu, Mu):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = curve["phiMn_ftk"].values
    y = curve["phiPn_k"].values

    p33 = np.percentile(y, 33)
    p66 = np.percentile(y, 66)

    m1 = y < p33
    m2 = (y >= p33) & (y < p66)
    m3 = y >= p66

    ax.plot(x[m1], y[m1], lw=2, label="Tension controlled")
    ax.plot(x[m2], y[m2], lw=2, label="Transition")
    ax.plot(x[m3], y[m3], "--", lw=2, label="Compression controlled")
    ax.plot([Mu], [Pu], "ks", ms=7, label="Demand point")
    ax.set_xlabel("ϕ Mn (ft-k)")
    ax.set_ylabel("ϕ Pn (k)")
    ax.grid(True, alpha=0.35)
    ax.legend()
    return fig


def plot_wall(inp: Inputs, out):
    fig, ax = plt.subplots(figsize=(6, 6))
    H = inp.Hcant_ft
    He = out["Hembed"]

    ax.plot([0, 0], [-He, H], color="black", lw=8, solid_capstyle="butt")
    ax.plot([-6, 0], [H, H], color="black", lw=2)
    ax.fill_between([-6, 0], [H + 2, H + 2], [H, H], color="#f2f2f2", edgecolor="black")

    for x in np.linspace(-5.5, -0.5, 10):
        ax.arrow(x, H + 1.5, 0, -1.0, head_width=0.12, head_length=0.18, fc="black", ec="black")

    ax.annotate("", xy=(2.2, H), xytext=(2.2, 0), arrowprops=dict(arrowstyle="<->"))
    ax.text(2.4, H / 2, "Hcant", rotation=90, va="center")
    ax.annotate("", xy=(2.2, 0), xytext=(2.2, -He), arrowprops=dict(arrowstyle="<->"))
    ax.text(2.4, -He / 2, "Hembed", rotation=90, va="center")

    ax.text(-3.6, H + 1.6, "ws (psf)")

    ax.axis("off")
    ax.set_aspect("equal")
    return fig


# ---------------------------------
# UI
# ---------------------------------
st.title("RC Cantilever Shoring Wall Design")
st.caption("Design-oriented Streamlit prototype based on the parameters and code checks visible in your sample image. This is not a sealed design document.")

with st.sidebar:
    st.header("Project Inputs")
    Hcant_ft = st.number_input("Height of cantilever, Hcant (ft)", min_value=1.0, value=10.0, step=0.5)
    ws_psf = st.number_input("Surcharge weight, ws (psf)", min_value=0.0, value=500.0, step=25.0)
    Pp_psf_per_ft = st.number_input("Allowable lateral soil-bearing pressure in embedment, Pp (psf/ft)", min_value=1.0, value=300.0, step=10.0)
    Pa_pcf = st.number_input("Lateral soil pressure, Pa (pcf)", min_value=1.0, value=35.0, step=1.0)
    include_seismic = st.checkbox("Include seismic ground shaking", value=True)
    Pe_psf_per_ft = st.number_input("Seismic pressure, Pe (psf/ft)", min_value=0.0, value=450.0, step=10.0, disabled=not include_seismic)
    gamma_b_pcf = st.number_input("Soil specific weight, γb (pcf)", min_value=50.0, value=110.0, step=5.0)
    gamma_c_pcf = st.number_input("Concrete unit weight, γc (pcf)", min_value=120.0, value=150.0, step=5.0)

    fc_ksi = st.number_input("Concrete strength, f'c (ksi)", min_value=2.5, value=4.0, step=0.5)
    fy_ksi = st.number_input("Rebar yield stress, fy (ksi)", min_value=40.0, value=60.0, step=5.0)

    D_in = st.number_input("Pile diameter, D (in)", min_value=12.0, value=44.0, step=1.0)
    S_ft = st.number_input("Pile spacing, S (ft)", min_value=1.0, value=4.03333, step=0.1)

    n_bars = st.number_input("No. of vertical bars", min_value=4, value=16, step=1)
    vert_bar = st.selectbox("Vertical rebar size", list(BAR_AREA.keys()), index=list(BAR_AREA.keys()).index("#11"))

    tie_bar = st.selectbox("Tie / spiral bar size", list(BAR_AREA.keys()), index=list(BAR_AREA.keys()).index("#6"))
    tie_spacing_in = st.number_input("Tie spacing (in)", min_value=1.0, value=8.0, step=1.0)
    cover_in = st.number_input("Clear cover (in)", min_value=1.5, value=3.0, step=0.5)

inp = Inputs(
    Hcant_ft=Hcant_ft, ws_psf=ws_psf, Pp_psf_per_ft=Pp_psf_per_ft, Pa_pcf=Pa_pcf,
    Pe_psf_per_ft=Pe_psf_per_ft, gamma_b_pcf=gamma_b_pcf, gamma_c_pcf=gamma_c_pcf,
    fc_ksi=fc_ksi, fy_ksi=fy_ksi, D_in=D_in, S_ft=S_ft, n_bars=n_bars,
    vert_bar=vert_bar, tie_bar=tie_bar, tie_spacing_in=tie_spacing_in,
    cover_in=cover_in, include_seismic=include_seismic
)

out = compute(inp)
overall_ok = out["fc_ok"] and out["D_ok"] and out["flex_ok"] and out["shear_ok"] and out["s_ok"] and out["rho_ok"]

# Summary
c1, c2, c3, c4 = st.columns([1.35, 1, 1, 1])
with c1:
    st.subheader("Design Summary")
    st.write(f"Hcant = **{fnum(inp.Hcant_ft)} ft**")
    st.write(f"ws = **{fnum(inp.ws_psf)} psf**")
    st.write(f"Pp = **{fnum(inp.Pp_psf_per_ft)} psf/ft**")
    st.write(f"Pa = **{fnum(inp.Pa_pcf)} pcf**")
    st.write(f"Pe = **{fnum(inp.Pe_psf_per_ft if inp.include_seismic else 0)} psf/ft**")
    st.write(f"D = **{fnum(inp.D_in)} in**")
    st.write(f"S = **{fnum(inp.S_ft,3)} ft**")
    st.write(f"Vertical bars = **{inp.n_bars} {inp.vert_bar}**")
    st.write(f"Ties = **{inp.tie_bar} @ {fnum(inp.tie_spacing_in)} in**")
with c2:
    st.metric("Pu", f"{fnum(out['Pu'])} kips")
    st.metric("Vu", f"{fnum(out['Vu'])} kips")
    st.metric("Mu", f"{fnum(out['Mu'])} ft-kips")
with c3:
    st.metric("ϕMn @ Pu", f"{fnum(out['phiMn_at_Pu'])} ft-kips")
    st.metric("ϕVn", f"{fnum(out['phiVn'])} kips")
    st.metric("Hembed", f"{fnum(out['Hembed'])} ft")
with c4:
    st.markdown(f"### {'ADEQUATE' if overall_ok else 'CHECK DESIGN'}")
    st.write(f"IBC f'c limit: **{'OK' if out['fc_ok'] else 'NG'}**")
    st.write(f"IBC D limit: **{'OK' if out['D_ok'] else 'NG'}**")
    st.write(f"Flexure: **{'OK' if out['flex_ok'] else 'NG'}**")
    st.write(f"Shear: **{'OK' if out['shear_ok'] else 'NG'}**")

tab1, tab2, tab3, tab4 = st.tabs(["Wall", "Analysis", "Code Checks", "Capacity"])

with tab1:
    a, b = st.columns([1.2, 1])
    with a:
        st.pyplot(plot_wall(inp, out), use_container_width=True)
    with b:
        st.markdown("### Notes")
        st.write("- The app focuses on the engineering parameters and code checks shown in your image.")
        st.write("- It does **not** try to replicate the handwritten style or static sheet appearance.")
        st.write("- The retained formulas were aligned to the visible expressions in the screenshots.")
        st.write("- Flexural-axial strength is based on a simplified circular RC strain-compatibility routine.")

with tab2:
    st.subheader("Determine Pile Section Forces at Cantilever Bottom")
    df = pd.DataFrame({
        "Parameter": ["Hb", "Hs", "He", "P", "V", "M", "Pu", "Vu", "Mu"],
        "Value": [out["Hb"], out["Hs"], out["He"], out["P"], out["V"], out["M"], out["Pu"], out["Vu"], out["Mu"]],
        "Unit": ["kips", "kips", "kips", "kips", "kips", "ft-kips", "kips", "kips", "ft-kips"]
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### Equations implemented")
    st.latex(r"H_b = 0.5 \, S \, P_a \, H_{cant}^{2}")
    st.latex(r"H_s = \frac{w_s \, S \, P_a \, H_{cant}}{\gamma_b}")
    st.latex(r"H_e = 0.5 \, S \, P_e \, H_{cant}^{2}")
    st.latex(r"P = S D w_s + 0.25 \pi \gamma_c D^2 H_{cant}")
    st.latex(r"V = H_b + H_s + H_e")
    st.latex(r"M = \left(\frac{H_b}{3} + \frac{2H_e}{3} + \frac{H_s}{2}\right)H_{cant}")
    st.latex(r"P_u = 1.2P,\quad V_u = 1.6V,\quad M_u = 1.6M")
    st.caption("For the axial load equation, D is internally converted to feet for unit consistency.")

with tab3:
    st.subheader("Code-Oriented Checks from the Image")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Pile limitations")
        st.write(f"f'c = {fnum(inp.fc_ksi)} ksi {'>' if out['fc_ok'] else '<'} 4 ksi  → **{'Satisfactory' if out['fc_ok'] else 'Not satisfactory'}**")
        st.write(f"D = {fnum(inp.D_in)} in {'>' if out['D_ok'] else '<'} {fnum(out['D_req_in'])} in  → **{'Satisfactory' if out['D_ok'] else 'Not satisfactory'}**")
        st.caption("Checks aligned to the visible IBC threshold style shown in the sample image.")

        st.markdown("#### Shear")
        st.write(f"Ao = {fnum(out['Ao'])} in²")
        st.write(f"Av = {fnum(out['Av'])} in²")
        st.write(f"Vc = {fnum(out['Vc'])} kips")
        st.write(f"Vs = {fnum(out['Vs'])} kips")
        st.write(f"ϕVn = {fnum(out['phiVn'])} kips {'>' if out['shear_ok'] else '<'} Vu = {fnum(out['Vu'])} kips")
    with c2:
        st.markdown("#### Transverse reinforcement")
        st.write(f"smax = {fnum(out['s_max_in'])} in")
        st.write(f"sprov = {fnum(inp.tie_spacing_in)} in  → **{'Satisfactory' if out['s_ok'] else 'Not satisfactory'}**")
        st.write(f"ρs,req = {fnum(out['rho_s_req'],3)}")
        st.write(f"ρs,prov = {fnum(out['rho_s_prov'],3)}  → **{'Satisfactory' if out['rho_ok'] else 'Not satisfactory'}**")

        st.markdown("#### Embedment")
        st.write(f"Estimated Hembed = **{fnum(out['Hembed'])} ft**")
        st.caption("Embedment is an engineering estimate tied to the visible depth-form expression from the sample, and should still be checked against your exact office methodology.")

with tab4:
    a, b = st.columns([1.3, 1])
    with a:
        st.pyplot(plot_interaction(out["curve"], out["Pu"], out["Mu"]), use_container_width=True)
    with b:
        st.subheader("Flexural & axial check")
        st.write(f"ϕMn at Pu = **{fnum(out['phiMn_at_Pu'])} ft-kips**")
        st.write(f"Mu = **{fnum(out['Mu'])} ft-kips**")
        st.write(f"Result = **{'Satisfactory' if out['flex_ok'] else 'Not satisfactory'}**")

summary_df = pd.DataFrame({
    "parameter": [
        "Hcant_ft", "ws_psf", "Pp_psf_per_ft", "Pa_pcf", "Pe_psf_per_ft",
        "gamma_b_pcf", "gamma_c_pcf", "fc_ksi", "fy_ksi", "D_in", "S_ft",
        "Pu_kips", "Vu_kips", "Mu_ftkips", "phiMn_at_Pu_ftkips", "phiVn_kips",
        "Hembed_ft", "overall_ok"
    ],
    "value": [
        inp.Hcant_ft, inp.ws_psf, inp.Pp_psf_per_ft, inp.Pa_pcf,
        inp.Pe_psf_per_ft if inp.include_seismic else 0.0, inp.gamma_b_pcf, inp.gamma_c_pcf,
        inp.fc_ksi, inp.fy_ksi, inp.D_in, inp.S_ft, out["Pu"], out["Vu"], out["Mu"],
        out["phiMn_at_Pu"], out["phiVn"], out["Hembed"], overall_ok
    ]
})
st.download_button("Download summary CSV", summary_df.to_csv(index=False).encode("utf-8"), "rc_cantilever_shoring_summary.csv", "text/csv")
