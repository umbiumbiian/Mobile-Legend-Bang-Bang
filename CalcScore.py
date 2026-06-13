import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Analisis Skor MLBB – Model Gabungan",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.hero {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a3e 50%, #0d1b2a 100%);
    border: 1px solid #2a2a5a; border-radius: 12px;
    padding: 2rem 2.5rem; margin-bottom: 1.5rem;
}
.hero h1 { color: #e8c84a; font-size: 1.9rem; font-weight: 700; margin: 0 0 0.3rem 0; }
.hero p  { color: #8888bb; font-size: 0.95rem; margin: 0; }
.step-card {
    background: #111128; border-left: 3px solid #e8c84a;
    border-radius: 8px; padding: 1rem 1.2rem; margin: 0.6rem 0;
}
.step-num { color: #e8c84a; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; }
.step-title { color: #ffffff; font-size: 1rem; font-weight: 600; margin: 0.2rem 0; }
.step-desc { color: #9999bb; font-size: 0.87rem; line-height: 1.5; }
.metric-box { background: #0d0d22; border: 1px solid #2a2a4a; border-radius: 8px; padding: 0.8rem 1rem; text-align: center; }
.metric-label { color: #7777aa; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { color: #e8c84a; font-size: 1.5rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.formula-box {
    background: #0a0a1f; border: 1px solid #3a3a6a; border-radius: 8px;
    padding: 1rem 1.4rem; font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem; color: #c8d8f0; line-height: 1.8;
}
.section-label { color: #e8c84a; font-size: 0.72rem; font-weight: 700; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 0.6rem; }
</style>
""", unsafe_allow_html=True)

# ── Data 4 Match (Ally + Enemy digabung) ─────────────────────────────────────
ALLY_DATA = [
    [7,5,5,20,42,25,52,7.8],[2,6,6,18,12,12,35,4.8],[0,4,12,11,5,19,52,6.9],
    [7,7,4,24,38,9,48,6.0],[10,5,5,27,4,36,65,9.3],[20,7,11,45,51,21,69,12.9],
    [2,5,26,8,4,23,62,9.7],[6,4,15,12,21,22,47,8.7],[6,9,14,16,4,24,44,6.6],
    [11,1,15,19,20,11,58,11.5],[13,6,4,35,69,18,63,9.6],[7,9,10,30,17,42,63,8.1],
    [0,5,16,5,0,15,59,7.3],[4,6,6,12,8,19,37,5.4],[3,2,9,18,6,7,44,7.8],
    [9,8,10,31,3,11,63,8.0],[5,5,10,21,97,31,50,8.2],[11,7,5,23,0,19,53,7.3],
    [3,10,8,18,0,12,37,4.3],[2,12,12,7,0,26,47,4.8],
]
ENEMY_DATA = [
    [1,3,11,14,45,9,44,7.1],[7,2,6,27,19,12,48,8.4],[6,6,5,23,36,32,41,6.7],
    [3,6,15,8,0,29,67,8.0],[10,6,8,29,0,18,67,8.5],[6,8,5,28,19,39,42,6.6],
    [11,7,2,19,15,15,50,7.1],[2,10,8,11,0,26,38,4.3],[2,12,9,18,66,11,42,4.2],
    [5,8,8,24,0,8,50,5.7],[1,6,20,16,0,39,75,9.7],[7,4,10,18,9,8,61,8.4],
    [10,5,3,18,69,19,46,8.0],[3,4,5,17,16,16,29,5.9],[7,8,9,30,6,15,57,7.0],
    [17,4,3,18,0,13,48,9.2],[16,6,6,36,35,24,52,10.5],[1,8,24,7,12,23,60,8.2],
    [7,8,3,14,53,22,24,5.3],[1,4,14,24,0,19,36,7.4],
]

MATCH_LABELS = (
    ["Match 1 - Ally"]*5 + ["Match 2 - Ally"]*5 + ["Match 3 - Ally"]*5 + ["Match 4 - Ally"]*5 +
    ["Match 1 - Enemy"]*5 + ["Match 2 - Enemy"]*5 + ["Match 3 - Enemy"]*5 + ["Match 4 - Enemy"]*5
)
TEAM_LABELS = ["Ally"]*20 + ["Enemy"]*20

COLS = ["K","D","A","Hero Dmg (%)","Turret Dmg (%)","Dmg Taken (%)","TFP (%)","Skor"]
LABELS = ["c (intercept)","K","D","A","Hero Dmg","Turret Dmg","Dmg Taken","TFP"]

# ── Gauss Elimination ─────────────────────────────────────────────────────────
def gauss_elimination(A, b):
    n = len(b)
    aug = np.hstack([A.astype(float), b.reshape(-1,1).astype(float)])
    steps = []
    for i in range(n):
        max_row = np.argmax(np.abs(aug[i:,i])) + i
        if max_row != i:
            aug[[i,max_row]] = aug[[max_row,i]]
            steps.append(f"Pivot: tukar baris {i+1} ↔ baris {max_row+1}")
        for j in range(i+1,n):
            if aug[i,i] != 0:
                factor = aug[j,i]/aug[i,i]
                aug[j] -= factor*aug[i]
                steps.append(f"R{j+1} ← R{j+1} − ({factor:.4f}) × R{i+1}")
    x = np.zeros(n)
    for i in range(n-1,-1,-1):
        x[i] = aug[i,-1]
        for j in range(i+1,n): x[i] -= aug[i,j]*x[j]
        x[i] /= aug[i,i]
    return x, aug, steps

def run_regression(data_array):
    X_raw = data_array[:,:7]
    y     = data_array[:,7]
    X     = np.hstack([np.ones((len(X_raw),1)), X_raw])
    AtA   = X.T @ X
    Atb   = X.T @ y
    coef, aug_final, steps = gauss_elimination(AtA, Atb)
    y_pred = X @ coef
    ss_res = np.sum((y-y_pred)**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot != 0 else 0
    return coef, y_pred, r2, AtA, Atb, steps, X, y

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label">📋 Data Gabungan (Ally + Enemy)</div>', unsafe_allow_html=True)
    df_all = pd.DataFrame(ALLY_DATA + ENEMY_DATA, columns=COLS)
    df_all.insert(0, "Team", TEAM_LABELS)
    df_all.insert(0, "Match", MATCH_LABELS)
    edited = st.data_editor(df_all, num_rows="dynamic", use_container_width=True, key="all_data")
    st.caption("Data ally dan enemy digabung. Model berlaku universal untuk semua pemain.")

data_arr = edited.drop(columns=["Match","Team"]).to_numpy().astype(float)
team_col = edited["Team"].tolist()

if len(data_arr) < 8:
    st.warning("Minimal 8 baris data.")
    st.stop()

coef, y_pred, r2, AtA, Atb, steps, X, y = run_regression(data_arr)
rmse = np.sqrt(np.mean((y-y_pred)**2))
max_err = np.max(np.abs(y-y_pred))

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>⚔️ Analisis Skor MLBB — Model Gabungan</h1>
  <p>Satu rumus universal untuk semua pemain · Eliminasi Gauss · 4 Match · 40 Data (20 Ally + 20 Enemy)</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📐 Step-by-Step",
    "🔢 Matriks & Eliminasi",
    "📊 Hasil & Koefisien",
    "📈 Visualisasi",
    "🧮 Kalkulator Skor",
])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — Step by Step
# ════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-label">Alur Metode</div>', unsafe_allow_html=True)
    steps_info = [
        ("STEP 01", "Asumsi: Satu Rumus Universal",
         "MLBB diasumsikan menggunakan satu rumus skor yang sama untuk semua pemain tanpa memandang tim. "
         "Validasi dari data: koefisien ally dan enemy terpisah sangat mirip, sehingga penggabungan dapat dibenarkan."),
        ("STEP 02", "Formulasi Model",
         "Skor = c + w₁·K + w₂·D + w₃·A + w₄·HeroDmg(%) + w₅·TurretDmg(%) + w₆·DmgTaken(%) + w₇·TFP(%). "
         "Tujuh variabel statistik dan satu intercept — total 8 parameter yang dicari."),
        ("STEP 03", "Susun Matriks A dan Vektor b",
         "40 baris data (20 ally + 20 enemy) masing-masing menjadi satu persamaan. "
         "Matriks A berukuran 40×8 (kolom pertama = 1 untuk intercept). Vektor b = 40 nilai Skor aktual."),
        ("STEP 04", "Bentuk Persamaan Normal: (AᵀA)x = Aᵀb",
         "Sistem overdetermined 40 persamaan × 8 variabel diselesaikan dengan Least Squares. "
         "Dikalikan Aᵀ dari kiri → sistem kuadrat 8×8."),
        ("STEP 05", "Augmented Matrix dan Eliminasi Gauss",
         "Gabungkan AᵀA dan Aᵀb jadi matriks 8×9. Lakukan forward elimination dengan partial pivoting "
         "→ matriks upper triangular → back substitution → 8 koefisien ditemukan."),
        ("STEP 06", "Verifikasi dengan R²",
         "Hitung y_pred = X·coef untuk semua 40 data. Bandingkan dengan y aktual. "
         "R² mendekati 1 berarti rumus yang ditemukan sangat mendekati cara MLBB menghitung skor aslinya."),
        ("STEP 07", "Interpretasi Bobot",
         "Setiap koefisien menunjukkan kontribusi statistik tersebut per satu satuan terhadap skor akhir. "
         "Koefisien positif = menaikkan skor, negatif = menurunkan skor."),
    ]
    for num, title, desc in steps_info:
        st.markdown(f"""
        <div class="step-card">
            <div class="step-num">{num}</div>
            <div class="step-title">{title}</div>
            <div class="step-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Rumus Skor yang Ditemukan</div>', unsafe_allow_html=True)
    terms = []
    for i, (l, c) in enumerate(zip(LABELS, coef)):
        if i == 0:
            terms.append(f"{c:.5f}")
        else:
            sign = "+" if c >= 0 else "−"
            terms.append(f"  {sign} {abs(c):.5f} · {l}")
    st.markdown(f'<div class="formula-box"><pre>Skor = {"chr(10)       ".join(terms)}</pre></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Urutan Pengaruh Statistik (dari terbesar)</div>', unsafe_allow_html=True)
    sorted_idx = np.argsort(np.abs(coef[1:]))[::-1]
    for rank, idx in enumerate(sorted_idx, 1):
        lbl = LABELS[idx+1]
        val = coef[idx+1]
        col_c = "#4ade80" if val >= 0 else "#f87171"
        arah = "menaikkan" if val >= 0 else "menurunkan"
        st.markdown(f"**{rank}.** `{lbl}` — koefisien <span style='color:{col_c}'>{val:+.5f}</span> → setiap +1 satuan **{arah}** skor sebesar {abs(val):.5f}", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# TAB 2 — Matriks & Eliminasi
# ════════════════════════════════════════════════════════════════════
with tab2:
    col_a2, col_b2 = st.columns(2)
    with col_a2:
        st.markdown('<div class="section-label">Matriks A (40×8)</div>', unsafe_allow_html=True)
        df_A = pd.DataFrame(X, columns=["1","K","D","A","HeroDmg","TurretDmg","DmgTaken","TFP"])
        st.dataframe(df_A.style.format("{:.0f}"), use_container_width=True, height=300)
    with col_b2:
        st.markdown('<div class="section-label">Vektor b (Skor)</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(y, columns=["Skor"]), use_container_width=True, height=300)

    st.markdown('<div class="section-label">Matriks AᵀA (8×8)</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(np.round(AtA,2), columns=LABELS, index=LABELS).style.format("{:.2f}"), use_container_width=True)

    st.markdown('<div class="section-label">Vektor Aᵀb</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(np.round(Atb,4), index=LABELS, columns=["Aᵀb"]).style.format("{:.4f}"), use_container_width=True)

    with st.expander("Tampilkan log operasi Eliminasi Gauss"):
        for i, s in enumerate(steps):
            st.markdown(f"`{i+1:03d}` {s}")

# ════════════════════════════════════════════════════════════════════
# TAB 3 — Hasil & Koefisien
# ════════════════════════════════════════════════════════════════════
with tab3:
    m1, m2, m3 = st.columns(3)
    m1.markdown(f'<div class="metric-box"><div class="metric-label">R²</div><div class="metric-value">{r2:.4f}</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-box"><div class="metric-label">RMSE</div><div class="metric-value">{rmse:.4f}</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-box"><div class="metric-label">Max Error</div><div class="metric-value">{max_err:.4f}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Koefisien Hasil Eliminasi Gauss</div>', unsafe_allow_html=True)
    rows_html = ""
    for lbl, c in zip(LABELS, coef):
        col_c = "#4ade80" if c > 0 else "#f87171"
        bar_w = min(abs(c)/max(np.abs(coef))*100, 100)
        rows_html += f"<tr><td style='color:#ccc;padding:6px 8px;'>{lbl}</td><td style='padding:6px 8px;'><div style='background:{col_c};width:{bar_w:.1f}%;height:6px;border-radius:3px;'></div></td><td style='padding:6px 8px;color:{col_c};font-family:JetBrains Mono,monospace;'>{c:+.6f}</td></tr>"
    st.markdown(f"<table style='width:100%;border-collapse:collapse;font-size:0.87rem;'><thead><tr><th style='color:#888;text-align:left;padding:6px 8px;'>Parameter</th><th style='color:#888;padding:6px 8px;'>Magnitude</th><th style='color:#888;padding:6px 8px;'>Koefisien</th></tr></thead><tbody>{rows_html}</tbody></table>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Aktual vs Prediksi (per pemain)</div>', unsafe_allow_html=True)
    df_v = pd.DataFrame({
        "Match": MATCH_LABELS[:len(y)],
        "Team": team_col[:len(y)],
        "Skor Aktual": y,
        "Skor Hasil Kalkulasi": np.round(y_pred, 4),
        "Selisih": np.round(y - y_pred, 4),
        "|Selisih|": np.round(np.abs(y - y_pred), 4),
    })
    st.dataframe(df_v.style
        .format({"Skor Aktual":"{:.1f}","Skor Hasil Kalkulasi":"{:.4f}","Selisih":"{:+.4f}","|Selisih|":"{:.4f}"})
        .background_gradient(subset=["|Selisih|"], cmap="YlOrRd"),
        use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# TAB 4 — Visualisasi
# ════════════════════════════════════════════════════════════════════
with tab4:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0d0d1a")
    fig.suptitle("Analisis Skor MLBB — Model Gabungan", color="#e8c84a", fontsize=13, fontweight="bold")

    for ax in axes:
        ax.set_facecolor("#111128")
        ax.tick_params(colors="#777")
        for sp in ["bottom","left"]: ax.spines[sp].set_color("#333")
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)

    # Warna per team
    colors_scatter = ["#4a90e8" if t == "Ally" else "#e84a4a" for t in team_col[:len(y)]]

    # Plot 1: Aktual vs Kalkulasi
    ax = axes[0]
    ax.scatter(y, y_pred, color=colors_scatter, s=60, zorder=5, edgecolors="#0d0d1a", linewidths=0.8)
    lims = [min(y.min(), y_pred.min())-0.3, max(y.max(), y_pred.max())+0.3]
    ax.plot(lims, lims, color="#aaa", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Skor Aktual", color="#aaa"); ax.set_ylabel("Skor Kalkulasi", color="#aaa")
    ax.set_title("Aktual vs Kalkulasi", color="#e8c84a", fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4a90e8",label="Ally"), Patch(color="#e84a4a",label="Enemy")], framealpha=0.2, labelcolor="#aaa")

    # Plot 2: Koefisien bar
    ax = axes[1]
    lbls_short = ["c","K","D","A","H.Dmg","T.Dmg","D.Tkn","TFP"]
    colors_bar = ["#4ade80" if c >= 0 else "#f87171" for c in coef]
    ax.bar(lbls_short, coef, color=colors_bar, edgecolor="#0d0d1a", linewidth=0.5)
    ax.axhline(0, color="#555", linewidth=0.8)
    ax.set_title("Bobot Tiap Statistik", color="#e8c84a", fontweight="bold")
    ax.set_ylabel("Koefisien", color="#aaa")
    ax.set_xticklabels(lbls_short, fontsize=8)

    # Plot 3: Residual
    ax = axes[2]
    residuals = y - y_pred
    ax.bar(range(1, len(residuals)+1), residuals,
           color=["#4a90e8" if t=="Ally" else "#e84a4a" for t in team_col[:len(y)]],
           edgecolor="#0d0d1a", linewidth=0.5)
    ax.axhline(0, color="#555", linewidth=0.8)
    ax.set_title("Distribusi Selisih", color="#e8c84a", fontweight="bold")
    ax.set_xlabel("Data ke-", color="#aaa"); ax.set_ylabel("Selisih", color="#aaa")
    ax.legend(handles=[Patch(color="#4a90e8",label="Ally"), Patch(color="#e84a4a",label="Enemy")], framealpha=0.2, labelcolor="#aaa")

    plt.tight_layout(pad=2)
    st.pyplot(fig)

# ════════════════════════════════════════════════════════════════════
# TAB 5 — Kalkulator Skor
# ════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-label">Kalkulator Skor Pemain</div>', unsafe_allow_html=True)
    st.caption("Masukkan statistik pemain. Skor dihitung langsung menggunakan bobot yang ditemukan melalui Eliminasi Gauss.")

    c1, c2 = st.columns(2)
    with c1:
        k_v  = st.number_input("Kill (K)",         0, 30, 7)
        d_v  = st.number_input("Death (D)",        0, 30, 3)
        a_v  = st.number_input("Assist (A)",       0, 30, 8)
        hd_v = st.number_input("Hero Dmg (%)",     0, 100, 25)
    with c2:
        td_v = st.number_input("Turret Dmg (%)",   0, 100, 20)
        dt_v = st.number_input("Dmg Taken (%)",    0, 100, 15)
        tf_v = st.number_input("TFP (%)",          0, 100, 60)

    x_new = np.array([1, k_v, d_v, a_v, hd_v, td_v, dt_v, tf_v])
    skor_kalkulasi = float(x_new @ coef)

    st.markdown(f"""
    <div style="text-align:center;padding:2rem;background:#0a0a1f;border-radius:12px;border:1px solid #3a3a6a;margin-top:1rem;">
        <div style="color:#888;font-size:0.8rem;letter-spacing:2px;text-transform:uppercase;">Hasil Kalkulasi Skor</div>
        <div style="color:#e8c84a;font-size:3.5rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{skor_kalkulasi:.2f}</div>
        <div style="color:#666;font-size:0.75rem;margin-top:0.5rem;">menggunakan rumus: Skor = c + Σ(wᵢ · statistikᵢ)</div>
    </div>""", unsafe_allow_html=True)

    with st.expander("Breakdown kontribusi tiap variabel"):
        total = 0
        for lbl, xi, wi in zip(LABELS, x_new, coef):
            contrib = xi * wi
            total += contrib
            col_c = "#4ade80" if contrib >= 0 else "#f87171"
            st.markdown(f"`{lbl}` = {xi:.0f} × {wi:.5f} = <span style='color:{col_c}'>{contrib:+.5f}</span>", unsafe_allow_html=True)
        st.markdown(f"**Total = {total:.5f}**")