import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Provision SNEM construction", page_icon="🏗️", layout="wide")

st.title("Provision SNEM en assurance construction")
st.caption("Chain Ladder sur données simulées, actualisation et sensibilité aux taux — Pamela Fagla, M2 Actuariat ISUP")

# ---------- Curseurs ----------
st.sidebar.header("Hypothèses")
freq = st.sidebar.slider("Nombre moyen de sinistres par an", 50, 300, 100, 10)
p_decl = st.sidebar.slider("Rapidité de déclaration", 0.10, 0.60, 0.30, 0.05,
                           help="Plus c'est bas, plus les sinistres se déclarent tard")
cout_moyen = st.sidebar.slider("Coût médian d'un sinistre (€)", 5000, 50000, 13000, 1000)
taux = st.sidebar.slider("Taux d'actualisation (%)", 0.0, 6.0, 3.0, 0.5) / 100
graine = st.sidebar.number_input("Graine aléatoire", value=42)

# ---------- Simulation ----------
def simuler(freq, p, cout, graine):
    rng = np.random.default_rng(int(graine))
    lignes = []
    for annee in range(2015, 2025):
        for _ in range(rng.poisson(freq)):
            delai = min(rng.geometric(p), 10)
            lignes.append([annee, delai, rng.lognormal(np.log(cout), 1)])
    df = pd.DataFrame(lignes, columns=["annee", "delai", "cout"])
    df["annee_decl"] = df["annee"] + df["delai"] - 1
    return df

def triangle(df, valeur=None):
    if valeur:
        t = pd.crosstab(df["annee"], df["delai"], values=df[valeur], aggfunc="sum").fillna(0)
    else:
        t = pd.crosstab(df["annee"], df["delai"])
    t = t.reindex(index=range(2015, 2025), columns=range(1, 11), fill_value=0).cumsum(axis=1).astype(float)
    for a in t.index:
        for j in t.columns:
            if a + j - 1 > 2024:
                t.loc[a, j] = np.nan
    return t

def facteurs(t):
    return [t.loc[t[j + 1].notna(), j + 1].sum() / t.loc[t[j + 1].notna(), j].sum() for j in range(1, 10)]

def completer(t, f):
    c = t.copy()
    for a in c.index:
        for j in range(2, 11):
            if pd.isna(c.loc[a, j]):
                c.loc[a, j] = c.loc[a, j - 1] * f[j - 2]
    return c

df_complet = simuler(freq, p_decl, cout_moyen, graine)
df = df_complet[df_complet["annee_decl"] <= 2024]

tri_nb, tri_mt = triangle(df), triangle(df, "cout")
f_nb, f_mt = facteurs(tri_nb), facteurs(tri_mt)
comp_nb, comp_mt = completer(tri_nb, f_nb), completer(tri_mt, f_mt)

connu_mt = tri_mt.ffill(axis=1)[10]
provision_annee = comp_mt[10] - connu_mt
tardifs = comp_nb[10] - tri_nb.ffill(axis=1)[10]
provision = provision_annee.sum()
vraie = df_complet.loc[df_complet["annee_decl"] > 2024, "cout"].sum()

# ---------- Flux et ALM ----------
flux = {}
for a in tri_mt.index:
    for j in range(2, 11):
        if pd.isna(tri_mt.loc[a, j]):
            flux[a + j - 1] = flux.get(a + j - 1, 0) + comp_mt.loc[a, j] - comp_mt.loc[a, j - 1]
flux = pd.Series(flux).sort_index()
t = flux.index - 2024
va = flux / (1 + taux) ** t
provision_act = va.sum()
duration = (t * va).sum() / va.sum()
sensibilite = duration / (1 + taux)

# ---------- Chiffres clés ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Provision SNEM", f"{provision/1e6:,.2f} M€".replace(",", " "),
          f"{(provision/vraie - 1)*100:+.1f} % vs vraie valeur" if vraie > 0 else None, delta_color="off")
c2.metric("Provision actualisée", f"{provision_act/1e6:,.2f} M€".replace(",", " "))
c3.metric("Duration", f"{duration:.2f} ans")
c4.metric("Sensibilité", f"{sensibilite:.2f} % / point de taux")

# ---------- Onglets ----------
o1, o2, o3, o4 = st.tabs(["Triangles", "Facteurs", "Provision", "Flux & taux"])

with o1:
    st.subheader("Triangle des nombres (cumulé)")
    st.dataframe(tri_nb.style.format("{:.0f}", na_rep=""))
    st.subheader("Triangle des montants (cumulé, €)")
    st.dataframe(tri_mt.style.format("{:,.0f}", na_rep=""))

with o2:
    st.dataframe(pd.DataFrame({"Nombres": f_nb, "Montants": f_mt},
                 index=[f"{j} → {j+1}" for j in range(1, 10)]).style.format("{:.3f}"))
    st.info("Facteur j → j+1 = somme de la colonne j+1 ÷ somme de la colonne j, sur les années où les deux sont connues.")

with o3:
    res = pd.DataFrame({"Sinistres tardifs": tardifs.round(1), "Provision (€)": provision_annee.round(0)})
    st.dataframe(res.style.format({"Sinistres tardifs": "{:.1f}", "Provision (€)": "{:,.0f}"}))
    st.bar_chart(provision_annee)
    st.success(f"Vraie provision simulée : {vraie:,.0f} € — Estimation Chain Ladder : {provision:,.0f} €")

with o4:
    st.subheader("Paiements futurs par année")
    st.bar_chart(flux)
    sens = pd.DataFrame({"Taux": [f"{r} %" for r in range(0, 7)],
                         "Provision actualisée (€)": [(flux / (1 + r/100) ** t).sum() for r in range(0, 7)]})
    st.subheader("Sensibilité de la provision aux taux")
    st.line_chart(sens.set_index("Taux"))
    st.caption("Plus les taux baissent, plus l'assureur doit mettre de côté aujourd'hui.")