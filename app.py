"""
Dashboard: Transição Demográfica e Urbanização
Índice de Envelhecimento (IE) x Grau de Urbanização (GU) — leitura por quartis.

Definições
    IE = (população 65+ / população 0-14) * 100
    GU = (população urbana / população total) * 100

Dados
    Se existir `dados.csv` nesta pasta, ele é usado. Colunas esperadas:
        municipio, uf, regiao, pop_urbana, pop_rural, pop_0_14, pop_65_mais
    Caso contrário, é gerada uma base SINTÉTICA de municípios brasileiros,
    com parâmetros por UF inspirados no Censo 2022 (apenas para demonstração).

Executar:  python app.py   ->  http://127.0.0.1:8050
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "dados.csv"

# --------------------------------------------------------------------------
# 1. Dados
# --------------------------------------------------------------------------
# UF: (região, nº municípios, GU médio %, IE médio)
UF_PARAMS = {
    "AC": ("Norte", 22, 70, 30), "AM": ("Norte", 62, 75, 27), "AP": ("Norte", 16, 88, 26),
    "PA": ("Norte", 144, 66, 34), "RO": ("Norte", 52, 72, 42), "RR": ("Norte", 15, 74, 24),
    "TO": ("Norte", 139, 76, 48),
    "AL": ("Nordeste", 102, 70, 45), "BA": ("Nordeste", 417, 70, 58), "CE": ("Nordeste", 184, 74, 55),
    "MA": ("Nordeste", 217, 62, 38), "PB": ("Nordeste", 223, 74, 62), "PE": ("Nordeste", 185, 79, 55),
    "PI": ("Nordeste", 224, 65, 58), "RN": ("Nordeste", 167, 76, 60), "SE": ("Nordeste", 75, 72, 47),
    "DF": ("Centro-Oeste", 1, 96, 50), "GO": ("Centro-Oeste", 246, 89, 52),
    "MS": ("Centro-Oeste", 79, 85, 50), "MT": ("Centro-Oeste", 141, 81, 44),
    "ES": ("Sudeste", 78, 83, 63), "MG": ("Sudeste", 853, 85, 70), "RJ": ("Sudeste", 92, 97, 78),
    "SP": ("Sudeste", 645, 95, 70),
    "PR": ("Sul", 399, 85, 68), "RS": ("Sul", 497, 85, 85), "SC": ("Sul", 295, 83, 68),
}


def gerar_dados_sinteticos(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    linhas = []
    for uf, (regiao, n, gu_med, ie_med) in UF_PARAMS.items():
        # GU: distribuição Beta centrada na média da UF (municípios pequenos mais rurais)
        m = gu_med / 100
        k = 9 if n > 1 else 400
        gu = rng.beta(m * k, (1 - m) * k, n)
        gu = np.clip(gu, 0.08, 0.999)
        pop_total = np.round(rng.lognormal(mean=9.3 + 2.2 * (gu - m), sigma=1.0, size=n)) + 800
        if uf == "DF":
            pop_total = np.array([2_817_000.0])
        # IE: base da UF + efeito positivo da urbanização + ruído multiplicativo.
        # No Sul/Sudeste o meio rural também é envelhecido (êxodo de jovens).
        efeito_urb = 0.9 if regiao in ("Sul", "Sudeste") else 1.4
        ie = ie_med * np.exp(efeito_urb * (gu - m) + rng.normal(0, 0.25, n))
        share_0_14 = np.clip(rng.normal(0.30 - 0.0012 * ie, 0.015, n), 0.10, 0.40)
        share_65 = share_0_14 * ie / 100

        pop_urb = np.round(pop_total * gu)
        for i in range(n):
            linhas.append(
                {
                    "municipio": f"{uf}-{i + 1:03d}",
                    "uf": uf,
                    "regiao": regiao,
                    "pop_urbana": int(pop_urb[i]),
                    "pop_rural": int(pop_total[i] - pop_urb[i]),
                    "pop_0_14": int(round(pop_total[i] * share_0_14[i])),
                    "pop_65_mais": int(round(pop_total[i] * share_65[i])),
                }
            )
    return pd.DataFrame(linhas)


def carregar_dados() -> tuple[pd.DataFrame, bool]:
    if CSV_PATH.exists():
        df, sintetico = pd.read_csv(CSV_PATH), False
    else:
        df, sintetico = gerar_dados_sinteticos(), True
    df["pop_total"] = df["pop_urbana"] + df["pop_rural"]
    df["gu"] = 100 * df["pop_urbana"] / df["pop_total"]
    df["ie"] = 100 * df["pop_65_mais"] / df["pop_0_14"].replace(0, np.nan)
    df = df.dropna(subset=["gu", "ie"])
    return df, sintetico


DF, SINTETICO = carregar_dados()
REGIOES = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
Q_LABELS = ["Q1 (mais rural)", "Q2", "Q3", "Q4 (mais urbano)"]

# --------------------------------------------------------------------------
# 2. Estatística
# --------------------------------------------------------------------------


def spearman(x: pd.Series, y: pd.Series) -> float:
    return x.rank().corr(y.rank())


def add_quartis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["quartil_gu"] = pd.qcut(df["gu"].rank(method="first"), 4, labels=Q_LABELS)
    return df


def tabela_quartis(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("quartil_gu", observed=True)
    t = pd.DataFrame(
        {
            "Quartil GU": Q_LABELS,
            "Municípios": g.size().values,
            "Faixa GU (%)": [f"{a:.1f} – {b:.1f}" for a, b in zip(g["gu"].min(), g["gu"].max())],
            "IE Q1": g["ie"].quantile(0.25).values,
            "IE Mediana": g["ie"].median().values,
            "IE Q3": g["ie"].quantile(0.75).values,
            "IE IIQ": (g["ie"].quantile(0.75) - g["ie"].quantile(0.25)).values,
            "IE agregado": (100 * g["pop_65_mais"].sum() / g["pop_0_14"].sum()).values,
            "% pop. rural": (100 * g["pop_rural"].sum() / g["pop_total"].sum()).values,
        }
    )
    for c in ["IE Q1", "IE Mediana", "IE Q3", "IE IIQ", "IE agregado", "% pop. rural"]:
        t[c] = t[c].round(1)
    return t


def interpretar(df: pd.DataFrame, t: pd.DataFrame) -> list:
    rho = spearman(df["gu"], df["ie"])
    r = df["gu"].corr(df["ie"])
    forca = (
        "muito fraca" if abs(rho) < 0.2 else "fraca" if abs(rho) < 0.4
        else "moderada" if abs(rho) < 0.6 else "forte"
    )
    sentido = "positiva" if rho > 0 else "negativa"
    med1, med4 = t["IE Mediana"].iloc[0], t["IE Mediana"].iloc[-1]
    iiq1, iiq4 = t["IE IIQ"].iloc[0], t["IE IIQ"].iloc[-1]
    gradiente = t["IE Mediana"].is_monotonic_increasing
    ie_geral_q = df["ie"].quantile([0.25, 0.5, 0.75])

    # sobreposição das caixas: Q3 do grupo mais rural vs Q1 do mais urbano
    sobrepoe = t["IE Q3"].iloc[0] > t["IE Q1"].iloc[-1]

    itens = [
        html.Li([html.B("Correlação: "),
                 f"Spearman ρ = {rho:.2f} (Pearson r = {r:.2f}) — associação {forca} e {sentido} "
                 "entre urbanização e envelhecimento."]),
        html.Li([html.B("Gradiente das medianas: "),
                 f"o IE mediano vai de {med1:.1f} no quartil mais rural para {med4:.1f} no mais urbano "
                 f"({med4 - med1:+.1f} pontos; {'cresce de forma monotônica' if gradiente else 'não é monotônico'} "
                 "entre os quartis)."]),
        html.Li([html.B("Dispersão (IIQ): "),
                 f"amplitude interquartil de {iiq1:.1f} no Q1 e {iiq4:.1f} no Q4. "
                 + ("Municípios urbanos são mais heterogêneos quanto ao envelhecimento."
                    if iiq4 > iiq1 else
                    "Municípios rurais são mais heterogêneos — convivem áreas de fronteira jovem "
                    "e zonas de êxodo envelhecidas.")]),
        html.Li([html.B("Sobreposição: "),
                 ("as caixas do Q1 e do Q4 se sobrepõem — a urbanização sozinha não explica o IE; "
                  "região e estágio da transição pesam bastante."
                  if sobrepoe else
                  "as caixas do Q1 e do Q4 não se sobrepõem — separação nítida entre rural e urbano.")]),
        html.Li([html.B("Distribuição geral do IE: "),
                 f"25% dos municípios têm IE abaixo de {ie_geral_q[0.25]:.1f}, metade abaixo de "
                 f"{ie_geral_q[0.5]:.1f} e 25% acima de {ie_geral_q[0.75]:.1f}. "
                 "IE > 100 indica mais idosos que crianças (transição avançada)."]),
    ]
    return itens


# --------------------------------------------------------------------------
# 3. Layout
# --------------------------------------------------------------------------
COR_Q = ["#2f7d4f", "#7fb069", "#6c9bd2", "#1f4e89"]  # rural (verde) -> urbano (azul)
COR_REG = {"Norte": "#2a9d8f", "Nordeste": "#e9c46a", "Centro-Oeste": "#f4a261",
           "Sudeste": "#e76f51", "Sul": "#5e60ce"}
LAYOUT_BASE = dict(template="plotly_white", margin=dict(l=50, r=20, t=50, b=45),
                   font=dict(family="Segoe UI, Arial", size=12))

app = Dash(__name__, title="Transição Demográfica e Urbanização")


def card(titulo, id_valor, sub=None):
    return html.Div(className="card kpi", children=[
        html.Div(titulo, className="kpi-titulo"),
        html.Div(id=id_valor, className="kpi-valor"),
        html.Div(sub or "", id=f"{id_valor}-sub", className="kpi-sub"),
    ])


app.layout = html.Div(className="pagina", children=[
    html.Header([
        html.H1("Transição Demográfica e Urbanização"),
        html.P("Índice de Envelhecimento × Grau de Urbanização nos municípios — leitura por quartis"),
        html.Div(
            "⚠ Dados sintéticos para demonstração. Coloque um dados.csv na pasta para usar dados reais."
            if SINTETICO else f"Fonte: {CSV_PATH.name}",
            className="aviso" if SINTETICO else "fonte",
        ),
    ]),

    html.Div(className="filtros card", children=[
        html.Div([html.Label("Região"),
                  dcc.Dropdown(REGIOES, REGIOES, multi=True, id="f-regiao")], style={"flex": 2}),
        html.Div([html.Label("UF"),
                  dcc.Dropdown(id="f-uf", multi=True, placeholder="Todas")], style={"flex": 2}),
        html.Div([html.Label("Escala do IE"),
                  dcc.RadioItems([{"label": " Linear", "value": "linear"},
                                  {"label": " Log", "value": "log"}],
                                 "linear", id="f-escala", inline=True)], style={"flex": 1}),
    ]),

    html.Div(className="kpis", children=[
        card("Municípios", "k-n"),
        card("IE mediano", "k-ie", "idosos por 100 jovens"),
        card("GU mediano", "k-gu", "% da população urbana"),
        card("Correlação (Spearman ρ)", "k-rho"),
        card("Razão IE Q4 / Q1", "k-razao", "medianas, urbano ÷ rural"),
    ]),

    html.Div(className="grade", children=[
        html.Div(className="card", children=[dcc.Graph(id="g-disp")]),
        html.Div(className="card", children=[dcc.Graph(id="g-box")]),
        html.Div(className="card", children=[dcc.Graph(id="g-hist")]),
        html.Div(className="card", children=[dcc.Graph(id="g-pop")]),
    ]),

    html.Div(className="grade", children=[
        html.Div(className="card", children=[
            html.H3("Estatísticas do IE por quartil de urbanização"),
            dash_table.DataTable(
                id="tabela",
                style_header={"fontWeight": "600", "backgroundColor": "#f1f4f8"},
                style_cell={"fontFamily": "Segoe UI, Arial", "fontSize": 13, "padding": "6px",
                            "textAlign": "center"},
                style_data_conditional=[
                    {"if": {"row_index": i}, "borderLeft": f"5px solid {c}"}
                    for i, c in enumerate(COR_Q)
                ],
            ),
            html.P("IIQ = intervalo interquartil (Q3 − Q1). IE agregado = soma de idosos ÷ soma de "
                   "jovens do grupo (pondera pela população).", className="nota"),
        ]),
        html.Div(className="card", children=[
            html.H3("Interpretação automática"),
            html.Ul(id="interpretacao"),
            html.Details([
                html.Summary("Como ler os quartis"),
                html.P("Os municípios são ordenados pelo Grau de Urbanização e divididos em 4 grupos "
                       "de mesmo tamanho (Q1 = 25% mais rurais … Q4 = 25% mais urbanos). Em cada grupo, "
                       "o boxplot mostra a distribuição do IE: a linha central é a mediana, a caixa vai "
                       "do 1º ao 3º quartil (50% centrais) e os pontos são valores atípicos. Medianas "
                       "crescentes de Q1 a Q4 indicam que a urbanização acompanha o avanço da transição "
                       "demográfica (menos fecundidade, maior longevidade)."),
            ]),
        ]),
    ]),
])


# --------------------------------------------------------------------------
# 4. Callbacks
# --------------------------------------------------------------------------
@app.callback(Output("f-uf", "options"), Input("f-regiao", "value"))
def opcoes_uf(regioes):
    regioes = regioes or REGIOES
    return sorted(DF.loc[DF["regiao"].isin(regioes), "uf"].unique())


@app.callback(
    Output("k-n", "children"), Output("k-ie", "children"), Output("k-gu", "children"),
    Output("k-rho", "children"), Output("k-rho-sub", "children"), Output("k-razao", "children"),
    Output("g-disp", "figure"), Output("g-box", "figure"), Output("g-hist", "figure"),
    Output("g-pop", "figure"), Output("tabela", "data"), Output("tabela", "columns"),
    Output("interpretacao", "children"),
    Input("f-regiao", "value"), Input("f-uf", "value"), Input("f-escala", "value"),
)
def atualizar(regioes, ufs, escala):
    df = DF[DF["regiao"].isin(regioes or REGIOES)]
    if ufs:
        df = df[df["uf"].isin(ufs)]
    if len(df) < 8:
        vazio = go.Figure().update_layout(**LAYOUT_BASE, title="Poucos municípios para quartis")
        return ("—",) * 6 + (vazio,) * 4 + ([], [], [html.Li("Selecione mais municípios.")])

    df = add_quartis(df)
    t = tabela_quartis(df)
    rho = spearman(df["gu"], df["ie"])
    forca = "fraca" if abs(rho) < 0.4 else "moderada" if abs(rho) < 0.6 else "forte"

    # --- Dispersão + tendência (OLS em log(IE) ~ GU) ---
    fig_d = go.Figure()
    for reg in REGIOES:
        d = df[df["regiao"] == reg]
        if d.empty:
            continue
        fig_d.add_trace(go.Scattergl(
            x=d["gu"], y=d["ie"], mode="markers", name=reg,
            marker=dict(color=COR_REG[reg], size=6, opacity=0.6, line=dict(width=0)),
            customdata=np.c_[d["municipio"], d["uf"], d["pop_total"]],
            hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>GU: %{x:.1f}%<br>"
                          "IE: %{y:.1f}<br>Pop.: %{customdata[2]:,.0f}<extra></extra>",
        ))
    b, a = np.polyfit(df["gu"], np.log(df["ie"]), 1)
    xs = np.linspace(df["gu"].min(), df["gu"].max(), 100)
    fig_d.add_trace(go.Scatter(x=xs, y=np.exp(a + b * xs), mode="lines", name="Tendência",
                               line=dict(color="#222", width=2.5, dash="dash")))
    for corte in df["gu"].quantile([0.25, 0.5, 0.75]):
        fig_d.add_vline(x=corte, line=dict(color="#999", width=1, dash="dot"))
    fig_d.add_hline(y=100, line=dict(color="#c0392b", width=1),
                    annotation_text="IE = 100", annotation_position="top left")
    fig_d.update_layout(**LAYOUT_BASE, title="IE × Grau de Urbanização (linhas pontilhadas = quartis do GU)",
                        xaxis_title="Grau de Urbanização (%)", yaxis_title="Índice de Envelhecimento",
                        yaxis_type=escala, legend=dict(orientation="h", y=-0.2))

    # --- Boxplot por quartil ---
    fig_b = go.Figure()
    for q, cor in zip(Q_LABELS, COR_Q):
        d = df[df["quartil_gu"] == q]
        fig_b.add_trace(go.Box(y=d["ie"], name=q, marker_color=cor, boxmean=True,
                               boxpoints="outliers", hovertemplate="IE: %{y:.1f}<extra></extra>"))
    fig_b.add_trace(go.Scatter(x=Q_LABELS, y=t["IE Mediana"], mode="lines+markers",
                               name="Mediana", line=dict(color="#222", width=2),
                               marker=dict(size=8)))
    fig_b.update_layout(**LAYOUT_BASE, title="Distribuição do IE por quartil de urbanização",
                        yaxis_title="Índice de Envelhecimento", yaxis_type=escala, showlegend=False)

    # --- Histograma do IE com quartis ---
    fig_h = go.Figure(go.Histogram(x=df["ie"], nbinsx=50, marker_color="#6c9bd2",
                                   hovertemplate="IE %{x}<br>%{y} municípios<extra></extra>"))
    for nome, v in zip(["Q1", "Mediana", "Q3"], df["ie"].quantile([0.25, 0.5, 0.75])):
        fig_h.add_vline(x=v, line=dict(color="#1f4e89", width=2 if nome == "Mediana" else 1, dash="dash"),
                        annotation_text=f"{nome}: {v:.0f}", annotation_position="top")
    fig_h.update_layout(**LAYOUT_BASE, title="Distribuição geral do IE (quartis marcados)",
                        xaxis_title="Índice de Envelhecimento", yaxis_title="Municípios", bargap=0.05)

    # --- População urbana x rural por quartil ---
    g = df.groupby("quartil_gu", observed=True)[["pop_urbana", "pop_rural"]].sum() / 1e6
    fig_p = go.Figure([
        go.Bar(x=Q_LABELS, y=g["pop_urbana"], name="Urbana", marker_color="#1f4e89",
               hovertemplate="%{y:.2f} mi<extra>Urbana</extra>"),
        go.Bar(x=Q_LABELS, y=g["pop_rural"], name="Rural", marker_color="#2f7d4f",
               hovertemplate="%{y:.2f} mi<extra>Rural</extra>"),
    ])
    fig_p.add_trace(go.Scatter(x=Q_LABELS, y=t["IE agregado"], yaxis="y2", name="IE agregado",
                               mode="lines+markers+text", text=t["IE agregado"], textposition="top center",
                               line=dict(color="#e76f51", width=3), marker=dict(size=9)))
    fig_p.update_layout(**LAYOUT_BASE, barmode="stack",
                        title="População urbana × rural e IE agregado por quartil",
                        yaxis=dict(title="População (milhões)"),
                        yaxis2=dict(title="IE agregado", overlaying="y", side="right", showgrid=False,
                                    rangemode="tozero"),
                        legend=dict(orientation="h", y=-0.2))

    cols = [{"name": c, "id": c} for c in t.columns]
    razao = t["IE Mediana"].iloc[-1] / t["IE Mediana"].iloc[0]
    return (
        f"{len(df):,}".replace(",", "."),
        f"{df['ie'].median():.1f}",
        f"{df['gu'].median():.1f}%",
        f"{rho:+.2f}",
        f"associação {forca} {'positiva' if rho > 0 else 'negativa'}",
        f"{razao:.2f}×",
        fig_d, fig_b, fig_h, fig_p,
        t.to_dict("records"), cols,
        interpretar(df, t),
    )


if __name__ == "__main__":
    app.run(debug=True)
