
def imputer_valeurs(df):
    df_impute = df.copy()
    
    # ✅ Identifier automatiquement les colonnes numériques (années)
    colonnes_annees = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # ✅ Calcul des valeurs globales d’imputation
    max_inflation = df.loc[df["Indicateur"] == "Inflation", colonnes_annees].max().max()
    max_chomage   = df.loc[df["Indicateur"] == "Chomage", colonnes_annees].max().max()
    min_gdp       = df.loc[df["Indicateur"] == "GDP", colonnes_annees].min().min()
    
    # ✅ Application de l’imputation selon l’indicateur
    for indicateur, group in df.groupby("Indicateur"):
        if indicateur == "Inflation":
            group[colonnes_annees] = group[colonnes_annees].fillna(max_inflation)
        elif indicateur == "Chomage":
            group[colonnes_annees] = group[colonnes_annees].fillna(max_chomage)
        elif indicateur == "GDP":
            group[colonnes_annees] = group[colonnes_annees].fillna(min_gdp)
        
        # Réintégration dans le DataFrame principal
        df_impute.loc[group.index, colonnes_annees] = group[colonnes_annees]
    
    return df_impute


# ============================
# ⚙️ Paramètres modifiables
# ============================

@dataclass
class MetricResult:
    country: str
    group: str
    n_obs: int
    risk_gdp: Optional[float]
    risk_infl: Optional[float]
    risk_unemp: Optional[float]


TARGET_INFL_LOWER = 2.0   # borne basse zone de confort inflation
TARGET_INFL_UPPER = 5.0   # borne haute zone de confort inflation
MIN_YEARS_REQUIRED = 1    # nb d'années min pour calculer des métriques fiables
WEIGHTS = {               # pondérations du score composite (somme = 1)
    "inflation": 0.40,
    "gdp": 0.35,
    "unemp": 0.25,
}
PERIOD = list(range(2010, 2025))


def _yoy_growth(series: pd.Series) -> pd.Series:
    # series.pct_change(periods=1) * 100.0
    return series

def _semivariance_negative(values: pd.Series) -> float:
    neg = values[values < 0]
    if len(neg) == 0:
        return 0.0
    return float(((neg - 0) ** 2).mean())


def _iqr_winsorize(s: pd.Series, k: float = 3.0) -> pd.Series:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return s.clip(lower=lo, upper=hi)

# ========== Calcul rolling par pays × année ==========


def compute_country_metrics(df_long: pd.DataFrame) -> pd.DataFrame:
    # Filtre période
    d = df_long[df_long["Année"].isin(PERIOD)].copy()

    results: List[MetricResult] = []

    for (grp, ctry), sub in d.groupby(["Groupe", "Pays"]):
        # --- PIB ---
        gdp = sub[sub["Indicateur"].str.lower() == "gdp"].sort_values("Année")
        risk_gdp = None
        if len(gdp) >= MIN_YEARS_REQUIRED:
            gdp_v = gdp[["Année", "Valeur"]].dropna()
            if not gdp_v.empty:
                gr = _yoy_growth(gdp_v.set_index("Année")["Valeur"]).dropna()
                if gr.size >= MIN_YEARS_REQUIRED - 1:
                    gr = _iqr_winsorize(gr)
                    vol = float(gr.std(ddof=1)) if gr.size > 1 else np.nan
                    downside = _semivariance_negative(gr)
                    risk_gdp = vol + 0.2 * downside  # pénalisation 20% du downside

        # --- Inflation ---
        infl = sub[sub["Indicateur"].str.lower() == "inflation"].sort_values("Année")
        risk_infl = None
        if len(infl) >= MIN_YEARS_REQUIRED:
            iv = infl["Valeur"].dropna()
            if iv.size >= MIN_YEARS_REQUIRED:
                iv = _iqr_winsorize(iv)
                vol = float(iv.std(ddof=1)) if iv.size > 1 else np.nan
                # écart moyen à la zone de confort [2,5]
                gap = (np.maximum(0, TARGET_INFL_LOWER - iv) + np.maximum(0, iv - TARGET_INFL_UPPER)).mean()
                risk_infl = vol + 0.7 * float(gap)  # 70% de poids sur l'écart de niveau

        # --- Chômage ---
        unemp = sub[sub["Indicateur"].str.lower().isin(["chômage", "chomage"])].sort_values("Année")
        risk_unemp = None
        if len(unemp) >= MIN_YEARS_REQUIRED:
            uv = unemp["Valeur"].dropna()
            if uv.size >= MIN_YEARS_REQUIRED:
                uv = _iqr_winsorize(uv)
                vol = float(uv.std(ddof=1)) if uv.size > 1 else np.nan
                spike = float(uv.quantile(0.95) - uv.median())
                risk_unemp = vol + 0.3 * spike

        n_obs = int(sub.dropna(subset=["Valeur"]).shape[0])
        results.append(MetricResult(ctry, grp, n_obs, risk_gdp, risk_infl, risk_unemp))

    out = pd.DataFrame([r.__dict__ for r in results])
    return out



def compute_country_metrics_per_year(df_long: pd.DataFrame,
                                     min_years: int = MIN_YEARS_REQUIRED,
                                     period: List[int] = PERIOD) -> pd.DataFrame:
    """
    Pour chaque (Groupe, Pays) et chaque année dans period, calcule les métriques:
    - risk_gdp : vol(YoY croissance) + 0.2 * downside (fenêtre historique jusqu'à l'année courante,
      en prenant les dernières `min_years` observations si disponibles)
    - risk_infl : vol(inflation) + 0.7 * gap (écart moyen à la zone [2,5])
    - risk_unemp: vol(chômage) + 0.3 * spike (95e percentile - median)
    Retourne DataFrame country × year avec colonnes:
    ['group','country','Année','n_obs','risk_gdp','risk_infl','risk_unemp']
    """
    df = df_long.copy()
    # homogénéiser nom colonnes
    if "Year" in df.columns and "Année" not in df.columns:
        df = df.rename(columns={"Year": "Année"})
    df["Indicateur"] = df["Indicateur"].astype(str)
    results = []

    # détection souple des libellés
    def is_gdp(lbl): return "gdp" in lbl.lower() or "pib" in lbl.lower()
    def is_infl(lbl): return "infl" in lbl.lower()
    def is_unemp(lbl): return "chomag" in lbl.lower() or "chômage" in lbl.lower() or "unemp" in lbl.lower()

    # itérer par pays
    grouped = df.groupby(["Groupe", "Pays"])
    for (grp, ctry), sub in grouped:
        # séparer séries par indicateur et indexer par année
        gdp_s = sub[sub["Indicateur"].apply(is_gdp)][["Année", "Valeur"]].dropna().set_index("Année").sort_index()["Valeur"]
        infl_s = sub[sub["Indicateur"].apply(is_infl)][["Année", "Valeur"]].dropna().set_index("Année").sort_index()["Valeur"]
        unemp_s = sub[sub["Indicateur"].apply(is_unemp)][["Année", "Valeur"]].dropna().set_index("Année").sort_index()["Valeur"]

        # années à évaluer : intersection entre period et années observées ? On gardera period
        for year in period:
            # fenetre: prendre les dernières min_years observations **≤ year**
            def last_window(series: pd.Series):
                s = series[series.index <= year]
                if s.empty:
                    return s
                return s.tail(min(len(s), min_years))

            # GDP
            risk_gdp = None
            gdp_w = last_window(gdp_s)
            if len(gdp_w) >= min_years:
                gr =  _yoy_growth(gdp_w).dropna()
                if gr.size >= max(1, min_years-1):
                    gr = _iqr_winsorize(gr)
                    vol = float(gr.std(ddof=1)) if gr.size > 1 else 0.0
                    downside = _semivariance_negative(gr)
                    risk_gdp = vol + 0.2 * downside

            # Inflation
            risk_infl = None
            infl_w = last_window(infl_s)
            if len(infl_w) >= min_years:
                iv = infl_w.copy()
                iv = _iqr_winsorize(iv)
                vol = float(iv.std(ddof=1)) if iv.size > 1 else 0.0
                # gap moyen à la zone [TARGET_INFL_LOWER, TARGET_INFL_UPPER]
                gap = (np.maximum(0, TARGET_INFL_LOWER - iv) + np.maximum(0, iv - TARGET_INFL_UPPER)).mean()
                risk_infl = vol + 0.7 * float(gap)

            # Chômage
            risk_unemp = None
            unemp_w = last_window(unemp_s)
            if len(unemp_w) >= min_years:
                uv = unemp_w.copy()
                uv = _iqr_winsorize(uv)
                vol = float(uv.std(ddof=1)) if uv.size > 1 else 0.0
                spike = float(uv.quantile(0.95) - uv.median())
                risk_unemp = vol + 0.3 * spike

            # nombre d'observations utiles dans la fenêtre (somme des non-nulls)
            n_obs = int(gdp_w.count() + infl_w.count() + unemp_w.count())

            results.append({
                "group": grp,
                "country": ctry,
                "Année": int(year),
                "n_obs": n_obs,
                "risk_gdp": risk_gdp,
                "risk_infl": risk_infl,
                "risk_unemp": risk_unemp
            })

    out = pd.DataFrame(results)
    # trier
    out = out.sort_values(["group", "country", "Année"]).reset_index(drop=True)
    return out


# ============================
# 📊 Fonctions utilitaires
# ============================

import pandas as pd
import numpy as np

def categorize_scores(x: pd.Series,
                      q_low: float = 0.5,
                      q_stress: float = 0.9,
                      ref_values: pd.Series | None = None) -> pd.Series:
    """
    Classe les scores en catégories 'Faible', 'Moyen', 'Élevé' selon la logique BCE/Fed.
    
    - Calibrage sur des quantiles fixes (médiane, 90e percentile)
    - Si ref_values est fourni, les seuils sont calibrés sur cette période de référence
      (garantissant la constance temporelle du stress)
    
    Paramètres
    ----------
    x : pd.Series
        Série de scores (risk ou score_composite)
    q_low : float
        Quantile inférieur (par défaut 0.5 = médiane)
    q_stress : float
        Quantile supérieur définissant la zone de stress (par défaut 0.9)
    ref_values : pd.Series | None
        Valeurs historiques utilisées pour calibrer les seuils.
        Si None, utilise x lui-même.
    
    Retour
    ------
    pd.Series
        Série de classes : 'Faible', 'Moyen', 'Élevé'
    """
    # Choisir base de calibration
    calib = ref_values.dropna() if ref_values is not None else x.dropna()

    if calib.nunique() <= 1:
        return pd.Series(["Moyen"] * len(x), index=x.index)

    # Seuils calibrés sur la période de référence
    ql = calib.quantile(q_low)
    qs = calib.quantile(q_stress)

    # Application des règles BCE/Fed
    classes = pd.cut(
        x,
        bins=[-np.inf, ql, qs, np.inf],
        labels=["Faible", "Moyen", "Élevé"],
        include_lowest=True
    ).astype(str)

    return classes





# ============================
# 🧪 Normalisation & Score (par année)
# ============================

def build_scores_per_year(metrics_year: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construit les scores normalisés et classes de risque par pays et par groupe, pour chaque année.
    """

    rows_pays = []
    rows_groupes = []

    for year, df_y in metrics_year.groupby("Année"):
        df = df_y.copy()

        # Normalisation 0..100
        for col, normcol in [("risk_gdp", "score_gdp"),
                             ("risk_infl", "score_infl"),
                             ("risk_unemp", "score_unemp")]:
            df[normcol] = _min_max_norm(df[col]) * 100.0

        # Score composite
        df["score_composite"] = (
            WEIGHTS["gdp"] * df["score_gdp"].fillna(df["score_gdp"].median()) +
            WEIGHTS["inflation"] * df["score_infl"].fillna(df["score_infl"].median()) +
            WEIGHTS["unemp"] * df["score_unemp"].fillna(df["score_unemp"].median())
        )

        # Rang pays
        df["rang_pays"] = df["score_composite"].rank(method="min", ascending=False).astype(int)

        # Catégorisation robuste
        df["classe_risque"] = categorize_scores(df["score_composite"])

        # Renommage
        df = df.rename(columns={
            "country": "Pays",
            "group": "Groupe",
            "score_gdp": "Score PIB (vol. croissance)",
            "score_infl": "Score Inflation (niv.+vol.)",
            "score_unemp": "Score Chômage (vol.)",
            "score_composite": "Score Composite (0-100)",
        })

        df["Année"] = year
        rows_pays.append(df)

        # Agrégation groupe (médiane)
        grp = (
            df.groupby("Groupe")
              .agg(**{
                  "Score PIB (vol. croissance)": ("Score PIB (vol. croissance)", "median"),
                  "Score Inflation (niv.+vol.)": ("Score Inflation (niv.+vol.)", "median"),
                  "Score Chômage (vol.)": ("Score Chômage (vol.)", "median"),
                  "Score Composite (0-100)": ("Score Composite (0-100)", "median"),
                  "nb_pays": ("Pays", "nunique"),
              })
              .reset_index()
        )

        grp["rang_groupe"] = grp["Score Composite (0-100)"].rank(method="min", ascending=False).astype(int)
        grp["classe_risque"] = categorize_scores(grp["Score Composite (0-100)"])
        grp["Année"] = year

        rows_groupes.append(grp)

    df_pays_year = pd.concat(rows_pays, axis=0).reset_index(drop=True)
    df_groupes_year = pd.concat(rows_groupes, axis=0).reset_index(drop=True)

    return df_pays_year, df_groupes_year



# ============================
# 🧵 Pipeline complet
# ============================


def compute_macro_risk_pipeline(df_input: pd.DataFrame,
                                export_excel: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pipeline de calcul des scores macroéconomiques.

    Paramètres
    ----------
    df_input : pd.DataFrame
        DataFrame au format large ou long (voir doc en tête).
    export_excel : str, optionnel
        Chemin complet du fichier .xlsx pour exporter les résultats.
        Exemple : "05-Result/scores/macro_scores.xlsx"

    Retour
    ------
    (df_pays, df_groupes)
    """
    # 🔹 Harmonisation du format
    df_long = ensure_long_format(df_input)

    # 🔹 Nettoyage basique
    df_long = df_long.dropna(subset=["Groupe", "Pays", "Indicateur"]).copy()
    df_long["Indicateur"] = df_long["Indicateur"].astype(str)

    # Harmoniser les libellés
    df_long["Indicateur"] = df_long["Indicateur"].str.replace("chomage", "Chômage", case=False)
    df_long["Indicateur"] = df_long["Indicateur"].str.replace("chômage", "Chômage", case=False)

    # 🔹 Calcul des métriques
    metrics = compute_country_metrics(df_long)

    # 🔹 Construction des scores
    df_pays, df_groupes = build_scores(metrics)

    # 🔹 Export si demandé
    if export_excel:
        folder = os.path.dirname(export_excel)
        if folder:  # si un dossier est indiqué
            os.makedirs(folder, exist_ok=True)

        with pd.ExcelWriter(export_excel, engine="xlsxwriter") as xw:
            df_pays.to_excel(xw, index=False, sheet_name="Scores_Pays")
            df_groupes.to_excel(xw, index=False, sheet_name="Scores_Groupes")

        print(f"✅ Résultats exportés vers : {export_excel}")

    return df_pays, df_groupes



def compute_macro_risk_pipeline_per_year(df_input: pd.DataFrame,
                                         export_excel: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pipeline complet de calcul macro par année
    """
    df_long = ensure_long_format(df_input)

    # Nettoyage basique
    df_long = df_long.dropna(subset=["Groupe", "Pays", "Indicateur"]).copy()
    df_long["Indicateur"] = df_long["Indicateur"].astype(str)
    df_long["Indicateur"] = df_long["Indicateur"].str.replace("chomage", "Chômage", case=False)
    df_long["Indicateur"] = df_long["Indicateur"].str.replace("chômage", "Chômage", case=False)

    # Calcul des métriques par année
    metrics_year = compute_country_metrics_per_year(df_long,
                                                    min_years=MIN_YEARS_REQUIRED,
                                                    period=PERIOD) # compute_country_metrics_per_year  compute_country_metrics

    # Construction des scores
    df_pays_year, df_groupes_year = build_scores_per_year(metrics_year)

    # Export Excel si demandé
    if export_excel:
        folder = os.path.dirname(export_excel)
        if folder:
            os.makedirs(folder, exist_ok=True)

        with pd.ExcelWriter(export_excel, engine="xlsxwriter") as xw:
            df_pays_year.to_excel(xw, index=False, sheet_name="Scores_Pays_Année")
            df_groupes_year.to_excel(xw, index=False, sheet_name="Scores_Groupes_Année")

        print(f"✅ Résultats exportés vers : {export_excel}")

    return df_pays_year, df_groupes_year


