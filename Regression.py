import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ----------------------------
# 1) Excel-Dateien einlesen
# ----------------------------
df_indikatoren = pd.read_excel(r"C:\Users\runte\Dropbox\Zwischenablage\Ergebnisse_Indikatoren_final.xlsx")
df_szenarien = pd.read_excel(r"C:\Users\runte\Dropbox\Zwischenablage\Ergebnisse_Stressoren_final.xlsx")

# ----------------------------------------------
# 2) Mergen über gemeinsame Spalte "Netz"
# ----------------------------------------------
df_merged = pd.merge(df_indikatoren, df_szenarien, on="Netz", how="inner")

# Leere Strings oder fehlerhafte Werte als NaN setzen
df_merged.replace(["", " "], pd.NA, inplace=True)

# Alle Spalten außer "Netz" in numerische Werte umwandeln
for col in df_merged.columns:
    if col != "Netz":
        df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')  # Ungültige Werte zu NaN umwandeln

# Fehlende Werte mit Spaltenmittelwert auffüllen
if df_merged.isnull().values.any():
    print("Warnung: Fehlende Werte gefunden. Ersetze mit Spaltenmittelwerten.")
    df_merged.fillna(df_merged.mean(numeric_only=True), inplace=True)

# Indikatoren- und Szenarien-Spalten identifizieren
indikatoren_spalten = [col for col in df_indikatoren.columns if col != "Netz"]
szenarien_spalten = [col for col in df_szenarien.columns if col != "Netz"]

# -----------------------------------
# 3) Regression für jedes Szenario
# -----------------------------------
ergebnisse_szenarien = {}

for szenario in szenarien_spalten:
    X = df_merged[indikatoren_spalten]  # Indikatoren als unabhängige Variablen
    y = df_merged[szenario]  # Szenario als abhängige Variable

    # Falls Szenario nur NaN oder konstante Werte enthält, Regression überspringen
    if y.isnull().all():
        print(f"Überspringe Regression für {szenario} (nur NaN-Werte).")
        continue
    if y.nunique() == 1:
        print(f"Überspringe Regression für {szenario} (nur konstante Werte).")
        continue

    # Konstanten-Term (Intercept) hinzufügen
    X_ols = sm.add_constant(X)

    # OLS-Modell anpassen
    model_ols = sm.OLS(y, X_ols).fit()

    # Ergebnisse speichern
    ergebnisse_szenarien[szenario] = model_ols

    # Konsolenausgabe der Ergebnisse
    print("=" * 70)
    print(f"Regressionsergebnisse für Szenario: {szenario}")
    print(model_ols.summary())

# ------------------------------------------------------------------------
# 4) Aggregierter Szenario-Score berechnen
# ------------------------------------------------------------------------
df_merged["Gesamt_Szenario_Score"] = df_merged[szenarien_spalten].mean(axis=1)

# Regression für den aggregierten Score
X_score = df_merged[indikatoren_spalten]
y_score = df_merged["Gesamt_Szenario_Score"]

# Falls aggregierter Score NaN oder konstant ist, Regression überspringen
if y_score.isnull().all():
    print("Überspringe Regression für aggregierten Szenario-Score (nur NaN-Werte).")
elif y_score.nunique() == 1:
    print("Überspringe Regression für aggregierten Szenario-Score (nur konstante Werte).")
else:
    X_score_ols = sm.add_constant(X_score)
    model_score = sm.OLS(y_score, X_score_ols).fit()
    print("=" * 70)
    print("Regressionsergebnisse für aggregierten Szenario-Score")
    print(model_score.summary())
