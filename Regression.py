import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ----------------------------
# 1) Excel-Dateien einlesen
# ----------------------------
df_indikatoren = pd.read_excel("C:\Users\runte\Dropbox\Zwischenablage\Ergebnisse_Indikatoren_final.xlsx")
df_stress = pd.read_excel("C:\Users\runte\Dropbox\Zwischenablage\Ergebnisse_Stressoren_final.xlsx")

# Beispielhafte Struktur:
# df_indikatoren: Spalten = ["Netz", "Ind1", "Ind2", "Ind3", ...]
# df_stress:      Spalten = ["Netz", "Stress1", "Stress2", "Stress3", ...]

# ----------------------------------------------
# 2) Mergen über gemeinsame Spalte "Netz"
# ----------------------------------------------
df_merged = pd.merge(df_indikatoren, df_stress, on="Netz", how="inner")

# Identifiziere Indikator-Spalten und Stressor-Spalten
indikator_spalten = [col for col in df_indikatoren.columns if col != "Netz"]
stress_spalten = [col for col in df_stress.columns if col != "Netz"]

# -----------------------------------
# 3) Regression pro Stress-Szenario
# -----------------------------------
# Dictionary zum Speichern der Ergebnisse
ergebnisse_stress = {}

for stress_col in stress_spalten:
    # Unabhängige Variablen (X) und abhängige Variable (y)
    X = df_merged[indikator_spalten]
    y = df_merged[stress_col]

    # Konstanten-Term (Intercept) hinzufügen (statsmodels macht das nicht automatisch)
    X_ols = sm.add_constant(X)

    # OLS-Modell anpassen
    model_ols = sm.OLS(y, X_ols).fit()

    # In einem Dictionary speichern, falls wir später nochmal darauf zugreifen wollen
    ergebnisse_stress[stress_col] = model_ols

    # Konsolenausgabe: Modell-Zusammenfassung (mit p-Werten, t-Tests etc.)
    print("=" * 70)
    print(f"Regressionsergebnisse für Stress-Szenario: {stress_col}")
    print(model_ols.summary())

    # 3a) Visualisierung: Koeffizienten + Konfidenzintervalle
    # -------------------------------------------------------
    params = model_ols.params  # Geschätzte Koeffizienten (inkl. Intercept)
    conf = model_ols.conf_int()  # Konfidenzintervalle
    # conf ist DataFrame mit zwei Spalten (unteres / oberes KI)

    # Für das Balkendiagramm lassen wir den Intercept weg (optional),
    # da er oft nicht so relevant für die Interpretation ist.
    # Du kannst ihn natürlich auch drin lassen.
    params_no_intercept = params.drop('const')
    conf_no_intercept = conf.drop('const')

    # Index (Indikatoren-Namen) und Werte
    ind_names = params_no_intercept.index
    coef_vals = params_no_intercept.values

    # KI-Abstände (nach unten und oben)
    # z.B. yerr erfordert "Fehler" in beide Richtungen
    lower_error = coef_vals - conf_no_intercept[0]
    upper_error = conf_no_intercept[1] - coef_vals

    # Erstelle Balkendiagramm
    plt.figure()  # Neues Figure-Fenster
    plt.bar(range(len(ind_names)), coef_vals,
            yerr=[lower_error, upper_error],
            capsize=5)
    plt.xticks(range(len(ind_names)), ind_names, rotation=45, ha="right")
    plt.title(f"Koeffizienten und 95%-KI: {stress_col}")
    plt.xlabel("Indikatoren")
    plt.ylabel("Regressionskoeffizient")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------
# 4) Aggregierter Score (Mittelwert) über alle Stress-Szenarien bilden
# ------------------------------------------------------------------------
df_merged["Score_alle_Stressoren"] = df_merged[stress_spalten].mean(axis=1)

# Regression für den neuen Score
X_score = df_merged[indikator_spalten]
y_score = df_merged["Score_alle_Stressoren"]

X_score_ols = sm.add_constant(X_score)
model_score = sm.OLS(y_score, X_score_ols).fit()

print("=" * 70)
print("Regressionsergebnisse für aggregierten Score über alle Stress-Szenarien")
print(model_score.summary())

# 4a) Visualisierung (Bar-Plot für Score)
params_score = model_score.params
conf_score = model_score.conf_int()

params_score_no_intercept = params_score.drop('const')
conf_score_no_intercept = conf_score.drop('const')

ind_names = params_score_no_intercept.index
coef_vals = params_score_no_intercept.values

lower_error = coef_vals - conf_score_no_intercept[0]
upper_error = conf_score_no_intercept[1] - coef_vals

plt.figure()
plt.bar(range(len(ind_names)), coef_vals,
        yerr=[lower_error, upper_error],
        capsize=5)
plt.xticks(range(len(ind_names)), ind_names, rotation=45, ha="right")
plt.title("Koeffizienten und 95%-KI: Aggregierter Score")
plt.xlabel("Indikatoren")
plt.ylabel("Regressionskoeffizient")
plt.tight_layout()
plt.show()
