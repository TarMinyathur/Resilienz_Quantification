import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import numpy as np

def lade_daten(indikatoren_path, szenarien_path):
    df_indikatoren = pd.read_excel(indikatoren_path)
    df_szenarien = pd.read_excel(szenarien_path)
    return df_indikatoren, df_szenarien

def preprocess_data(df_indikatoren, df_szenarien):
    df_merged = pd.merge(df_indikatoren, df_szenarien, on="Netz", how="inner")
    df_merged.replace(["", " "], pd.NA, inplace=True)
    for col in df_merged.columns:
        if col != "Netz":
            df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')
    if df_merged.isnull().values.any():
        print("Warnung: Fehlende Werte gefunden. Ersetze mit Spaltenmittelwerten.")
        df_merged.fillna(df_merged.mean(numeric_only=True), inplace=True)
    return df_merged

def run_regression(df_merged, indikatoren_spalten, szenarien_spalten, output_dir):
    ergebnisse_szenarien = {}

    os.makedirs(output_dir, exist_ok=True)

    for szenario in szenarien_spalten:
        X = df_merged[indikatoren_spalten]
        y = df_merged[szenario]

        if y.isnull().all():
            print(f"Überspringe Regression für {szenario} (nur NaN-Werte).")
            continue
        if y.nunique() == 1:
            print(f"Überspringe Regression für {szenario} (nur konstante Werte).")
            continue

        X_ols = sm.add_constant(X)
        model_ols = sm.OLS(y, X_ols).fit()
        ergebnisse_szenarien[szenario] = model_ols

        print("=" * 70)
        print(f"Regressionsergebnisse für Szenario: {szenario}")
        print(model_ols.summary())

        plot_regression(model_ols, szenario, indikatoren_spalten, output_dir)
        save_summary(model_ols, szenario, output_dir)

    return ergebnisse_szenarien

def plot_regression(model_ols, szenario, indikatoren_spalten, output_dir):
    params_score = model_ols.params
    conf_score = model_ols.conf_int()
    p_values = model_ols.pvalues
    params_score_no_intercept = params_score.drop('const')
    conf_score_no_intercept = conf_score.drop('const')
    p_values_no_intercept = p_values.drop('const')

    ind_names = params_score_no_intercept.index
    coef_vals = params_score_no_intercept.values
    lower_error = coef_vals - conf_score_no_intercept[0]
    upper_error = conf_score_no_intercept[1] - coef_vals

    # Sternchen + Farben für signifikante Koeffizienten
    ind_labels = []
    colors = []
    for ind, val, pval in zip(ind_names, coef_vals, p_values_no_intercept):
        label = f"{ind}*" if pval < 0.05 else ind
        ind_labels.append(label)
        colors.append('green' if val > 0 else 'red')

    # Custom X-Positionen mit mehr Abstand
    x_pos = np.arange(0, len(ind_names) * 2, 2)  # z.B. Abstand von 2 Einheiten statt 1

    plt.figure(figsize=(12,8))
    plt.bar(x_pos, coef_vals,
            yerr=[lower_error, upper_error],
            capsize=5, color=colors)

        # X-Achse hinzufügen
    plt.xticks(x_pos, ind_labels, rotation=45, ha="right")
    plt.axhline(y=0, color='grey', linewidth=0.5, linestyle="-")  # Horizontale Linie bei y=0 als Referenz

    plt.title(f"Koeffizienten und 95%-Konfidenz-Intervall: {szenario}")
    plt.xlabel("Indikatoren", labelpad=10)
    plt.gca().xaxis.set_label_coords(0.5, 0.05)  # x=0.5 zentriert, y negativ = in Plot verschieben
    plt.ylabel("Regressionskoeffizient")

    # Legende hinzufügen
    legend_patches = [
        mpatches.Patch(color='green', label='Positiver Koeffizient'),
        mpatches.Patch(color='red', label='Negativer Koeffizient'),
        mpatches.Patch(color='white', label='* = signifikant bei p < 0.05', edgecolor='black')
    ]
    plt.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.5),
               ncol=3, frameon=False)

    # Fix: mehr Platz unten reservieren!
    plt.subplots_adjust(bottom=0.3)

    #plt.tight_layout()

    plot_path = os.path.join(output_dir, f"regression_{szenario}.png")
    plt.savefig(plot_path, dpi=400)
    plt.close()
    print(f"Plot für {szenario} gespeichert unter: {plot_path}")

def save_summary(model_ols, szenario, output_dir):
    with open(os.path.join(output_dir, f"regression_summary_{szenario}.txt"), "w") as f:
        f.write(model_ols.summary().as_text())

# --------------------------
# Main-Workflow
# --------------------------

def main():
    indikatoren_path =
    szenarien_path =
    output_dir =

    df_indikatoren, df_szenarien = lade_daten(indikatoren_path, szenarien_path)
    df_merged = preprocess_data(df_indikatoren, df_szenarien)

    indikatoren_spalten = [col for col in df_indikatoren.columns if col != "Netz"]
    szenarien_spalten = [col for col in df_szenarien.columns if col != "Netz"]

    run_regression(df_merged, indikatoren_spalten, szenarien_spalten, output_dir)

if __name__ == "__main__":
    main()
