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

    # DataFrame zur Sammlung der Regressionsergebnisse
    df_results = pd.DataFrame(index=indikatoren_spalten)

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

        # Ergebnisse extrahieren
        params = model_ols.params.drop('const')
        std_errors = model_ols.bse.drop('const')
        pvalues = model_ols.pvalues.drop('const')
        conf_int = model_ols.conf_int().drop('const')

        # Sternchen-Logik für wissenschaftliche Notation
        params_stars = []
        pvalues_stars = []

        for param, pval in zip(params, pvalues):
            if pval < 0.001:
                star = "***"
            elif pval < 0.01:
                star = "**"
            elif pval < 0.05:
                star = "*"
            else:
                star = ""
            params_stars.append(f"{param:.4f}{star}")
            pvalues_stars.append(f"{pval:.4f}{star}")

        # DataFrame füllen
        df_results[f'coeff_{szenario}'] = params_stars
        df_results[f'std_error_{szenario}'] = std_errors.round(4)
        df_results[f'P>t_{szenario}'] = pvalues_stars
        df_results[f'conf_int_{szenario}'] = conf_int.apply(lambda row: f"[{row[0]:.3f}, {row[1]:.3f}]", axis=1)

        # Plots und Summary speichern
        plot_regression(model_ols, szenario, indikatoren_spalten, output_dir)
        save_summary(model_ols, szenario, output_dir)

    # Excel Export am Ende
    export_path = os.path.join(output_dir, "regression_results.xlsx")
    df_results.to_excel(export_path)
    print(f"Gesamtergebnis-DataFrame wurde als Excel exportiert: {export_path}")

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

    plt.title(f"Coefficients and 95% Confidence Interval: {szenario}")
    plt.xlabel("Indicators", labelpad=10)
    plt.gca().xaxis.set_label_coords(0.5, 0.05)  # x=0.5 centers it, y positive moves it inside the plot
    plt.ylabel("Regression Coefficient")

    # Add legend
    legend_patches = [
        mpatches.Patch(color='green', label='Positive Coefficient'),
        mpatches.Patch(color='red', label='Negative Coefficient'),
        mpatches.Patch(facecolor='white', label='* = significant at p < 0.05')
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
    # Path to the single Excel file containing both sheets
    excel_file = r"C:\Users\runte\Dropbox\Zwischenablage\Regression_Plots\Ergebnisse_final.xlsx"

    # Define the output directory (unchanged)
    output_dir = r"C:\Users\runte\Dropbox\Zwischenablage\Regression_Plots"

    # Read each worksheet into a separate DataFrame
    df_indikatoren = pd.read_excel(excel_file, sheet_name="Indikatoren_final")
    df_szenarien = pd.read_excel(excel_file, sheet_name="Stressoren_final")

    #df_indikatoren, df_szenarien = lade_daten(indikatoren_path, szenarien_path)
    df_merged = preprocess_data(df_indikatoren, df_szenarien)

    indikatoren_spalten = [col for col in df_indikatoren.columns if col != "Netz"]
    szenarien_spalten = [col for col in df_szenarien.columns if col != "Netz"]

    run_regression(df_merged, indikatoren_spalten, szenarien_spalten, output_dir)

if __name__ == "__main__":
    main()
