import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from matplotlib.lines import Line2D

def remove_multicollinearity_vif(X, threshold=10.0):
    """
    Entfernt iterativ Variablen mit hohem VIF, bis alle Variablen
    unter dem angegebenen 'threshold' liegen.
    """
    X_reduced = X.copy()
    excluded_columns = []  # Hier speichern wir (Spaltenname, VIF) der entfernten Spalten

    while True:
        # Schritt 1: VIF für jede Spalte berechnen
        vif_values = []
        for i in range(X_reduced.shape[1]):
            val = variance_inflation_factor(X_reduced.values, i)
            vif_values.append(val)

        vif_series = pd.Series(vif_values, index=X_reduced.columns)
        max_vif = vif_series.max()

        # Schritt 2: Prüfen, ob größter VIF über dem Schwellwert liegt
        if max_vif > threshold:
            drop_col = vif_series.idxmax()
            print(f"  -> Entferne '{drop_col}' wegen VIF={max_vif:.2f} (Threshold={threshold})")
            excluded_columns.append((drop_col, max_vif))  # speichern
            X_reduced.drop(columns=[drop_col], inplace=True)

            # Falls wir auf 1 Spalte oder weniger runterfallen, brechen wir ab
            if X_reduced.shape[1] <= 1:
                break
        else:
            # Alle VIFs < threshold: Abbruch
            break

    # Geben X_reduced und die Liste der entfernten Spalten zurück
    return X_reduced, excluded_columns


def lade_daten(indikatoren_path, szenarien_path):
    df_indikatoren = pd.read_excel(indikatoren_path)
    df_szenarien = pd.read_excel(szenarien_path)
    return df_indikatoren, df_szenarien


def preprocess_data(df_indikatoren, df_szenarien):
    df_filtered_szenario = df_szenarien[~(df_szenarien.drop(columns="Netz").fillna(0) == 0).all(axis=1)]
    # Die Angabe how="inner" bei pd.merge(...). Ein Inner Join behält nur Zeilen mit übereinstimmendem Schlüssel in beiden DataFrames.
    df_merged = pd.merge(df_indikatoren, df_filtered_szenario, on="Netz", how="inner", suffixes=("", "_szenario"))
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
    excluded_by_scenario = {}  # neues Dict: Szenarioname → Liste (Spalte, VIF)

    # Leeres DataFrame, das später dynamisch wächst
    df_results = pd.DataFrame()

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

        print(f"\n=== Stepwise VIF-Check für Szenario '{szenario}' ===")
        X_n, excluded_info = remove_multicollinearity_vif(X, threshold=10.0)
        excluded_by_scenario[szenario] = excluded_info

        # Falls nach VIF-Filter keine Spalten übrig sind, Szenario überspringen
        if X_n.shape[1] == 0:
            print(f"Keine Indikatoren mehr übrig nach VIF-Filter. Szenario '{szenario}' wird übersprungen.")
            continue

        X_ols = sm.add_constant(X_n)
        model_ols = sm.OLS(y, X_ols).fit()
        ergebnisse_szenarien[szenario] = model_ols

        print("=" * 70)
        print(f"Regressionsergebnisse für Szenario: {szenario}")
        print(model_ols.summary())

        # Ergebnisse extrahieren (ohne 'const')
        params = model_ols.params.drop('const', errors='ignore')
        std_errors = model_ols.bse.drop('const', errors='ignore')
        pvalues = model_ols.pvalues.drop('const', errors='ignore')
        conf_int = model_ols.conf_int().drop('const', errors='ignore')

        # --- 1) Series für Koeffizienten mit Sternchen ---
        params_stars_series = pd.Series(index=params.index, dtype="object")
        for idx in params.index:
            param_val = params.loc[idx]
            p_val = pvalues.loc[idx]
            if p_val < 0.001:
                star = "***"
            elif p_val < 0.01:
                star = "**"
            elif p_val < 0.05:
                star = "*"
            else:
                star = ""
            params_stars_series[idx] = f"{param_val:.4f}{star}"

        # --- 2) Series für P-Werte mit Sternchen (optional, selbes Prinzip) ---
        pvalues_stars_series = pd.Series(index=pvalues.index, dtype="object")
        for idx in pvalues.index:
            p_val = pvalues.loc[idx]
            if p_val < 0.001:
                star = "***"
            elif p_val < 0.01:
                star = "**"
            elif p_val < 0.05:
                star = "*"
            else:
                star = ""
            pvalues_stars_series[idx] = f"{p_val:.4f}{star}"

        # --- 3) Series für Konfidenzintervalle (als String) ---
        conf_str_series = pd.Series(index=conf_int.index, dtype="object")
        for idx in conf_int.index:
            ci_low, ci_high = conf_int.loc[idx]
            conf_str_series[idx] = f"[{ci_low:.3f}, {ci_high:.3f}]"

        # --- 4) Series für t-Werte ---
        tvalues_series = model_ols.tvalues.drop('const', errors='ignore').round(4)

        # Stelle sicher, dass alle Index-Zeilen vorhanden sind
        df_results = df_results.reindex(index=df_results.index.union(params.index))

        # ... und dann die noch vorhandenen Variablen bestücken:
        df_results.loc[params.index, f'coeff_{szenario}'] = params_stars_series
        df_results.loc[std_errors.index, f'std_error_{szenario}'] = std_errors.round(4)
        df_results.loc[tvalues_series.index, f't_value_{szenario}'] = tvalues_series
        df_results.loc[pvalues.index, f'P>t_{szenario}'] = pvalues_stars_series
        df_results.loc[conf_int.index, f'conf_int_{szenario}'] = conf_str_series

        # Plots und Summary speichern
        plot_regression(model_ols, szenario, X_n.columns, output_dir, excluded_info)
        save_summary(model_ols, szenario, output_dir)

        # === Modellmetriken in separatem DataFrame sammeln ===
        if 'df_model_metrics' not in locals():
            df_model_metrics = pd.DataFrame(columns=[
                "R² (Modellgüte)", "Adj. R²", "F-Statistik", "F-Test p-Wert",
                "AIC", "BIC", "Log-Likelihood", "Anzahl Beobachtungen"
            ])

        df_model_metrics.loc[szenario] = {
            "R² (Modellgüte)": round(model_ols.rsquared, 4),
            "Adj. R²": round(model_ols.rsquared_adj, 4),
            "F-Statistik": round(model_ols.fvalue, 4),
            "F-Test p-Wert": round(model_ols.f_pvalue, 4),
            "AIC": round(model_ols.aic, 4),
            "BIC": round(model_ols.bic, 4),
            "Log-Likelihood": round(model_ols.llf, 4),
            "Anzahl Beobachtungen": int(model_ols.nobs)
        }

    # Ausgabe der zusammengefassten Ergebnisse in der Konsole
    print("\nTabellarische Zusammenfassung der Ergebnisse:")
    print(df_results.to_string())

    # Excel Export am Ende
    export_path = os.path.join(output_dir, "regression_results_min_less_indi.xlsx")
    with pd.ExcelWriter(export_path, engine='openpyxl', mode='w') as writer:
        df_results.to_excel(writer, sheet_name='Regressionsdetails')

        # Transponieren für bessere Lesbarkeit (jede Zeile = Szenario)

        df_model_metrics.to_excel(writer, sheet_name='Modellmetriken')

        # === Neues Sheet: Ausgeschlossene Indikatoren ===
        excluded_rows = []
        for szenario, ex_list in excluded_by_scenario.items():
            for col, vif in ex_list:
                excluded_rows.append({
                    "Indikator": col,
                    "VIF-Wert": round(vif, 4)
                })

        df_excluded = pd.DataFrame(excluded_rows)
        df_excluded.to_excel(writer, sheet_name='Ausgeschlossene Indikatoren', index=False)

    print(f"Gesamtergebnis-DataFrame wurde als Excel exportiert: {export_path}")
    return ergebnisse_szenarien


def plot_regression(model_ols, szenario, indikatoren_spalten, output_dir, excluded_info):
    params_score = model_ols.params
    conf_score = model_ols.conf_int()
    p_values = model_ols.pvalues

    # Entferne den Intercept (const)
    params_score_no_intercept = params_score.drop('const')
    conf_score_no_intercept = conf_score.drop('const')
    p_values_no_intercept = p_values.drop('const')

    ind_names = params_score_no_intercept.index
    coef_vals = params_score_no_intercept.values
    lower_error = coef_vals - conf_score_no_intercept[0].values
    upper_error = conf_score_no_intercept[1].values - coef_vals

    # Erstelle Labels: Füge bei signifikanter Variable (*) hinzu
    ind_labels = []
    for ind, pval in zip(ind_names, p_values_no_intercept):
        label = f"{ind}*" if pval < 0.05 else ind
        ind_labels.append(label)

    # Custom X-Positionen mit mehr Abstand
    x_pos = np.arange(0, len(ind_names) * 2, 2)  # z.B. Abstand von 2 Einheiten statt 1

    plt.figure(figsize=(12, 8))
    # Statt Balken zeichnen wir einzelne Punkte (x) mit Fehlerbalken
    for i, (x, coef, low_err, up_err, pval) in enumerate(zip(x_pos, coef_vals, lower_error, upper_error, p_values_no_intercept)):
        color = 'green' if coef > 0 else 'red'
        plt.errorbar(x, coef, yerr=[[low_err], [up_err]], fmt='x', markersize=10, color=color, capsize=5)

    plt.xticks(x_pos, ind_labels, rotation=45, ha="right")
    plt.axhline(y=0, color='grey', linewidth=0.5, linestyle="-")  # Horizontale Linie bei y=0 als Referenz

    plt.title(f"Regression Coefficients and 95% Confidence Intervals: {szenario}")
    plt.xlabel("Indicators", labelpad=10)
    plt.gca().xaxis.set_label_coords(0.5, 0.05)
    plt.ylabel("Regression Coefficient")

    # Legende erstellen
    positive_marker = Line2D([0], [0], marker='x', color='green', linestyle='None', markersize=10, label='Positive Coefficients')
    negative_marker = Line2D([0], [0], marker='x', color='red', linestyle='None', markersize=10, label='Negative Coefficients')
    significance_note = Line2D([0], [0], marker='None', color='none', label='* = significance (p < 0.05)')
    plt.legend(handles=[positive_marker, negative_marker, significance_note],
               loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=3, frameon=False)

    # # Platz für ausgeschlossene Variablen (VIF-Info)
    # if len(excluded_info) > 0:
    #     box_text = "Excluded (VIF > 10):\n"
    #     for col_name, vif_val in excluded_info:
    #         box_text += f"  - {col_name}: VIF={vif_val:.2f}\n"
    # else:
    #     box_text = "Keine Indikatoren ausgeschlossen (alle VIF <= 10)"
    # props = dict(boxstyle='round', facecolor='white', alpha=0.8, ec='black')
    # plt.gca().text(1.02, 0.95, box_text, transform=plt.gca().transAxes,
    #                verticalalignment='top', fontsize=9, bbox=props)
    #
    plt.subplots_adjust(bottom=0.3)

    plot_path = os.path.join(output_dir, f"regression_less_indi_{szenario}.png")
    plt.savefig(plot_path, dpi=400)
    plt.close()
    print(f"Plot für {szenario} gespeichert unter: {plot_path}")


def save_summary(model_ols, szenario, output_dir):
    with open(os.path.join(output_dir, f"regression_summary_less_indi_{szenario}.txt"), "w") as f:
        f.write(model_ols.summary().as_text())


# --------------------------
# Main-Workflow
# --------------------------

def main():
    # Pfad zur Excel-Datei, welche beide Arbeitsblätter enthält
    excel_file = r"C:\Users\runte\Dropbox\Zwischenablage\Regression_Plots\Ergebnisse_final_EP_min_less_indi.xlsx"

    # Output-Verzeichnis
    output_dir = r"C:\Users\runte\Dropbox\Zwischenablage\Regression_Plots"

    # Lese die beiden Arbeitsblätter in separate DataFrames
    df_indikatoren = pd.read_excel(excel_file, sheet_name="Indikatoren_final")
    df_szenarien = pd.read_excel(excel_file, sheet_name="Stressoren_final")

    df_merged = preprocess_data(df_indikatoren, df_szenarien)

    indikatoren_spalten = [col for col in df_indikatoren.columns if col != "Netz"]
    szenarien_spalten = [col for col in df_szenarien.columns if col != "Netz"]

    run_regression(df_merged, indikatoren_spalten, szenarien_spalten, output_dir)


if __name__ == "__main__":
    main()
