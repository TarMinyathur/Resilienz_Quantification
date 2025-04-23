import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from matplotlib.lines import Line2D
from scipy.stats import shapiro
import statsmodels.formula.api as smf



def remove_multicollinearity_vif(X, threshold):
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


def run_regression(df_merged, indikatoren_spalten, szenarien_spalten, output_dir, Name , threshold, regression_type):
    ergebnisse_szenarien = {}
    excluded_by_scenario = {}
    df_results = pd.DataFrame()
    os.makedirs(output_dir, exist_ok=True)
    normaltest_rows = []

    for szenario in szenarien_spalten:
        X = df_merged[indikatoren_spalten].copy()
        y = df_merged[szenario].copy()

        if y.isnull().all() or y.nunique() == 1:
            print(f"Überspringe Regression für {szenario} (nur NaN oder konstante Werte).")
            continue

        # === Transformiere y für Beta-Regression ===
        if regression_type == "beta":
            eps = 1e-6
            y = (y * (len(y) - 1) + 0.5) / len(y)  # Transformation in (0,1)
            y = y.clip(eps, 1 - eps)  # Falls nach Rundung noch 0/1 entstehen

        # === Shapiro-Wilk-Test für Zielvariable (y) ===
        y_clean = y.dropna()
        if len(y_clean) >= 3:
            stat_y, p_y = shapiro(y_clean)
            print(f"\nShapiro-Wilk-Test für Zielvariable '{szenario}':")
            print(f"  Teststatistik = {stat_y:.2f}, p-Wert = {p_y:.2f}")
            if p_y < 0.05:
                print("  ⚠️ Achtung: Die Zielvariable ist möglicherweise nicht normalverteilt.")
        else:
            print(f"Zu wenige Werte für Normalverteilungstest bei '{szenario}'.")

        # === VIF-Check ===
        print(f"\n=== Stepwise VIF-Check für Szenario '{szenario}' ===")
        X_n, excluded_info = remove_multicollinearity_vif(X, threshold)
        excluded_by_scenario[szenario] = excluded_info

        if X_n.shape[1] == 0:
            print(f"Keine Indikatoren nach VIF-Filter übrig: {szenario}")
            continue

        # === Regressionsmodell je nach Typ ===
        if regression_type == "ols":
            X_ols = sm.add_constant(X_n)
            model = sm.OLS(y, X_ols).fit()
        elif regression_type == "beta":
            # Formel für GLM mit logit-Link
            data = X_n.copy()
            data['y'] = y
            data = data.rename(columns=lambda x: x.replace(" ", "_").replace("-", "_"))
            formula = 'y ~ ' + ' + '.join(col for col in data.columns if col != 'y')
            model = smf.glm(formula=formula, data=data,
                            family=sm.families.Binomial(link=sm.families.links.Logit())).fit()
        else:
            raise ValueError("Ungültiger Regressions-Typ. Bitte 'ols' oder 'beta' angeben.")

        ergebnisse_szenarien[szenario] = model

        # Plots und Summary speichern
        plot_regression(model, szenario, X_n.columns, output_dir, excluded_info, regression_type, threshold, Name)
        save_summary(model, szenario, output_dir, regression_type,threshold, Name)

        # === Shapiro-Wilk-Test für Residuen ===
        # === Residuen für OLS prüfen ===
        if regression_type == "ols":
            residuals = model.resid.dropna()
            if len(residuals) >= 3:
                stat_r, p_r = shapiro(residuals)
                print(f"Shapiro-Test für Residuen von '{szenario}': stat = {stat_r:.2f}, p = {p_r:.2f}")

        indikatoren = [col for col in indikatoren_spalten if col in model.params.index]
        params = model.params.loc[indikatoren]
        std_errors = model.bse.loc[indikatoren]
        pvalues = model.pvalues.loc[indikatoren]
        conf_int = model.conf_int().loc[indikatoren]

        if regression_type == "ols":
            X_std = (X_n - X_n.mean()) / X_n.std()
            y_std = (y - y.mean()) / y.std()
            X_std_const = sm.add_constant(X_std)
            model_std = sm.OLS(y_std, X_std_const).fit()
            standardized_betas = model_std.params.loc[indikatoren]
        else:
            standardized_betas = pd.Series(index=indikatoren, data=[np.nan] * len(indikatoren))

        params_stars_series = pd.Series(index=params.index, dtype="object")
        for idx in params.index:
            p_val = pvalues.loc[idx]
            param_val = params.loc[idx]
            if p_val < 0.001:
                star = "***"
            elif p_val < 0.01:
                star = "**"
            elif p_val < 0.05:
                star = "*"
            else:
                star = ""
            params_stars_series[idx] = f"{param_val:.2f}{star}"

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
            pvalues_stars_series[idx] = f"{p_val:.2f}{star}"

        # Stelle sicher, dass alle Regressor-Namen im Index von df_results vorhanden sind
        df_results = df_results.reindex(index=df_results.index.union(params.index))

        # Schreibe Parameterwerte und Standardfehler
        df_results.loc[params.index, f'coeff_{szenario}'] = params_stars_series
        df_results.loc[std_errors.index, f'std_error_{szenario}'] = std_errors.round(2)
        df_results.loc[standardized_betas.index, f'std_beta_{szenario}'] = standardized_betas.round(2)

        # Statistische Testwerte (t oder z) – je nach Regressions-Typ
        if regression_type == "ols":
            test_stats = model.tvalues.drop(labels=['const', 'Intercept'], errors='ignore').round(2)
            df_results.loc[test_stats.index, f't_value_{szenario}'] = test_stats
        else:
            test_stats = (params / std_errors).round(2)  # z-Werte bei GLM
            df_results.loc[test_stats.index, f'Z_value_{szenario}'] = test_stats

        # P-Werte
        df_results.loc[pvalues.index, f'P>t_{szenario}'] = pvalues.round(2)

        # Konfidenzintervall als formatierter String vorbereiten
        conf_colname = f'conf_int_{szenario}'
        conf_formatted = conf_int.loc[params.index].apply(
            lambda x: f"[{x[0]:.2f}, {x[1]:.2f}]", axis=1
        )

        # Sicherheitshalber vorbereiten (nicht zwingend, aber sauber)
        if conf_colname not in df_results.columns:
            df_results[conf_colname] = np.nan

        # Zuweisung der Konfidenzintervalle
        df_results.loc[params.index, conf_colname] = conf_formatted

        if 'df_model_metrics' not in locals():
            df_model_metrics = pd.DataFrame(columns=[
                "R²", "Adjusted R²", "F-Statistic", "F-Test p-value", "AIC", "BIC",
                "Log-Likelihood", "Number of Observations"
            ])

        if regression_type == "ols":
            r_squared = round(model.rsquared, 2)
            adj_r_squared = round(model.rsquared_adj, 2)
            f_stat = round(model.fvalue, 2)
            f_pval = round(model.f_pvalue, 2)
        else:
            r_squared = adj_r_squared = f_stat = f_pval = np.nan

        df_model_metrics.loc[szenario] = {
            "R²": r_squared,
            "Adjusted R²": adj_r_squared,
            "F-Statistic": f_stat,
            "F-Test p-value": f_pval,
            "AIC": round(model.aic, 2),
            "BIC": round(model.bic, 2),
            "Log-Likelihood": round(model.llf, 2),
            "Number of Observations": int(model.nobs)
        }


    # === Shapiro-Wilk-Tests für Zielvariable & Residuen (nach der Regression) ===
    normaltest_rows = []

    for szenario in ergebnisse_szenarien:
        model = ergebnisse_szenarien[szenario]
        y_vals = df_merged[szenario].dropna()

        # Zielvariable testen
        if len(y_vals) >= 3:
            stat_y, p_y = shapiro(y_vals)
            comment_y = "nicht normalverteilt" if p_y < 0.05 else "normalverteilt"
            normaltest_rows.append({
                "Szenario": szenario,
                "Testtyp": "Zielvariable",
                "Shapiro-Statistik": round(stat_y, 2),
                "p-Wert": round(p_y, 2),
                "Interpretation": comment_y
            })
        else:
            normaltest_rows.append({
                "Szenario": szenario,
                "Testtyp": "Zielvariable",
                "Shapiro-Statistik": None,
                "p-Wert": None,
                "Interpretation": "nicht geprüft (zu wenig Werte)"
            })

        # Nur bei OLS: Residuen testen
        if regression_type == "ols":
            try:
                resid_vals = model.resid.dropna()
                if len(resid_vals) >= 3:
                    stat_r, p_r = shapiro(resid_vals)
                    comment_r = "nicht normalverteilt" if p_r < 0.05 else "normalverteilt"
                else:
                    stat_r, p_r = None, None
                    comment_r = "nicht geprüft (zu wenig Werte)"
            except AttributeError:
                stat_r, p_r = None, None
                comment_r = "nicht getestet (nicht verfügbar für dieses Modell)"

            normaltest_rows.append({
                "Szenario": szenario,
                "Testtyp": "Residuen",
                "Shapiro-Statistik": stat_r if stat_r is None else round(stat_r, 2),
                "p-Wert": p_r if p_r is None else round(p_r, 2),
                "Interpretation": comment_r
            })

    df_normaltests = pd.DataFrame(normaltest_rows)

    # Excel Export am Ende
    export_path = os.path.join(output_dir, f"regression_results_{Name}_{regression_type}_{threshold}.xlsx")
    with pd.ExcelWriter(export_path, engine='openpyxl', mode='w') as writer:
        df_results.to_excel(writer, sheet_name='Regressionsdetails')

        # Transponieren für bessere Lesbarkeit (jede Zeile = Szenario)

        df_model_metrics.to_excel(writer, sheet_name='Modellmetriken')

        df_normaltests.to_excel(writer, sheet_name='Normalverteilungstests', index=False)

        # === Neues Sheet: Ausgeschlossene Indikatoren ===
        excluded_rows = []
        for szenario, ex_list in excluded_by_scenario.items():
            for col, vif in ex_list:
                excluded_rows.append({
                    "Indikator": col,
                    "VIF-Wert": round(vif, 2)
                })

        df_excluded = pd.DataFrame(excluded_rows)
        df_excluded.to_excel(writer, sheet_name='Ausgeschlossene Indikatoren', index=False)

    print(f"Gesamtergebnis-DataFrame wurde als Excel exportiert: {export_path}")
    return ergebnisse_szenarien


def plot_regression(models, szenario, indikatoren_spalten, output_dir, excluded_info, regression_type, threshold, Name):

    # Formatierter Regressionstyp für Beschriftungen
    reg_label = "Linear Regression" if regression_type == "ols" else "Beta-Regression"
    reg_suffix = regression_type.lower()

    params_score = models.params
    conf_score = models.conf_int()
    p_values = models.pvalues

    # Entferne den Intercept (const)
    indikator_namen = [col for col in params_score.index if col.lower() not in ['const', 'intercept']]
    params_score_no_intercept = params_score[indikator_namen]
    conf_score_no_intercept = conf_score.loc[indikator_namen]
    p_values_no_intercept = p_values[indikator_namen]

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

    plt.title(f"{reg_label}: Coefficients and 95% Confidence Intervals: – {szenario}")
    plt.xlabel("Indicators", labelpad=10)
    plt.gca().xaxis.set_label_coords(0.5, 0.05)
    plt.ylabel("Regression Coefficient")

    # Legende erstellen
    positive_marker = Line2D([0], [0], marker='x', color='green', linestyle='None', markersize=10, label='Positive Coefficients')
    negative_marker = Line2D([0], [0], marker='x', color='red', linestyle='None', markersize=10, label='Negative Coefficients')
    significance_note = Line2D([0], [0], marker='None', color='none', label='* = significance (p < 0.05)')
    plt.legend(handles=[positive_marker, negative_marker, significance_note],
               loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=3, frameon=False)

    plt.subplots_adjust(bottom=0.3)

    plot_path = os.path.join(output_dir, f"regression_{Name}_{reg_suffix}_{szenario}_{threshold}.png")
    plt.savefig(plot_path, dpi=400)
    plt.close()

    print(f"Plot für {szenario} gespeichert unter: {plot_path}")


    # === Q-Q-Plot für Residuen (nur wenn vorhanden) ===
    try:
        resid = models.resid.dropna()
        plt.figure(figsize=(6, 6))
        sm.qqplot(resid, line='45', fit=True, markersize=4)
        plt.title(f"Q-Q-Plot der Residuen – {szenario}")
        plt.grid(True)
        plt.tight_layout()
        qq_path = os.path.join(output_dir, f"qqplot_resid_{szenario}.png")
        plt.savefig(qq_path, dpi=300)

        plt.close()
        print(f"Q-Q-Plot gespeichert unter: {qq_path}")

    except AttributeError:
        print(f"Keine Residuen verfügbar für Q-Q-Plot bei {szenario} ({reg_label})")


def save_summary(models, szenario, output_dir,regression_type,threshold, Name):
    with open(os.path.join(output_dir, f"regression_summary_{Name}_{szenario}{regression_type}{threshold}.txt"), "w") as f:
        f.write(models.summary().as_text())


# --------------------------
# Main-Workflow
# --------------------------

def main():
    # Pfad zur Excel-Datei, welche beide Arbeitsblätter enthält
    #Ergebnisse_final_EP_min_Netze_AVG
    #Ergebnisse_final_EP_min_Netze_min_Indi
    excel_file = r"C:\Users\runte\Dropbox\Zwischenablage\Regression_Plots\Ergebnisse_final_EP_min_Netze_AVG.xlsx"

    # Output-Verzeichnis
    output_dir = r"C:\Users\runte\Dropbox\Zwischenablage\Regression_Plots"

    # Lese die beiden Arbeitsblätter in separate DataFrames
    df_indikatoren = pd.read_excel(excel_file, sheet_name="Indikatoren_final")
    df_szenarien = pd.read_excel(excel_file, sheet_name="Stressoren_final")

    # ---- Spalte in df_szenarien umbenennen ----
    umzubenennende_spalte = "high_EE_generation"   # <- Hier alten Namen eintragen
    neuer_spaltenname = "IT Attack on Renewables"       # <- Hier neuen Namen eintragen

    if umzubenennende_spalte in df_szenarien.columns:
        df_szenarien = df_szenarien.rename(columns={umzubenennende_spalte: neuer_spaltenname})
    else:
        print(f"Warnung: Spalte '{umzubenennende_spalte}' nicht in Stressoren_final gefunden.")

    # Liste der zu ignorierenden Indikatoren
    zu_ignorierende_indikatoren = ["Time required","Flexibility Average", "Flexibility Feasible Operating Region scaled", "Redundancy Average", "Redundancy N-3", "Self Sufficiency At Bus Level", "Self Sufficiency System"]  # <- Hier anpassen
    #zu_ignorierende_indikatoren = ["Time required"]  # <- Hier anpassen

    Name = "ols_AVG_4"

    df_merged = preprocess_data(df_indikatoren, df_szenarien)

    # Erstelle die Liste der zu verwendenden Indikatoren (alle außer "Netz" und ignorierte)
    indikatoren_spalten = [
        col for col in df_indikatoren.columns
        if col != "Netz" and col not in zu_ignorierende_indikatoren
    ]

    szenarien_spalten = [col for col in df_szenarien.columns if col != "Netz"]

    run_regression(df_merged, indikatoren_spalten, szenarien_spalten, output_dir, Name, threshold=4, regression_type="ols")


if __name__ == "__main__":
    main()
