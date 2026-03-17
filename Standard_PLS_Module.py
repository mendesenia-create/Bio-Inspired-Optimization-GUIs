import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import scipy.io as sio
import pandas as pd
from scipy.stats import f as f_dist, linregress
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.patches import Ellipse

# =============================================================================
#  CLASSE PRINCIPAL
# =============================================================================

class Standard_PLS_App:
    def __init__(self, root_master=None):
        if root_master:
            self.root = tk.Toplevel(root_master)
        else:
            self.root = tk.Tk()
        
        self.root.title("Standard PLS (Full Spectrum) - Maxilab Style")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        # --- Variáveis de Dados ---
        self.Xcal = None
        self.Ycal = None
        self.Xval = None
        self.Yval = None
        self.Xpred = None
        self.Ypred = None
        self.xaxis = None 
        
        # --- Variáveis do Modelo ---
        self.pls_model = None
        self.model_results = {}
        self.prediction_results = {}
        
        # --- Variáveis da GUI ---
        self.num_lvs = tk.StringVar(value="5")
        self.validation_type = tk.StringVar(value="cv") 
        self.preprocess_var = tk.StringVar(value="mean_center") 
        
        # Controle dos plots
        self.plot_cal_var = tk.BooleanVar(value=True)
        self.plot_val_var = tk.BooleanVar(value=False)

        self._setup_layout()

    def _setup_layout(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#FF8C00", height=60)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        black_band = tk.Frame(header_frame, bg="black", height=25)
        black_band.pack(fill=tk.X, side=tk.TOP, pady=(15, 0))
        lbl_title = tk.Label(black_band, text="Standard PLS Regression (Full Spectrum)", 
                             bg="black", fg="white", font=("Arial", 12, "bold"))
        lbl_title.pack(pady=2)

        # Container Principal
        main_container = tk.Frame(self.root, bg="#f0f0f0")
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = tk.Frame(main_container, bg="#f0f0f0", width=350)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        right_frame = tk.Frame(main_container, bg="#f0f0f0")
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # --- ESQUERDA: LOAD DATA ---
        lbl_load = tk.LabelFrame(left_frame, text=" LOAD THE DATA ", font=("Arial", 10, "bold"), bg="#f0f0f0")
        lbl_load.pack(fill=tk.X, ipady=10)

        self.btn_load = tk.Button(lbl_load, text="Load Data (.mat)", command=self.load_data, 
                                  bg="#34495e", fg="white", font=("Arial", 10, "bold"), height=2)
        self.btn_load.pack(fill=tk.X, padx=10, pady=10)

        self.lbl_status_xcal = self._create_data_row(lbl_load, "Xcal:")
        self.lbl_status_ycal = self._create_data_row(lbl_load, "Ycal:")
        self.lbl_status_xval = self._create_data_row(lbl_load, "Xval (Opt):")
        self.lbl_status_yval = self._create_data_row(lbl_load, "Yval (Opt):")
        
        tk.Frame(lbl_load, height=10, bg="#f0f0f0").pack() 

        self.btn_plot_raw = tk.Button(lbl_load, text="PLOT RAW SPECTRA", command=self.plot_raw_spectra, 
                                      state=tk.DISABLED, bg="#95a5a6", fg="black")
        self.btn_plot_raw.pack(fill=tk.X, padx=20, pady=10)

        # --- DIREITA: MODEL & PREDICTION ---
        lbl_model = tk.LabelFrame(right_frame, text=" MODEL PARAMETERS ", font=("Arial", 10, "bold"), bg="#f0f0f0")
        lbl_model.pack(fill=tk.X, pady=(0, 10), ipady=5)

        frame_config = tk.Frame(lbl_model, bg="#f0f0f0")
        frame_config.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(frame_config, text="Latent Variables (LVs):", bg="#f0f0f0").grid(row=0, column=0, sticky='e')
        tk.Entry(frame_config, textvariable=self.num_lvs, width=5).grid(row=0, column=1, padx=5, sticky='w')

        self.btn_estimate = tk.Button(frame_config, text="ESTIMATE MODEL", command=self.run_model, 
                                      state=tk.DISABLED, bg="#27ae60", fg="white", font=("Arial", 9, "bold"))
        self.btn_estimate.grid(row=0, column=2, padx=20)

        frame_radios = tk.Frame(lbl_model, bg="#f0f0f0")
        frame_radios.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(frame_radios, text="Preprocessing:", bg="#f0f0f0", font=("Arial", 9, "bold")).grid(row=0, column=0, sticky='w')
        tk.Radiobutton(frame_radios, text="Mean Centering", variable=self.preprocess_var, value="mean_center", bg="#f0f0f0").grid(row=1, column=0, sticky='w')
        tk.Radiobutton(frame_radios, text="None", variable=self.preprocess_var, value="none", bg="#f0f0f0").grid(row=2, column=0, sticky='w')

        tk.Label(frame_radios, text="Validation Method:", bg="#f0f0f0", font=("Arial", 9, "bold")).grid(row=0, column=1, sticky='w', padx=(30,0))
        tk.Radiobutton(frame_radios, text="Full Cross-Validation", variable=self.validation_type, value="cv", bg="#f0f0f0").grid(row=1, column=1, sticky='w', padx=(30,0))
        self.radio_ext = tk.Radiobutton(frame_radios, text="External Test Set (Req. Xval)", variable=self.validation_type, value="test", bg="#f0f0f0", state=tk.DISABLED)
        self.radio_ext.grid(row=2, column=1, sticky='w', padx=(30,0))
        
        # CAIXA DE TEXTO COM MÉTRICAS (Expandida)
        tk.Label(lbl_model, text="Comprehensive Model Metrics:", font=("Arial", 9, "bold"), bg="#f0f0f0").pack(anchor='w', padx=10, pady=(10,0))
        
        # Frame com Scrollbar para as métricas
        metrics_frame = tk.Frame(lbl_model)
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.txt_metrics = tk.Text(metrics_frame, height=12, bg="white", fg="#2c3e50", font=("Courier New", 9))
        scroll_y = tk.Scrollbar(metrics_frame, orient="vertical", command=self.txt_metrics.yview)
        self.txt_metrics.configure(yscrollcommand=scroll_y.set)
        self.txt_metrics.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        self.btn_view_results = tk.Button(lbl_model, text="VIEW FIT RESULTS", command=self.open_fit_results_window,
                                          state=tk.DISABLED, bg="#abf5a4", fg="white", font=("Arial", 11, "bold"), height=2)
        self.btn_view_results.pack(fill=tk.X, padx=50, pady=10)

        # Botões de Exportação Lado a Lado
        export_frame = tk.Frame(lbl_model, bg="#f0f0f0")
        export_frame.pack(fill=tk.X, padx=10, pady=5)

        self.btn_export_txt = tk.Button(export_frame, text="Export to Notepad (.txt)", command=self.export_metrics_txt,
                                        state=tk.DISABLED, bg="#000000", fg="white")
        self.btn_export_txt.pack(side=tk.RIGHT, padx=5)

        self.btn_export_xls = tk.Button(export_frame, text="Export to Excel (.xlsx)", command=self.export_metrics_excel,
                                        state=tk.DISABLED, bg="#27ae60", fg="white")
        self.btn_export_xls.pack(side=tk.RIGHT, padx=5)

        # --- PREDICTION ---
        lbl_pred = tk.LabelFrame(right_frame, text=" PREDICTION (Optional) ", font=("Arial", 10, "bold"), bg="#f0f0f0")
        lbl_pred.pack(fill=tk.X, ipady=5)

        pred_frame_inner = tk.Frame(lbl_pred, bg="#f0f0f0")
        pred_frame_inner.pack(fill=tk.X, padx=10, pady=10)

        self.lbl_status_xpred = tk.Label(pred_frame_inner, text="Xpred: Not Loaded", bg="#ecf0f1", relief="sunken", width=20, anchor='w')
        self.lbl_status_xpred.grid(row=0, column=0, padx=5)
        self.lbl_status_ypred = tk.Label(pred_frame_inner, text="Ypred: Not Loaded", bg="#ecf0f1", relief="sunken", width=20, anchor='w')
        self.lbl_status_ypred.grid(row=0, column=1, padx=5)

        self.btn_go_pred = tk.Button(pred_frame_inner, text="GO / VIEW PREDICTION", command=self.run_prediction,
                                     state=tk.DISABLED, bg="#f7e3ff", fg="black", font=("Arial", 9, "bold"))
        self.btn_go_pred.grid(row=0, column=2, padx=15)

    def _create_data_row(self, parent, label_text):
        f = tk.Frame(parent, bg="#f0f0f0")
        f.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(f, text=label_text, width=10, anchor='e', bg="#f0f0f0").pack(side=tk.LEFT)
        lbl_val = tk.Label(f, text="Not Loaded", bg="white", relief="sunken", anchor="w")
        lbl_val.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        return lbl_val

    # =============================================================================
    #  CARREGAMENTO DE DADOS
    # =============================================================================
    def load_data(self):
        file_path = filedialog.askopenfilename(title="Select .mat Data File", filetypes=[("MAT Files", "*.mat")])
        if not file_path: return
        try:
            mat = sio.loadmat(file_path)
            self.Xcal, self.Ycal, self.Xval, self.Yval, self.Xpred, self.Ypred = None, None, None, None, None, None
            self.txt_metrics.delete(1.0, tk.END)
            
            keys = mat.keys()
            def find_key(patterns):
                for k in keys:
                    if k.startswith('__'): continue
                    for p in patterns:
                        if p.lower() == k.lower(): return k
                return None

            k_xcal = find_key(['Xcal', 'X_cal', 'X_train', 'Train', 'X'])
            k_ycal = find_key(['Ycal', 'Y_cal', 'Y_train', 'Group_Train', 'Y'])
            k_xval = find_key(['Xval', 'X_val', 'Validation', 'X_test']) 
            k_yval = find_key(['Yval', 'Y_val', 'Group_Val', 'Y_test'])
            k_xpred = find_key(['Xpred', 'X_pred', 'Prediction'])
            k_ypred = find_key(['Ypred', 'Y_pred'])
            k_axis = find_key(['xaxis', 'axis', 'wavenumbers', 'wavelengths'])

            if k_xcal and k_ycal:
                self.Xcal = mat[k_xcal].astype(float)
                self.Ycal = mat[k_ycal].astype(float).flatten()
                self.lbl_status_xcal.config(text=f"Loaded ({self.Xcal.shape})", bg="#dff9fb")
                self.lbl_status_ycal.config(text=f"Loaded ({self.Ycal.shape})", bg="#dff9fb")
                self.btn_plot_raw.config(state=tk.NORMAL)
                self.btn_estimate.config(state=tk.NORMAL)
            else:
                messagebox.showerror("Error", "Could not find mandatory variables (Xcal/Ycal).")
                return

            if k_xval and k_yval:
                self.Xval = mat[k_xval].astype(float)
                self.Yval = mat[k_yval].astype(float).flatten()
                self.lbl_status_xval.config(text=f"Loaded ({self.Xval.shape})", bg="#dff9fb")
                self.lbl_status_yval.config(text=f"Loaded ({self.Yval.shape})", bg="#dff9fb")
                self.radio_ext.config(state=tk.NORMAL)
            else:
                self.lbl_status_xval.config(text="Not Found", bg="#f0f0f0")
                self.lbl_status_yval.config(text="Not Found", bg="#f0f0f0")
                self.radio_ext.config(state=tk.DISABLED)
                self.validation_type.set("cv")

            if k_xpred:
                self.Xpred = mat[k_xpred].astype(float)
                self.lbl_status_xpred.config(text=f"Loaded ({self.Xpred.shape})", bg="#dff9fb")
                self.btn_go_pred.config(state=tk.NORMAL)
                if k_ypred:
                    self.Ypred = mat[k_ypred].astype(float).flatten()
                    self.lbl_status_ypred.config(text=f"Loaded ({self.Ypred.shape})", bg="#dff9fb")
                else:
                    self.lbl_status_ypred.config(text="Not Found (Blind)")
            
            if k_axis: self.xaxis = mat[k_axis].flatten()
            else: self.xaxis = np.arange(self.Xcal.shape[1])

            messagebox.showinfo("Success", "Data loaded successfully.")
        except Exception as e:
            messagebox.showerror("Loading Error", str(e))

    def plot_raw_spectra(self):
        if self.Xcal is None: return
        # Create figure and use show method to avoid backend errors
        fig = plt.figure(figsize=(10, 6))
        plt.plot(self.xaxis, self.Xcal.T)
        plt.title(f"Raw Spectra ({self.Xcal.shape[0]} samples)")
        plt.xlabel("Variables"); plt.ylabel("Intensity"); plt.grid(True)
        fig.show()

    # =============================================================================
    #  CÁLCULO DE MÉTRICAS DETALHADAS
    # =============================================================================
    def calculate_detailed_stats(self, y_true, y_pred, prefix):
        """Calcula todas as métricas quimiométricas padrão."""
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        residuals = y_true - y_pred
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        bias = np.mean(residuals)
        r2 = r2_score(y_true, y_pred)
        r_corr = np.corrcoef(y_true, y_pred)[0, 1]
        
        slope, intercept, _, _, _ = linregress(y_true, y_pred)
        press = np.sum(residuals**2)
        
        std_dev = np.std(y_true, ddof=1)
        data_range = np.max(y_true) - np.min(y_true)
        
        rpd = std_dev / rmse if rmse > 0 else np.nan
        rer = data_range / rmse if rmse > 0 else np.nan
        
        stats = {
            f'RMSE{prefix}': rmse,
            f'R2_{prefix}': r2,
            f'r_corr_{prefix}': r_corr,
            f'Bias_{prefix}': bias,
            f'Slope_{prefix}': slope,
            f'Offset_{prefix}': intercept,
            f'RPD_{prefix}': rpd,
            f'RER_{prefix}': rer,
            f'PRESS_{prefix}': press
        }
        return stats

    def run_model(self):
        try:
            n_comp = int(self.num_lvs.get())
            do_scale = (self.preprocess_var.get() == "none")
            
            # 1. Fit PLS
            self.pls_model = PLSRegression(n_components=n_comp, scale=do_scale)
            self.pls_model.fit(self.Xcal, self.Ycal)
            
            # 2. Calibração (Train)
            y_cal_pred = self.pls_model.predict(self.Xcal).flatten()
            cal_stats = self.calculate_detailed_stats(self.Ycal, y_cal_pred, "C") # C para Calibração
            
            # Salvar resultados base
            self.model_results = {
                'Y_Cal_True': self.Ycal, 'Y_Cal_Pred': y_cal_pred,
                'Loadings': self.pls_model.x_loadings_,
                'Coef': self.pls_model.coef_,
                'Scores': self.pls_model.x_scores_,
                'LVs': n_comp
            }
            self.model_results.update(cal_stats)
            
            # 3. Validação
            val_method = self.validation_type.get()
            val_stats = {}
            
            if val_method == "cv":
                kf = KFold(n_splits=10, shuffle=True, random_state=42)
                y_cv_pred = cross_val_predict(self.pls_model, self.Xcal, self.Ycal, cv=kf, n_jobs=-1).flatten()
                val_stats = self.calculate_detailed_stats(self.Ycal, y_cv_pred, "CV")
                self.model_results.update(val_stats)
                self.model_results.update({'Y_Val_True': self.Ycal, 'Y_Val_Pred': y_cv_pred, 'Val_Type': 'Cross-Validation'})
                
            elif val_method == "test":
                y_val_pred = self.pls_model.predict(self.Xval).flatten()
                val_stats = self.calculate_detailed_stats(self.Yval, y_val_pred, "P") # P para Predição/Validação
                self.model_results.update(val_stats)
                self.model_results.update({'Y_Val_True': self.Yval, 'Y_Val_Pred': y_val_pred, 'Val_Type': 'External Validation'})

            # --- GERAR RELATÓRIO DE TEXTO (FORMATADO) ---
            report = f"MODEL SUMMARY ({n_comp} LVs)\n"
            report += "="*50 + "\n"
            report += "CALIBRATION METRICS:\n"
            report += f"  RMSEC : {cal_stats['RMSEC']:.4f}\n"
            report += f"  R²    : {cal_stats['R2_C']:.4f}\n"
            report += f"  r     : {cal_stats['r_corr_C']:.4f}\n"
            report += f"  Bias  : {cal_stats['Bias_C']:.4f}\n"
            report += f"  RPD   : {cal_stats['RPD_C']:.4f}\n"
            report += f"  RER   : {cal_stats['RER_C']:.4f}\n"
            report += f"  Slope : {cal_stats['Slope_C']:.4f}\n"
            report += f"  Offset: {cal_stats['Offset_C']:.4f}\n"
            report += "-"*50 + "\n"
            
            if val_method == "cv":
                report += "CROSS-VALIDATION METRICS:\n"
                suffix = "CV"
            else:
                report += "EXTERNAL VALIDATION METRICS:\n"
                suffix = "P"
                
            report += f"  RMSE{suffix} : {val_stats[f'RMSE{suffix}']:.4f}\n"
            report += f"  R²     : {val_stats[f'R2_{suffix}']:.4f}\n"
            report += f"  r      : {val_stats[f'r_corr_{suffix}']:.4f}\n"
            report += f"  Bias   : {val_stats[f'Bias_{suffix}']:.4f}\n"
            report += f"  RPD    : {val_stats[f'RPD_{suffix}']:.4f}\n"
            report += f"  RER    : {val_stats[f'RER_{suffix}']:.4f}\n"
            report += f"  Slope  : {val_stats[f'Slope_{suffix}']:.4f}\n"
            report += f"  Offset : {val_stats[f'Offset_{suffix}']:.4f}\n"
            report += f"  PRESS  : {val_stats[f'PRESS_{suffix}']:.4f}\n"

            self.txt_metrics.delete(1.0, tk.END)
            self.txt_metrics.insert(tk.END, report)

            messagebox.showinfo("Model Created", "PLS Model estimated successfully!")
            self.btn_view_results.config(state=tk.NORMAL)
            self.btn_export_xls.config(state=tk.NORMAL)
            self.btn_export_txt.config(state=tk.NORMAL)
            if self.Xpred is not None: self.btn_go_pred.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Model Error", str(e))

    # =============================================================================
    #  JANELA DE RESULTADOS
    # =============================================================================
    def open_fit_results_window(self):
        top = tk.Toplevel(self.root)
        top.title("VIEW FIT RESULTS")
        top.geometry("600x520")
        top.configure(bg="#f0f0f0")
        
        header = tk.Label(top, text="VIEW FIT RESULTS", bg="yellow", fg="black", font=("Arial", 12, "bold"), pady=10)
        header.pack(fill=tk.X)
        tk.Frame(top, bg="black", height=5).pack(fill=tk.X) 

        chk_frame = tk.Frame(top, bg="#f0f0f0")
        chk_frame.pack(pady=10)
        tk.Checkbutton(chk_frame, text="Calibration", variable=self.plot_cal_var, bg="#f0f0f0", font=("Arial", 10)).pack(side=tk.LEFT, padx=20)
        tk.Checkbutton(chk_frame, text="Validation", variable=self.plot_val_var, bg="#f0f0f0", font=("Arial", 10)).pack(side=tk.LEFT, padx=20)

        btn_frame = tk.Frame(top, bg="#f0f0f0")
        btn_frame.pack(expand=True, padx=20, pady=5)

        btns = [
            ("Predicted vs Actual", self.plot_pred_vs_actual),
            ("EJCR (Confidence Ellipse)", self.plot_ejcr_popup),
            ("Regression Coefficients", self.plot_reg_coeffs),
            ("Residuals", self.plot_residuals),
            ("PLS Loadings", self.plot_loadings),
        ]

        for i, (text, cmd) in enumerate(btns):
            b = tk.Button(btn_frame, text=text, command=cmd, width=25, height=2, bg="white")
            b.grid(row=i//2, column=i%2, padx=10, pady=5)
        
        tk.Button(top, text="SAVE ALL PLOTS (TIFF / 300 DPI)", command=self.save_all_plots_tiff, 
                  bg="#27ae60", fg="white", font=("Arial", 10, "bold"), width=35).pack(pady=10)
        tk.Button(top, text="DONE", command=top.destroy, bg="#e74c3c", fg="white", font=("Arial", 10, "bold")).pack(pady=5)

    # =============================================================================
    #  PREDIÇÃO
    # =============================================================================
    def run_prediction(self):
        if self.pls_model is None: return
        try:
            y_pred_est = self.pls_model.predict(self.Xpred).flatten()
            self.prediction_results = {'Y_Pred_Est': y_pred_est}
            msg = "Prediction done.\n"
            
            if self.Ypred is not None:
                # Calcular métricas completas para a predição também
                pred_stats = self.calculate_detailed_stats(self.Ypred, y_pred_est, "PredSet")
                self.prediction_results.update(pred_stats)
                self.prediction_results['Y_Pred_True'] = self.Ypred
                
                msg += f"RMSE: {pred_stats['RMSEPredSet']:.4f}\n"
                msg += f"R²:   {pred_stats['R2_PredSet']:.4f}\n"
                msg += f"RPD:  {pred_stats['RPD_PredSet']:.4f}"
            
            messagebox.showinfo("Prediction", msg)
            self.open_prediction_window()
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    def open_prediction_window(self):
        top = tk.Toplevel(self.root)
        top.title("PREDICTION RESULTS"); top.geometry("400x200")
        tk.Label(top, text="PREDICTION RESULTS", bg="#9b59b6", fg="white", font=("Arial", 12, "bold"), pady=10).pack(fill=tk.X)
        btn_frame = tk.Frame(top); btn_frame.pack(expand=True)
        tk.Button(btn_frame, text="Predicted vs Actual (Pred)", command=self.plot_prediction_fit, width=25).pack(pady=5)
        tk.Button(btn_frame, text="Save Prediction to Excel", command=self.save_prediction_excel, width=25).pack(pady=5)

    # =============================================================================
    #  PLOTAGEM E EXPORTAÇÃO
    # =============================================================================
    def save_all_plots_tiff(self):
        folder_path = filedialog.askdirectory(title="Select Folder to Save TIFFs")
        if not folder_path: return
        try:
            tasks = [("Pred_vs_Actual.tiff", self._create_fig_pred_actual), ("Residuals.tiff", self._create_fig_residuals),
                     ("Coefficients.tiff", self._create_fig_coeffs), ("Loadings.tiff", self._create_fig_loadings), ("EJCR.tiff", self._create_fig_ejcr)]
            count = 0
            for filename, func in tasks:
                fig = func()
                if fig:
                    fig.savefig(os.path.join(folder_path, filename), format='tiff', dpi=300, bbox_inches='tight'); plt.close(fig); count += 1
            messagebox.showinfo("Success", f"{count} graphs saved in:\n{folder_path}")
        except Exception as e: messagebox.showerror("Save Error", str(e))

    # Funções de criação de figuras com CORREÇÃO DE DIMENSÕES
    def _create_fig_pred_actual(self):
        fig, ax = plt.subplots(figsize=(8, 6)); res = self.model_results
        if self.plot_cal_var.get(): ax.scatter(res['Y_Cal_True'], res['Y_Cal_Pred'], c='blue', alpha=0.6, label='Calibration')
        if self.plot_val_var.get(): ax.scatter(res['Y_Val_True'], res['Y_Val_Pred'], c='red', marker='^', alpha=0.6, label=res.get('Val_Type', 'Validation'))
        limits = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(limits, limits, 'k--', alpha=0.75); ax.set_xlabel('Measured Y'); ax.set_ylabel('Predicted Y'); ax.set_title("Predicted vs Actual"); ax.legend(); return fig

    def _create_fig_residuals(self):
        fig, ax = plt.subplots(figsize=(8, 6)); res = self.model_results
        if self.plot_cal_var.get(): ax.scatter(res['Y_Cal_Pred'], res['Y_Cal_True'] - res['Y_Cal_Pred'], c='blue', alpha=0.6, label='Calibration')
        if self.plot_val_var.get(): ax.scatter(res['Y_Val_Pred'], res['Y_Val_True'] - res['Y_Val_Pred'], c='red', marker='^', alpha=0.6, label=res.get('Val_Type', 'Validation'))
        ax.axhline(0, color='black', linestyle='--'); ax.set_xlabel('Predicted Y'); ax.set_ylabel('Residuals'); ax.set_title('Residuals Plot'); ax.legend(); return fig

    def _create_fig_loadings(self):
        fig, ax = plt.subplots(figsize=(8, 6)); loads = self.model_results['Loadings']
        for i in range(loads.shape[1]): ax.plot(self.xaxis, loads[:, i], label=f'LV {i+1}')
        ax.set_xlabel('Variables'); ax.set_ylabel('Loading Weight'); ax.legend(); ax.set_title('PLS Loadings'); return fig

    def _create_fig_coeffs(self):
        fig, ax = plt.subplots(figsize=(8, 6)); coef = self.model_results['Coef']
        # CORREÇÃO CRÍTICA: Flattening coeficientes para evitar erro de dimensão
        ax.plot(self.xaxis, coef.flatten()); ax.axhline(0, color='k', linestyle='--'); ax.set_xlabel('Variables'); ax.set_ylabel('Coefficient Value'); ax.set_title('Regression Coefficients'); return fig

    def _create_fig_ejcr(self):
        fig, ax = plt.subplots(figsize=(8, 6)); res = self.model_results; datasets = []
        if self.plot_cal_var.get(): datasets.append((res['Y_Cal_True'], res['Y_Cal_Pred'], 'blue', 'Calibration'))
        if self.plot_val_var.get(): datasets.append((res['Y_Val_True'], res['Y_Val_Pred'], 'red', res.get('Val_Type', 'Validation')))
        if not datasets: plt.close(fig); return None
        for y_t, y_p, color, label in datasets:
            ejcr_res = self.calculate_ejcr(y_t, y_p)
            if ejcr_res:
                slope, intercept, w, h, angle = ejcr_res
                ellipse = Ellipse(xy=(intercept, slope), width=2*w, height=2*h, angle=angle, edgecolor=color, facecolor='none', linewidth=2, label=f'{label} 95%')
                ax.add_patch(ellipse); ax.plot(intercept, slope, marker='o', color=color)
        ax.plot(0, 1, 'k+', markersize=12, markeredgewidth=2, label='Ideal (0, 1)'); ax.set_xlabel('Intercept'); ax.set_ylabel('Slope'); ax.set_title('Elliptical Joint Confidence Region (EJCR)'); ax.grid(True); ax.legend(); ax.autoscale(); return fig

    def calculate_ejcr(self, y_true, y_pred, confidence=0.95):
        n = len(y_true); 
        if n <= 2: return None
        slope, intercept, _, _, _ = linregress(y_true, y_pred)
        residuals = y_pred - (intercept + slope * y_true)
        MSE = np.sum(residuals**2) / (n - 2)
        F_val = f_dist.ppf(confidence, 2, n - 2)
        X_mat = np.vstack([np.ones(n), y_true]).T
        XtX_inv = np.linalg.inv(np.dot(X_mat.T, X_mat))
        eig_vals, eig_vecs = np.linalg.eigh(XtX_inv)
        axis_lengths = np.sqrt(2 * F_val * MSE * eig_vals)
        angle = np.degrees(np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0]))
        return slope, intercept, axis_lengths[0], axis_lengths[1], angle

    # CORREÇÃO: Usando fig.show() em vez de plt.show(fig)
    def plot_pred_vs_actual(self): 
        f = self._create_fig_pred_actual(); 
        if f: f.show()
    def plot_residuals(self): 
        f = self._create_fig_residuals(); 
        if f: f.show()
    def plot_loadings(self): 
        f = self._create_fig_loadings(); 
        if f: f.show()
    def plot_reg_coeffs(self): 
        f = self._create_fig_coeffs(); 
        if f: f.show()
    def plot_ejcr_popup(self): 
        f = self._create_fig_ejcr(); 
        if f: f.show()
        else: messagebox.showinfo("Info", "Select Calibration or Validation checkbox.")
    
    def plot_prediction_fit(self):
        if 'Y_Pred_True' not in self.prediction_results: messagebox.showinfo("Info", "Reference Y not available."); return
        plt.figure(); plt.scatter(self.prediction_results['Y_Pred_True'], self.prediction_results['Y_Pred_Est'], c='purple')
        limits = [min(self.prediction_results['Y_Pred_True']), max(self.prediction_results['Y_Pred_True'])]
        plt.plot(limits, limits, 'k--'); plt.xlabel("Measured"); plt.ylabel("Predicted"); plt.title("Prediction Set"); plt.show()

    def export_metrics_excel(self):
        if not self.model_results: return
        f = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if f:
            data = {k: v for k, v in self.model_results.items() if isinstance(v, (int, float, str))}
            pd.DataFrame([data]).to_excel(f, index=False); messagebox.showinfo("Export", "Metrics exported to Excel!")

    def export_metrics_txt(self):
        if not self.model_results: return
        f = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text File", "*.txt")])
        if f:
            txt_content = self.txt_metrics.get(1.0, tk.END)
            with open(f, 'w') as file:
                file.write(txt_content)
            messagebox.showinfo("Export", "Metrics exported to .txt!")

    def save_prediction_excel(self):
        if not self.prediction_results: return
        f = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if f:
            df = pd.DataFrame({'Y_Pred_Est': self.prediction_results['Y_Pred_Est']})
            for k, v in self.prediction_results.items():
                if isinstance(v, (int, float, str)): df[k] = v
                elif isinstance(v, np.ndarray) and v.size == df.shape[0]: df[k] = v
            df.to_excel(f, index=False); messagebox.showinfo("Export", "Prediction data saved!")

if __name__ == "__main__":
    app = Standard_PLS_App()
    app.root.mainloop()