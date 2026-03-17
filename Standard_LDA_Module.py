import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import scipy.io as sio
import pandas as pd
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, cross_val_predict
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score, 
                             precision_score, f1_score, roc_curve, auc, mean_squared_error)
from sklearn.preprocessing import LabelEncoder, label_binarize

# =============================================================================
#  CLASSE PRINCIPAL - STANDARD LDA (MAXILAB STYLE)
# =============================================================================

class Standard_LDA_App:
    def __init__(self, root_master=None):
        if root_master:
            self.root = tk.Toplevel(root_master)
        else:
            self.root = tk.Tk()
        
        self.root.title("Standard Discriminant Analysis - Maxilab Style")
        self.root.geometry("1250x850")
        self.root.configure(bg="#f0f0f0")
        
        # --- Dados ---
        self.X_train = None
        self.y_train = None
        self.X_test = None 
        self.y_test = None
        self.X_pred = None
        self.class_labels = None 
        
        # --- Modelo ---
        self.model = None
        self.le = LabelEncoder()
        self.model_results = {}
        
        # --- GUI Vars ---
        self.discrim_type = tk.StringVar(value="Linear") 
        self.cv_type = tk.StringVar(value="k-fold") 
        self.k_folds = tk.StringVar(value="5")
        
        self.plot_train_var = tk.BooleanVar(value=True)
        self.plot_test_var = tk.BooleanVar(value=False)

        self._setup_layout()

    def _setup_layout(self):
        # HEADER
        header_frame = tk.Frame(self.root, bg="#FF8C00", height=60)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        black_band = tk.Frame(header_frame, bg="black", height=25)
        black_band.pack(fill=tk.X, side=tk.TOP, pady=(15, 0))
        lbl_title = tk.Label(black_band, text="Standard Discriminant Analysis (LDA/QDA)", 
                             bg="black", fg="white", font=("Arial", 12, "bold"))
        lbl_title.pack(pady=2)

        # MAIN CONTAINER
        main_container = tk.Frame(self.root, bg="#f0f0f0")
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = tk.Frame(main_container, bg="#f0f0f0", width=350)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        right_frame = tk.Frame(main_container, bg="#f0f0f0")
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # ESQUERDA: LOAD DATA
        lbl_load = tk.LabelFrame(left_frame, text=" LOAD DATA ", font=("Arial", 10, "bold"), bg="#f0f0f0")
        lbl_load.pack(fill=tk.X, ipady=10)

        self.btn_load = tk.Button(lbl_load, text="Load Data (.mat)", command=self.load_data, 
                                  bg="#8e44ad", fg="white", font=("Arial", 10, "bold"), height=2)
        self.btn_load.pack(fill=tk.X, padx=10, pady=10)

        self.lbl_status_train = self._create_data_row(lbl_load, "Cal (Train):")
        self.lbl_status_test = self._create_data_row(lbl_load, "Val (Test):")
        self.lbl_status_pred = self._create_data_row(lbl_load, "Pred (Blind):")

        # DIREITA: PARAMETERS
        lbl_model = tk.LabelFrame(right_frame, text=" ANALYSIS PARAMETERS ", font=("Arial", 10, "bold"), bg="#f0f0f0")
        lbl_model.pack(fill=tk.X, pady=(0, 10), ipady=5)

        frame_conf = tk.Frame(lbl_model, bg="#f0f0f0")
        frame_conf.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(frame_conf, text="Discriminant Type:", bg="#f0f0f0", font=("Arial", 9, "bold")).grid(row=0, column=0, sticky='w')
        discrim_options = ["Linear", "Quadratic", "Diagonal (Naive Bayes)", "Pseudo-Linear (Shrinkage)"]
        cbox_discrim = ttk.Combobox(frame_conf, textvariable=self.discrim_type, values=discrim_options, state="readonly", width=25)
        cbox_discrim.grid(row=0, column=1, padx=5, sticky='w')

        tk.Label(frame_conf, text="Cross-Validation:", bg="#f0f0f0", font=("Arial", 9, "bold")).grid(row=1, column=0, sticky='w', pady=5)
        cv_frame = tk.Frame(frame_conf, bg="#f0f0f0")
        cv_frame.grid(row=1, column=1, sticky='w')
        tk.Radiobutton(cv_frame, text="K-Fold", variable=self.cv_type, value="k-fold", bg="#f0f0f0", command=self._toggle_k_entry).pack(side=tk.LEFT)
        self.entry_k = tk.Entry(cv_frame, textvariable=self.k_folds, width=3)
        self.entry_k.pack(side=tk.LEFT, padx=2)
        tk.Radiobutton(cv_frame, text="Leave-One-Out", variable=self.cv_type, value="loo", bg="#f0f0f0", command=self._toggle_k_entry).pack(side=tk.LEFT, padx=10)

        self.btn_run = tk.Button(frame_conf, text="RUN ANALYSIS", command=self.run_analysis, 
                                 state=tk.DISABLED, bg="#27ae60", fg="white", font=("Arial", 9, "bold"))
        self.btn_run.grid(row=0, column=2, rowspan=2, padx=20, sticky='ns')

        # Metrics Box
        tk.Label(lbl_model, text="Metrics:", font=("Arial", 9, "bold"), bg="#f0f0f0").pack(anchor='w', padx=10, pady=(10,0))
        metrics_frame = tk.Frame(lbl_model)
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.txt_metrics = tk.Text(metrics_frame, height=14, bg="white", fg="#2c3e50", font=("Courier New", 9))
        scroll = tk.Scrollbar(metrics_frame, command=self.txt_metrics.yview)
        self.txt_metrics.configure(yscrollcommand=scroll.set)
        self.txt_metrics.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Buttons Box
        btn_box = tk.Frame(lbl_model, bg="#f0f0f0")
        btn_box.pack(fill=tk.X, padx=10, pady=5)
        self.btn_view_results = tk.Button(btn_box, text="VIEW RESULTS (GRAPHS)", command=self.open_results_window,
                                          state=tk.DISABLED, bg="#c6f1b5", fg="white", font=("Arial", 10, "bold"), height=2)
        self.btn_view_results.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
        self.btn_export_xls = tk.Button(btn_box, text="Export Excel", command=self.export_excel, state=tk.DISABLED, bg="#27ae60", fg="white")
        self.btn_export_xls.pack(side=tk.RIGHT, padx=5)

    def _create_data_row(self, parent, label_text):
        f = tk.Frame(parent, bg="#f0f0f0")
        f.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(f, text=label_text, width=12, anchor='e', bg="#f0f0f0").pack(side=tk.LEFT)
        lbl = tk.Label(f, text="Not Loaded", bg="white", relief="sunken", anchor="w")
        lbl.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        return lbl

    def _toggle_k_entry(self):
        if self.cv_type.get() == 'k-fold': self.entry_k.config(state=tk.NORMAL)
        else: self.entry_k.config(state=tk.DISABLED)

    # =============================================================================
    #  LOAD DATA
    # =============================================================================
    def load_data(self):
        file_path = filedialog.askopenfilename(title="Select .mat Data File", filetypes=[("MAT Files", "*.mat")])
        if not file_path: return
        try:
            mat = sio.loadmat(file_path)
            self.X_train, self.y_train, self.X_test, self.y_test, self.X_pred = None, None, None, None, None
            self.txt_metrics.delete(1.0, tk.END)
            keys = mat.keys()
            def find_key(patterns):
                for k in keys:
                    if k.startswith('__'): continue
                    for p in patterns: 
                        if p.lower() == k.lower(): return k
                return None

            k_xtrain = find_key(['X_train', 'Train', 'Xcal', 'X_cal', 'X'])
            k_ytrain = find_key(['Y_train', 'Group_Train', 'Ycal', 'Y_cal', 'Y', 'Group'])
            if k_xtrain and k_ytrain:
                self.X_train = mat[k_xtrain].astype(float)
                raw_y = mat[k_ytrain].flatten()
                self.y_train = self.le.fit_transform(raw_y)
                self.class_labels = self.le.classes_
                self.lbl_status_train.config(text=f"Loaded ({self.X_train.shape})", bg="#dff9fb")
                self.btn_run.config(state=tk.NORMAL)
            else: messagebox.showerror("Error", "Mandatory Train data (X/Y) not found."); return

            k_xtest = find_key(['X_test', 'Test', 'Xval', 'X_val'])
            k_ytest = find_key(['Y_test', 'Group_Test', 'Yval', 'Y_val'])
            if k_xtest and k_ytest:
                self.X_test = mat[k_xtest].astype(float)
                self.y_test = self.le.transform(mat[k_ytest].flatten())
                self.lbl_status_test.config(text=f"Loaded ({self.X_test.shape})", bg="#dff9fb")
            else:
                self.lbl_status_test.config(text="Not Found", bg="#f0f0f0")

            k_xpred = find_key(['X_pred', 'Prediction', 'Unknown'])
            if k_xpred:
                self.X_pred = mat[k_xpred].astype(float)
                self.lbl_status_pred.config(text=f"Loaded ({self.X_pred.shape})", bg="#dff9fb")
            else: self.lbl_status_pred.config(text="Not Found", bg="#f0f0f0")

            messagebox.showinfo("Success", f"Data loaded!\nClasses: {self.class_labels}")
        except Exception as e: messagebox.showerror("Load Error", str(e))

    # =============================================================================
    #  METRICS
    # =============================================================================
    def calculate_rmse_classification(self, y_true, y_prob):
        n_classes = len(self.class_labels)
        y_true_dummy = label_binarize(y_true, classes=range(n_classes))
        if n_classes == 2: y_true_dummy = np.hstack((1 - y_true_dummy, y_true_dummy))
        if y_prob.shape != y_true_dummy.shape: return np.nan
        mse = np.mean((y_true_dummy - y_prob)**2)
        return np.sqrt(mse)

    def calculate_metrics_block(self, y_true, y_pred, y_prob, title):
        pcc = accuracy_score(y_true, y_pred) * 100
        rmse = self.calculate_rmse_classification(y_true, y_prob)
        n_classes = len(self.class_labels)
        try:
            if n_classes == 2:
                from sklearn.metrics import roc_auc_score
                auc_score = roc_auc_score(y_true, y_prob[:, 1])
            else:
                from sklearn.metrics import roc_auc_score
                auc_score = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except: auc_score = 0.5

        cm = confusion_matrix(y_true, y_pred)
        sens = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
        specs = []
        for i in range(n_classes):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        spec = np.mean(specs) * 100

        return {f'{title}_PCC': pcc, f'{title}_RMSE': rmse, f'{title}_AUC': auc_score,
                f'{title}_Sens': sens, f'{title}_Spec': spec, f'{title}_CM': cm,
                f'{title}_Y_True': y_true, f'{title}_Y_Pred': y_pred, f'{title}_Y_Prob': y_prob}

    def run_analysis(self):
        try:
            dtype = self.discrim_type.get()
            if dtype == "Linear": self.model = LinearDiscriminantAnalysis(solver='svd')
            elif dtype == "Quadratic": self.model = QuadraticDiscriminantAnalysis()
            elif dtype == "Diagonal (Naive Bayes)": self.model = GaussianNB()
            elif dtype == "Pseudo-Linear (Shrinkage)": self.model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            
            cv_method = self.cv_type.get()
            if cv_method == 'k-fold':
                k = int(self.k_folds.get()); cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42); cv_name = f"{k}-Fold CV"
            else: cv = LeaveOneOut(); cv_name = "Leave-One-Out CV"
            
            y_cv_pred = cross_val_predict(self.model, self.X_train, self.y_train, cv=cv)
            y_cv_prob = cross_val_predict(self.model, self.X_train, self.y_train, cv=cv, method='predict_proba')
            self.model_results = self.calculate_metrics_block(self.y_train, y_cv_pred, y_cv_prob, "CV")
            
            self.model.fit(self.X_train, self.y_train)
            y_cal_pred = self.model.predict(self.X_train); y_cal_prob = self.model.predict_proba(self.X_train)
            self.model_results.update(self.calculate_metrics_block(self.y_train, y_cal_pred, y_cal_prob, "Cal"))
            
            test_metrics = {}
            if self.X_test is not None:
                y_test_pred = self.model.predict(self.X_test); y_test_prob = self.model.predict_proba(self.X_test)
                test_metrics = self.calculate_metrics_block(self.y_test, y_test_pred, y_test_prob, "Test")
                self.model_results.update(test_metrics)

            report = f"ANALYSIS REPORT\nDiscriminant: {dtype}\nValidation: {cv_name}\n" + "="*40 + "\n"
            def append_stats(title, key_prefix):
                r = f"{title}\n"
                r += f"  RMSE = {self.model_results[f'{key_prefix}_RMSE']:.4f}\n"
                r += f"  AUC  = {self.model_results[f'{key_prefix}_AUC']:.4f}\n"
                r += f"  PCC  = {self.model_results[f'{key_prefix}_PCC']:.2f} %\n"
                r += f"  Sens = {self.model_results[f'{key_prefix}_Sens']:.2f} %\n"
                r += f"  Spec = {self.model_results[f'{key_prefix}_Spec']:.2f} %\n"
                r += "-"*40 + "\n"
                return r

            report += append_stats("CALIBRATION", "Cal")
            report += append_stats("CROSS-VALIDATION", "CV")
            if test_metrics: report += append_stats("EXTERNAL PREDICTION", "Test")
            
            self.txt_metrics.delete(1.0, tk.END); self.txt_metrics.insert(tk.END, report)
            messagebox.showinfo("Done", "Analysis complete!"); self.btn_view_results.config(state=tk.NORMAL); self.btn_export_xls.config(state=tk.NORMAL)
        except Exception as e: messagebox.showerror("Execution Error", str(e))

    # =============================================================================
    #  VISUALIZAÇÃO
    # =============================================================================
    def open_results_window(self):
        top = tk.Toplevel(self.root); top.title("LDA RESULTS DASHBOARD"); top.geometry("600x480"); top.configure(bg="#f0f0f0")
        tk.Label(top, text="VIEW RESULTS", bg="yellow", fg="black", font=("Arial", 12, "bold"), pady=10).pack(fill=tk.X)
        tk.Frame(top, bg="black", height=5).pack(fill=tk.X)
        
        chk_frame = tk.Frame(top, bg="#f0f0f0"); chk_frame.pack(pady=10)
        tk.Checkbutton(chk_frame, text="Calibration", variable=self.plot_train_var, bg="#f0f0f0").pack(side=tk.LEFT, padx=10)
        tk.Checkbutton(chk_frame, text="Test/Validation", variable=self.plot_test_var, bg="#f0f0f0").pack(side=tk.LEFT, padx=10)

        btn_frame = tk.Frame(top, bg="#f0f0f0"); btn_frame.pack(expand=True)
        
        # ADICIONADO: Probability Plot
        btns = [("Score Plot (LD1 x LD2)", self.plot_scores), 
                ("Confusion Matrix", self.plot_cm), 
                ("ROC Curve", self.plot_roc),
                ("Probability/Prediction Plot", self.plot_probabilities)] # <--- NOVO
        
        if self.discrim_type.get() not in ["Linear", "Pseudo-Linear (Shrinkage)"]: btns.pop(0) # Remove Score plot if not LDA
            
        for i, (txt, cmd) in enumerate(btns): tk.Button(btn_frame, text=txt, command=cmd, width=30, height=2, bg="white").pack(pady=3)
        tk.Button(top, text="SAVE ALL (TIFF)", command=self.save_tiff, bg="#78eca9", fg="white", font=("Arial", 10, "bold"), width=30).pack(pady=10)

    # GRÁFICOS
    def _create_fig_scores(self):
        f = plt.figure(figsize=(8,6)); ax = f.add_subplot(111); X_r = self.model.transform(self.X_train)
        if X_r.shape[1] < 2: ax.text(0.5, 0.5, "Not enough dimensions for 2D Plot", ha='center'); return f
        if self.plot_train_var.get(): sns.scatterplot(x=X_r[:,0], y=X_r[:,1], hue=self.class_labels[self.y_train], style=['Cal']*len(self.y_train), ax=ax, s=100, alpha=0.7)
        if self.plot_test_var.get() and self.X_test is not None:
            X_rt = self.model.transform(self.X_test); sns.scatterplot(x=X_rt[:,0], y=X_rt[:,1], hue=self.class_labels[self.y_test], style=['Test']*len(self.y_test), ax=ax, markers=['X'], s=120, legend=False)
        ax.set_title("Discriminant Scores"); ax.set_xlabel("LD1"); ax.set_ylabel("LD2"); ax.grid(True); return f

    def _create_fig_cm(self):
        f, axes = plt.subplots(1, 2, figsize=(12, 5)); plotted = False
        if self.plot_train_var.get(): sns.heatmap(self.model_results['Cal_CM'], annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=self.class_labels, yticklabels=self.class_labels); axes[0].set_title("Confusion Matrix (Cal)"); plotted = True
        else: axes[0].axis('off')
        if self.plot_test_var.get() and 'Test_CM' in self.model_results: sns.heatmap(self.model_results['Test_CM'], annot=True, fmt='d', cmap='Greens', ax=axes[1], xticklabels=self.class_labels, yticklabels=self.class_labels); axes[1].set_title("Confusion Matrix (Test)"); plotted = True
        else: axes[1].axis('off')
        if not plotted: plt.close(f); return None
        return f

    def _create_fig_roc(self):
        f, ax = plt.subplots(figsize=(8, 6))
        def add_roc(y_true, y_prob, label_suf):
            n = len(self.class_labels); y_bin = label_binarize(y_true, classes=range(n))
            if n == 2: fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1]); auc_v = auc(fpr, tpr); ax.plot(fpr, tpr, label=f'{self.class_labels[1]} {label_suf} (AUC={auc_v:.2f})')
            else:
                for i in range(n): fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i]); auc_v = auc(fpr, tpr); ax.plot(fpr, tpr, label=f'{self.class_labels[i]} {label_suf} (AUC={auc_v:.2f})')
        if self.plot_train_var.get(): add_roc(self.y_train, self.model_results['Cal_Y_Prob'], "(Cal)")
        if self.plot_test_var.get() and 'Test_Y_Prob' in self.model_results: add_roc(self.y_test, self.model_results['Test_Y_Prob'], "(Test)")
        ax.plot([0, 1], [0, 1], 'k--'); ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate'); ax.set_title('ROC Curve'); ax.legend(); return f

    def _create_fig_probabilities(self):
        # NOVO GRÁFICO: Sample vs Probability
        f, ax = plt.subplots(figsize=(10, 6))
        
        if self.plot_train_var.get():
            y_prob = self.model_results['Cal_Y_Prob']
            # Plot probabilidade da classe PREDITA para cada amostra
            max_probs = np.max(y_prob, axis=1)
            ax.plot(max_probs, 'o', label='Calibration', alpha=0.7)
            
        if self.plot_test_var.get() and 'Test_Y_Prob' in self.model_results:
            y_prob_test = self.model_results['Test_Y_Prob']
            max_probs_test = np.max(y_prob_test, axis=1)
            # Offset x-axis for test samples
            x_start = len(self.model_results['Cal_Y_Prob']) if self.plot_train_var.get() else 0
            x_idx = np.arange(x_start, x_start + len(max_probs_test))
            ax.plot(x_idx, max_probs_test, 'x', label='Test', alpha=0.9)

        ax.axhline(0.5, color='r', linestyle='--', label='Threshold (0.5)')
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Probability of Predicted Class")
        ax.set_title("Prediction Probability per Sample")
        ax.legend()
        ax.grid(True)
        return f

    # SHOW FUNCTIONS
    def plot_scores(self): 
        f = self._create_fig_scores(); 
        if f: f.show()
    def plot_cm(self): 
        f = self._create_fig_cm(); 
        if f: f.show()
    def plot_roc(self): 
        f = self._create_fig_roc(); 
        if f: f.show()
    def plot_probabilities(self):
        f = self._create_fig_probabilities();
        if f: f.show()

    def save_tiff(self):
        d = filedialog.askdirectory(title="Select Folder to Save TIFFs")
        if not d: return
        try:
            tasks = [("LDA_Scores.tiff", self._create_fig_scores), ("LDA_ConfusionMatrix.tiff", self._create_fig_cm), 
                     ("LDA_ROC.tiff", self._create_fig_roc), ("LDA_Probabilities.tiff", self._create_fig_probabilities)]
            c = 0
            for name, func in tasks:
                fig = func()
                if fig: fig.savefig(os.path.join(d, name), format='tiff', dpi=300); plt.close(fig); c += 1
            messagebox.showinfo("Success", f"{c} Images saved!")
        except Exception as e: messagebox.showerror("Error", str(e))

    def export_excel(self):
        if not self.model_results: return
        f = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if f:
            data = {k: v for k, v in self.model_results.items() if isinstance(v, (int, float, str))}
            pd.DataFrame([data]).to_excel(f, index=False); messagebox.showinfo("Export", "Metrics exported to Excel!")

if __name__ == "__main__":
    app = Standard_LDA_App()
    app.root.mainloop()