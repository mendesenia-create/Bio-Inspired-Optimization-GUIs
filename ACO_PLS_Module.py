import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import scipy.io as sio
from joblib import Parallel, delayed, parallel_backend
import webbrowser
import random
import ctypes
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import f as f_dist, linregress

# =============================================================================
#  FUNÇÕES AUXILIARES (MOTOR MATEMÁTICO ROBUSTO)
# =============================================================================

def calculate_pls_metrics(y_true, y_pred):
    """Calcula RMSE, R2, Bias, RPD e REP(%) com robustez."""
    if y_true is None or y_pred is None or len(y_true) < 2:
        return np.inf, 0, 0, 0, 0
    
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    
    std_dev = np.std(y_true, ddof=1)
    rpd = std_dev / rmse if rmse > 0 else 0
    
    mean_y = np.mean(y_true)
    rep = (rmse / mean_y) * 100 if mean_y != 0 else 0
    
    return rmse, r2, bias, rpd, rep

def fcrit(p, n1, n2):
    return f_dist.ppf(1 - p, n1, n2)

# =============================================================================
#  MOTOR DO ALGORITMO ACO
# =============================================================================

class ACO_PLS_Selector:
    def __init__(self, X_train, Y_train, X_val_orig, Y_val_orig, X_pred_orig, Y_pred_orig,
                 n_ants=20, n_iterations=100, decay_rate=0.05, 
                 initial_pheromone=1.0, alpha_aco=1.0, beta_aco=0.0,
                 min_vars=5, max_vars_ratio=0.8):
        
        self.X_cal = X_train
        self.y_cal = Y_train.flatten()
        self.X_val = X_val_orig
        self.y_val = Y_val_orig.flatten() if Y_val_orig is not None else None
        self.X_pred = X_pred_orig
        self.y_pred = Y_pred_orig.flatten() if Y_pred_orig is not None else None
        
        self.n_samples, self.n_features = self.X_cal.shape
        self.n_ants = int(n_ants)
        self.n_iterations = int(n_iterations)
        self.decay_rate = float(decay_rate)
        self.initial_pheromone = float(initial_pheromone)
        self.alpha_aco = float(alpha_aco) 
        self.beta_aco = float(beta_aco) 
        self.min_vars = int(min_vars)
        self.max_vars = int(self.n_features * float(max_vars_ratio))
        
        self.pheromones = np.full(self.n_features, initial_pheromone)
        self.best_subset_indices = None
        self.best_rmse = float('inf')
        self.history_rmse = []
        self.cycle_best_subsets = []

    def _evaluate_subset_cost(self, selected_indices):
        if len(selected_indices) < self.min_vars: return float('inf')
        X_subset = self.X_cal[:, selected_indices]
        max_lvs = min(10, X_subset.shape[0] - 1, X_subset.shape[1])
        if max_lvs < 1: return float('inf')
        
        rmsecv_per_lv = []
        for lv in range(1, max_lvs + 1):
            pls = PLSRegression(n_components=lv)
            y_cv = cross_val_predict(pls, X_subset, self.y_cal, cv=5, n_jobs=1) 
            rmse = np.sqrt(mean_squared_error(self.y_cal, y_cv))
            rmsecv_per_lv.append(rmse)
        return min(rmsecv_per_lv) if rmsecv_per_lv else float('inf')

    def _generate_ant_subset(self):
        total_pheromones = np.sum(self.pheromones)
        if total_pheromones == 0: probabilities = np.full(self.n_features, 1.0 / self.n_features)
        else: probabilities = self.pheromones**self.alpha_aco / total_pheromones
            
        r = np.random.rand(self.n_features)
        selected_indices = np.where(r < probabilities)[0]
        
        if len(selected_indices) < self.min_vars:
            needed = self.min_vars - len(selected_indices)
            pool = [idx for idx in np.argsort(self.pheromones)[::-1] if idx not in selected_indices]
            selected_indices = np.concatenate((selected_indices, pool[:needed]))
            
        if len(selected_indices) > self.max_vars:
            selected_indices = sorted(selected_indices, key=lambda x: self.pheromones[x], reverse=True)
            selected_indices = np.array(selected_indices[:self.max_vars])
            
        return np.unique(selected_indices).astype(int)

    def _update_pheromones(self, ant_results):
        self.pheromones *= (1 - self.decay_rate)
        valid_results = [res for res in ant_results if res[1] != float('inf')]
        if not valid_results: return
        sorted_results = sorted(valid_results, key=lambda x: x[1])
        num_depositors = max(1, int(self.n_ants * 0.1))
        for i in range(min(num_depositors, len(sorted_results))):
            subset, rmse = sorted_results[i]
            if rmse > 0:
                deposit = 1.0 / rmse
                self.pheromones[subset] += deposit
        self.pheromones = np.clip(self.pheromones, 1e-6, 1000.0)

    def _evaluate_single_ant(self):
        subset = self._generate_ant_subset()
        cost = self._evaluate_subset_cost(subset)
        return (subset, cost)

    def run(self):
        parallel_executor = Parallel(n_jobs=-1, backend='threading')
        
        for iteration in range(self.n_iterations):
            ant_results = parallel_executor(delayed(self._evaluate_single_ant)() for _ in range(self.n_ants))
            
            current_iter_best_subset = None
            current_iter_best_cost = float('inf')
            
            for subset, cost in ant_results:
                if cost < self.best_rmse:
                    self.best_rmse, self.best_subset_indices = cost, subset
                if cost < current_iter_best_cost:
                    current_iter_best_cost, current_iter_best_subset = cost, subset
            
            self._update_pheromones(ant_results)
            self.history_rmse.append(self.best_rmse)
            self.cycle_best_subsets.append(current_iter_best_subset if current_iter_best_subset is not None else self.best_subset_indices)

        if self.best_subset_indices is None or len(self.best_subset_indices) == 0:
            return {'error': 'No solution found'}

        # CONSTRUÇÃO DO MODELO FINAL
        selected_vars = self.best_subset_indices
        X_cal_sel = self.X_cal[:, selected_vars]
        
        max_lvs_final = min(15, X_cal_sel.shape[0]-1, X_cal_sel.shape[1])
        rmsecv_per_lv = []
        with parallel_backend('threading'):
            for lv in range(1, max_lvs_final + 1):
                pls = PLSRegression(n_components=lv)
                y_cv = cross_val_predict(pls, X_cal_sel, self.y_cal, cv=10, n_jobs=-1)
                rmsecv_per_lv.append(np.sqrt(mean_squared_error(self.y_cal, y_cv)))
        
        optimal_lvs = np.argmin(rmsecv_per_lv) + 1
        final_rmsecv = min(rmsecv_per_lv)
        
        final_y_cv = cross_val_predict(PLSRegression(optimal_lvs), X_cal_sel, self.y_cal, cv=10, n_jobs=-1)
        r_cv = r2_score(self.y_cal, final_y_cv)
        
        final_pls = PLSRegression(n_components=optimal_lvs)
        final_pls.fit(X_cal_sel, self.y_cal)
        
        Model = {'selected_variables': selected_vars.tolist(), 'num_selected_variables': len(selected_vars),
                 'optimal_lvs': optimal_lvs, 'RMSECV': final_rmsecv, 'Rcv': r_cv}
        
        y_cal_est = final_pls.predict(X_cal_sel)
        Model.update({'Ycal_est': y_cal_est, 'Ycal': self.y_cal})
        Model['RMSEC'], Model['R2_cal'], Model['Bias_cal'], Model['RPD_cal'], Model['REP_cal'] = calculate_pls_metrics(self.y_cal, y_cal_est)
        
        if self.X_val is not None and self.y_val is not None:
            y_val_est = final_pls.predict(self.X_val[:, selected_vars])
            Model.update({'Yval_est': y_val_est, 'Yval': self.y_val})
            Model['RMSEV'], Model['R2_val'], Model['Bias_val'], Model['RPD_val'], Model['REP_val'] = calculate_pls_metrics(self.y_val, y_val_est)
        else:
             for k in ['RMSEV', 'R2_val', 'Bias_val', 'RPD_val', 'REP_val']: Model[k] = np.nan

        if self.X_pred is not None and self.y_pred is not None:
            y_pred_est = final_pls.predict(self.X_pred[:, selected_vars])
            Model.update({'Ypred_est': y_pred_est, 'Ypred': self.y_pred})
            Model['RMSEP'], Model['R2_pred'], Model['Bias_pred'], Model['RPD_pred'], Model['REP_pred'] = calculate_pls_metrics(self.y_pred, y_pred_est)
        else:
            for k in ['RMSEP', 'R2_pred', 'Bias_pred', 'RPD_pred', 'REP_pred']: Model[k] = np.nan
            
        return Model

# =============================================================================
#  CLASSE DA INTERFACE GRÁFICA (VISUAL GOLD STANDARD)
# =============================================================================
class ACO_PLS_App:
    def __init__(self, root_master=None):
        if root_master:
            self.root = tk.Toplevel(root_master)
        else:
            self.root = tk.Tk()
        
        self.root.title("ACO-PLS Interface")
        self.root.geometry("1600x900")
        
        # Variáveis de Estado
        self.all_repetitions_data = []
        self.selected_repetition_index = -1
        self.Model_ACO_PLS_global = None
        self.optimization_evolution_global = None
        self.Xcal_data = None; self.Ycal_data = None
        self.Xval_data = None; self.Yval_data = None
        self.Xpred_data = None; self.Ypred_data = None
        self.entries = {}
        self.xaxis_data = None
        
        # Variáveis para a janela de exportação
        self.check_vars = {}

        self._setup_gui_components()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        self.root.destroy()

    def get_safe_scalar_value(self, value):
        if isinstance(value, np.ndarray):
            if value.size == 1: return float(value.item())
            try: return float(value.flatten()[0])
            except: return np.nan
        if isinstance(value, (int, float)): return float(value)
        return str(value)

    # --- CÁLCULO ELIPSE (Helper interno para GUI) ---
    def calculate_ejcr_gui(self, y_real, y_pred):
        try:
            y_real, y_pred = y_real.flatten(), y_pred.flatten()
            Iv = len(y_real)
            if Iv < 3: return None
            slope, intercept, _, _, _ = linregress(y_real, y_pred)
            R = np.array([[Iv, np.sum(y_real)], [np.sum(y_real), np.sum(y_real**2)]])
            g = np.array([[np.sum(y_pred)], [np.sum(y_real * y_pred)]])
            if np.linalg.det(R) == 0: return None
            b = np.linalg.inv(R) @ g
            n, m = b[0,0], b[1,0]
            sfit = np.sqrt(np.sum((y_pred - (m * y_real + n))**2) / (Iv - 2))
            f_critico = fcrit(0.05, 2, Iv - 2)
            d1 = 2 * sfit**2 * f_critico
            a1, b1, c1 = np.sum(y_real**2), -2 * np.sum(y_real), Iv
            denominator = -b1**2 + 4 * a1 * c1
            if denominator == 0: return None
            discriminant_sqrt_term = 4 * c1 * d1 / denominator
            if discriminant_sqrt_term < 0: return None
            limx1 = np.sqrt(discriminant_sqrt_term)
            xx1 = np.linspace(-limx1, limx1, 101)
            sqrt_term_inner = b1**2 * xx1**2 - 4 * c1 * (a1 * xx1**2 - d1)
            sqrt_term_inner[sqrt_term_inner < 0] = 0
            yy11 = (-b1 * xx1 + np.sqrt(sqrt_term_inner)) / (2 * c1)
            yy21 = (-b1 * xx1 - np.sqrt(sqrt_term_inner)) / (2 * c1)
            return {'x_coords': xx1 + m, 'y1_coords': yy11 + n, 'y2_coords': yy21 + n, 'ideal_point_x': 1.0, 'ideal_point_y': 0.0}
        except: return None

    # --- EXPORTAÇÃO ---
    def export_model_to_matlab(self):
        if not self.Model_ACO_PLS_global: return
        file_path = filedialog.asksaveasfilename(defaultextension=".mat", filetypes=[("MATLAB Files", "*.mat")])
        if not file_path: return
        try:
            mat_data = {
                'Model': self.Model_ACO_PLS_global, 'Evolution': self.optimization_evolution_global,
                'Data': {'Xcal': self.Xcal_data, 'Ycal': self.Ycal_data, 'Xval': self.Xval_data, 'Yval': self.Yval_data, 'Xpred': self.Xpred_data, 'Ypred': self.Ypred_data}
            }
            sio.savemat(file_path, mat_data)
            messagebox.showinfo("Success", f"Exported to:\n{os.path.abspath(file_path)}")
        except Exception as e: messagebox.showerror("Error", str(e))

    def export_metrics_to_excel(self):
        if not self.Model_ACO_PLS_global: return
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if not file_path: return
        try:
            data = []
            for k, v in self.Model_ACO_PLS_global.items():
                if k not in ['Ycal_est', 'Yval_est', 'Ypred_est', 'Ycal', 'Yval', 'Ypred', 'selected_variables']:
                    data.append([k, self.get_safe_scalar_value(v)])
            pd.DataFrame(data, columns=['Metric', 'Value']).to_excel(file_path, index=False)
            messagebox.showinfo("Success", "Metrics exported.")
        except Exception as e: messagebox.showerror("Error", str(e))

    # --- NOVO SISTEMA DE EXPORTAÇÃO DE GRÁFICOS ---
    
    def plot_single_graph_on_ax(self, ax, ptype):
        """Função auxiliar para desenhar um único gráfico em um eixo dado."""
        model_data = self.Model_ACO_PLS_global
        evolution_data = self.optimization_evolution_global
        
        if ptype == 'selected_variables':
            mean_spec = np.mean(self.Xcal_data, axis=0)
            xaxis = self.xaxis_data if self.xaxis_data is not None else np.arange(len(mean_spec))
            ax.plot(xaxis, mean_spec, color='gray', alpha=0.5)
            sel = model_data.get('selected_variables', [])
            if len(sel) > 0:
                ax.plot(xaxis[sel], mean_spec[sel], 'ro', markersize=3)
            ax.set_title("Selected Variables", fontweight='bold')
            ax.set_xlabel('Variables', fontweight='bold'); ax.set_ylabel('Intensity', fontweight='bold')
            
        elif ptype == 'calibration':
            yc, yce = np.asarray(model_data['Ycal']), np.asarray(model_data['Ycal_est'])
            ax.plot(yc, yc, 'k-'); ax.plot(yc, yce, 'ob', alpha=0.6)
            ax.set_title("Calibration", fontweight='bold'); ax.grid(True)
            ax.set_xlabel("True Value", fontweight='bold'); ax.set_ylabel("Predicted Value", fontweight='bold')
            
        elif ptype == 'validation':
            if 'Yval' in model_data and np.asarray(model_data['Yval']).size > 0:
                yv, yve = np.asarray(model_data['Yval']), np.asarray(model_data['Yval_est'])
                ax.plot(yv, yv, 'k-'); ax.plot(yv, yve, 'or', alpha=0.6)
                ax.set_title("External Validation", fontweight='bold')
            else: 
                ax.text(0.5, 0.5, "No Validation", ha='center')
                ax.set_title("Validation", fontweight='bold')
            ax.grid(True); ax.set_xlabel("True Value", fontweight='bold'); ax.set_ylabel("Predicted Value", fontweight='bold')
            
        elif ptype == 'prediction':
            if 'Ypred' in model_data and np.asarray(model_data['Ypred']).size > 0:
                yp, ype = np.asarray(model_data['Ypred']), np.asarray(model_data['Ypred_est'])
                ax.plot(yp, yp, 'k-'); ax.plot(yp, ype, 'sk', alpha=0.6)
                ax.set_title("Prediction", fontweight='bold')
            else: 
                ax.text(0.5, 0.5, "No Prediction", ha='center')
                ax.set_title("Prediction", fontweight='bold')
            ax.grid(True); ax.set_xlabel("True Value", fontweight='bold'); ax.set_ylabel("Predicted Value", fontweight='bold')
            
        elif ptype == 'evolution':
            cyc = evolution_data['iterations']
            cost = evolution_data['cost_evolution']
            ax.plot(cyc, cost, 'b-o', markersize=4)
            ax.set_title("RMSECV Evolution", fontweight='bold'); ax.grid(True)
            ax.set_xlabel("Cycle", fontweight='bold'); ax.set_ylabel("RMSECV", fontweight='bold')
            
        elif ptype == 'ejcr':
            if 'Ypred' in model_data and np.asarray(model_data['Ypred']).size > 3:
                ejcr = self.calculate_ejcr_gui(np.asarray(model_data['Ypred']), np.asarray(model_data['Ypred_est']))
                if ejcr:
                    xl = np.concatenate([ejcr['x_coords'], ejcr['x_coords'][::-1]])
                    yl = np.concatenate([ejcr['y1_coords'], ejcr['y2_coords'][::-1]])
                    ax.plot(xl, yl, 'b-')
                    ax.plot(ejcr['ideal_point_x'], ejcr['ideal_point_y'], 'ro', markersize=6)
            ax.set_title("Confidence Ellipse", fontweight='bold'); ax.grid(True)
            ax.set_xlabel("Slope", fontweight='bold'); ax.set_ylabel("Intercept", fontweight='bold')

    def save_batch_plots(self, selected_plots, file_format, top_window):
        """Salva todos os gráficos selecionados de uma vez."""
        if not selected_plots:
            messagebox.showwarning("Warning", "No graphs selected!")
            return
            
        # Pede um prefixo/nome base para os arquivos
        initial_file = f"Repetition_{self.selected_repetition_index + 1}"
        file_path = filedialog.asksaveasfilename(
            defaultextension=file_format,
            initialfile=initial_file,
            title="Save Selected Graphs (Choose base name)",
            filetypes=[(f"{file_format.upper()} files", f"*{file_format}"), ("All files", "*.*")]
        )
        
        if not file_path: return
        
        # Remove a extensão se o usuário digitou, para adicionar os sufixos corretamente
        if file_path.lower().endswith(file_format):
            base_path = file_path[:-len(file_format)]
        else:
            base_path = file_path

        try:
            for ptype in selected_plots:
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111)
                self.plot_single_graph_on_ax(ax, ptype)
                
                # Nome do arquivo: BaseName_PlotType.ext
                save_name = f"{base_path}_{ptype}{file_format}"
                fig.tight_layout()
                fig.savefig(save_name, dpi=300)
                plt.close(fig)
            
            messagebox.showinfo("Success", f"Saved {len(selected_plots)} graphs successfully!")
            top_window.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error saving graphs: {str(e)}")

    def save_all_plots_composite(self):
        """Salva o Dashboard completo (Composite)"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png", 
            filetypes=[("PNG", "*.png"), ("TIFF", "*.tiff"), ("JPG", "*.jpg"), ("PDF", "*.pdf")]
        )
        if not file_path: return
        fig = plt.figure(figsize=(18, 9))
        self.draw_results_plots(fig, None, self.Model_ACO_PLS_global, self.optimization_evolution_global)
        fig.savefig(file_path, dpi=300)
        plt.close(fig)
        webbrowser.open(file_path)

    def open_export_plot_options(self):
        """Janela popup atualizada com Checkboxes e TIFF."""
        if not self.Model_ACO_PLS_global: return
        
        top = tk.Toplevel(self.root)
        top.title("Export Graphs")
        top.geometry("350x450")
        top.transient(self.root)
        top.grab_set()
        
        tk.Label(top, text="Select Graphs to Export:", font=('Arial', 11, 'bold')).pack(pady=10)
        
        # Opções de gráficos
        plot_options = [
            ("Selected Variables", "selected_variables"),
            ("Calibration", "calibration"),
            ("Validation", "validation"),
            ("Prediction", "prediction"),
            ("RMSECV Evolution", "evolution"),
            ("Confidence Ellipse (EJCR)", "ejcr")
        ]
        
        self.check_vars = {}
        
        # Frame para checkboxes
        chk_frame = ttk.Frame(top)
        chk_frame.pack(pady=5, padx=20, fill='x')
        
        for text, key in plot_options:
            var = tk.BooleanVar(value=False)
            self.check_vars[key] = var
            ttk.Checkbutton(chk_frame, text=text, variable=var).pack(anchor='w', pady=2)
            
        # Seleção de Formato (incluindo TIFF)
        tk.Label(top, text="File Format:", font=('Arial', 10)).pack(pady=(15, 5))
        format_combo = ttk.Combobox(top, values=[".png", ".tiff", ".jpg", ".pdf", ".eps"], state="readonly", width=10)
        format_combo.set(".tiff") # Default TIFF como pedido
        format_combo.pack()
        
        # Botão Salvar Selecionados
        def run_batch_save():
            selected = [key for key, var in self.check_vars.items() if var.get()]
            fmt = format_combo.get()
            self.save_batch_plots(selected, fmt, top)
            
        ttk.Button(top, text="Save Selected Graphs", command=run_batch_save).pack(pady=20, fill='x', padx=20)
        
        ttk.Separator(top, orient='horizontal').pack(fill='x', padx=10, pady=5)
        
        # Botão Dashboard
        ttk.Button(top, text="Save Full Dashboard (All-in-One)", command=self.save_all_plots_composite).pack(pady=5, fill='x', padx=20)


    # --- PLOTAGEM DO DASHBOARD PRINCIPAL ---
    def draw_results_plots(self, figure, canvas, model_data, evolution_data, highlighted_cycle=None):
        figure.clear()
        if not model_data: return
        axs = figure.subplots(2, 3)
        figure.suptitle(f"Results for Repetition {self.selected_repetition_index + 1}", fontsize=12, fontweight='bold', y=0.98)

        # Usar a mesma lógica auxiliar para desenhar no dashboard
        # 1. Selected Variables
        try:
            mean_spec = np.mean(self.Xcal_data, axis=0)
            xaxis = self.xaxis_data if self.xaxis_data is not None else np.arange(len(mean_spec))
            axs[0,0].plot(xaxis, mean_spec, color='gray', alpha=0.5)
            sel = model_data.get('selected_variables', [])
            if len(sel) > 0:
                axs[0,0].plot(xaxis[sel], mean_spec[sel], 'ro', markersize=3)
            axs[0,0].set_title("Selected Variables", fontsize=9, fontweight='bold')
            axs[0,0].set_xlabel('Variables', fontsize=9, fontweight='bold'); axs[0,0].set_ylabel('Intensity', fontsize=9, fontweight='bold')
        except: axs[0,0].text(0.5, 0.5, "Error", ha='center')

        # 2. Calibration
        try:
            yc, yce = np.asarray(model_data['Ycal']), np.asarray(model_data['Ycal_est'])
            axs[0,1].plot(yc, yc, 'k-'); axs[0,1].plot(yc, yce, 'ob', alpha=0.6)
            axs[0,1].set_title("Calibration", fontsize=9, fontweight='bold'); axs[0,1].grid(True)
            axs[0,1].set_xlabel("True Value", fontsize=9, fontweight='bold'); axs[0,1].set_ylabel("Predicted Value", fontsize=9, fontweight='bold')
        except: axs[0,1].text(0.5, 0.5, "N/A", ha='center')

        # 3. Validation
        try:
            if 'Yval' in model_data and np.asarray(model_data['Yval']).size > 0:
                yv, yve = np.asarray(model_data['Yval']), np.asarray(model_data['Yval_est'])
                axs[0,2].plot(yv, yv, 'k-'); axs[0,2].plot(yv, yve, 'or', alpha=0.6)
                axs[0,2].set_title("External Validation", fontsize=9, fontweight='bold')
            else: axs[0,2].text(0.5, 0.5, "No Validation", ha='center'); axs[0,2].set_title("Validation", fontsize=9, fontweight='bold')
            axs[0,2].grid(True); axs[0,2].set_xlabel("True Value", fontsize=9, fontweight='bold'); axs[0,2].set_ylabel("Predicted Value", fontsize=9, fontweight='bold')
        except: axs[0,2].text(0.5, 0.5, "N/A", ha='center')

        # 4. Prediction
        try:
            if 'Ypred' in model_data and np.asarray(model_data['Ypred']).size > 0:
                yp, ype = np.asarray(model_data['Ypred']), np.asarray(model_data['Ypred_est'])
                axs[1,0].plot(yp, yp, 'k-'); axs[1,0].plot(yp, ype, 'sk', alpha=0.6)
                axs[1,0].set_title("Prediction", fontsize=9, fontweight='bold')
            else: axs[1,0].text(0.5, 0.5, "No Prediction", ha='center'); axs[1,0].set_title("Prediction", fontsize=9, fontweight='bold')
            axs[1,0].grid(True); axs[1,0].set_xlabel("True Value", fontsize=9, fontweight='bold'); axs[1,0].set_ylabel("Predicted Value", fontsize=9, fontweight='bold')
        except: axs[1,0].text(0.5, 0.5, "N/A", ha='center')

        # 5. Evolution
        try:
            cyc = evolution_data['iterations']
            cost = evolution_data['cost_evolution']
            axs[1,1].plot(cyc, cost, 'b-o', markersize=4)
            if highlighted_cycle:
                idx = np.where(np.array(cyc) == highlighted_cycle)[0]
                if idx.size > 0: axs[1,1].plot(cyc[idx[0]], cost[idx[0]], 'ro', markersize=8, zorder=5)
            axs[1,1].set_title("RMSECV Evolution", fontsize=9, fontweight='bold'); axs[1,1].grid(True)
            axs[1,1].set_xlabel("Cycle", fontsize=9, fontweight='bold'); axs[1,1].set_ylabel("RMSECV", fontsize=9, fontweight='bold')
        except: axs[1,1].text(0.5, 0.5, "N/A", ha='center')

        # 6. EJCR
        try:
            if 'Ypred' in model_data and np.asarray(model_data['Ypred']).size > 3:
                ejcr = self.calculate_ejcr_gui(np.asarray(model_data['Ypred']), np.asarray(model_data['Ypred_est']))
                if ejcr:
                    xl = np.concatenate([ejcr['x_coords'], ejcr['x_coords'][::-1]])
                    yl = np.concatenate([ejcr['y1_coords'], ejcr['y2_coords'][::-1]])
                    axs[1,2].plot(xl, yl, 'b-')
                    axs[1,2].plot(ejcr['ideal_point_x'], ejcr['ideal_point_y'], 'ro', markersize=6)
            axs[1,2].set_title("Confidence Ellipse", fontsize=9, fontweight='bold'); axs[1,2].grid(True)
            axs[1,2].set_xlabel("Slope", fontsize=9, fontweight='bold'); axs[1,2].set_ylabel("Intercept", fontsize=9, fontweight='bold')
        except: axs[1,2].text(0.5, 0.5, "N/A", ha='center')

        if canvas:
            figure.tight_layout(rect=[0, 0, 1, 0.95])
            canvas.draw()

    # --- LOGICA DA INTERFACE ---
    def clear_frame(self, frame):
        if frame:
            for widget in frame.winfo_children(): widget.destroy()

    def display_summary_report(self, all_data):
        self.clear_frame(self.initial_summary_frame)
        if not all_data: ttk.Label(self.initial_summary_frame, text="No results.").pack(); return
        
        rmsep_values = [self.get_safe_scalar_value(rep.get('Model', {}).get('RMSEP', np.inf)) for rep in all_data]
        if not rmsep_values: return
        best_rep_idx = np.argmin(rmsep_values)
        
        header_text = f"Best Result: Repetition {best_rep_idx + 1} (RMSEP = {rmsep_values[best_rep_idx]:.4f})"
        ttk.Label(self.initial_summary_frame, text=header_text, font=('Arial', 11, 'bold'), foreground='green').pack(anchor='w', padx=10, pady=(5,0))
        ttk.Separator(self.initial_summary_frame, orient='horizontal').pack(fill='x', padx=10, pady=5)
        ttk.Label(self.initial_summary_frame, text="Summary of All Repetitions:", font=('Arial', 10, 'underline')).pack(anchor='w', padx=10, pady=(5,5))
        
        for i, rep in enumerate(all_data):
            rmsep = rmsep_values[i]
            line = f"Repetition {i+1}: RMSEP = {rmsep:.4f}"
            font_style = ('Courier New', 10, 'bold') if i == best_rep_idx else ('Courier New', 10)
            fg_color = 'green' if i == best_rep_idx else 'black'
            ttk.Label(self.initial_summary_frame, text=line, font=font_style, foreground=fg_color).pack(anchor='w', padx=10)

    def display_initial_details_report(self, model):
        self.clear_frame(self.initial_details_frame)
        metrics = ['optimal_lvs', 'num_selected_variables', 'RMSECV', 'Rcv', 
                   'RMSEC', 'R2_cal', 'RPD_cal', 'REP_cal', 
                   'RMSEP', 'R2_pred', 'RPD_pred', 'REP_pred']
        
        r = 0
        for m in metrics:
            if m in model:
                val = model[m]
                val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
                ttk.Label(self.initial_details_frame, text=m+":", font=('Arial',9,'bold')).grid(row=r, column=0, sticky='e', padx=(10,5), pady=1)
                ttk.Label(self.initial_details_frame, text=val_str).grid(row=r, column=1, sticky='w', padx=5, pady=1)
                r+=1

    def view_initial_details(self, event=None):
        try:
            rep_idx_str = self.initial_repetition_selector.get()
            if not rep_idx_str: return
            idx = int(rep_idx_str) - 1
            self.selected_repetition_index = idx
            
            data = self.all_repetitions_data[idx]
            self.Model_ACO_PLS_global = data['Model']
            self.optimization_evolution_global = data['Evolution']
            
            self.display_initial_details_report(self.Model_ACO_PLS_global)
            self.draw_results_plots(self.initial_plot_figure, self.initial_plot_canvas, self.Model_ACO_PLS_global, self.optimization_evolution_global)
            
            self.export_graphics_button.config(state=tk.NORMAL)
            self.export_excel_button.config(state=tk.NORMAL)
            self.export_matlab_button.config(state=tk.NORMAL)
            
            self.status_label.config(text=f"Rep. {idx+1}: Details loaded.")
        except: messagebox.showerror("Error", "Select valid repetition.")

    def on_repetition_select(self, event):
        sel = self.summary_tree.selection()
        if not sel: return
        item = self.summary_tree.item(sel[0])
        idx = int(item['values'][0]) - 1
        self.selected_repetition_index = idx
        data = self.all_repetitions_data[idx]
        self.Model_ACO_PLS_global = data['Model']
        self.optimization_evolution_global = data['Evolution']
        
        for i in self.cycle_tree.get_children(): self.cycle_tree.delete(i)
        cycles, cost_evol = self.optimization_evolution_global.get('iterations', []), self.optimization_evolution_global.get('cost_evolution', [])
        if cost_evol:
            best_c = min(cost_evol)
            for i, (c, cost) in enumerate(zip(cycles, cost_evol)):
                tags = ('best_cycle',) if cost == best_c else ()
                self.cycle_tree.insert("", "end", values=(int(c), f"{cost:.4f}"), tags=tags)
        
        self.draw_results_plots(self.interactive_plot_figure, self.interactive_plot_canvas, self.Model_ACO_PLS_global, self.optimization_evolution_global)
        self.status_label.config(text=f"Analyzing Rep. {idx+1}.")

    def on_cycle_select(self, event):
        sel = self.cycle_tree.selection()
        if not sel: return
        item = self.cycle_tree.item(sel[0])
        cyc = int(item['values'][0])
        self.draw_results_plots(self.interactive_plot_figure, self.interactive_plot_canvas, self.Model_ACO_PLS_global, self.optimization_evolution_global, highlighted_cycle=cyc)

    def get_entry_value(self, entry, name, dtype):
        v = entry.get().strip()
        if not v: raise ValueError(f"Empty {name}")
        return dtype(v)

    def run_multi_repetitions_handler(self):
        if self.Xcal_data is None: messagebox.showerror("Error", "Load Data first."); return
        try:
            self.all_repetitions_data = []
            for t in [self.summary_tree, self.cycle_tree]:
                for i in t.get_children(): t.delete(i)
            self.clear_frame(self.initial_summary_frame)
            self.clear_frame(self.initial_details_frame)
            self.initial_repetition_selector.set('')
            self.initial_repetition_selector['values'] = []
            
            params = {
                'n_ants': self.get_entry_value(self.entries['entry_ants'], "Ants", int),
                'n_iterations': self.get_entry_value(self.entries['entry_iterations'], "Iter", int),
                'decay_rate': self.get_entry_value(self.entries['entry_decay'], "Decay", float),
                'initial_pheromone': self.get_entry_value(self.entries['entry_pheromone'], "Pheromone", float),
                'alpha_aco': self.get_entry_value(self.entries['entry_alpha'], "Alpha", float),
                'beta_aco': self.get_entry_value(self.entries['entry_beta'], "Beta", float),
                'min_vars': self.get_entry_value(self.entries['entry_min_vars'], "Min Vars", int),
                'max_vars_ratio': self.get_entry_value(self.entries['entry_max_vars'], "Max Vars", float)
            }
            reps = self.get_entry_value(self.entries['entry_repetitions'], "Reps", int)
            
            for i in range(reps):
                self.status_label.config(text=f"Running Repetition {i+1}/{reps}..."); self.root.update()
                
                aco = ACO_PLS_Selector(self.Xcal_data, self.Ycal_data, self.Xval_data, self.Yval_data, self.Xpred_data, self.Ypred_data, **params)
                model = aco.run()
                
                if 'error' in model: continue
                
                evolution = {
                    'iterations': list(range(1, len(aco.history_rmse)+1)),
                    'cost_evolution': aco.history_rmse,
                    'cycle_vars': aco.cycle_best_subsets
                }
                self.all_repetitions_data.append({'Model': model, 'Evolution': evolution})
            
            self.display_summary_report(self.all_repetitions_data)
            self.initial_repetition_selector['values'] = list(range(1, len(self.all_repetitions_data)+1))
            if self.all_repetitions_data:
                self.initial_repetition_selector.set("1")
                self.view_initial_details()
                self.view_details_button.config(state=tk.NORMAL)
                
            for i, r in enumerate(self.all_repetitions_data):
                m = r['Model']
                self.summary_tree.insert("", "end", values=(i+1, m.get('optimal_lvs',0), 
                                                            f"{m.get('RMSECV',0):.4f}", f"{m.get('RMSEP',0):.4f}", 
                                                            f"{m.get('R2_cal',0):.4f}", f"{m.get('R2_pred',0):.4f}",
                                                            f"{m.get('RPD_pred',0):.2f}", f"{m.get('REP_pred',0):.2f}"))
            
            self.notebook.select(0)
            self.status_label.config(text="Execution Finished.")
            messagebox.showinfo("Done", "All repetitions finished.")
            
        except Exception as e:
            self.status_label.config(text="Error"); messagebox.showerror("Error", str(e))

    def load_data(self):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("MAT Files", "*.mat")])
            if not file_path: return
            mat = sio.loadmat(file_path)
            keys = mat.keys()
            def fk(pats):
                for k in keys:
                    if k.startswith('__'): continue
                    for p in pats: 
                        if p.lower() == k.lower(): return k
                return None
            kx, ky = fk(['Xcal','X_train']), fk(['Ycal','Y_train'])
            if kx and ky: self.Xcal_data, self.Ycal_data = mat[kx].astype(float), mat[ky].astype(float).flatten()
            else: raise Exception("Training data missing")
            vx, vy = fk(['Xval','X_val']), fk(['Yval','Y_val'])
            if vx and vy: self.Xval_data, self.Yval_data = mat[vx].astype(float), mat[vy].astype(float).flatten()
            else: self.Xval_data, self.Yval_data = None, None
            px, py = fk(['Xpred','X_test']), fk(['Ypred','Y_test'])
            if px: self.Xpred_data, self.Ypred_data = mat[px].astype(float), mat[py].astype(float).flatten() if py else None
            else: self.Xpred_data, self.Ypred_data = None, None
            if 'xaxis' in mat: self.xaxis_data = np.asarray(mat['xaxis']).flatten()
            else: self.xaxis_data = np.arange(self.Xcal_data.shape[1])
            self.status_label.config(text="Data Loaded.")
            messagebox.showinfo("Success", "Data Loaded.")
        except Exception as e: messagebox.showerror("Error", str(e))

    def reset_all(self):
        self.all_repetitions_data = []
        self.clear_frame(self.initial_details_frame)
        self.clear_frame(self.initial_summary_frame)
        for t in [self.summary_tree, self.cycle_tree]:
            for i in t.get_children(): t.delete(i)
        self.initial_plot_figure.clear(); self.initial_plot_canvas.draw()
        self.interactive_plot_figure.clear(); self.interactive_plot_canvas.draw()
        self.status_label.config(text="Reset.")

    def _setup_gui_components(self):
        try: self.root.iconbitmap('aco_icon.ico')
        except: pass
        
        style = ttk.Style(self.root); style.theme_use("clam"); style.configure('TNotebook.Tab', font=('Arial', 11, 'bold')); style.configure('best_cycle.Treeview', background='lightcyan')
        
        self.status_label = tk.Label(self.root, text="Status: Waiting...", bd=1, relief=tk.SUNKEN, anchor=tk.W, padx=10)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        header_frame = tk.Frame(self.root, bg="#2c3e50", height=40); header_frame.pack(fill=tk.X)
        header_text_frame = tk.Frame(header_frame, bg="#2c3e50"); header_text_frame.pack() 
        tk.Label(header_text_frame, text="Regression Analysis with Ant Colony Optimization (ACO-PLS) \U0001F41C ", fg="white", bg="#2c3e50", font=("Arial", 14, "bold")).pack(side=tk.LEFT, pady=5, padx=10)
        tk.Label(header_text_frame, text="Developer: Enia Mendes", fg="white", bg="#2c3e50", font=("Arial", 9)).pack(side=tk.LEFT, pady=5, padx=10)

        main_frame = ttk.Frame(self.root); main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        control_frame = ttk.LabelFrame(main_frame, text="Controls and Parameters"); control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=5)
        tk.Button(control_frame, text="Load Data (.mat)", command=self.load_data, bg="#4682B4", fg="white").pack(pady=5, padx=5, fill=tk.X)
        
        param_frame = ttk.LabelFrame(control_frame, text="ACO Parameters"); param_frame.pack(pady=10, padx=5, fill=tk.X)
        aco_params = [("No. Ants:", "30", 'entry_ants'), ("No. Iterations:", "30", 'entry_iterations'), ("Evaporation Rate:", "0.05", 'entry_decay'), 
                      ("Initial Pheromone:", "1.0", 'entry_pheromone'), ("Alpha (Pheromone):", "1.0", 'entry_alpha'), ("Beta (Heuristic):", "0.1", 'entry_beta'), 
                      ("Min Vars No.:", "5", 'entry_min_vars'), ("Max Vars Ratio:", "0.8", 'entry_max_vars')]
        for i, (text, val, key) in enumerate(aco_params):
            ttk.Label(param_frame, text=text).grid(row=i, column=0, sticky="w", padx=2, pady=2)
            entry = ttk.Entry(param_frame, width=10); entry.insert(0, val); entry.grid(row=i, column=1, padx=5, pady=2, sticky="ew")
            self.entries[key] = entry
            
        ttk.Label(control_frame, text="Repetitions:").pack(pady=(10,0))
        entry_repetitions = ttk.Entry(control_frame); entry_repetitions.insert(0, "2"); entry_repetitions.pack(pady=2, padx=5, fill=tk.X); self.entries['entry_repetitions'] = entry_repetitions
        
        tk.Button(control_frame, text="Run Repetitions", command=self.run_multi_repetitions_handler, font=("Arial", 11, "bold"), bg="#8A2BE2", fg="white").pack(pady=10, fill=tk.X, padx=5)
        tk.Button(control_frame, text="Reset All", command=self.reset_all, bg="#B0C4DE", fg="black").pack(pady=5, fill=tk.X, padx=5)
        
        results_frame = ttk.Frame(main_frame); results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.notebook = ttk.Notebook(results_frame); self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # TAB 1: Initial Analysis
        tab1 = ttk.Frame(self.notebook); self.notebook.add(tab1, text='Initial Analysis')
        main_pane = ttk.PanedWindow(tab1, orient=tk.HORIZONTAL); main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        left_container = ttk.Frame(main_pane, width=400); main_pane.add(left_container, weight=1)
        left_pane = ttk.PanedWindow(left_container, orient=tk.VERTICAL); left_pane.pack(fill=tk.BOTH, expand=True)
        
        top_left_frame = ttk.LabelFrame(left_pane, text="Repetitions Summary"); left_pane.add(top_left_frame, weight=1)
        controls_frame = ttk.Frame(top_left_frame); controls_frame.pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Label(controls_frame, text="Analyze Rep.:").pack(side=tk.LEFT, padx=(0,2))
        self.initial_repetition_selector = ttk.Combobox(controls_frame, state="readonly", width=8); self.initial_repetition_selector.pack(side=tk.LEFT, padx=5)
        self.initial_repetition_selector.bind("<<ComboboxSelected>>", self.view_initial_details)
        self.view_details_button = tk.Button(controls_frame, text="View Details", command=self.view_initial_details, state=tk.DISABLED); self.view_details_button.pack(side=tk.LEFT, padx=5)
        
        self.export_graphics_button = tk.Button(controls_frame, text="Export Graphs...", command=self.open_export_plot_options, state=tk.DISABLED); self.export_graphics_button.pack(side=tk.LEFT, padx=2)
        self.export_excel_button = tk.Button(controls_frame, text="Export to Excel", command=self.export_metrics_to_excel, state=tk.DISABLED); self.export_excel_button.pack(side=tk.LEFT, padx=2)
        self.export_matlab_button = tk.Button(controls_frame, text="port to MATLAB (.mat)", command=self.export_model_to_matlab, state=tk.DISABLED); self.export_matlab_button.pack(side=tk.LEFT, padx=2)

        canvas = tk.Canvas(top_left_frame); scrollbar = ttk.Scrollbar(top_left_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set); canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.initial_summary_frame = ttk.Frame(canvas); canvas.create_window((0, 0), window=self.initial_summary_frame, anchor="nw")
        self.initial_summary_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        bottom_left_frame = ttk.LabelFrame(left_pane, text="Selected Repetition Metrics"); left_pane.add(bottom_left_frame, weight=2)
        self.initial_details_frame = ttk.Frame(bottom_left_frame); self.initial_details_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        right_frame = ttk.LabelFrame(main_pane, text="Graphic Dashboard"); main_pane.add(right_frame, weight=3)
        self.initial_plot_figure = plt.figure(); self.initial_plot_canvas = FigureCanvasTkAgg(self.initial_plot_figure, master=right_frame)
        self.initial_plot_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # TAB 2: Interactive Analysis
        tab2 = ttk.Frame(self.notebook); self.notebook.add(tab2, text='Interactive Analysis')
        paned_window_interactive = ttk.PanedWindow(tab2, orient=tk.VERTICAL); paned_window_interactive.pack(fill=tk.BOTH, expand=True)
        
        summary_frame = ttk.LabelFrame(paned_window_interactive, text="Repetitions Summary"); paned_window_interactive.add(summary_frame, weight=2)
        cols = ("Rep.", "LV", "RMSECV", "RMSEP", "R2 Cal", "R2 Pred", "RPD Pred", "REP Pred")
        self.summary_tree = ttk.Treeview(summary_frame, columns=cols, show='headings')
        for col in cols: self.summary_tree.heading(col, text=col); self.summary_tree.column(col, width=80, anchor=tk.CENTER)
        
        ttk.Scrollbar(summary_frame, orient="vertical", command=self.summary_tree.yview).pack(side='right', fill='y'); self.summary_tree.pack(fill=tk.BOTH, expand=True); self.summary_tree.bind('<<TreeviewSelect>>', self.on_repetition_select)
        
        cycle_frame = ttk.LabelFrame(paned_window_interactive, text="Cycles Evolution (Click to highlight)"); paned_window_interactive.add(cycle_frame, weight=2)
        cycle_cols = ("Cycle", "RMSECV"); self.cycle_tree = ttk.Treeview(cycle_frame, columns=cycle_cols, show='headings')
        self.cycle_tree.heading("Cycle", text="Cycle"); self.cycle_tree.column("Cycle", width=50, anchor=tk.CENTER)
        self.cycle_tree.heading("RMSECV", text="RMSECV"); self.cycle_tree.column("RMSECV", width=100, anchor=tk.CENTER)
        self.cycle_tree.tag_configure('best_cycle', background='lightcyan')
        vsb_cycle = ttk.Scrollbar(cycle_frame, orient="vertical", command=self.cycle_tree.yview); self.cycle_tree.configure(yscrollcommand=vsb_cycle.set); vsb_cycle.pack(side=tk.RIGHT, fill='y'); self.cycle_tree.pack(fill=tk.BOTH, expand=True); self.cycle_tree.bind('<<TreeviewSelect>>', self.on_cycle_select)
        
        plot_frame_interactive = ttk.LabelFrame(paned_window_interactive, text="Interactive Graphical Visualization"); paned_window_interactive.add(plot_frame_interactive, weight=4)
        self.interactive_plot_figure = plt.figure(); self.interactive_plot_canvas = FigureCanvasTkAgg(self.interactive_plot_figure, master=plot_frame_interactive)
        self.interactive_plot_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.interactive_plot_canvas, plot_frame_interactive); toolbar.update()

if __name__ == "__main__":
    try: ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except: pass
    app = ACO_PLS_App()
    app.root.mainloop()