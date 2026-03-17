import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import scipy.io as sio
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import random
import math
from scipy.stats import f as f_dist
import pandas as pd
import webbrowser
from joblib import Parallel, delayed, parallel_backend
import ctypes  # Para alta resolução no Windows

# =============================================================================
#  FUNÇÕES AUXILIARES E ALGORITMO BAT-PLS (BACKEND)
# =============================================================================

def calculate_pls_metrics(y_true, y_pred):
    """
    Calcula RMSE, R2, Bias, RPD e REP(%).
    """
    if y_true is None or y_pred is None or len(y_true) < 2:
        return np.inf, 0, 0, 0, 0
    
    # 1. RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # 2. R2
    r2 = r2_score(y_true, y_pred)
    
    # 3. Bias (Viés)
    bias = np.mean(y_pred.flatten() - y_true.flatten())
    
    # 4. RPD (Desvio Padrão / RMSE)
    std_dev = np.std(y_true, ddof=1) # ddof=1 para desvio padrão amostral
    rpd = std_dev / rmse if rmse > 0 else 0
    
    # 5. REP (%) = (RMSE / mean(y_true)) * 100
    mean_y = np.mean(y_true)
    rep = (rmse / mean_y) * 100 if mean_y != 0 else 0
    
    return rmse, r2, bias, rpd, rep

def fcrit(p, n1, n2):
    return f_dist.ppf(1 - p, n1, n2)

def gcost_pls_py(X_cal, y_cal, x_bin, n_min, n_max, max_lvs):
    with parallel_backend('threading'):
        selected_vars = np.where(x_bin)[0]
        num_selected = len(selected_vars)
        
        # Penalidade se o número de variáveis estiver fora dos limites
        if num_selected < n_min or num_selected > n_max:
            return 1e10
        
        X_cal_sel = X_cal[:, selected_vars]
        
        # Limita LVs ao tamanho da matriz
        current_max_lvs = min(max_lvs, X_cal_sel.shape[0] - 1, X_cal_sel.shape[1])
        if current_max_lvs < 1:
            return 1e10
            
        rmsecv_per_lv = []
        # Validação cruzada rápida para o custo
        for lv in range(1, current_max_lvs + 1):
            pls = PLSRegression(n_components=lv)
            y_cv_pred = cross_val_predict(pls, X_cal_sel, y_cal, cv=5, n_jobs=-1) 
            rmsecv = np.sqrt(mean_squared_error(y_cal, y_cv_pred))
            rmsecv_per_lv.append(rmsecv)
            
        return min(rmsecv_per_lv) if rmsecv_per_lv else 1e10

class BAT_PLS:
    def __init__(self, params):
        self.params = params
        self.cost_evolution_ = []

    def fit(self, X_cal, y_cal, X_val=None, y_val=None, X_pred=None, y_pred=None):
        n_samples, n_features = X_cal.shape
        p = self.params
        n_min = p['n_min']
        n_max = p['n_max']
        alpha = p['alpha']
        gamma = p['gamma']
        n_bats = p['mbats']
        n_iter = p['numberofiterations']
        fmin = p['fmin']
        fmax = p['fmax']
        
        sigma = 0.5
        max_lvs = min(n_samples - 1, 15)

        # Inicialização dos morcegos
        positions = np.zeros((n_bats, n_features))
        for i in range(n_bats):
            n_select = random.randint(n_min, n_max)
            temp = np.random.permutation(n_features)
            positions[i, temp[:n_select]] = sigma + (1 - sigma) * np.random.rand(n_select)
            positions[i, temp[n_select:]] = sigma * np.random.rand(n_features - n_select)
            
        velocities = np.zeros((n_bats, n_features))
        loudness = np.ones(n_bats)
        pulse_rate = np.random.rand(n_bats)
        initial_pulse_rate = pulse_rate.copy()

        # Avaliação Inicial
        with Parallel(n_jobs=-1, backend='threading') as parallel:
             costs = parallel(delayed(gcost_pls_py)(X_cal, y_cal, pos > sigma, n_min, n_max, max_lvs) for pos in positions)
        costs = np.array(costs)
        
        best_idx = np.argmin(costs)
        best_cost = costs[best_idx]
        best_position = positions[best_idx, :].copy()
        self.cost_evolution_.append(best_cost)

        # Loop Principal
        for t in range(1, n_iter):
            print(f"Iteration {t+1}/{n_iter} – Best RMSECV: {best_cost:.4f}")
            new_positions = positions.copy()
            mean_loudness = np.mean(loudness)
            
            for i in range(n_bats):
                beta = np.random.rand()
                freq = fmin + (fmax - fmin) * beta
                
                # Atualiza velocidade e posição
                velocities[i, :] += (positions[i, :] - best_position) * freq
                new_positions[i, :] = positions[i, :] + velocities[i, :]
                
                # Random Walk
                if np.random.rand() > pulse_rate[i]:
                    new_positions[i, :] += (2 * np.random.rand() - 1) * mean_loudness

            new_positions = np.clip(new_positions, 0, 1)

            # Avalia novas posições
            with Parallel(n_jobs=-1, backend='threading') as parallel:
                new_costs = parallel(delayed(gcost_pls_py)(X_cal, y_cal, n_pos > sigma, n_min, n_max, max_lvs) for n_pos in new_positions)
            new_costs = np.array(new_costs)

            # Atualiza morcegos (Loudness e Pulse Rate)
            for i in range(n_bats):
                if np.random.rand() < loudness[i] and new_costs[i] < costs[i]:
                    positions[i, :] = new_positions[i, :].copy()
                    costs[i] = new_costs[i]
                    loudness[i] *= alpha
                    pulse_rate[i] = initial_pulse_rate[i] * (1 - math.exp(-gamma * t))
            
            # Atualiza melhor global
            current_best_idx = np.argmin(costs)
            if costs[current_best_idx] < best_cost:
                best_cost = costs[current_best_idx]
                best_position = positions[current_best_idx, :].copy()
            
            self.cost_evolution_.append(best_cost)
            
        # --- Construção do Modelo Final ---
        selected_vars = np.where(best_position > sigma)[0]
        Model = {'selected_variables': selected_vars.tolist(), 'num_selected_variables': len(selected_vars)}
        Evolution = {'cost_evolution': self.cost_evolution_, 'iterations': list(range(1, n_iter + 1))}

        if not selected_vars.any():
            return Model, Evolution

        X_cal_sel = X_cal[:, selected_vars]
        final_max_lvs = min(max_lvs, X_cal_sel.shape[0] - 1, X_cal_sel.shape[1])
        if final_max_lvs < 1: final_max_lvs = 1
        
        # Otimiza LVs finais com CV de 10 folds
        with parallel_backend('threading'):
            rmsecv_per_lv = [np.sqrt(mean_squared_error(y_cal, cross_val_predict(PLSRegression(lv), X_cal_sel, y_cal, cv=10, n_jobs=-1))) for lv in range(1, final_max_lvs + 1)]

        optimal_lvs = np.argmin(rmsecv_per_lv) + 1
        Model['optimal_lvs'] = optimal_lvs
        Model['RMSECV'] = min(rmsecv_per_lv) if rmsecv_per_lv else np.inf
        
        # --- CÁLCULO DO Rcv (Q²) ---
        # Recalcula a predição CV final para obter o Rcv exato
        final_y_cv = cross_val_predict(PLSRegression(optimal_lvs), X_cal_sel, y_cal, cv=10, n_jobs=-1)
        Model['Rcv'] = r2_score(y_cal, final_y_cv) 

        # Ajuste Final
        final_pls = PLSRegression(n_components=optimal_lvs).fit(X_cal_sel, y_cal)
        
        # Calibração
        y_cal_est = final_pls.predict(X_cal_sel)
        Model.update({'Ycal_est': y_cal_est, 'Ycal': y_cal})
        Model['RMSEC'], Model['R2_cal'], Model['Bias_cal'], Model['RPD_cal'], Model['REP_cal'] = calculate_pls_metrics(y_cal, y_cal_est)

        # Validação
        if X_val is not None and y_val is not None and X_val.size > 0:
            y_val_est = final_pls.predict(X_val[:, selected_vars])
            Model.update({'Yval_est': y_val_est, 'Yval': y_val})
            Model['RMSEV'], Model['R2_val'], Model['Bias_val'], Model['RPD_val'], Model['REP_val'] = calculate_pls_metrics(y_val, y_val_est)
        else:
            for k in ['RMSEV', 'R2_val', 'Bias_val', 'RPD_val', 'REP_val']: Model[k] = np.nan

        # Predição
        if X_pred is not None and y_pred is not None and X_pred.size > 0:
            y_pred_est = final_pls.predict(X_pred[:, selected_vars])
            Model.update({'Ypred_est': y_pred_est, 'Ypred': y_pred})
            Model['RMSEP'], Model['R2_pred'], Model['Bias_pred'], Model['RPD_pred'], Model['REP_pred'] = calculate_pls_metrics(y_pred, y_pred_est)
        else:
            for k in ['RMSEP', 'R2_pred', 'Bias_pred', 'RPD_pred', 'REP_pred']: Model[k] = np.nan
            
        return Model, Evolution

# =============================================================================
#  CLASSE DA INTERFACE GRÁFICA (VISUAL PADRONIZADO E COMPLETO)
# =============================================================================
class BAT_PLS_App:
    def __init__(self, root_master=None):
        if root_master:
            self.root = tk.Toplevel(root_master)
        else:
            self.root = tk.Tk()
        
        self.root.title("BAT-PLS Interface")
        self.root.geometry("1600x900")
        
        # Variáveis de Estado
        self.all_repetitions_data = []
        self.selected_repetition_index = -1
        self.Model_BAT_global = None
        self.optimization_evolution_global = None
        self.Train_data = None
        self.Ycal_data = None
        self.Val_data = None
        self.Yval_data = None
        self.Pred_data = None
        self.Ypred_data = None
        self.entries = {}
        self.xaxis_data = None 

        self._setup_gui_components()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        self.root.destroy()

    def get_safe_scalar_value(self, value, default_nan=np.nan):
        if isinstance(value, np.ndarray):
            if value.size == 1: return float(value.item())
            else:
                try: return float(value.flatten()[0])
                except (IndexError, TypeError, ValueError): return default_nan
        elif isinstance(value, (int, float)): return float(value)
        return str(value)

    # --- CÁLCULOS AUXILIARES (ELIPSE) ---
    def calculate_ejcr(self, y_real, y_pred):
        try:
            y_real, y_pred = y_real.flatten(), y_pred.flatten()
            Iv = len(y_real)
            if Iv < 3: return None
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
        except Exception: return None

    # --- FUNÇÕES DE EXPORTAÇÃO ---
    def export_model_to_matlab(self):
        if not self.Model_BAT_global:
            messagebox.showwarning("Warning", "No model selected.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".mat", filetypes=[("MATLAB Files", "*.mat")], title="Export Model to MATLAB")
        if not file_path: return
        try:
            mat_data = {'Model': self.Model_BAT_global, 'Evolution': self.optimization_evolution_global, 'Data': {'X_train': self.Train_data, 'Y_train': self.Ycal_data, 'X_val': self.Val_data, 'Y_val': self.Yval_data, 'X_test': self.Pred_data, 'Y_test': self.Ypred_data}}
            sio.savemat(file_path, mat_data)
            messagebox.showinfo("Export Complete", f"Model exported to:\n{os.path.abspath(file_path)}")
        except Exception as e: messagebox.showerror("Export Error", str(e))

    def export_metrics_to_excel(self):
        if not self.Model_BAT_global:
            messagebox.showwarning("Warning", "No model selected.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Workbook", "*.xlsx")], title="Export Metrics to Excel")
        if not file_path: return
        try:
            data_for_df = []
            for key, value in self.Model_BAT_global.items():
                if key in ['Ycal_est', 'Yval_est', 'Ypred_est', 'Ycal', 'Yval', 'Ypred']: continue
                if isinstance(value, list): formatted_value = ', '.join(map(str, value))
                else: formatted_value = self.get_safe_scalar_value(value)
                data_for_df.append([key, formatted_value])
            df = pd.DataFrame(data_for_df, columns=['Metric', 'Value'])
            df.to_excel(file_path, index=False)
            messagebox.showinfo("Export Complete", f"Metrics saved to:\n{os.path.abspath(file_path)}")
        except Exception as e: messagebox.showerror("Export Error", str(e))

    # --- PLOTAGEM ---
    def draw_results_plots(self, target_figure, target_canvas, model_data, evolution_data, highlighted_cycle=None):
        target_figure.clear()
        axs = target_figure.subplots(2, 3)
        target_figure.suptitle(f'Results for Repetition {self.selected_repetition_index + 1}', fontsize=12, fontweight='bold', y=0.98)

        # 1. Selected Variables
        try:
            if self.Train_data is not None:
                mean_spectrum = np.mean(self.Train_data, axis=0)
                xaxis = self.xaxis_data if self.xaxis_data is not None else np.arange(len(mean_spectrum))
                axs[0,0].plot(xaxis, mean_spectrum, color='gray', alpha=0.5)
                selected_vars_indices = model_data.get('selected_variables', [])
                if selected_vars_indices:
                    axs[0,0].plot(xaxis[selected_vars_indices], mean_spectrum[selected_vars_indices], 'ro', markersize=3, label='Selected')
            axs[0,0].set_title('Selected Variables', fontsize=9, fontweight='bold')
            axs[0,0].set_xlabel('Variables', fontsize=9, fontweight='bold'); axs[0,0].set_ylabel('Intensity', fontsize=9, fontweight='bold')
        except Exception: axs[0,0].text(0.5, 0.5, "Error", ha='center')

        # 2. Calibration
        try:
            y_c = np.asarray(model_data.get('Ycal')); y_c_est = np.asarray(model_data.get('Ycal_est'))
            axs[0,1].plot(y_c, y_c, 'k-'); axs[0,1].plot(y_c, y_c_est, 'ob', alpha=0.6)
            axs[0,1].set_title('Calibration', fontsize=9, fontweight='bold'); axs[0,1].grid(True)
            axs[0,1].set_xlabel("True Value", fontsize=9, fontweight='bold'); axs[0,1].set_ylabel("Predicted Value", fontsize=9, fontweight='bold')
        except Exception: axs[0,1].text(0.5, 0.5, "N/A", ha='center')

        # 3. Validation
        try:
            if 'Yval_est' in model_data and 'Yval' in model_data and np.asarray(model_data['Yval']).size > 0:
                y_v = np.asarray(model_data.get('Yval')); y_v_est = np.asarray(model_data.get('Yval_est'))
                axs[0,2].plot(y_v, y_v, 'k-'); axs[0,2].plot(y_v, y_v_est, 'or', alpha=0.6)
                axs[0,2].set_title('External Validation', fontsize=9, fontweight='bold')
            else:
                 axs[0,2].text(0.5, 0.5, "No Ext. Validation", ha='center'); axs[0,2].set_title('Validation', fontsize=9, fontweight='bold')
            axs[0,2].grid(True); axs[0,2].set_xlabel("True Value", fontsize=9, fontweight='bold'); axs[0,2].set_ylabel("Predicted Value", fontsize=9, fontweight='bold')
        except Exception: axs[0,2].text(0.5, 0.5, "N/A", ha='center')

        # 4. Prediction
        try:
             if 'Ypred_est' in model_data and 'Ypred' in model_data and np.asarray(model_data['Ypred']).size > 0:
                y_p = np.asarray(model_data.get('Ypred')); y_p_est = np.asarray(model_data.get('Ypred_est'))
                axs[1,0].plot(y_p, y_p, 'k-'); axs[1,0].plot(y_p, y_p_est, 'sk', alpha=0.6)
             else:
                axs[1,0].text(0.5, 0.5, "No Prediction Data", ha='center')
             axs[1,0].set_title('Prediction', fontsize=9, fontweight='bold'); axs[1,0].grid(True)
             axs[1,0].set_xlabel("True Value", fontsize=9, fontweight='bold'); axs[1,0].set_ylabel("Predicted Value", fontsize=9, fontweight='bold')
        except Exception: axs[1,0].text(0.5, 0.5, "N/A", ha='center')

        # 5. Evolution
        try:
            if evolution_data:
                cycles = np.asarray(evolution_data.get('iterations', [])).flatten()
                costs = np.asarray(evolution_data.get('cost_evolution', [])).flatten()
                axs[1,1].plot(cycles, costs, 'b-o', markersize=4)
                if highlighted_cycle is not None and len(cycles) > 0:
                    idx = np.where(cycles == highlighted_cycle)[0]
                    if idx.size > 0: axs[1,1].plot(cycles[idx[0]], costs[idx[0]], 'ro', markersize=8, zorder=5)
                axs[1,1].set_title('RMSECV Evolution', fontsize=9, fontweight='bold'); axs[1,1].grid(True)
                axs[1,1].set_xlabel("Cycle", fontsize=9, fontweight='bold'); axs[1,1].set_ylabel("RMSECV", fontsize=9, fontweight='bold')
        except Exception: axs[1,1].text(0.5, 0.5, "N/A", ha='center')

        # 6. EJCR
        try:
             if 'Ypred_est' in model_data and 'Ypred' in model_data and np.asarray(model_data['Ypred']).size > 2:
                ejcr_data = self.calculate_ejcr(np.asarray(model_data['Ypred']), np.asarray(model_data['Ypred_est']))
                if ejcr_data:
                    x_loop = np.concatenate([ejcr_data['x_coords'], ejcr_data['x_coords'][::-1]])
                    y_loop = np.concatenate([ejcr_data['y1_coords'], ejcr_data['y2_coords'][::-1]])
                    axs[1,2].plot(x_loop, y_loop, 'b-'); axs[1,2].plot(ejcr_data['ideal_point_x'], ejcr_data['ideal_point_y'], 'ro', markersize=6)
                else: axs[1,2].text(0.5, 0.5, "Calc Error", ha='center')
             else: axs[1,2].text(0.5, 0.5, "Insufficient Data", ha='center')
             axs[1,2].set_title('Confidence Ellipse', fontsize=9, fontweight='bold'); axs[1,2].grid(True)
             axs[1,2].set_xlabel('Slope', fontsize=9, fontweight='bold'); axs[1,2].set_ylabel('Intercept', fontsize=9, fontweight='bold')
        except Exception: axs[1,2].text(0.5, 0.5, "Error", ha='center')

        if target_canvas:
            target_figure.tight_layout(rect=[0, 0, 1, 0.95]); target_canvas.draw()

    def save_all_plots_composite(self, model, evolution):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("TIFF files", "*.tiff"), ("JPEG files", "*.jpg"), ("All files", "*.*")], title="Save Composite")
        if not file_path: return
        fig = plt.figure(figsize=(18, 9)); self.draw_results_plots(fig, None, model, evolution)
        try: fig.savefig(file_path, dpi=300); plt.close(fig); messagebox.showinfo("Saved", f"Graph saved to:\n{os.path.abspath(file_path)}"); webbrowser.open(file_path)
        except Exception as e: plt.close(fig); messagebox.showerror("Error", f"Could not save: {e}")

    def save_single_plot(self, plot_type, model_data, evolution_data):
        plot_details = {'vars': ('Selected Variables', 0, 0), 'calibration': ('Calibration', 0, 1), 'validation': ('External Validation', 0, 2), 'prediction': ('Prediction', 1, 0), 'evolution': ('RMSECV Evolution', 1, 1), 'ejcr': ('Confidence Ellipse (EJCR)', 1, 2)}
        if plot_type not in plot_details: return
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("TIFF files", "*.tiff"), ("JPEG files", "*.jpg"), ("All files", "*.*")], title=f"Save {plot_details[plot_type][0]} Graph")
        if not file_path: return
        fig, ax = plt.subplots(figsize=(8, 6)); title = plot_details[plot_type][0]; fig.suptitle(f'{title} (Repetition {self.selected_repetition_index + 1})', fontsize=14, fontweight='bold')
        try:
            if plot_type == 'vars':
                if self.Train_data is not None:
                    mean_spectrum = np.mean(self.Train_data, axis=0)
                    xaxis = self.xaxis_data if self.xaxis_data is not None else np.arange(len(mean_spectrum))
                    ax.plot(xaxis, mean_spectrum, color='gray', alpha=0.5)
                    sel_idx = model_data.get('selected_variables', [])
                    if sel_idx: ax.plot(xaxis[sel_idx], mean_spectrum[sel_idx], 'ro', markersize=3)
                ax.set_xlabel('Variables', fontweight='bold'); ax.set_ylabel('Intensity', fontweight='bold')
            elif plot_type == 'calibration':
                y_c = np.asarray(model_data.get('Ycal')); y_c_est = np.asarray(model_data.get('Ycal_est'))
                ax.plot(y_c, y_c, 'k-'); ax.plot(y_c, y_c_est, 'ob', alpha=0.6); ax.set_xlabel("True Value", fontweight='bold'); ax.set_ylabel("Predicted Value", fontweight='bold')
            elif plot_type == 'validation':
                if 'Yval' in model_data and np.asarray(model_data['Yval']).size > 0:
                    y_v = np.asarray(model_data.get('Yval')); y_v_est = np.asarray(model_data.get('Yval_est'))
                    ax.plot(y_v, y_v, 'k-'); ax.plot(y_v, y_v_est, 'or', alpha=0.6)
                ax.set_xlabel("True Value", fontweight='bold'); ax.set_ylabel("Predicted Value", fontweight='bold')
            elif plot_type == 'prediction':
                if 'Ypred' in model_data and np.asarray(model_data['Ypred']).size > 0:
                    y_p = np.asarray(model_data.get('Ypred')); y_p_est = np.asarray(model_data.get('Ypred_est'))
                    ax.plot(y_p, y_p, 'k-'); ax.plot(y_p, y_p_est, 'sk', alpha=0.6)
                ax.set_xlabel("True Value", fontweight='bold'); ax.set_ylabel("Predicted Value", fontweight='bold')
            elif plot_type == 'evolution':
                 cycles = np.asarray(evolution_data.get('iterations', [])).flatten(); costs = np.asarray(evolution_data.get('cost_evolution', [])).flatten()
                 ax.plot(cycles, costs, 'b-o', markersize=4); ax.set_xlabel("Cycle", fontweight='bold'); ax.set_ylabel("RMSECV", fontweight='bold')
            elif plot_type == 'ejcr':
                if 'Ypred' in model_data:
                    ejcr_data = self.calculate_ejcr(np.asarray(model_data['Ypred']), np.asarray(model_data['Ypred_est']))
                    if ejcr_data:
                        x_loop = np.concatenate([ejcr_data['x_coords'], ejcr_data['x_coords'][::-1]]); y_loop = np.concatenate([ejcr_data['y1_coords'], ejcr_data['y2_coords'][::-1]])
                        ax.plot(x_loop, y_loop, 'b-'); ax.plot(ejcr_data['ideal_point_x'], ejcr_data['ideal_point_y'], 'ro', markersize=6)
                ax.set_xlabel('Slope', fontweight='bold'); ax.set_ylabel('Intercept', fontweight='bold')
            ax.grid(True); fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(file_path, dpi=300); plt.close(fig); messagebox.showinfo("Saved", f"Graph saved to:\n{os.path.abspath(file_path)}")
        except Exception as e: plt.close(fig); messagebox.showerror("Error", f"Error: {e}")

    def open_export_plot_options(self):
        if not self.Model_BAT_global: messagebox.showwarning("Warning", "Please select a repetition first."); return
        top = tk.Toplevel(self.root); top.title("Graph Export Options"); top.geometry("350x250"); top.transient(self.root); top.grab_set()
        tk.Label(top, text="Select which graph to export:", font=('Arial', 10, 'bold')).pack(pady=(10,5))
        ttk.Button(top, text="Save All Together (1 Image)", command=lambda: [self.save_all_plots_composite(self.Model_BAT_global, self.optimization_evolution_global), top.destroy()]).pack(pady=4, padx=20, fill=tk.X)
        ttk.Separator(top, orient='horizontal').pack(fill='x', padx=10, pady=5)
        plot_options = [("Selected Variables", "vars"), ("Calibration", "calibration"), ("Validation", "validation"), ("Prediction", "prediction"), ("RMSECV Evolution", "evolution"), ("Confidence Ellipse", "ejcr")]
        for label, ptype in plot_options: btn = ttk.Button(top, text=f"Save Only {label}", command=lambda p=ptype: [self.save_single_plot(p, self.Model_BAT_global, self.optimization_evolution_global), top.destroy()]); btn.pack(pady=2, padx=20, fill=tk.X)

    # --- MAIN FUNCTIONS ---
    def display_detailed_metrics_grid(self, model):
        for widget in self.detailed_metrics_frame.winfo_children(): widget.destroy()
        
        # DEFINIÇÃO DOS NOMES E ORDEM DAS MÉTRICAS NA TELA
        priority_order = ['optimal_lvs', 'num_selected_variables', 'RMSECV', 'Rcv', 
                          'RMSEC', 'R2_cal', 'RPD_cal', 'REP_cal', 
                          'RMSEP', 'R2_pred', 'RPD_pred', 'REP_pred']
                          
        pretty_names = {
            'optimal_lvs': 'LVs',
            'num_selected_variables': 'Selected Vars (Intervalos)',
            'RMSECV': 'RMSECV',
            'Rcv': 'Rcv (Q²)',
            'RMSEC': 'RMSEC (ppm)', 'R2_cal': 'R² Cal', 'RPD_cal': 'RPD Cal', 'REP_cal': 'REP(%) Cal',
            'RMSEP': 'RMSEP (ppm)', 'R2_pred': 'R² Pred', 'RPD_pred': 'RPD Pred', 'REP_pred': 'REP(%) Pred'
        }
        
        ttk.Label(self.detailed_metrics_frame, text="Model Metrics:", font=('Arial', 10, 'bold', 'underline')).grid(row=0, column=0, columnspan=2, pady=(5, 10))
        
        row_idx = 1
        for key in priority_order:
            if key in model:
                val = model[key]
                val_str = f"{val:.4f}" if isinstance(val, (float, np.floating)) else str(val)
                label = pretty_names.get(key, key) + ":"
                ttk.Label(self.detailed_metrics_frame, text=label, font=('Arial', 9, 'bold')).grid(row=row_idx, column=0, sticky='e', padx=(10, 5), pady=2)
                ttk.Label(self.detailed_metrics_frame, text=val_str, font=('Arial', 9)).grid(row=row_idx, column=1, sticky='w', padx=5, pady=2)
                row_idx += 1

    def load_data(self):
        file_path = filedialog.askopenfilename(title="Select .mat data file", filetypes=[("MATLAB files", "*.mat")])
        if not file_path: self.status_label.config(text="Status: Load cancelled."); return
        try:
            self.status_label.config(text="Status: Reading .mat file..."); self.root.update_idletasks()
            mat = sio.loadmat(file_path)
            self.Train_data, self.Ycal_data = np.asarray(mat['Xcal'], dtype=float), np.asarray(mat['Ycal'], dtype=float).flatten()
            if 'xaxis' in mat: self.xaxis_data = np.asarray(mat['xaxis']).flatten()
            else: self.xaxis_data = np.arange(self.Train_data.shape[1])
            loaded = ["Train (cal)"]
            if 'Xval' in mat and 'Yval' in mat:
                self.Val_data, self.Yval_data = np.asarray(mat.get('Xval'), dtype=float), np.asarray(mat.get('Yval'), dtype=float).flatten()
                if self.Val_data.size > 0: loaded.append("Validation (val)")
            if 'Xpred' in mat and 'Ypred' in mat:
                self.Pred_data, self.Ypred_data = np.asarray(mat.get('Xpred'), dtype=float), np.asarray(mat.get('Ypred'), dtype=float).flatten()
                if self.Pred_data.size > 0: loaded.append("Test (pred)")
            msg = f"Data loaded! Found: {', '.join(loaded)}."
            self.status_label.config(text=f"Status: {msg}"); messagebox.showinfo("Success", msg)
        except Exception as e:
            self.Train_data, self.Ycal_data, self.Val_data, self.Yval_data, self.Pred_data, self.Ypred_data = (None,) * 6
            self.status_label.config(text="Status: Load failed."); messagebox.showerror("Error Loading Data", str(e))

    def get_entry_value(self, entry, T): return T(entry.get().strip())

    def reset_all(self):
        self.Train_data, self.Ycal_data, self.Val_data, self.Yval_data, self.Pred_data, self.Ypred_data = (None,) * 6
        self.clear_previous_results()
        self.status_label.config(text="Status: Ready for new execution.")

    def clear_previous_results(self):
        self.all_repetitions_data, self.selected_repetition_index, self.Model_BAT_global, self.optimization_evolution_global = [], -1, None, None
        for tree in [self.summary_tree, self.cycle_tree]:
            if tree:
                for i in tree.get_children(): tree.delete(i)
        if self.classic_repetition_selector: self.classic_repetition_selector['values'] = []; self.classic_repetition_selector.set('Select')
        if self.initial_plot_figure: self.initial_plot_figure.clear(); self.initial_plot_canvas.draw()
        if self.interactive_plot_figure: self.interactive_plot_figure.clear(); self.interactive_plot_canvas.draw()
        if self.notebook: self.notebook.select(0)
        for btn in [self.export_graphics_button, self.export_excel_button, self.export_matlab_button]:
            if btn: btn.config(state=tk.DISABLED)

    def run_multi_repetitions_handler(self):
        if self.Train_data is None: messagebox.showerror("Error", "Load Training Data first."); return
        try:
            self.clear_previous_results()
            params = {k: self.get_entry_value(v, T) for k, v, T in [
                ('mbats', self.entries['entry_mbats'], int), ('numberofiterations', self.entries['entry_iterations'], int), ('n_min', self.entries['entry_n_min'], int),
                ('n_max', self.entries['entry_n_max'], int), ('alpha', self.entries['entry_alpha'], float), ('gamma', self.entries['entry_gamma'], float),
                ('fmin', self.entries['entry_fmin'], float), ('fmax', self.entries['entry_fmax'], float)]}
            num_repetitions = self.get_entry_value(self.entries['entry_repetitions'], int)
            
            for i in range(num_repetitions):
                self.status_label.config(text=f"Status: Running Repetition {i+1}/{num_repetitions}..."); self.root.update_idletasks()
                bat_model = BAT_PLS(params)
                model, evolution = bat_model.fit(self.Train_data, self.Ycal_data, self.Val_data, self.Yval_data, self.Pred_data, self.Ypred_data)
                self.all_repetitions_data.append({'Model': model, 'Evolution': evolution})
            
            self.status_label.config(text="Status: Generating summary..."); self.root.update_idletasks()
            
            rep_numbers = [str(i + 1) for i in range(len(self.all_repetitions_data))]
            for i, rep in enumerate(self.all_repetitions_data):
                m = rep.get('Model', {})
                self.summary_tree.insert("", "end", values=(i + 1, f"{m.get('RMSEC', np.inf):.4f}", f"{m.get('RMSEP', np.inf):.4f}", f"{m.get('R2_cal', 0):.4f}", f"{m.get('R2_pred', 0):.4f}"))
            
            if rep_numbers:
                self.classic_repetition_selector['values'] = rep_numbers
                self.classic_repetition_selector.set(rep_numbers[0])
                self.classic_view_details_handler() 
            
            self.status_label.config(text="Analysis finished."); self.notebook.select(0)
            messagebox.showinfo("Process Complete", f"All {num_repetitions} repetitions have been executed.")
        except Exception as e: self.status_label.config(text="Status: Error in execution."); messagebox.showerror("Execution Error", f"An error occurred: {e}")

    def classic_view_details_handler(self, event=None):
        rep_idx_str = self.classic_repetition_selector.get()
        if not rep_idx_str or rep_idx_str == "Select": return
        self.selected_repetition_index = int(rep_idx_str) - 1
        selected_data = self.all_repetitions_data[self.selected_repetition_index]
        self.Model_BAT_global = selected_data.get('Model', {})
        self.optimization_evolution_global = selected_data.get('Evolution', {})
        self.display_detailed_metrics_grid(self.Model_BAT_global)
        self.draw_results_plots(self.initial_plot_figure, self.initial_plot_canvas, self.Model_BAT_global, self.optimization_evolution_global)
        for btn in [self.export_graphics_button, self.export_excel_button, self.export_matlab_button]: btn.config(state=tk.NORMAL)

    def on_interactive_repetition_select(self, event):
        selection = self.summary_tree.selection();
        if not selection: return
        item = self.summary_tree.item(selection[0]); rep_number = int(item['values'][0]); self.selected_repetition_index = rep_number - 1
        selected_data = self.all_repetitions_data[self.selected_repetition_index]
        self.Model_BAT_global = selected_data.get('Model'); self.optimization_evolution_global = selected_data.get('Evolution')
        for i in self.cycle_tree.get_children(): self.cycle_tree.delete(i)
        cycles, cost_evol = self.optimization_evolution_global.get('iterations', []), self.optimization_evolution_global.get('cost_evolution', [])
        best_idx = np.argmin(cost_evol) if len(cost_evol) > 0 else -1
        for i, (c, cost) in enumerate(zip(cycles, cost_evol)):
            self.cycle_tree.insert("", "end", values=(c, f"{cost:.4f}"), tags=('best',) if i == best_idx else ())
        self.draw_results_plots(self.interactive_plot_figure, self.interactive_plot_canvas, self.Model_BAT_global, self.optimization_evolution_global)

    def on_cycle_select(self, event):
        selection = self.cycle_tree.selection();
        if not selection: return
        item = self.cycle_tree.item(selection[0]); cycle_num = int(item['values'][0])
        self.draw_results_plots(self.interactive_plot_figure, self.interactive_plot_canvas, self.Model_BAT_global, self.optimization_evolution_global, highlighted_cycle=cycle_num)

    def _setup_gui_components(self):
        try: self.root.iconbitmap('bat_icon.ico')
        except tk.TclError: print("Warning: 'bat_icon.ico' not found.")
        self.root.configure(bg="#f0f0f0"); style = ttk.Style(self.root); style.theme_use("clam")
        self.status_label = tk.Label(self.root, text="Status: Waiting...", bd=1, relief=tk.SUNKEN, anchor=tk.W, padx=10)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        header_frame = tk.Frame(self.root, bg="#2c3e50"); header_frame.pack(fill=tk.X)
        header_text_frame = tk.Frame(header_frame, bg="#2c3e50"); header_text_frame.pack(expand=True)
        tk.Label(header_text_frame, text="BAT Variable Selection Algorithm \U0001F987", fg="white", bg="#2c3e50", font=("Arial", 14, "bold")).pack(side=tk.LEFT, pady=5, padx=10)
        tk.Label(header_text_frame, text="Developer: Enia Mendes", fg="white", bg="#2c3e50", font=("Arial", 8)).pack(side=tk.LEFT, pady=5)
        main_frame = ttk.Frame(self.root); main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        control_frame = ttk.LabelFrame(main_frame, text="Controls and Parameters"); control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=5)
        tk.Button(control_frame, text="Load Data (.mat)", command=self.load_data, bg="#4682B4", fg="white").pack(pady=5, fill=tk.X, padx=5)
        param_frame = ttk.LabelFrame(control_frame, text="BAT Parameters"); param_frame.pack(pady=10, padx=5, fill=tk.X)
        bat_params_setup = [("No. Bats:", 30, 'entry_mbats'), ("No. Iterations:", 100, 'entry_iterations'), ("Min Vars No.:", 2, 'entry_n_min'), ("Max Vars No.:", 20, 'entry_n_max'), ("Alpha (Loudness):", 0.5, 'entry_alpha'), ("Gamma (Pulse):", 0.4, 'entry_gamma'), ("Min Freq.:", 0.0, 'entry_fmin'), ("Max Freq.:", 0.05, 'entry_fmax')]
        for i, (text, val, key) in enumerate(bat_params_setup):
            tk.Label(param_frame, text=text).grid(row=i, column=0, sticky="w", padx=2, pady=2); entry = ttk.Entry(param_frame, width=10); entry.insert(0, val); entry.grid(row=i, column=1, padx=5, pady=2); self.entries[key] = entry
        tk.Label(control_frame, text="Repetitions:").pack(pady=(10,0)); entry_repetitions = ttk.Entry(control_frame); entry_repetitions.insert(0, "2"); entry_repetitions.pack(pady=2, padx=5, fill=tk.X); self.entries['entry_repetitions'] = entry_repetitions
        tk.Button(control_frame, text="Run Analysis", command=self.run_multi_repetitions_handler, font=("Arial", 11, "bold"), bg="#8A2BE2", fg="white").pack(pady=10, fill=tk.X, padx=5)
        tk.Button(control_frame, text="Reset All", command=self.reset_all, bg="#B0C4DE").pack(pady=5, fill=tk.X, padx=5)
        results_frame = ttk.Frame(main_frame); results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True); self.notebook = ttk.Notebook(results_frame); self.notebook.pack(fill=tk.BOTH, expand=True)
        
        tab1 = ttk.Frame(self.notebook); self.notebook.add(tab1, text='Initial Analysis')
        tab1_paned = ttk.PanedWindow(tab1, orient=tk.HORIZONTAL); tab1_paned.pack(fill=tk.BOTH, expand=True)
        left_pane_tab1 = ttk.Frame(tab1_paned, width=400); tab1_paned.add(left_pane_tab1, weight=2)
        right_pane_tab1 = ttk.LabelFrame(tab1_paned, text="Graphic Dashboard"); tab1_paned.add(right_pane_tab1, weight=5)
        classic_controls = ttk.Frame(left_pane_tab1); classic_controls.pack(fill=tk.X, pady=5, padx=5)
        ttk.Label(classic_controls, text="Repetition:").pack(side=tk.LEFT, padx=(0,2))
        self.classic_repetition_selector = ttk.Combobox(classic_controls, state="readonly", width=8); self.classic_repetition_selector.pack(side=tk.LEFT, padx=2)
        self.classic_view_btn = tk.Button(classic_controls, text="View Details", command=self.classic_view_details_handler); self.classic_view_btn.pack(side=tk.LEFT, padx=2)
        self.classic_repetition_selector.bind("<<ComboboxSelected>>", self.classic_view_details_handler)
        self.export_graphics_button = tk.Button(classic_controls, text="Export Graphs...", command=self.open_export_plot_options, state=tk.DISABLED); self.export_graphics_button.pack(side=tk.LEFT, padx=2)
        self.export_excel_button = tk.Button(classic_controls, text="Export to Excel", command=self.export_metrics_to_excel, state=tk.DISABLED); self.export_excel_button.pack(side=tk.LEFT, padx=2)
        self.export_matlab_button = tk.Button(classic_controls, text="Export to MATLAB (.mat)", command=self.export_model_to_matlab, state=tk.DISABLED); self.export_matlab_button.pack(side=tk.LEFT, padx=2)
        classic_report_container = ttk.LabelFrame(left_pane_tab1, text="Execution Summary"); classic_report_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0,5))
        self.detailed_metrics_frame = ttk.Frame(classic_report_container); self.detailed_metrics_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.initial_plot_figure = plt.figure(); self.initial_plot_canvas = FigureCanvasTkAgg(self.initial_plot_figure, master=right_pane_tab1); self.initial_plot_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        tab2 = ttk.Frame(self.notebook); self.notebook.add(tab2, text='Interactive Analysis')
        paned_window = ttk.PanedWindow(tab2, orient=tk.VERTICAL); paned_window.pack(fill=tk.BOTH, expand=True)
        summary_frame = ttk.LabelFrame(paned_window, text="Repetitions Summary (Click to analyze)"); paned_window.add(summary_frame, weight=2)
        cols = ("Rep.", "RMSEC", "RMSEP", "R² Cal", "R² Pred"); self.summary_tree = ttk.Treeview(summary_frame, columns=cols, show='headings')
        for col in cols: self.summary_tree.heading(col, text=col); self.summary_tree.column(col, width=100, anchor=tk.CENTER)
        vsb_summary = ttk.Scrollbar(summary_frame, orient="vertical", command=self.summary_tree.yview); self.summary_tree.configure(yscrollcommand=vsb_summary.set); vsb_summary.pack(side='right', fill='y'); self.summary_tree.pack(fill=tk.BOTH, expand=True)
        self.summary_tree.bind('<<TreeviewSelect>>', self.on_interactive_repetition_select)
        cycle_frame = ttk.LabelFrame(paned_window, text="Cycles Evolution (Click to highlight on graph)"); paned_window.add(cycle_frame, weight=2)
        cycle_cols = ("Cycle", "Cost (RMSECV)"); self.cycle_tree = ttk.Treeview(cycle_frame, columns=cycle_cols, show='headings')
        self.cycle_tree.heading("Cycle", text="Cycle"); self.cycle_tree.column("Cycle", width=80, anchor=tk.CENTER); self.cycle_tree.heading("Cost (RMSECV)", text="Cost"); self.cycle_tree.column("Cost (RMSECV)", width=150)
        self.cycle_tree.tag_configure('best', background='lightgreen')
        vsb_cycle = ttk.Scrollbar(cycle_frame, orient="vertical", command=self.cycle_tree.yview); self.cycle_tree.configure(yscrollcommand=vsb_cycle.set); vsb_cycle.pack(side='right', fill='y'); self.cycle_tree.pack(fill=tk.BOTH, expand=True)
        self.cycle_tree.bind('<<TreeviewSelect>>', self.on_cycle_select)
        plot_frame = ttk.LabelFrame(paned_window, text="Interactive Graphical Visualization"); paned_window.add(plot_frame, weight=4)
        self.interactive_plot_figure = plt.figure(); self.interactive_plot_canvas = FigureCanvasTkAgg(self.interactive_plot_figure, master=plot_frame); self.interactive_plot_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.interactive_plot_canvas, plot_frame); toolbar.update(); self.interactive_plot_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# =============================================================================
#  BLOCO PRINCIPAL (COM CORREÇÃO DE DPI PARA WINDOWS)
# =============================================================================

# Tenta configurar a alta resolução de DPI no Windows
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass  # Falha silenciosa em sistemas não-Windows ou versões antigas

if __name__ == "__main__":
    app = BAT_PLS_App()
    app.root.mainloop()