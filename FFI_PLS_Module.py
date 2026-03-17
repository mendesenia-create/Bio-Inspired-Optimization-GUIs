import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import scipy.io as sio
import webbrowser
import ctypes  # Para alta resolução no Windows
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import f as f_dist, linregress
from joblib import parallel_backend

# =============================================================================
#  FUNÇÕES AUXILIARES E ALGORITMO FFI-PLS (BACKEND)
# =============================================================================

def fcrit(p, n1, n2):
    return f_dist.ppf(1 - p, n1, n2)

def fun_part(col, part):
    if part <= 1: return [np.arange(col)]
    
    rest = col % part
    inter = col // part
    
    ranges = []
    start = 0
    for i in range(part):
        end = start + inter + (1 if i < rest else 0)
        ranges.append(np.arange(start, end))
        start = end
    return ranges

def reg_pls(Xcal, Ycal, A):
    with parallel_backend('threading'):
        y_true = Ycal.ravel()
        rmsecv_per_lv = []
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42) 
        
        for n_components in range(1, A + 1):
            pls = PLSRegression(n_components=n_components, scale=False)
            y_pred = cross_val_predict(pls, Xcal, y_true, cv=kf, n_jobs=-1) 
            mse = mean_squared_error(y_true, y_pred)
            rmsecv_per_lv.append(np.sqrt(mse))
        
        rmsecv_per_lv = np.array(rmsecv_per_lv)

        if rmsecv_per_lv.size > 0:
            min_rmsecv = np.min(rmsecv_per_lv)
            best_n_lvs = np.argmin(rmsecv_per_lv) + 1
        else:
            min_rmsecv = np.nan
            best_n_lvs = 0

        y_pred_cv = np.array([])
        if best_n_lvs > 0:
            best_pls = PLSRegression(n_components=best_n_lvs)
            y_pred_cv = cross_val_predict(best_pls, Xcal, y_true, cv=kf, n_jobs=-1) 

        results = {
            'best_n_lvs': best_n_lvs,
            'RMSECV_min': min_rmsecv,
            'RMSECV': rmsecv_per_lv,
            'ypred': y_pred_cv.reshape(-1, 1)
        }
        return results

def calculate_pls_metrics(y_true, y_pred):
    if y_true is None or y_pred is None or len(y_true) < 2:
        return np.inf, 0
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2

def ffipls(Xcal, Ycal, Xval, Yval, Xpred, Ypred, intervalos, VL, fpop, ciclos, w0, gama, alfa, num_repetitions):
    all_repetitions_results = []
    
    for rep_idx in range(num_repetitions):
        print(f"\n--- Starting Repetition {rep_idx + 1}/{num_repetitions} ---")
        alfa_dinamico = float(alfa) 
        n_samples, n_variables = Xcal.shape
        nsampval = Xval.shape[0] if Xval.size > 0 else 0
        X_intervals = fun_part(n_variables, intervalos)

        F = np.random.rand(fpop, intervalos) 
        FF = np.round(F).astype(int) 
        
        for k in range(fpop):
            if not np.any(FF[k, :]):
                FF[k, np.random.randint(intervalos)] = 1
        
        evolution = {'cycles': [], 'min_rmse_evolution': [], 'selected_intervals_evolution': []}
        best_overall_rmse = np.inf
        best_overall_FF_solution = None

        for s in range(ciclos):
            rmsep = np.zeros(fpop)
            for k in range(fpop):
                selected_interval_indices = np.where(FF[k, :] == 1)[0]
                if selected_interval_indices.size == 0:
                    rmsep[k] = np.inf
                    continue
                
                z = np.concatenate([X_intervals[i] for i in selected_interval_indices]).astype(int)
                Xcalff = Xcal[:, z] 
                pls_results = reg_pls(Xcalff, Ycal, VL) 
                rmsep[k] = pls_results['RMSECV_min'] 
            
            current_best_rmse_in_cycle = np.min(rmsep)
            if current_best_rmse_in_cycle < best_overall_rmse:
                best_overall_rmse = current_best_rmse_in_cycle
                best_overall_FF_solution = FF[np.argmin(rmsep), :].copy()

            print(f"Cycle {s+1}/{ciclos}, Best RMSE: {np.min(rmsep):.4f}")
            
            fitness = 1.0 / (rmsep + 1e-12) 
            fitness[np.isinf(fitness)] = 0
            F_new = F.copy() 
            
            for i in range(fpop):
                found_brighter = False
                for j in range(fpop):
                    if fitness[j] > fitness[i]:
                        firefly_i_pos, firefly_j_pos = F[i, :], F[j, :]
                        rij = np.sqrt(np.sum((firefly_j_pos - firefly_i_pos)**2))
                        random_walk = alfa_dinamico * (np.random.rand(intervalos) - 0.5)
                        attraction = w0 * np.exp(-gama * rij**2) * (firefly_j_pos - firefly_i_pos)
                        F_new[i, :] = firefly_i_pos + attraction + random_walk
                        found_brighter = True
                        break

                if not found_brighter:
                    random_walk_self = alfa_dinamico * (np.random.rand(intervalos) - 0.5)
                    F_new[i, :] = F[i, :] + random_walk_self
            
            best_idx_current_gen = np.argmax(fitness)
            F_new[best_idx_current_gen, :] = F[best_idx_current_gen, :] 
            
            F = F_new
            FF = np.clip(np.round(F), 0, 1).astype(int) 
            
            for k in range(fpop):
                if not np.any(FF[k, :]): FF[k, np.random.randint(intervalos)] = 1
            
            best_idx_cycle = np.argmin(rmsep)
            evolution['cycles'].append(s + 1)
            evolution['min_rmse_evolution'].append(rmsep[best_idx_cycle])
            evolution['selected_intervals_evolution'].append(np.where(FF[best_idx_cycle, :] == 1)[0] + 1)
            
            alfa_dinamico *= 0.97

        if best_overall_FF_solution is None:
             print(f"ERROR: Repetition {rep_idx + 1} failed.")
             all_repetitions_results.append({'Model_FF_iPLS': 'FAILED', 'optimization_evolution': evolution}); continue
        
        l = np.where(best_overall_FF_solution == 1)[0] + 1
        
        if l.size == 0:
            print(f"ERROR: Repetition {rep_idx + 1} failed (no intervals).")
            all_repetitions_results.append({'Model_FF_iPLS': 'FAILED', 'optimization_evolution': evolution}); continue
            
        h = np.concatenate([X_intervals[i-1] for i in l]).astype(int)
        Xcal2 = Xcal[:, h]
        final_model_results = {}
        
        final_pls_cal = reg_pls(Xcal2, Ycal, VL) 
        VL_final = final_pls_cal['best_n_lvs']
        
        final_model = PLSRegression(n_components=VL_final, scale=False)
        final_model.fit(Xcal2, Ycal)
        
        y_cal_pred = final_model.predict(Xcal2)
        final_model_results['RMSEC'] = np.sqrt(mean_squared_error(Ycal, y_cal_pred))
        final_model_results['R2_cal'] = r2_score(Ycal, y_cal_pred)
        final_model_results['BIAS_cal'] = np.mean(y_cal_pred - Ycal)
        final_model_results['RPD_cal'] = 1 / np.sqrt(1 - final_model_results['R2_cal']) if (1 - final_model_results['R2_cal']) > 0 else np.nan

        if nsampval == 0:
            final_model_results['val_method'] = 'Cross-Validation'
            final_model_results['RMSECV'] = final_pls_cal['RMSECV_min']
            y_val_cv_pred = final_pls_cal['ypred']
            final_model_results['R2_cv'] = r2_score(Ycal, y_val_cv_pred)
            final_model_results['BIAS_cv'] = np.mean(y_val_cv_pred - Ycal)
            final_model_results['RPD_cv'] = 1 / np.sqrt(1 - final_model_results['R2_cv']) if (1 - final_model_results['R2_cv']) > 0 else np.nan
            final_model_results['Yval_cv'] = y_val_cv_pred
        else:
            Xval2 = Xval[:,h]
            y_val_pred = final_model.predict(Xval2)
            final_model_results['val_method'] = 'External Validation'
            final_model_results['RMSEV'] = np.sqrt(mean_squared_error(Yval, y_val_pred))
            final_model_results['R2_val'] = r2_score(Yval, y_val_pred)
            final_model_results['BIAS_val'] = np.mean(y_val_pred - Yval)
            final_model_results['RPD_val'] = 1 / np.sqrt(1 - final_model_results['R2_val']) if (1 - final_model_results['R2_val']) > 0 else np.nan
            final_model_results['Yval_est'] = y_val_pred

        Xpred2 = Xpred[:,h]
        y_pred_pred = final_model.predict(Xpred2)
        final_model_results['RMSEP'] = np.sqrt(mean_squared_error(Ypred, y_pred_pred))
        final_model_results['R2_pred'] = r2_score(Ypred, y_pred_pred)
        final_model_results['BIAS_pred'] = np.mean(y_pred_pred - Ypred)
        final_model_results['RPD_pred'] = 1 / np.sqrt(1 - final_model_results['R2_pred']) if (1 - final_model_results['R2_pred']) > 0 else np.nan

        final_model_results['no_of_lv'] = VL_final
        final_model_results['Ycal_est'] = y_cal_pred
        final_model_results['Ypred_est'] = y_pred_pred
        final_model_results['selected_intervals'] = l
        
        allint_map = []; last_end = 0
        for i, interval_indices in enumerate(X_intervals):
            allint_map.append([i + 1, last_end + 1, last_end + len(interval_indices)])
            last_end += len(interval_indices)
        final_model_results['allint'] = np.array(allint_map)

        all_repetitions_results.append({
            'Model_FF_iPLS': final_model_results,
            'optimization_evolution': evolution
        })

    return all_repetitions_results


# =============================================================================
#  CLASSE DA INTERFACE GRÁFICA (COM EXPORTAÇÃO SELETIVA)
# =============================================================================

class FFI_PLS_App:
    def __init__(self, root_master=None):
        if root_master:
            self.root = tk.Toplevel(root_master)
        else:
            self.root = tk.Tk()
        
        self.root.title("FFI-PLS Interface")
        self.root.geometry("1600x900")
        
        # Variáveis de Estado
        self.all_repetitions_data = []
        self.selected_repetition_index = -1
        self.Model_FF_iPLS_global = None
        self.optimization_evolution_global = None
        self.Xcal_data = None
        self.Ycal_data = None
        self.Xval_data = None
        self.Yval_data = None
        self.Xpred_data = None
        self.Ypred_data = None
        self.xaxis_data = None
        self.entries = {}
        
        self.check_vars = {} # Para checkboxes

        self._setup_gui_components()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        self.root.destroy()

    def get_entry_value(self, entry, field_name, tipo="int"):
        value = entry.get().strip()
        if not value: raise ValueError(f"The '{field_name}' field cannot be empty.")
        try:
            if tipo == "int": return int(value)
            elif tipo == "float": return float(value)
        except ValueError: raise ValueError(f"Invalid value for field '{field_name}'.")

    def get_safe_scalar_value(self, value, default_nan=np.nan):
        if isinstance(value, np.ndarray):
            if value.size == 1: return float(value.item())
            else:
                try: return float(value.flatten()[0])
                except (IndexError, TypeError, ValueError): return default_nan
        elif isinstance(value, (int, float)): return float(value)
        return str(value)

    def clear_report_frame(self):
        if self.classic_report_frame:
            for widget in self.classic_report_frame.winfo_children():
                widget.destroy()

    def calculate_ejcr(self, y_real, y_pred):
        try:
            y_real, y_pred = y_real.flatten(), y_pred.flatten()
            Iv = len(y_real)
            if Iv < 3: return None
            
            # Correção com linregress para robustez
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
        except Exception: return None

    # --- NOVO SISTEMA DE EXPORTAÇÃO DE GRÁFICOS ---

    def plot_single_graph_on_ax(self, ax, plot_type):
        """Desenha um gráfico específico em um eixo fornecido (Backend de desenho)."""
        model_data = self.Model_FF_iPLS_global
        evolution_data = self.optimization_evolution_global
        
        if plot_type == 'intervals':
            ax.plot(self.xaxis_data, self.Xcal_data.T, color='gray', alpha=0.3)
            allint_map = np.asarray(model_data.get('allint', []))
            selected_intervals = np.asarray(model_data.get('selected_intervals', [])).flatten()
            
            if allint_map.size > 0 and selected_intervals.size > 0:
                for int_idx in selected_intervals:
                    matching_rows = allint_map[allint_map[:, 0].astype(int) == int(int_idx)]
                    if matching_rows.size > 0:
                        start_var, end_var = int(matching_rows[0, 1]) - 1, int(matching_rows[0, 2]) - 1
                        if 0 <= start_var < len(self.xaxis_data) and 0 <= end_var < len(self.xaxis_data):
                            ax.axvspan(self.xaxis_data[start_var], self.xaxis_data[end_var], color='red', alpha=0.4)
            ax.set_title("Selected Intervals", fontweight='bold')
            ax.set_xlabel("Variables", fontweight='bold'); ax.set_ylabel("Intensity", fontweight='bold')

        elif plot_type == 'calibration':
            yc, yce = self.Ycal_data, np.asarray(model_data.get('Ycal_est'))
            ax.plot(yc, yc, 'k-')
            ax.plot(yc, yce, 'ob', alpha=0.6)
            ax.set_title("Calibration", fontweight='bold'); ax.grid(True)
            ax.set_xlabel("True Value", fontweight='bold'); ax.set_ylabel("Predicted Value", fontweight='bold')
        
        elif plot_type == 'validation':
            if 'Yval_est' in model_data and self.Yval_data.size > 0:
                yv, yve = self.Yval_data, np.asarray(model_data.get('Yval_est'))
                ax.plot(yv, yv, 'k-')
                ax.plot(yv, yve, 'or', alpha=0.6)
                ax.set_title("External Validation", fontweight='bold')
            elif 'Yval_cv' in model_data:
                yc, ycv = self.Ycal_data, np.asarray(model_data.get('Yval_cv'))
                ax.plot(yc, yc, 'k-')
                ax.plot(yc, ycv, 'og', alpha=0.6)
                ax.set_title("Cross-Validation", fontweight='bold')
            else:
                ax.text(0.5, 0.5, "No Validation Data", ha='center')
            ax.grid(True); ax.set_xlabel("True Value", fontweight='bold'); ax.set_ylabel("Predicted Value", fontweight='bold')

        elif plot_type == 'prediction':
            if self.Ypred_data is not None and self.Ypred_data.size > 0:
                yp, ype = self.Ypred_data, np.asarray(model_data.get('Ypred_est'))
                ax.plot(yp, yp, 'k-')
                ax.plot(yp, ype, 'sk', alpha=0.6)
                ax.set_title("Prediction", fontweight='bold')
            else:
                ax.text(0.5, 0.5, "No Prediction Data", ha='center')
            ax.grid(True); ax.set_xlabel("True Value", fontweight='bold'); ax.set_ylabel("Predicted Value", fontweight='bold')

        elif plot_type == 'rmsecv_evo':
            if evolution_data:
                cycles = np.asarray(evolution_data.get('cycles', [])).flatten()
                min_rmse = np.asarray(evolution_data.get('min_rmse_evolution', [])).flatten()
                ax.plot(cycles, min_rmse, 'b-o', markersize=4)
                ax.set_title("RMSECV Evolution", fontweight='bold')
            else:
                ax.text(0.5, 0.5, "No Evolution Data", ha='center')
            ax.grid(True); ax.set_xlabel("Cycle", fontweight='bold'); ax.set_ylabel("RMSECV", fontweight='bold')

        elif plot_type == 'ejcr':
            if self.Ypred_data is not None and self.Ypred_data.size > 3:
                ejcr_data = self.calculate_ejcr(self.Ypred_data, np.asarray(model_data.get('Ypred_est')))
                if ejcr_data:
                    x_loop = np.concatenate([ejcr_data['x_coords'], ejcr_data['x_coords'][::-1]])
                    y_loop = np.concatenate([ejcr_data['y1_coords'], ejcr_data['y2_coords'][::-1]])
                    ax.plot(x_loop, y_loop, 'b-')
                    ax.plot(ejcr_data['ideal_point_x'], ejcr_data['ideal_point_y'], 'ro', markersize=6, label='Ideal (1,0)')
                else:
                    ax.text(0.5, 0.5, "Calculation Error", ha='center')
            else:
                ax.text(0.5, 0.5, "Insufficient Data", ha='center')
            ax.set_title("Confidence Ellipse", fontweight='bold'); ax.grid(True)
            ax.set_xlabel('Slope', fontweight='bold'); ax.set_ylabel('Intercept', fontweight='bold')

    def open_export_plot_options(self):
        """Abre janela para selecionar quais gráficos exportar."""
        if not self.Model_FF_iPLS_global:
            messagebox.showwarning("Warning", "Select a repetition and click 'View Details' first.")
            return

        top = tk.Toplevel(self.root)
        top.title("Export Graphs")
        top.geometry("350x450")
        top.transient(self.root)
        top.grab_set()

        tk.Label(top, text="Select Graphs to Export:", font=('Arial', 11, 'bold')).pack(pady=10)
        
        plot_options = [
            ("Selected Intervals", "intervals"),
            ("Calibration", "calibration"),
            ("Validation", "validation"),
            ("Prediction", "prediction"),
            ("RMSECV Evolution", "rmsecv_evo"),
            ("Confidence Ellipse (EJCR)", "ejcr")
        ]
        
        self.check_vars = {}
        chk_frame = ttk.Frame(top)
        chk_frame.pack(pady=5, padx=20, fill='x')
        
        for text, key in plot_options:
            var = tk.BooleanVar(value=False)
            self.check_vars[key] = var
            ttk.Checkbutton(chk_frame, text=text, variable=var).pack(anchor='w', pady=2)
            
        tk.Label(top, text="File Format:", font=('Arial', 10)).pack(pady=(15, 5))
        format_combo = ttk.Combobox(top, values=[".tiff", ".png", ".jpg", ".pdf", ".eps"], state="readonly", width=10)
        format_combo.set(".tiff") 
        format_combo.pack()
        
        def run_batch_save():
            selected = [key for key, var in self.check_vars.items() if var.get()]
            fmt = format_combo.get()
            self.save_batch_plots(selected, fmt, top)
            
        ttk.Button(top, text="Save Selected Graphs", command=run_batch_save).pack(pady=20, fill='x', padx=20)
        
        ttk.Separator(top, orient='horizontal').pack(fill='x', padx=10, pady=5)
        
        # Mantém o botão de salvar o Dashboard completo
        ttk.Button(top, text="Save Full Dashboard (All-in-One)", command=lambda: [self.save_all_plots_composite(self.Model_FF_iPLS_global), top.destroy()]).pack(pady=5, fill='x', padx=20)

    def save_batch_plots(self, selected_plots, file_format, top_window):
        """Salva os gráficos selecionados em lote."""
        if not selected_plots:
            messagebox.showwarning("Warning", "No graphs selected!")
            return
            
        initial_file = f"Repetition_{self.selected_repetition_index + 1}"
        file_path = filedialog.asksaveasfilename(
            defaultextension=file_format,
            initialfile=initial_file,
            title="Choose base name (e.g. Result_Rep1)",
            filetypes=[(f"{file_format.upper()} files", f"*{file_format}"), ("All files", "*.*")]
        )
        
        if not file_path: return
        
        if file_path.lower().endswith(file_format):
            base_path = file_path[:-len(file_format)]
        else:
            base_path = file_path

        try:
            for ptype in selected_plots:
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111)
                self.plot_single_graph_on_ax(ax, ptype)
                
                save_name = f"{base_path}_{ptype}{file_format}"
                fig.tight_layout()
                fig.savefig(save_name, dpi=300)
                plt.close(fig)
            
            messagebox.showinfo("Success", f"Saved {len(selected_plots)} graphs successfully!")
            top_window.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error saving graphs: {str(e)}")

    # --- PLOTAGEM DO DASHBOARD PRINCIPAL ---

    def draw_results_plots(self, target_figure, target_canvas, model_data, highlighted_cycle=None):
        target_figure.clear()
        axs = target_figure.subplots(2, 3)
        target_figure.suptitle(f'Results for Repetition {self.selected_repetition_index + 1}', fontsize=12, fontweight='bold', y=0.98)
        
        # Reuse logic for dashboard consistency
        self.plot_single_graph_on_ax(axs[0,0], 'intervals')
        self.plot_single_graph_on_ax(axs[0,1], 'calibration')
        self.plot_single_graph_on_ax(axs[0,2], 'validation')
        self.plot_single_graph_on_ax(axs[1,0], 'prediction')
        self.plot_single_graph_on_ax(axs[1,1], 'rmsecv_evo')
        
        # Highlight logic for evolution plot
        if highlighted_cycle is not None and self.optimization_evolution_global:
            cycles = np.asarray(self.optimization_evolution_global.get('cycles', [])).flatten()
            min_rmse = np.asarray(self.optimization_evolution_global.get('min_rmse_evolution', [])).flatten()
            cycle_index = np.where(cycles == highlighted_cycle)[0]
            if cycle_index.size > 0: 
                axs[1,1].plot(cycles[cycle_index[0]], min_rmse[cycle_index[0]], 'ro', markersize=8, zorder=5)

        self.plot_single_graph_on_ax(axs[1,2], 'ejcr')
        
        if target_canvas: 
            target_figure.tight_layout(rect=[0, 0, 1, 0.95])
            target_canvas.draw()

    def display_summary_report(self, all_data):
        self.clear_report_frame()
        if not all_data: ttk.Label(self.classic_report_frame, text="No results to display.").pack(); return
        rmsep_values = [self.get_safe_scalar_value(rep.get('Model_FF_iPLS', {}).get('RMSEP', np.inf)) for rep in all_data]
        if not rmsep_values: return
        best_rep_idx = np.argmin(rmsep_values)
        header_text = f"Best Result: Repetition {best_rep_idx + 1} (RMSEP = {rmsep_values[best_rep_idx]:.4f})"
        ttk.Label(self.classic_report_frame, text=header_text, font=('Arial', 11, 'bold'), foreground='green').pack(anchor='w', padx=10, pady=(5,0))
        separator = ttk.Separator(self.classic_report_frame, orient='horizontal'); separator.pack(fill='x', padx=10, pady=5)
        summary_header = ttk.Label(self.classic_report_frame, text="Summary of All Repetitions:", font=('Arial', 10, 'underline'))
        summary_header.pack(anchor='w', padx=10, pady=(5,5))
        for i, rep in enumerate(all_data):
            rmsep = rmsep_values[i]
            line = f"Repetition {i+1}: RMSEP = {rmsep:.4f}"
            font_style = ('Courier New', 10, 'bold') if i == best_rep_idx else ('Courier New', 10)
            fg_color = 'green' if i == best_rep_idx else 'black'
            label = ttk.Label(self.classic_report_frame, text=line, font=font_style, foreground=fg_color)
            label.pack(anchor='w', padx=10)

    def display_full_report_grid(self, model):
        self.clear_report_frame()
        title_label = ttk.Label(self.classic_report_frame, text=f"Full Report for Repetition {self.selected_repetition_index + 1}", font=('Arial', 12, 'bold'))
        title_label.grid(row=0, column=0, columnspan=6, pady=(5, 15), padx=10)
        ignore_keys = {'rawX', 'rawY', 'Ycal_est', 'Yval_est', 'Ypred_est', 'Yval_cv', 'selected_intervals', 'allint', 'xaxislabels'}
        metrics_to_display = [(k, v) for k, v in model.items() if k not in ignore_keys]
        num_columns = 3
        for i, (key, value) in enumerate(metrics_to_display):
            row = (i // num_columns) + 1; col_base = (i % num_columns) * 2
            try:
                safe_val = self.get_safe_scalar_value(value); formatted_val = str(value).replace("'", "")
                if isinstance(safe_val, float): formatted_val = f"{int(safe_val)}" if safe_val.is_integer() else f"{safe_val:.4f}"
            except Exception: formatted_val = str(value)
            key_label = ttk.Label(self.classic_report_frame, text=f"{key}:", font=('Arial', 9, 'bold')); key_label.grid(row=row, column=col_base, sticky='e', padx=(10, 2), pady=2)
            value_label = ttk.Label(self.classic_report_frame, text=formatted_val); value_label.grid(row=row, column=col_base + 1, sticky='w', padx=(0, 20), pady=2)

    def save_all_plots_composite(self, model):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("TIFF files", "*.tiff"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            title="Save Composite Graph Image"
        )
        if not file_path: return
        
        fig = plt.figure(figsize=(18, 9))
        self.draw_results_plots(fig, None, model) 
        try:
            fig.savefig(file_path, dpi=300)
            plt.close(fig)
            messagebox.showinfo("Graph Saved", f"The composite graph was saved to:\n{os.path.abspath(file_path)}")
            webbrowser.open(file_path)
        except Exception as e:
            messagebox.showerror("Error Saving", f"Could not save the graph: {e}")

    # --- FUNÇÃO DE EXPORTAÇÃO .MAT ---
    def export_model_to_matlab(self):
        if not self.Model_FF_iPLS_global:
            messagebox.showwarning("Warning", "No model selected. Please view the details of a repetition first.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".mat",
            filetypes=[("MATLAB Files", "*.mat")],
            title="Export Model to MATLAB"
        )
        if not file_path: return

        try:
            # Preparar os dados para o formato do MATLAB
            mat_data = {
                'Model': self.Model_FF_iPLS_global,
                'Evolution': self.optimization_evolution_global,
                'Data': {
                    'Xcal': self.Xcal_data,
                    'Ycal': self.Ycal_data,
                    'Xval': self.Xval_data if self.Xval_data is not None else [],
                    'Yval': self.Yval_data if self.Yval_data is not None else [],
                    'Xpred': self.Xpred_data,
                    'Ypred': self.Ypred_data,
                    'xaxis': self.xaxis_data
                }
            }
            sio.savemat(file_path, mat_data)
            messagebox.showinfo("Export Complete", f"Model successfully exported to:\n{os.path.abspath(file_path)}")
        except Exception as e:
            messagebox.showerror("Export Error", f"An error occurred while exporting to MATLAB: {e}")
    # --------------------------------------

    def export_metrics_to_excel(self):
        if not self.Model_FF_iPLS_global:
            messagebox.showwarning("Warning", "No model selected. Please view the details of a repetition first.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Workbook", "*.xlsx")],
            title="Export Metrics to Excel"
        )
        if not file_path: return

        try:
            ignore_keys = {'rawX', 'rawY', 'Ycal_est', 'Yval_est', 'Ypred_est', 'Yval_cv', 'allint', 'xaxislabels'}
            metrics_to_export = {k: v for k, v in self.Model_FF_iPLS_global.items() if k not in ignore_keys}
            
            # Formatting data for the DataFrame
            data_for_df = []
            for key, value in metrics_to_export.items():
                if isinstance(value, np.ndarray):
                    # Convert arrays to a readable string
                    formatted_value = np.array2string(value.flatten(), separator=', ')
                else:
                    formatted_value = self.get_safe_scalar_value(value)
                data_for_df.append([key, formatted_value])
                
            df = pd.DataFrame(data_for_df, columns=['Metric', 'Value'])
            df.to_excel(file_path, index=False)
            messagebox.showinfo("Export Complete", f"The metrics were successfully saved to:\n{os.path.abspath(file_path)}")
        except Exception as e:
            messagebox.showerror("Export Error", f"An error occurred while exporting to Excel: {e}")

    # --- MAIN FUNCTIONS ---

    def load_data(self):
        try:
            file_path = filedialog.askopenfilename(title="Select .mat file with all data", filetypes=[("MATLAB files", "*.mat")])
            if not file_path: return
            self.status_label.config(text="Status: Loading data..."); self.root.update_idletasks()
            mat_content = sio.loadmat(file_path)
            var_map = {'Xcal': 'Xcal_data', 'Ycal': 'Ycal_data', 'Xpred': 'Xpred_data', 'Ypred': 'Ypred_data', 'Xval': 'Xval_data', 'Yval': 'Yval_data', 'xaxis': 'xaxis_data'}
            loaded_vars_info = []
            for mat_name, py_name in var_map.items():
                if mat_name in mat_content:
                    data = np.asarray(mat_content[mat_name], dtype=float)
                    if 'Y' in mat_name: data = data.reshape(-1, 1)
                    if 'xaxis' in mat_name: data = data.flatten()
                    setattr(self, py_name, data); loaded_vars_info.append(f"   - '{mat_name}' loaded ({data.shape})")
                else:
                    if mat_name in ['Xval', 'Yval', 'xaxis']: setattr(self, py_name, np.array([]))
                    else: raise ValueError(f"The required variable '{mat_name}' was not found in the .mat file.")
            if self.Xcal_data.shape[1] != self.Xpred_data.shape[1] or (self.Xval_data.size > 0 and self.Xcal_data.shape[1] != self.Xval_data.shape[1]):
                raise ValueError("Column count inconsistency between datasets.")
            if self.xaxis_data.size == 0 or self.xaxis_data.size != self.Xcal_data.shape[1]:
                self.xaxis_data = np.arange(1, self.Xcal_data.shape[1] + 1)
            self.status_label.config(text="Status: Data loaded."); messagebox.showinfo("Success", "Data loaded from file!\n\n" + "\n".join(loaded_vars_info))
        except Exception as e:
            messagebox.showerror("Error Loading Data", str(e)); self.status_label.config(text="Status: Load failed.")

    def run_multi_repetitions_handler(self):
        if self.Xcal_data is None: messagebox.showerror("Error", "Load Data first."); return
        try:
            self.reset_all()
            params = {"num_repetitions": self.get_entry_value(self.entries["Repetitions:"], "Repetitions"), 
                      "intervalos": self.get_entry_value(self.entries["Intervals:"], "Intervals"), 
                      "VL": self.get_entry_value(self.entries["VL (Lat. Vars.):"], "VL"), 
                      "fpop": self.get_entry_value(self.entries["F_pop (Population):"], "Population"), 
                      "ciclos": self.get_entry_value(self.entries["Cycles:"], "Cycles"), 
                      "w0": self.get_entry_value(self.entries["w0:"], "w0", "float"), 
                      "gama": self.get_entry_value(self.entries["Gama:"], "Gama", "float"), 
                      "alfa": self.get_entry_value(self.entries["Alfa:"], "Alfa", "float")}
            
            self.progress_bar.pack(pady=(5,10), fill=tk.X, padx=5)
            self.progress_bar['maximum'] = params["num_repetitions"]
            
            temp_results = []
            for i in range(params["num_repetitions"]):
                self.status_label.config(text=f"Status: Running repetition {i + 1} of {params['num_repetitions']}...")
                # We call the backend function for a single repetition
                single_rep_data = ffipls(self.Xcal_data, self.Ycal_data, self.Xval_data, self.Yval_data, self.Xpred_data, self.Ypred_data, params["intervalos"], params["VL"], params["fpop"], params["ciclos"], params["w0"], params["gama"], params["alfa"], 1)
                temp_results.extend(single_rep_data) # Add the repetition result
                
                # Update the progress bar
                self.progress_bar['value'] = i + 1
                self.root.update_idletasks() # Force GUI update

            self.all_repetitions_data = temp_results
            self.progress_bar.pack_forget() # Hide progress bar on completion
            
            self.status_label.config(text="Status: Optimization complete. Displaying results..."); self.root.update_idletasks()
            self.display_summary_report(self.all_repetitions_data)
            for i in self.summary_tree.get_children(): self.summary_tree.delete(i)
            rep_numbers = [str(i + 1) for i, _ in enumerate(self.all_repetitions_data)]
            for i, rep in enumerate(self.all_repetitions_data):
                model = rep.get('Model_FF_iPLS', {})
                if isinstance(model, str) and model == 'FAILED':
                    self.summary_tree.insert("", "end", values=(i + 1, "FAILED", "FAILED", "FAILED", "FAILED")); continue
                self.summary_tree.insert("", "end", values=(i + 1, f"{model.get('RMSEC', np.nan):.3f}", f"{model.get('RMSEV', model.get('RMSECV', np.nan)):.3f}", f"{model.get('RMSEP', np.nan):.3f}", f"{model.get('R2_pred', np.nan):.3f}"))
            self.classic_repetition_selector['values'] = rep_numbers; self.classic_repetition_selector.set("Select"); self.classic_view_details_button.config(state=tk.NORMAL)
            self.status_label.config(text=f"Execution finished."); messagebox.showinfo("Process Complete!", f"All {params['num_repetitions']} repetitions have been executed.")
        except Exception as e:
            self.status_label.config(text="Status: Execution error."); messagebox.showerror("Execution Error", f"An error occurred: {e}"); print(f"DEBUG: {repr(e)}")
            self.progress_bar.pack_forget() # Ensure bar is hidden on error

    def display_classic_details(self):
        try:
            rep_number_str = self.classic_repetition_selector.get()
            if not rep_number_str or rep_number_str == "Select": raise ValueError("Please select a repetition.")
            rep_number = int(rep_number_str); self.selected_repetition_index = rep_number - 1
            selected_data = self.all_repetitions_data[self.selected_repetition_index]
            self.Model_FF_iPLS_global = selected_data.get('Model_FF_iPLS'); self.optimization_evolution_global = selected_data.get('optimization_evolution')
            if self.Model_FF_iPLS_global is None or (isinstance(self.Model_FF_iPLS_global, str) and self.Model_FF_iPLS_global == 'FAILED'):
                raise ValueError("Model data not found or this repetition failed.")
            self.display_full_report_grid(self.Model_FF_iPLS_global)
            self.draw_results_plots(self.classic_plot_figure, self.classic_plot_canvas, self.Model_FF_iPLS_global)
            
            # ATIVAR BOTÕES
            self.classic_generate_plot_button.config(state=tk.NORMAL)
            self.classic_export_excel_button.config(state=tk.NORMAL) 
            self.classic_export_matlab_button.config(state=tk.NORMAL)
            
            self.status_label.config(text=f"Rep. {rep_number}: Report and graphs displayed.")
        except (ValueError, IndexError) as e: 
            messagebox.showerror("Selection Error", str(e))
            self.classic_generate_plot_button.config(state=tk.DISABLED)
            self.classic_export_excel_button.config(state=tk.DISABLED) 
            self.classic_export_matlab_button.config(state=tk.DISABLED)
        except Exception as e: 
            messagebox.showerror("Error", f"Error viewing details: {e}")
            self.classic_generate_plot_button.config(state=tk.DISABLED)
            self.classic_export_excel_button.config(state=tk.DISABLED) 
            self.classic_export_matlab_button.config(state=tk.DISABLED)

    def on_repetition_select(self, event):
        selection = self.summary_tree.selection();
        if not selection: return
        item = self.summary_tree.item(selection[0]); rep_number = int(item['values'][0]); self.selected_repetition_index = rep_number - 1
        selected_data = self.all_repetitions_data[self.selected_repetition_index]
        self.Model_FF_iPLS_global = selected_data.get('Model_FF_iPLS'); self.optimization_evolution_global = selected_data.get('optimization_evolution')
        if self.Model_FF_iPLS_global is None or (isinstance(self.Model_FF_iPLS_global, str) and self.Model_FF_iPLS_global == 'FAILED'):
            messagebox.showerror("Error", f"Data for repetition {rep_number} is missing or failed."); return
        for i in self.cycle_tree.get_children(): self.cycle_tree.delete(i)
        cycles = np.asarray(self.optimization_evolution_global.get('cycles', [])).flatten(); min_rmse = np.asarray(self.optimization_evolution_global.get('min_rmse_evolution', [])).flatten(); intervals_evol = self.optimization_evolution_global.get('selected_intervals_evolution', [])
        best_cycle_idx = np.argmin(min_rmse) if min_rmse.size > 0 else -1
        for i in range(len(cycles)):
            intervals_list = [int(val) for val in np.asarray(intervals_evol[i]).flatten()]
            tags = ('best',) if i == best_cycle_idx else ()
            self.cycle_tree.insert("", "end", values=(int(cycles[i]), f"{min_rmse[i]:.4f}", str(intervals_list)), tags=tags)
        self.draw_results_plots(self.interactive_plot_figure, self.interactive_plot_canvas, self.Model_FF_iPLS_global)
        self.notebook.select(1); self.status_label.config(text=f"Analyzing Rep. {rep_number}. Click a cycle for details.")

    def on_cycle_select(self, event):
        selection = self.cycle_tree.selection()
        if not selection: return
        item = self.cycle_tree.item(selection[0]); cycle_num = int(item['values'][0])
        self.draw_results_plots(self.interactive_plot_figure, self.interactive_plot_canvas, self.Model_FF_iPLS_global, highlighted_cycle=cycle_num)
        self.status_label.config(text=f"Viewing Cycle {cycle_num} of Rep. {self.selected_repetition_index+1}.")

    def reset_all(self):
        self.all_repetitions_data, self.selected_repetition_index, self.Model_FF_iPLS_global, self.optimization_evolution_global = [], -1, None, None
        if self.summary_tree:
            for i in self.summary_tree.get_children(): self.summary_tree.delete(i)
        if self.cycle_tree:
            for i in self.cycle_tree.get_children(): self.cycle_tree.delete(i)
        self.clear_report_frame()
        if self.classic_repetition_selector: 
            self.classic_repetition_selector['values'] = []
            self.classic_repetition_selector.set('Select')
            self.classic_view_details_button.config(state=tk.DISABLED)
            self.classic_generate_plot_button.config(state=tk.DISABLED)
            self.classic_export_excel_button.config(state=tk.DISABLED) 
            self.classic_export_matlab_button.config(state=tk.DISABLED)
        if self.classic_plot_figure: self.classic_plot_figure.clear(); self.classic_plot_canvas.draw()
        if self.interactive_plot_figure: self.interactive_plot_figure.clear(); self.interactive_plot_canvas.draw()
        if self.notebook: self.notebook.select(0)
        self.status_label.config(text="Status: Ready.")

    def _setup_gui_components(self):
        try: self.root.iconbitmap('firefly.ico')
        except tk.TclError: print("Warning: 'firefly.ico' not found.")
        self.root.configure(bg="#f0f0f0"); style = ttk.Style(self.root); style.theme_use("clam")
        header_frame = tk.Frame(self.root, bg="#2c3e50"); header_frame.pack(fill=tk.X)
        header_text_frame = tk.Frame(header_frame, bg="#2c3e50"); header_text_frame.pack(expand=True)
        tk.Label(header_text_frame, text="Firefly Variable Selection Algorithm \U0001F41D", fg="white", bg="#2c3e50", font=("Arial", 14, "bold")).pack(side=tk.LEFT, pady=5, padx=10)
        tk.Label(header_text_frame, text="Developer: Enia Mendes", fg="white", bg="#2c3e50", font=("Arial", 8)).pack(side=tk.LEFT, pady=5)
        main_frame = ttk.Frame(self.root); main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        control_frame = ttk.LabelFrame(main_frame, text="Controls and Parameters"); control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=5)
        tk.Button(control_frame, text="Load Data (.mat)", command=self.load_data, bg="#4682B4", fg="white").pack(pady=5, fill=tk.X, padx=5)
        param_frame = ttk.LabelFrame(control_frame, text="FF-iPLS Parameters"); param_frame.pack(pady=10, padx=5, fill=tk.X)
        param_labels = ["Intervals:", "I_max:", "VL (Lat. Vars.):", "F_pop (Population):", "Cycles:", "w0:", "Gama:", "Alfa:", "Repetitions:"]
        default_values = ["10", "20", "5", "20", "20", "0.97", "1.0", "0.5", "4"]
        for i, label in enumerate(param_labels):
            tk.Label(param_frame, text=label).grid(row=i, column=0, sticky="w", padx=2, pady=2); entry = tk.Entry(param_frame, width=10); entry.insert(0, default_values[i]); entry.grid(row=i, column=1, padx=5, pady=2, sticky="ew"); self.entries[label] = entry
        run_button = tk.Button(control_frame, text="Run Repetitions", command=self.run_multi_repetitions_handler, font=("Arial", 11, "bold"), bg="#8A2BE2", fg="white")
        run_button.pack(pady=10, fill=tk.X, padx=5)
        tk.Button(control_frame, text="Reset All", command=self.reset_all, bg="#B0C4DE").pack(pady=5, fill=tk.X, padx=5)

        self.progress_bar = ttk.Progressbar(control_frame, orient='horizontal', mode='determinate')

        results_frame = ttk.Frame(main_frame); results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.notebook = ttk.Notebook(results_frame); self.notebook.pack(fill=tk.BOTH, expand=True)
        tab1 = ttk.Frame(self.notebook); self.notebook.add(tab1, text='Initial Analysis')
        tab1_paned_window = ttk.PanedWindow(tab1, orient=tk.HORIZONTAL); tab1_paned_window.pack(fill=tk.BOTH, expand=True)
        left_pane_tab1 = ttk.Frame(tab1_paned_window, width=400); tab1_paned_window.add(left_pane_tab1, weight=2)
        right_pane_tab1 = ttk.LabelFrame(tab1_paned_window, text="Graphic Dashboard"); tab1_paned_window.add(right_pane_tab1, weight=5)
        classic_controls_frame = ttk.Frame(left_pane_tab1); classic_controls_frame.pack(fill=tk.X, pady=5, padx=5)
        ttk.Label(classic_controls_frame, text="Repetition:").pack(side=tk.LEFT, padx=(0,2))
        self.classic_repetition_selector = ttk.Combobox(classic_controls_frame, state="readonly", width=8); self.classic_repetition_selector.pack(side=tk.LEFT, padx=2)
        self.classic_view_details_button = tk.Button(classic_controls_frame, text="View Details", command=self.display_classic_details, state=tk.DISABLED); self.classic_view_details_button.pack(side=tk.LEFT, padx=2)
        
        # --- MODIFIED AND NEW BUTTONS ---
        self.classic_generate_plot_button = tk.Button(classic_controls_frame, text="Export Graphs...", command=self.open_export_plot_options, state=tk.DISABLED); self.classic_generate_plot_button.pack(side=tk.LEFT, padx=2)
        self.classic_export_excel_button = tk.Button(classic_controls_frame, text="Export to Excel", command=self.export_metrics_to_excel, state=tk.DISABLED); self.classic_export_excel_button.pack(side=tk.LEFT, padx=2)
        
        # --- NEW BUTTON ---
        self.classic_export_matlab_button = tk.Button(classic_controls_frame, text="Export to MATLAB (.mat)", command=self.export_model_to_matlab, state=tk.DISABLED); self.classic_export_matlab_button.pack(side=tk.LEFT, padx=2)

        classic_report_container = ttk.LabelFrame(left_pane_tab1, text="Execution Summary"); classic_report_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0,5))
        canvas = tk.Canvas(classic_report_container); scrollbar = ttk.Scrollbar(classic_report_container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set); canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.classic_report_frame = ttk.Frame(canvas); canvas.create_window((0, 0), window=self.classic_report_frame, anchor="nw")
        def on_frame_configure(event): canvas.configure(scrollregion=canvas.bbox("all"))
        self.classic_report_frame.bind("<Configure>", on_frame_configure)
        self.classic_plot_figure = plt.figure(); self.classic_plot_canvas = FigureCanvasTkAgg(self.classic_plot_figure, master=right_pane_tab1)
        self.classic_plot_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        tab2 = ttk.Frame(self.notebook); self.notebook.add(tab2, text='Interactive Analysis')
        paned_window = ttk.PanedWindow(tab2, orient=tk.VERTICAL); paned_window.pack(fill=tk.BOTH, expand=True)
        summary_frame = ttk.LabelFrame(paned_window, text="Repetitions Summary (Click to analyze)"); paned_window.add(summary_frame, weight=2)
        cols = ("Rep.", "RMSEC", "RMSEV/CV", "RMSEP", "R2_Pred"); self.summary_tree = ttk.Treeview(summary_frame, columns=cols, show='headings')
        for col in cols: self.summary_tree.heading(col, text=col); self.summary_tree.column(col, width=100, anchor=tk.CENTER)
        vsb_summary = ttk.Scrollbar(summary_frame, orient="vertical", command=self.summary_tree.yview); self.summary_tree.configure(yscrollcommand=vsb_summary.set); vsb_summary.pack(side='right', fill='y'); self.summary_tree.pack(fill=tk.BOTH, expand=True)
        self.summary_tree.bind('<<TreeviewSelect>>', self.on_repetition_select)
        cycle_frame = ttk.LabelFrame(paned_window, text="Cycles Evolution (Click to highlight on graph)"); paned_window.add(cycle_frame, weight=2)
        cycle_cols = ("Cycle", "RMSEP", "Intervals"); self.cycle_tree = ttk.Treeview(cycle_frame, columns=cycle_cols, show='headings')
        self.cycle_tree.heading("Cycle", text="Cycle"); self.cycle_tree.column("Cycle", width=50, anchor=tk.CENTER); self.cycle_tree.heading("RMSEP", text="RMSEP"); self.cycle_tree.column("RMSEP", width=100, anchor=tk.CENTER)

        self.cycle_tree.heading("Intervals", text="Selected Intervals"); self.cycle_tree.column("Intervals", width=400); self.cycle_tree.tag_configure('best', background='lightgreen')

        vsb_cycle = ttk.Scrollbar(cycle_frame, orient="vertical", command=self.cycle_tree.yview); self.cycle_tree.configure(yscrollcommand=vsb_cycle.set); vsb_cycle.pack(side='right', fill='y'); self.cycle_tree.pack(fill=tk.BOTH, expand=True)
        self.cycle_tree.bind('<<TreeviewSelect>>', self.on_cycle_select)
        plot_frame = ttk.LabelFrame(paned_window, text="Interactive Graphical Visualization"); paned_window.add(plot_frame, weight=4)
        self.interactive_plot_figure = plt.figure(); self.interactive_plot_canvas = FigureCanvasTkAgg(self.interactive_plot_figure, master=plot_frame); self.interactive_plot_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.interactive_plot_canvas, plot_frame); toolbar.update(); self.interactive_plot_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.status_label = tk.Label(self.root, text="Status: Waiting...", bd=1, relief=tk.SUNKEN, anchor=tk.W, padx=10)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

# =============================================================================
#  BLOCO PRINCIPAL (COM CORREÇÃO DE DPI PARA WINDOWS)
# =============================================================================

# Tenta configurar a alta resolução de DPI no Windows
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass  # Falha silenciosa em sistemas não-Windows ou versões antigas

if __name__ == "__main__":
    app = FFI_PLS_App()
    app.root.mainloop()