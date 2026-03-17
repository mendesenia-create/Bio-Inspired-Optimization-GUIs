import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
import scipy.io as sio
import webbrowser
import random
import math
import pandas as pd
import ctypes  # Para alta resolução no Windows
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc)
from sklearn.preprocessing import StandardScaler, label_binarize
from joblib import Parallel, delayed, parallel_backend

# =============================================================================
#  CÓDIGO-MÃE (MOTOR MATEMÁTICO DO BAT-LDA)
# =============================================================================

class BAT_LDA:
    def __init__(self, num_bats=30, num_iter=100, min_features=2, max_features=20, 
                 alpha=0.5, gamma=0.5, f_min=0.0, f_max=0.05, random_state=None, cv_folds=5):
        self.num_bats = int(num_bats)
        self.num_iter = int(num_iter)
        self.min_features = int(min_features)
        self.max_features = int(max_features)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.random_state = random_state
        self.cv_folds = int(cv_folds)
        self.best_subset_mask_ = None
        self.history_accuracy_ = []
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

    def _evaluate_subset(self, X, y, subset_mask, cv):
        # Evita subconjuntos vazios ou classes insuficientes para CV
        if not subset_mask.any() or np.min(np.unique(y, return_counts=True)[1]) < self.cv_folds:
            return 0.0
        
        X_subset = X[:, subset_mask]
        
        # Evita erro de singularidade no LDA (features >= samples)
        if X_subset.shape[1] >= X_subset.shape[0]: 
            return 0.0
            
        acc_scores = []
        try:
            for train_idx, test_idx in cv.split(X, y):
                X_train_fold, y_train_fold = X_subset[train_idx], y[train_idx]
                X_test_fold, y_test_fold = X_subset[test_idx], y[test_idx]
                
                # Checagem extra dentro do fold
                if X_train_fold.shape[1] >= X_train_fold.shape[0]: 
                    continue
                
                clf = LDA(solver='lsqr', shrinkage='auto')
                clf.fit(X_train_fold, y_train_fold)
                y_pred = clf.predict(X_test_fold)
                acc_scores.append(accuracy_score(y_test_fold, y_pred))
            
            return float(np.mean(acc_scores)) if acc_scores else 0.0
        except Exception:
            return 0.0

    def fit(self, X, y):
        n_features = X.shape[1]
        sigma = 0.5
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Inicialização
        positions = np.random.rand(self.num_bats, n_features)
        velocities = np.zeros((self.num_bats, n_features))
        loudness = np.ones(self.num_bats)
        pulse_rate = np.random.rand(self.num_bats)
        initial_pulse_rate = pulse_rate.copy()
        
        # Garantir restrições iniciais
        subsets = (positions > sigma)
        for i in range(self.num_bats):
            num_selected = np.sum(subsets[i])
            if not (self.min_features <= num_selected <= self.max_features):
                num_to_select = random.randint(self.min_features, self.max_features)
                selected_indices = np.random.choice(n_features, num_to_select, replace=False)
                subsets[i] = False; subsets[i, selected_indices] = True
                # Atualiza a posição contínua para refletir a binária
                positions[i] = np.where(subsets[i], 0.6 + 0.4*np.random.rand(), 0.4*np.random.rand())
        
        # Avaliação Inicial
        with Parallel(n_jobs=-1, backend='threading') as parallel:
            costs = parallel(delayed(self._evaluate_subset)(X, y, s, cv) for s in subsets)
            
        costs = np.array(costs)
        best_idx = np.argmax(costs)
        best_accuracy = costs[best_idx]
        best_position = positions[best_idx, :].copy()
        self.history_accuracy_.append(best_accuracy)
        
        # Loop Principal do Morcego
        for t in range(self.num_iter):
            # print(f"Iteration {t+1}/{self.num_iter} – Best CV Accuracy: {best_accuracy:.4f}")
            mean_loudness = np.mean(loudness)
            
            for i in range(self.num_bats):
                beta = np.random.rand()
                freq = self.f_min + (self.f_max - self.f_min) * beta
                
                # Atualiza Velocidade e Posição
                velocities[i, :] += (positions[i, :] - best_position) * freq
                new_position = positions[i, :] + velocities[i, :]
                
                # Random Walk (Local Search)
                if np.random.rand() > pulse_rate[i]:
                    new_position += (2 * np.random.rand(n_features) - 1) * mean_loudness
                
                new_position = np.clip(new_position, 0, 1)
                new_subset = (new_position > sigma)
                num_selected = np.sum(new_subset)
                
                # Avalia apenas se respeitar limites de features
                new_cost = -1.0
                if (self.min_features <= num_selected <= self.max_features):
                    new_cost = self._evaluate_subset(X, y, new_subset, cv)

                # Critério de Aceitação (Simulated Annealing-like)
                # Nota: Para acurácia, queremos MAIOR, então new_cost > costs[i]
                if (np.random.rand() < loudness[i]) and (new_cost >= costs[i]):
                    positions[i, :] = new_position.copy()
                    costs[i] = new_cost
                    loudness[i] *= self.alpha
                    pulse_rate[i] = initial_pulse_rate[i] * (1 - math.exp(-self.gamma * (t+1)))
            
            # Atualiza Melhor Global
            current_best_idx = np.argmax(costs)
            if costs[current_best_idx] > best_accuracy:
                best_accuracy = costs[current_best_idx]
                best_position = positions[current_best_idx, :].copy()
            
            self.history_accuracy_.append(best_accuracy)
            
        self.best_subset_mask_ = (best_position > sigma)
        
        # Verificação final de segurança
        if not (self.min_features <= np.sum(self.best_subset_mask_) <= self.max_features):
             return {"error": "Final solution is outside variable limits."}
             
        return {"best_subset_mask": self.best_subset_mask_, "cost_evolution": self.history_accuracy_, "iterations": list(range(1, self.num_iter + 2))}

# =============================================================================
#  CLASSE DA INTERFACE GRÁFICA (PADRONIZADA E ROBUSTA)
# =============================================================================

class BAT_LDA_App:
    def __init__(self, root_master=None):
        if root_master:
            self.root = tk.Toplevel(root_master)
        else:
            self.root = tk.Tk()
        
        self.root.title("BAT-LDA Interface")
        self.root.geometry("1600x900")
        
        # Variáveis de Estado
        self.all_repetitions_data = []
        self.selected_repetition_index = -1
        self.Model_BAT_global = None
        self.evolution_global = None
        self.X_train_data = None
        self.Y_train_data = None
        self.X_val_data = None
        self.Y_val_data = None
        self.X_test_data = None
        self.Y_test_data = None
        self.variables_data = None
        self.entries = {}

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

    # --- FUNÇÕES DE EXPORTAÇÃO ---
    def export_model_to_matlab(self):
        if not self.Model_BAT_global:
            messagebox.showwarning("Warning", "No model selected.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".mat", filetypes=[("MATLAB Files", "*.mat")], title="Export Model to MATLAB")
        if not file_path: return
        try:
            mat_data = {
                'Model': self.Model_BAT_global, 'Evolution': self.evolution_global,
                'Data': {'X_train': self.X_train_data, 'Y_train': self.Y_train_data, 'X_val': self.X_val_data if self.X_val_data is not None else [], 'Y_val': self.Y_val_data if self.Y_val_data is not None else [], 'X_test': self.X_test_data, 'Y_test': self.Y_test_data}
            }
            sio.savemat(file_path, mat_data)
            messagebox.showinfo("Export Complete", f"Model successfully exported to:\n{os.path.abspath(file_path)}")
        except Exception as e: messagebox.showerror("Export Error", f"Error exporting to MATLAB: {e}")

    def export_metrics_to_excel(self):
        if not self.Model_BAT_global:
            messagebox.showwarning("Warning", "No model selected.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Workbook", "*.xlsx")], title="Export Metrics to Excel")
        if not file_path: return
        try:
            data_for_df = []
            skip_keys = [k for k in self.Model_BAT_global.keys() if 'confusion_matrix' in k or 'roc_auc_data' in k or 'var_sel' in k or 'classes' in k]
            for key, value in self.Model_BAT_global.items():
                if key in skip_keys: continue
                formatted_value = self.get_safe_scalar_value(value)
                if isinstance(formatted_value, float): formatted_value = f"{formatted_value:.4f}"
                data_for_df.append([key, formatted_value])
            df = pd.DataFrame(data_for_df, columns=['Metric', 'Value'])
            df.to_excel(file_path, index=False)
            messagebox.showinfo("Export Complete", f"Metrics saved to:\n{os.path.abspath(file_path)}")
        except Exception as e: messagebox.showerror("Export Error", f"Error exporting to Excel: {e}")

    def save_all_plots_composite(self, model_data, evolution_data):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("TIFF files", "*.tiff"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            title="Save Composite Graph Image"
        )
        if not file_path: return
        fig = plt.figure(figsize=(18, 9))
        self.draw_classification_dashboard(fig, None, model_data, evolution_data)
        try:
            fig.savefig(file_path, dpi=300); plt.close(fig)
            messagebox.showinfo("Graph Saved", f"Composite graph saved to:\n{os.path.abspath(file_path)}")
            webbrowser.open(file_path)
        except Exception as e: plt.close(fig); messagebox.showerror("Error Saving", f"Could not save graph: {e}")

    def save_single_plot(self, plot_type, model_data, evolution_data):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("TIFF files", "*.tiff"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            title=f"Save {plot_type} Graph"
        )
        if not file_path: return
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle(f'{plot_type} (Repetition {self.selected_repetition_index + 1})', fontsize=14, fontweight='bold')
        try:
            class_labels = model_data.get('classes', [])
            if plot_type == 'Confusion Matrix (Train)':
                cm = np.array(model_data.get('confusion_matrix_train', []))
                if cm.size > 0: sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=class_labels, yticklabels=class_labels); ax.set_xlabel('Predicted', fontweight='bold'); ax.set_ylabel('True', fontweight='bold')
            elif plot_type == 'Confusion Matrix (Test)':
                cm = np.array(model_data.get('confusion_matrix_test', []))
                if cm.size > 0: sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=class_labels, yticklabels=class_labels); ax.set_xlabel('Predicted', fontweight='bold'); ax.set_ylabel('True', fontweight='bold')
            elif plot_type == 'ROC Curve (Test)':
                roc_data = model_data.get('roc_auc_data_test', {})
                if roc_data:
                    for class_id, data in roc_data.items(): ax.plot(data['fpr'], data['tpr'], label=f'{class_id} (AUC = {data["auc"]:.2f})')
                    ax.plot([0, 1], [0, 1], 'k--'); ax.legend(fontsize='small'); ax.grid(True)
            elif plot_type == 'Accuracy Evolution':
                cost_evolution = evolution_data.get('cost_evolution', [])
                ax.plot(evolution_data.get('iterations', []), cost_evolution, 'o-')
                ax.set_xlabel('Cycle', fontweight='bold'); ax.set_ylabel('CV Accuracy', fontweight='bold'); ax.grid(True)
            elif plot_type == 'Selected Variables':
                var_sel_indices = model_data.get('var_sel_indices', [])
                if var_sel_indices:
                    ax.stem(var_sel_indices, np.ones_like(var_sel_indices))
                    xlim_max = len(self.variables_data[0]) if self.variables_data is not None and self.variables_data.ndim > 1 else (self.X_train_data.shape[1] if self.X_train_data is not None else 1)
                    ax.set_xlim(0, xlim_max); ax.set_yticks([]); ax.set_xlabel('Variables', fontweight='bold')
            elif plot_type == 'Global Metrics (Bar)':
                metrics_to_plot = {'Accuracy': model_data.get('accuracy_test', 0), 'Sensitivity': model_data.get('sensitivity_recall_test', 0), 'Specificity': model_data.get('specificity_test', 0)}
                ax.bar(metrics_to_plot.keys(), metrics_to_plot.values(), color=['#4CAF50', '#2196F3', '#FFC107'])
                ax.set_ylim(0, 1.1); ax.set_ylabel('Score', fontweight='bold')
                for i, v in enumerate(metrics_to_plot.values()): ax.text(i, v + 0.02, f"{v:.3f}", ha='center')

            fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(file_path, dpi=300); plt.close(fig)
            messagebox.showinfo("Graph Saved", f"Graph saved to:\n{os.path.abspath(file_path)}")
        except Exception as e: plt.close(fig); messagebox.showerror("Error Saving", f"Error: {e}")

    def open_export_plot_options(self):
        if not self.Model_BAT_global: messagebox.showwarning("Warning", "Select a repetition first."); return
        top = tk.Toplevel(self.root); top.title("Graph Export Options"); top.geometry("350x320"); top.transient(self.root); top.grab_set()
        ttk.Button(top, text="Save All Together (Dashboard)", command=lambda: [self.save_all_plots_composite(self.Model_BAT_global, self.evolution_global), top.destroy()]).pack(pady=4, padx=20, fill=tk.X)
        ttk.Separator(top, orient='horizontal').pack(fill='x', padx=10, pady=5)
        plot_options = [("Confusion Matrix (Train)", "Confusion Matrix (Train)"), ("Confusion Matrix (Test)", "Confusion Matrix (Test)"),
                        ("ROC Curve (Test)", "ROC Curve (Test)"), ("Accuracy Evolution", "Accuracy Evolution"),
                        ("Selected Variables", "Selected Variables"), ("Global Metrics (Bar)", "Global Metrics (Bar)")]
        for label, ptype in plot_options: ttk.Button(top, text=f"Save Only {label}", command=lambda p=ptype: [self.save_single_plot(p, self.Model_BAT_global, self.evolution_global), top.destroy()]).pack(pady=2, padx=20, fill=tk.X)

    # --- CÁLCULO DE MÉTRICAS E PLOTAGEM ---
    def _get_classification_metrics(self, y_true, y_pred_classes, y_pred_scores, classes):
        metrics = {}; n_classes = len(classes); cm = confusion_matrix(y_true, y_pred_classes, labels=classes); metrics['confusion_matrix'] = cm.tolist()
        specificity_per_class = []
        for i in range(n_classes):
            tn = cm.sum() - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i]); fp = np.sum(cm[:, i]) - cm[i, i]
            specificity_per_class.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        metrics['accuracy'] = accuracy_score(y_true, y_pred_classes)
        metrics['sensitivity_recall'] = recall_score(y_true, y_pred_classes, average='macro', zero_division=0)
        metrics['specificity'] = np.mean(specificity_per_class)
        metrics['precision'] = precision_score(y_true, y_pred_classes, average='macro', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred_classes, average='macro', zero_division=0)
        roc_data = {}; y_true_binarized = label_binarize(y_true, classes=classes)
        if n_classes == 2:
            y_scores = y_pred_scores[:, 1]; fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=classes[1]); roc_auc = auc(fpr, tpr)
            roc_data['binary'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': roc_auc}
        elif n_classes > 2 and y_true_binarized.shape[1] == y_pred_scores.shape[1]:
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_pred_scores[:, i]); roc_auc = auc(fpr, tpr)
                roc_data[f'class_{classes[i]}'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': roc_auc}
        metrics['roc_auc_data'] = roc_data
        return metrics

    def draw_classification_dashboard(self, figure, canvas, model_results, evolution_results, stage='final', cycle_data=None):
        figure.clear()
        if not model_results or model_results.get('error'): return
        
        # Título dinâmico
        title = f"Classification Results - Repetition {self.selected_repetition_index + 1}"
        if stage == 'cycle': title += f" (Cycle {cycle_data['cycle']})"
        figure.suptitle(title, fontsize=12, fontweight='bold', y=0.98)
        
        gs = figure.add_gridspec(2, 3)
        class_labels = model_results.get('classes', [])
        
        # 1. CM Train
        ax1 = figure.add_subplot(gs[0, 0]); cm_train = np.array(model_results.get('confusion_matrix_train', []))
        if cm_train.size > 0: sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=ax1, xticklabels=class_labels, yticklabels=class_labels); ax1.set_title('Confusion Matrix (Train)', fontsize=9, fontweight='bold'); ax1.set_xlabel('Predicted', fontsize=9, fontweight='bold'); ax1.set_ylabel('True', fontsize=9, fontweight='bold')
        else: ax1.text(0.5, 0.5, "N/A", ha='center')
        
        # 2. CM Test
        ax2 = figure.add_subplot(gs[0, 1]); cm_test = np.array(model_results.get('confusion_matrix_test', []))
        if cm_test.size > 0: sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=ax2, xticklabels=class_labels, yticklabels=class_labels); ax2.set_title('Confusion Matrix (Test)', fontsize=9, fontweight='bold'); ax2.set_xlabel('Predicted', fontsize=9, fontweight='bold'); ax2.set_ylabel('True', fontsize=9, fontweight='bold')
        else: ax2.text(0.5, 0.5, "N/A", ha='center')

        # 3. ROC Test
        ax3 = figure.add_subplot(gs[0, 2]); roc_data = model_results.get('roc_auc_data_test', {})
        if roc_data:
            for class_id, data in roc_data.items(): ax3.plot(data['fpr'], data['tpr'], label=f'{class_id} (AUC={data["auc"]:.2f})')
            ax3.plot([0, 1], [0, 1], 'k--'); ax3.set_title('ROC Curve (Test)', fontsize=9, fontweight='bold'); ax3.legend(fontsize='small'); ax3.grid(True)
        else: ax3.text(0.5, 0.5, "N/A", ha='center')

        # 4. Evolution
        ax4 = figure.add_subplot(gs[1, 0]); cost_evolution = evolution_results.get('cost_evolution', [])
        ax4.plot(evolution_results.get('iterations', []), cost_evolution, 'o-'); ax4.set_title('Accuracy Evolution', fontsize=9, fontweight='bold'); ax4.set_xlabel('Cycle', fontsize=9, fontweight='bold'); ax4.set_ylabel('CV Accuracy', fontsize=9, fontweight='bold'); ax4.grid(True)

        # 5. Selected Vars (Pode mudar se for visualização de ciclo)
        ax5 = figure.add_subplot(gs[1, 1])
        var_sel = cycle_data['vars'] if stage == 'cycle' and cycle_data else model_results.get('var_sel_indices', [])
        
        if len(var_sel) > 0:
            ax5.stem(var_sel, np.ones_like(var_sel)); ax5.set_title(f'Selected Vars ({len(var_sel)})', fontsize=9, fontweight='bold'); ax5.set_yticks([])
            max_x = self.X_train_data.shape[1] if self.X_train_data is not None else 100; ax5.set_xlim(0, max_x)
        else: ax5.text(0.5, 0.5, "No variables\nselected", ha='center', va='center')

        # 6. Metrics Bar
        ax6 = figure.add_subplot(gs[1, 2]); metrics = {'Acc': model_results.get('accuracy_test', 0), 'Sens': model_results.get('sensitivity_recall_test', 0), 'Spec': model_results.get('specificity_test', 0)}
        ax6.bar(metrics.keys(), metrics.values(), color=['#4CAF50', '#2196F3', '#FFC107']); ax6.set_title('Global Metrics (Test)', fontsize=9, fontweight='bold'); ax6.set_ylim(0, 1.1)
        for i, v in enumerate(metrics.values()): ax6.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=8)

        figure.tight_layout(rect=[0, 0, 1, 0.95])
        if canvas: canvas.draw()

    # --- INTERFACE E AUXILIARES ---
    def clear_frame(self, frame):
        if frame:
            for widget in frame.winfo_children(): widget.destroy()

    def display_initial_summary_report(self, all_data):
        self.clear_frame(self.initial_summary_frame)
        if not all_data: return
        
        # Filtra repetições válidas
        valid_reps = [r for r in all_data if 'error' not in r.get('Model', {})]
        
        if not valid_reps:
            ttk.Label(self.initial_summary_frame, text="All repetitions failed.", foreground='red').pack()
            return
            
        acc_values = [r['Model'].get('accuracy_test', -1) for r in valid_reps]
        
        # Índice dentro dos válidos
        best_valid_idx = np.argmax(acc_values)
        best_rep = valid_reps[best_valid_idx]
        
        # Índice original
        best_idx = self.all_repetitions_data.index(best_rep)
        
        ttk.Label(self.initial_summary_frame, text=f"Best: Rep. {best_idx + 1} (Test Acc = {np.max(acc_values):.4f})", font=('Arial', 11, 'bold'), foreground='green').pack(anchor='w', padx=10, pady=5)
        ttk.Separator(self.initial_summary_frame, orient='horizontal').pack(fill='x', padx=10, pady=5)
        
        for i, rep in enumerate(all_data):
            model = rep.get('Model', {})
            txt = f"Rep {i+1}: " + (f"Acc = {model.get('accuracy_test', 0):.4f}" if 'error' not in model else "FAILED")
            font = ('Courier New', 10, 'bold') if i == best_idx else ('Courier New', 10)
            color = 'green' if i == best_idx else ('red' if 'error' in model else 'black')
            ttk.Label(self.initial_summary_frame, text=txt, font=font, foreground=color).pack(anchor='w', padx=10)

    def display_initial_details_report(self, model):
        self.clear_frame(self.initial_details_frame)
        wanted = {'number_of_sel_var': 'No. Vars', 'accuracy_train': 'Acc Train', 'accuracy_test': 'Acc Test', 'sensitivity_recall_test': 'Sens Test', 'specificity_test': 'Spec Test'}
        row = 0
        for key, name in wanted.items():
            if key in model:
                val = model[key]
                val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
                ttk.Label(self.initial_details_frame, text=f"{name}:", font=('Arial', 9, 'bold')).grid(row=row, column=0, sticky='e', padx=5)
                ttk.Label(self.initial_details_frame, text=val_str).grid(row=row, column=1, sticky='w', padx=5); row+=1

    def view_initial_details(self, event=None):
        try:
            val = self.initial_repetition_selector.get()
            if not val: return
            idx = int(val) - 1
            self.selected_repetition_index = idx
            self.Model_BAT_global = self.all_repetitions_data[idx]['Model']
            self.evolution_global = self.all_repetitions_data[idx]['Evolution']
            if 'error' in self.Model_BAT_global: messagebox.showwarning("Error", "This repetition failed."); return
            self.display_initial_details_report(self.Model_BAT_global)
            self.draw_classification_dashboard(self.initial_plot_figure, self.initial_plot_canvas, self.Model_BAT_global, self.evolution_global)
            self.export_graphics_button.config(state=tk.NORMAL)
            self.export_excel_button.config(state=tk.NORMAL)
            self.export_matlab_button.config(state=tk.NORMAL)
        except Exception: pass

    def run_multi_repetitions_handler(self):
        if self.X_train_data is None: messagebox.showerror("Error", "Load Data first."); return
        try:
            self.reset_all()
            params = {'num_bats': int(self.entries['mbats'].get()), 'num_iter': int(self.entries['numberofiterations'].get()), 'min_features': int(self.entries['n_min'].get()), 'max_features': int(self.entries['n_max'].get()), 'alpha': float(self.entries['alpha'].get()), 'gamma': float(self.entries['gamma'].get()), 'f_min': float(self.entries['fmin'].get()), 'f_max': float(self.entries['fmax'].get()), 'random_state': 42, 'cv_folds': 5}
            num_reps = int(self.entries['repetitions'].get())
            
            scaler = StandardScaler().fit(self.X_train_data)
            X_train_s, X_test_s = scaler.transform(self.X_train_data), scaler.transform(self.X_test_data)
            X_val_s = scaler.transform(self.X_val_data) if self.X_val_data is not None else None
            
            for i in range(num_reps):
                self.status_label.config(text=f"Running Repetition {i+1}/{num_reps}..."); self.root.update_idletasks()
                bat = BAT_LDA(**params)
                res = bat.fit(X_train_s, self.Y_train_data)
                
                if 'error' in res: 
                    self.all_repetitions_data.append({'Model': {'error': res['error']}, 'Evolution': {}})
                    continue
                
                mask = res['best_subset_mask']
                idxs = np.where(mask)[0]
                
                final_lda = LDA()
                final_lda.fit(X_train_s[:, mask], self.Y_train_data)
                
                model_data = {'var_sel_indices': idxs.tolist(), 'number_of_sel_var': len(idxs), 'classes': final_lda.classes_.tolist()}
                datasets = {'train': (X_train_s, self.Y_train_data), 'test': (X_test_s, self.Y_test_data)}
                if X_val_s is not None: datasets['val'] = (X_val_s, self.Y_val_data)
                
                for name, (X, y) in datasets.items():
                    y_p = final_lda.predict(X[:, mask])
                    y_s = final_lda.predict_proba(X[:, mask])
                    mets = self._get_classification_metrics(y, y_p, y_s, final_lda.classes_)
                    for k, v in mets.items(): model_data[f'{k}_{name}'] = v
                
                self.all_repetitions_data.append({'Model': model_data, 'Evolution': {'iterations': res['iterations'], 'cost_evolution': res['cost_evolution']}})
            
            # Atualiza GUI
            self.display_initial_summary_report(self.all_repetitions_data)
            
            # Preenche Combobox com repetições
            rep_list = [str(i+1) for i in range(num_reps)]
            self.initial_repetition_selector['values'] = rep_list
            
            if self.all_repetitions_data: 
                self.initial_repetition_selector.current(0)
                self.view_details_button.config(state=tk.NORMAL)
                self.view_initial_details()
            
            for i, rep in enumerate(self.all_repetitions_data):
                m = rep.get('Model', {})
                if 'error' not in m: 
                    self.summary_tree.insert("", "end", values=(i+1, f"{m.get('accuracy_train',0):.4f}", "N/A", f"{m.get('accuracy_test',0):.4f}"))
                else: 
                    self.summary_tree.insert("", "end", values=(i+1, "FAIL", "FAIL", "FAIL"))
            
            self.notebook.select(0)
            self.status_label.config(text="Execution Finished.")
            messagebox.showinfo("Success", "All repetitions finished.")
            
        except Exception as e: 
            self.status_label.config(text="Error.")
            messagebox.showerror("Error", str(e))

    def load_data(self):
        file_path = filedialog.askopenfilename(title="Select .mat file", filetypes=[("MAT Files", "*.mat")])
        if not file_path: return
        try:
            mat = sio.loadmat(file_path)
            keys = mat.keys()
            # Smart Key Search
            def find(patterns):
                for k in keys:
                    for p in patterns:
                        if p.lower() == k.lower(): return k
                return None
            
            ktX, ktY = find(['Train', 'X_train']), find(['Group_Train', 'Y_train'])
            kteX, kteY = find(['Test', 'X_test']), find(['Group_Test', 'Y_test'])
            
            if ktX and ktY and kteX and kteY:
                self.X_train_data = mat[ktX].astype(float)
                self.Y_train_data = mat[ktY].astype(float).flatten()
                self.X_test_data = mat[kteX].astype(float)
                self.Y_test_data = mat[kteY].astype(float).flatten()
                
                kvX, kvY = find(['Val', 'Validation']), find(['Group_Val', 'Y_val'])
                if kvX and kvY: 
                    self.X_val_data, self.Y_val_data = mat[kvX].astype(float), mat[kvY].astype(float).flatten()
                else: 
                    self.X_val_data, self.Y_val_data = None, None
                
                self.variables_data = mat.get('Variables', None)
                self.status_label.config(text="Data Loaded Successfully.")
                messagebox.showinfo("Success", "Data Loaded.")
            else: 
                raise KeyError("Missing Train/Test matrices.")
        except Exception as e: 
            messagebox.showerror("Load Error", str(e))

    def reset_all(self):
        self.all_repetitions_data = []
        self.Model_BAT_global = None
        self.evolution_global = None
        if self.summary_tree: [self.summary_tree.delete(i) for i in self.summary_tree.get_children()]
        if self.cycle_tree: [self.cycle_tree.delete(i) for i in self.cycle_tree.get_children()]
        self.clear_frame(self.initial_summary_frame)
        self.clear_frame(self.initial_details_frame)
        if self.initial_repetition_selector:
            self.initial_repetition_selector['values'] = []
            self.initial_repetition_selector.set('')
        if self.view_details_button: self.view_details_button.config(state=tk.DISABLED)
        # Disable export buttons
        if self.export_graphics_button: self.export_graphics_button.config(state=tk.DISABLED)
        if self.export_excel_button: self.export_excel_button.config(state=tk.DISABLED)
        if self.export_matlab_button: self.export_matlab_button.config(state=tk.DISABLED)
        
        self.initial_plot_figure.clear()
        self.initial_plot_canvas.draw()
        self.interactive_plot_figure.clear()
        self.interactive_plot_canvas.draw()
        self.status_label.config(text="Ready.")

    def on_interactive_repetition_select(self, event):
        sel = self.summary_tree.selection()
        if not sel: return
        idx = int(self.summary_tree.item(sel[0])['values'][0]) - 1
        model = self.all_repetitions_data[idx]['Model']
        evo = self.all_repetitions_data[idx]['Evolution']
        if 'error' in model: return
        
        [self.cycle_tree.delete(i) for i in self.cycle_tree.get_children()]
        costs = evo.get('cost_evolution', [])
        for i, c in enumerate(costs): 
            self.cycle_tree.insert("", "end", values=(i+1, f"{c:.4f}"))
        
        self.draw_classification_dashboard(self.interactive_plot_figure, self.interactive_plot_canvas, model, evo)

    def on_cycle_select(self, event):
        pass 

    def _setup_gui_components(self):
        try: self.root.iconbitmap('bat_icon.ico')
        except: pass
        style = ttk.Style(self.root); style.theme_use("clam")
        
        # Cria Status Label PRIMEIRO
        self.status_label = tk.Label(self.root, text="Status: Waiting...", bd=1, relief=tk.SUNKEN, anchor=tk.W, padx=10)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
        header = tk.Frame(self.root, bg="#2c3e50", height=40); header.pack(fill=tk.X)
        tk.Label(header, text="Classification Analysis with Bat Algorithm (BAT-LDA) \U0001F987", fg="white", bg="#2c3e50", font=("Arial", 14, "bold")).pack(pady=10)
        
        main = ttk.Frame(self.root); main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left = ttk.LabelFrame(main, text="Controls"); left.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))
        tk.Button(left, text="Load Data (.mat)", command=self.load_data, bg="#4682B4", fg="white").pack(pady=5, fill=tk.X, padx=5)
        
        params = [("No. Bats:", "30", "mbats"), ("No. Iterations:", "100", "numberofiterations"), ("Min Vars:", "2", "n_min"), ("Max Vars:", "20", "n_max"),
                  ("Alpha:", "0.5", "alpha"), ("Gamma:", "0.5", "gamma"), ("Min Freq:", "0.0", "fmin"), ("Max Freq:", "0.05", "fmax")]
        pf = ttk.LabelFrame(left, text="BAT Parameters"); pf.pack(pady=10, padx=5)
        for i, (txt, val, key) in enumerate(params):
            ttk.Label(pf, text=txt).grid(row=i, column=0, sticky='w'); e = ttk.Entry(pf, width=10); e.insert(0, val); e.grid(row=i, column=1); self.entries[key] = e
        
        ttk.Label(left, text="Repetitions:").pack(pady=(10,0)); e_rep = ttk.Entry(left); e_rep.insert(0, "5"); e_rep.pack(pady=2, padx=5); self.entries['repetitions'] = e_rep
        
        tk.Button(left, text="Run Analysis", command=self.run_multi_repetitions_handler, bg="#8A2BE2", fg="white", font=("Arial", 11, "bold")).pack(pady=10, fill=tk.X, padx=5)
        tk.Button(left, text="Reset All", command=self.reset_all, bg="#B0C4DE").pack(pady=5, fill=tk.X, padx=5)
        
        right = ttk.Frame(main); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.notebook = ttk.Notebook(right); self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1
        t1 = ttk.Frame(self.notebook); self.notebook.add(t1, text="Initial Analysis")
        pan1 = ttk.PanedWindow(t1, orient=tk.HORIZONTAL); pan1.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Painel Esquerdo da Tab 1
        l_pan1 = ttk.Frame(pan1, width=400); pan1.add(l_pan1, weight=1)
        
        # Bloco de Cima (Resumo + Controles)
        l_top = ttk.LabelFrame(l_pan1, text="Repetitions Summary"); l_top.pack(fill=tk.BOTH, expand=True)
        ctrl_frame = ttk.Frame(l_top); ctrl_frame.pack(fill=tk.X, pady=5)
        ttk.Label(ctrl_frame, text="Repetition:").pack(side=tk.LEFT)
        self.initial_repetition_selector = ttk.Combobox(ctrl_frame, width=5, state="readonly")
        self.initial_repetition_selector.pack(side=tk.LEFT, padx=5)
        self.initial_repetition_selector.bind("<<ComboboxSelected>>", self.view_initial_details)
        self.view_details_button = tk.Button(ctrl_frame, text="View Details", command=self.view_initial_details, state=tk.DISABLED); self.view_details_button.pack(side=tk.LEFT)
        
        # Botões de Exportação
        btn_frame = ttk.Frame(l_top); btn_frame.pack(fill=tk.X, pady=2)
        self.export_graphics_button = tk.Button(btn_frame, text="Export Graphs...", command=self.open_export_plot_options, state=tk.DISABLED); self.export_graphics_button.pack(side=tk.LEFT, padx=2)
        self.export_excel_button = tk.Button(btn_frame, text="Export Excel", command=self.export_metrics_to_excel, state=tk.DISABLED); self.export_excel_button.pack(side=tk.LEFT, padx=2)
        self.export_matlab_button = tk.Button(btn_frame, text="Export MAT", command=self.export_model_to_matlab, state=tk.DISABLED); self.export_matlab_button.pack(side=tk.LEFT, padx=2)

        # Canvas para a Lista de Resumo
        canvas = tk.Canvas(l_top)
        scrollbar = ttk.Scrollbar(l_top, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.initial_summary_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=self.initial_summary_frame, anchor="nw")
        self.initial_summary_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Bloco de Baixo (Detalhes)
        l_bot = ttk.LabelFrame(l_pan1, text="Metrics"); l_bot.pack(fill=tk.BOTH, expand=True)
        self.initial_details_frame = ttk.Frame(l_bot); self.initial_details_frame.pack(fill=tk.BOTH, expand=True, padx=5)
        
        # Painel Direito (Dashboard)
        r_pan1 = ttk.LabelFrame(pan1, text="Dashboard"); pan1.add(r_pan1, weight=3)
        self.initial_plot_figure = plt.figure(); self.initial_plot_canvas = FigureCanvasTkAgg(self.initial_plot_figure, master=r_pan1); self.initial_plot_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Tab 2
        t2 = ttk.Frame(self.notebook); self.notebook.add(t2, text="Interactive Analysis")
        pan2 = ttk.PanedWindow(t2, orient=tk.VERTICAL); pan2.pack(fill=tk.BOTH, expand=True)
        sum_frame = ttk.LabelFrame(pan2, text="Summary Table"); pan2.add(sum_frame, weight=1)
        self.summary_tree = ttk.Treeview(sum_frame, columns=("Rep", "Train", "Val", "Test"), show="headings"); 
        for c in ["Rep", "Train", "Val", "Test"]: self.summary_tree.heading(c, text=c); self.summary_tree.column(c, width=80)
        self.summary_tree.pack(fill=tk.BOTH, expand=True); self.summary_tree.bind("<<TreeviewSelect>>", self.on_interactive_repetition_select)
        
        cyc_frame = ttk.LabelFrame(pan2, text="Cycles"); pan2.add(cyc_frame, weight=1)
        self.cycle_tree = ttk.Treeview(cyc_frame, columns=("Cycle", "Acc"), show="headings"); 
        self.cycle_tree.heading("Cycle", text="Cycle"); self.cycle_tree.heading("Acc", text="Accuracy"); self.cycle_tree.pack(fill=tk.BOTH, expand=True)
        self.cycle_tree.bind("<<TreeviewSelect>>", self.on_cycle_select)
        
        plt_frame = ttk.LabelFrame(pan2, text="Viz"); pan2.add(plt_frame, weight=3)
        self.interactive_plot_figure = plt.figure(); self.interactive_plot_canvas = FigureCanvasTkAgg(self.interactive_plot_figure, master=plt_frame); self.interactive_plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.interactive_plot_canvas, plt_frame).update()

# =============================================================================
#  BLOCO PRINCIPAL
# =============================================================================
try: ctypes.windll.shcore.SetProcessDpiAwareness(1)
except: pass

if __name__ == "__main__":
    app = BAT_LDA_App()
    app.root.mainloop()