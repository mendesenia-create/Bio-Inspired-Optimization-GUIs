import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys

def resource_path(relative_path):
    """ Retorna o caminho absoluto para arquivos, funcionando tanto em script quanto em exe """
    try:
        # PyInstaller cria uma pasta temporária em _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# --- IMPORTAÇÃO DOS MÓDULOS DAS INTERFACES ---
try:
    # Algoritmos de Seleção (Otimização)
    from ACO_LDA_Module import ACO_LDA_App
    from BAT_LDA_Module import BAT_LDA_App
    from ACO_PLS_Module import ACO_PLS_App
    from BAT_PLS_Module import BAT_PLS_App
    from FFI_PLS_Module import FFI_PLS_App
    
    # Métodos Padrão (Full Spectrum)
    from Standard_LDA_Module import Standard_LDA_App
    from Standard_PLS_Module import Standard_PLS_App
    
except ImportError as e:
    print(f"Erro crítico de importação: {e}")
    print("Verifique se todos os 7 arquivos de módulo estão na mesma pasta.")

class MainMenuApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("LQAQ - Chemometric Tools Suite") # Título em Inglês
        self.root.geometry("900x650")
        self.root.configure(bg="#f0f0f0")
        
        # Tentar colocar ícone
        try: self.root.iconbitmap(resource_path('firefly.ico'))
        except: pass

        self._setup_header()
        self._setup_buttons()
        self._setup_footer()

    def _setup_header(self):
        # Cabeçalho Estilizado
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=90)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        
        # Título Principal
        title_label = tk.Label(header_frame, text="Laboratório de Química Analítica e Quimiometria", 
                               fg="white", bg="#2c3e50", font=("Helvetica", 20, "bold"))
        title_label.pack(pady=(20, 5))
        
        subtitle_label = tk.Label(header_frame, text="Universidade Estadual da Paraiba - UEPB", 
                                  fg="#ecf0f1", bg="#2c3e50", font=("Helvetica", 12))
        subtitle_label.pack(pady=(0, 15))

    def _setup_buttons(self):
        # Container principal com scroll se necessário (aqui simplificado)
        main_container = tk.Frame(self.root, bg="#f0f0f0")
        main_container.pack(expand=True, fill=tk.BOTH, padx=40, pady=20)

        # ========================================================
        # SEÇÃO 1: CLASSIFICAÇÃO (Classification)
        # ========================================================
        class_frame = tk.LabelFrame(main_container, text=" Classification Algorithms (LDA) ", 
                                    bg="#f0f0f0", font=("Arial", 12, "bold"), fg="#2980b9")
        class_frame.pack(fill=tk.X, pady=10, ipady=10)

        btn_frame_class = tk.Frame(class_frame, bg="#f0f0f0")
        btn_frame_class.pack(expand=True, pady=5)

        # Botão Standard (Baseline) - Cor Cinza/Azul Escuro para destacar que é o básico
        self.create_button(btn_frame_class, "LDA", "Full Spectrum Baseline", self.open_std_lda, "#34495e")
        
        # Separador visual (Espaço)
        tk.Frame(btn_frame_class, width=20, bg="#f0f0f0").pack(side=tk.LEFT)

        # Botões de Otimização - Cores Vibrantes
        self.create_button(btn_frame_class, "ACO-LDA", "Ant Colony Optimization", self.open_aco_lda, "#3498db")
        self.create_button(btn_frame_class, "BAT-LDA", "Bat Algorithm", self.open_bat_lda, "#9b59b6")

        # ========================================================
        # SEÇÃO 2: REGRESSÃO (Regression)
        # ========================================================
        reg_frame = tk.LabelFrame(main_container, text=" Regression Algorithms (PLS) ", 
                                  bg="#f0f0f0", font=("Arial", 12, "bold"), fg="#27ae60")
        reg_frame.pack(fill=tk.X, pady=20, ipady=10)

        btn_frame_reg = tk.Frame(reg_frame, bg="#f0f0f0")
        btn_frame_reg.pack(expand=True, pady=5)

        # Botão Standard (Baseline)
        self.create_button(btn_frame_reg, "PLS", "Full Spectrum Baseline", self.open_std_pls, "#16a085")

        # Separador visual
        tk.Frame(btn_frame_reg, width=20, bg="#f0f0f0").pack(side=tk.LEFT)

        # Botões de Otimização
        self.create_button(btn_frame_reg, "ACO-PLS", "Ant Colony Optimization", self.open_aco_pls, "#2ecc71")
        self.create_button(btn_frame_reg, "BAT-PLS", "Bat Algorithm", self.open_bat_pls, "#1abc9c")
        self.create_button(btn_frame_reg, "FA-iPLS", "Firefly Algorithm", self.open_ffi_pls, "#e67e22")

    def create_button(self, parent, text, subtext, command, color):
        # Cria um frame para o botão e o subtítulo ficarem juntos
        frame = tk.Frame(parent, bg="#f0f0f0")
        frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        btn = tk.Button(frame, text=text, command=command, bg=color, fg="white", 
                        font=("Arial", 11, "bold"), width=16, height=2, cursor="hand2", relief="flat")
        btn.pack()
        
        # Efeito de hover simples
        btn.bind("<Enter>", lambda e: btn.config(bg=self.lighten_color(color)))
        btn.bind("<Leave>", lambda e: btn.config(bg=color))
        
        lbl = tk.Label(frame, text=subtext, bg="#f0f0f0", fg="#7f8c8d", font=("Arial", 8))
        lbl.pack(pady=(2, 0))

    def lighten_color(self, color):
        # Função auxiliar simples para clarear a cor no hover (opcional)
        # Se der erro de cor, retorne a cor original
        return color 

    def _setup_footer(self):
        footer_frame = tk.Frame(self.root, bg="#ecf0f1", height=40)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        credits = tk.Label(footer_frame, text="Developed by: Me. Enia Mendes & Dr. Germano Veras (LQAQ-UFPB)", 
                           bg="#ecf0f1", fg="#95a5a6", font=("Arial", 9, "italic"))
        credits.pack(pady=10)

    # --- FUNÇÕES PARA ABRIR AS JANELAS ---
    
    # -- Standard Methods --
    def open_std_lda(self):
        try: Standard_LDA_App(self.root)
        except Exception as e: messagebox.showerror("Error", f"Failed to open Standard LDA: {e}")

    def open_std_pls(self):
        try: Standard_PLS_App(self.root)
        except Exception as e: messagebox.showerror("Error", f"Failed to open Standard PLS: {e}")

    # -- Classification --
    def open_aco_lda(self):
        try: ACO_LDA_App(self.root)
        except Exception as e: messagebox.showerror("Error", f"Failed to open ACO-LDA: {e}")

    def open_bat_lda(self):
        try: BAT_LDA_App(self.root)
        except Exception as e: messagebox.showerror("Error", f"Failed to open BAT-LDA: {e}")

    # -- Regression --
    def open_aco_pls(self):
        try: ACO_PLS_App(self.root)
        except Exception as e: messagebox.showerror("Error", f"Failed to open ACO-PLS: {e}")

    def open_bat_pls(self):
        try: BAT_PLS_App(self.root)
        except Exception as e: messagebox.showerror("Error", f"Failed to open BAT-PLS: {e}")

    def open_ffi_pls(self):
        try: FFI_PLS_App(self.root)
        except Exception as e: messagebox.showerror("Error", f"Failed to open FFI-PLS: {e}")

if __name__ == "__main__":
    app = MainMenuApp()
    app.root.mainloop()