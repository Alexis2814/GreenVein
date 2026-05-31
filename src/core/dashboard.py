import os
import sys
import threading
import time
import subprocess
import tkinter as tk 
from PIL import Image

# 🔥 ÉP MATPLOTLIB VẼ TRONG RAM, CẤM ĐỤNG VÀO GIAO DIỆN
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
plt.show = lambda: None 

# 🌟 VÁ LỖI ĐƯỜNG DẪN
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) 
SRC_DIR = os.path.dirname(CURRENT_DIR)                   
PROJECT_DIR = os.path.dirname(SRC_DIR)                   

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

import customtkinter as ctk
from test_agent import test_and_compare

# ============ ĐỒNG BỘ DPI CẤP OS ============
try:
    import win32gui
    import win32con
    HAS_WIN32 = True
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(2) 
    except:
        try: windll.user32.SetProcessDPIAware()
        except: pass
except ImportError:
    HAS_WIN32 = False

ctk.set_appearance_mode("Dark")  
ctk.set_default_color_theme("blue") 

# 🔥 BẮT LUỒNG PRINT AN TOÀN CHỐNG SẬP APP
class SafeRedirectText(object):
    def __init__(self, text_widget):
        self.text_widget = text_widget
        
    def write(self, string):
        self.text_widget.after(0, self._insert_text, string)
        
    def _insert_text(self, string):
        self.text_widget.insert(ctk.END, string)
        self.text_widget.see(ctk.END)
        
    def flush(self): pass

class GreenVeinDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("GREENVEIN - TRẠM ĐIỀU KHIỂN AI & MÔ PHỎNG SUMO 3D")
        self.geometry("1500x900") 

        # ============ BỐ CỤC GIAO DIỆN ============
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar_frame.pack_propagate(False) 

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="♻️ GREENVEIN AI", font=ctk.CTkFont(size=22, weight="bold"))
        self.logo_label.pack(padx=20, pady=(30, 20))

        self.btn_run_ai = ctk.CTkButton(self.sidebar_frame, text="🚀 CHẠY ĐỒ ÁN", fg_color="#28a745", hover_color="#218838", font=ctk.CTkFont(weight="bold"), command=self.start_simulation)
        self.btn_run_ai.pack(padx=20, pady=15, ipady=10, fill=tk.X)

        self.btn_stop = ctk.CTkButton(self.sidebar_frame, text="🛑 Dừng Khẩn Cấp", fg_color="#dc3545", hover_color="#c82333", command=self.stop_simulation)
        self.btn_stop.pack(padx=20, pady=(0, 20), fill=tk.X)

        self.divider = ctk.CTkFrame(self.sidebar_frame, height=2, fg_color="#555555")
        self.divider.pack(padx=20, pady=10, fill=tk.X)

        self.lbl_chart = ctk.CTkLabel(self.sidebar_frame, text="📊 KHO BIỂU ĐỒ KẾT QUẢ", font=ctk.CTkFont(weight="bold"))
        self.lbl_chart.pack(padx=20, pady=(10, 0), anchor="w")

        self.chart_menu = ctk.CTkOptionMenu(self.sidebar_frame, dynamic_resizing=False, command=self.display_chart)
        self.chart_menu.pack(padx=20, pady=10, fill=tk.X)
        
        self.btn_refresh_chart = ctk.CTkButton(self.sidebar_frame, text="🔄 Làm mới danh sách", fg_color="transparent", border_width=1, text_color=("gray10", "#DCE4EE"), command=self.refresh_chart_list)
        self.btn_refresh_chart.pack(padx=20, pady=0, fill=tk.X)

        self.info_label = ctk.CTkLabel(self.sidebar_frame, text="ℹ️ Chú ý:\nKéo thanh gờ ở giữa\nđể thay đổi kích thước.", justify="left", text_color="gray")
        self.info_label.pack(padx=20, pady=40, side=tk.BOTTOM, anchor="sw")

        self.paned_window = tk.PanedWindow(self, orient=tk.VERTICAL, bg="#1e1e1e", sashwidth=10, sashrelief=tk.RAISED, bd=0)
        self.paned_window.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.top_container = ctk.CTkFrame(self.paned_window, corner_radius=10)
        self.paned_window.add(self.top_container, stretch="always", height=550) 
        
        self.tabview = ctk.CTkTabview(self.top_container, corner_radius=10)
        self.tabview.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tab_sumo = self.tabview.add("🖥️ Mô Phỏng SUMO")
        self.tab_chart = self.tabview.add("📊 Biểu Đồ Phân Tích")

        self.sumo_frame = tk.Frame(self.tab_sumo, bg="black")
        self.sumo_frame.pack(fill=tk.BOTH, expand=True) 
        
        self.sumo_text_waiting = tk.Label(self.sumo_frame, text="Hệ thống đã sẵn sàng...", bg="black", fg="gray", font=("Arial", 14))
        self.sumo_text_waiting.place(relx=0.5, rely=0.5, anchor="center")
        
        self.sumo_hwnd = None
        self.sumo_frame.bind("<Configure>", self.on_frame_resize)

        self.chart_label = ctk.CTkLabel(self.tab_chart, text="Chưa có biểu đồ nào được chọn.", text_color="gray", font=ctk.CTkFont(size=18))
        self.chart_label.pack(fill=tk.BOTH, expand=True)

        self.bottom_container = ctk.CTkFrame(self.paned_window, corner_radius=10)
        self.paned_window.add(self.bottom_container, stretch="never", height=250)

        self.log_label = ctk.CTkLabel(self.bottom_container, text="🖥️ LIVE TERMINAL TRACKER:", font=ctk.CTkFont(weight="bold", size=14))
        self.log_label.pack(anchor="w", padx=10, pady=5)

        self.console_text = ctk.CTkTextbox(self.bottom_container, bg_color="#1e1e1e", text_color="#00ff00", font=("Consolas", 13))
        self.console_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        sys.stdout = SafeRedirectText(self.console_text)
        self.is_running = False
        self.refresh_chart_list()

    def refresh_chart_list(self):
        report_dir = os.path.join(PROJECT_DIR, "reports")
        if not os.path.exists(report_dir): os.makedirs(report_dir)
        files = [f for f in os.listdir(report_dir) if f.endswith(".png") and "ai_vs_baseline" in f]
        files.sort(reverse=True) 
        if files:
            self.chart_menu.configure(values=files)
            self.chart_menu.set(files[0])
            self.display_chart(files[0])

    def display_chart(self, filename):
        if filename == "Trống": return
        filepath = os.path.join(PROJECT_DIR, "reports", filename)
        try:
            img = Image.open(filepath)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(960, 540))
            self.chart_label.configure(image=ctk_img, text="")
        except Exception as e: pass

    # ==========================================================
    # 🔥 BẢN VÁ V98: HACK LÕI TRACI (GIẢI QUYẾT LỖI CONNECTION ACTIVE)
    # ==========================================================
    def start_simulation(self):
        if not self.is_running:
            self.console_text.delete("1.0", ctk.END)
            self.tabview.set("🖥️ Mô Phỏng SUMO") 
            self.is_running = True
            
            # Khởi chạy luồng dọn dẹp riêng biệt để Main GUI không bị đứng hình
            threading.Thread(target=self._async_cleanup, daemon=True).start()

    def _async_cleanup(self):
        print("🔄 Đang dọn dẹp tàn dư của kịch bản cũ...")
        
        # 1. TIÊU DIỆT SUMO TRƯỚC TIÊN (Giết chết tiến trình để cắt đứt socket)
        if HAS_WIN32:
            subprocess.run(["taskkill", "/f", "/im", "sumo-gui.exe"], capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
            subprocess.run(["taskkill", "/f", "/im", "sumo.exe"], capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        
        time.sleep(0.5) 
        
        # 2. XÓA BỘ NHỚ TRACI (Sát thủ diệt lỗi 'Connection default is already active')
        try:
            import traci
            # Cố gắng đóng nhẹ nhàng (lúc này socket đã đứt nên có thể văng lỗi)
            traci.close()
        except Exception: 
            pass
            
        try:
            import traci
            # Can thiệp sâu vào lõi của traci để xóa sạch bóng ma 'default'
            if "default" in traci.getConnectionLabels():
                del traci._connections["default"]
        except Exception:
            pass
            
        time.sleep(0.5) 
        
        # Báo cho giao diện chính tiến hành chạy tiếp
        self.after(0, self._start_sim_processes)

    def _start_sim_processes(self):
        # 1. Xóa bỏ khung cũ chứa "xác" của SUMO
        if hasattr(self, 'sumo_frame') and self.sumo_frame.winfo_exists():
            self.sumo_frame.destroy()
            
        # 2. Xây khung mới tinh
        self.sumo_frame = tk.Frame(self.tab_sumo, bg="black")
        self.sumo_frame.pack(fill=tk.BOTH, expand=True)
        self.sumo_hwnd = None
        self.sumo_frame.bind("<Configure>", self.on_frame_resize)
        
        self.sumo_text_waiting = tk.Label(self.sumo_frame, text="Đang chuẩn bị dữ liệu AI (Chờ đồ họa SUMO 1.5s)...\n", bg="black", fg="gray", font=("Arial", 14))
        self.sumo_text_waiting.place(relx=0.5, rely=0.5, anchor="center")
        
        print("🚀 Khởi chạy hệ thống AI GreenVein...")
        
        self.update_idletasks()
        self.update()
        tk_hwnd = int(self.sumo_frame.winfo_id()) 
        
        # 3. Phân phát 2 luồng công việc chính
        threading.Thread(target=self.run_test_in_background, daemon=True).start()
        if HAS_WIN32:
            threading.Thread(target=self.capture_sumo_window, args=(tk_hwnd,), daemon=True).start()

    def run_test_in_background(self):
        try: test_and_compare(ep_to_load=500)
        except Exception as e: print(f"❌ LỖI HỆ THỐNG AI: {str(e)}")
        finally:
            self.is_running = False
            self.after(500, self.update_ui_after_test)

    def update_ui_after_test(self):
        self.refresh_chart_list()
        try: self.tabview.set("📊 Biểu Đồ Phân Tích")
        except: pass

    # ==========================================================
    # 🔥 BẮT CÓC SUMO HOÀN TOÀN TÁCH BIỆT KHỎI TKINTER
    # ==========================================================
    def capture_sumo_window(self, safe_tk_hwnd):
        print("🔍 Đang đồng bộ hóa đồ họa 3D...")
        for _ in range(15000): 
            time.sleep(0.02)
            if not self.is_running: break 
            
            hwnds = []
            def callback(hwnd, hwnds_list):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if "SUMO" in title.upper() and ".SUMOCFG" in title.upper() and "TRẠM ĐIỀU KHIỂN" not in title.upper():
                        hwnds_list.append(hwnd)
            win32gui.EnumWindows(callback, hwnds)
            
            if hwnds:
                self.sumo_hwnd = hwnds[0]
                time.sleep(1.5) 
                
                try:
                    if safe_tk_hwnd == 0 or not win32gui.IsWindow(safe_tk_hwnd): break

                    style = win32gui.GetWindowLong(self.sumo_hwnd, win32con.GWL_STYLE)
                    style = style & ~(win32con.WS_POPUP | win32con.WS_CAPTION | win32con.WS_THICKFRAME | win32con.WS_MINIMIZEBOX | win32con.WS_MAXIMIZEBOX)
                    style = style | win32con.WS_CHILD   
                    win32gui.SetWindowLong(self.sumo_hwnd, win32con.GWL_STYLE, style)
                    
                    win32gui.SetParent(self.sumo_hwnd, safe_tk_hwnd)
                    
                    self.after(0, self._destroy_waiting_label)
                        
                    def aggressive_snap_thread():
                        end_time = time.time() + 1.5 
                        while time.time() < end_time and self.sumo_hwnd:
                            if not win32gui.IsWindow(safe_tk_hwnd): break
                            rect = win32gui.GetClientRect(safe_tk_hwnd)
                            real_width = rect[2] - rect[0]
                            real_height = rect[3] - rect[1]
                            if real_width > 10 and real_height > 10:
                                win32gui.MoveWindow(self.sumo_hwnd, 0, 0, real_width, real_height, True)
                            
                            win32gui.ShowWindow(self.sumo_hwnd, win32con.SW_SHOWMAXIMIZED)
                            time.sleep(0.05) 
                    
                    threading.Thread(target=aggressive_snap_thread, daemon=True).start()
                    print(f"🎉 Đã nhúng SUMO thành công vào giao diện!")
                except Exception as e:
                    print(f"⚠️ LỖI NHÚNG: {e}")
                break

    def _destroy_waiting_label(self):
        if hasattr(self, 'sumo_text_waiting') and self.sumo_text_waiting.winfo_exists():
            self.sumo_text_waiting.destroy()

    def on_frame_resize(self, event):
        if self.sumo_hwnd and event.width > 50 and event.height > 50:
            self.after(5, self.force_snap_physical)

    def force_snap_physical(self):
        if self.sumo_hwnd and hasattr(self, 'sumo_frame') and self.sumo_frame.winfo_exists():
            try:
                tk_hwnd = int(self.sumo_frame.winfo_id())
                rect = win32gui.GetClientRect(tk_hwnd)
                w = rect[2] - rect[0]
                h = rect[3] - rect[1]
                if w > 10 and h > 10:
                    win32gui.MoveWindow(self.sumo_hwnd, 0, 0, w, h, True)
            except: pass

    def stop_simulation(self):
        # Dọn dẹp mạnh tay y hệt như lúc Start
        if HAS_WIN32:
            subprocess.run(["taskkill", "/f", "/im", "sumo-gui.exe"], capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
            subprocess.run(["taskkill", "/f", "/im", "sumo.exe"], capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            import traci
            traci.close()
        except: pass
        try:
            import traci
            if "default" in traci.getConnectionLabels():
                del traci._connections["default"]
        except: pass
            
        self.is_running = False
        self.quit()
        sys.exit()

if __name__ == "__main__":
    app = GreenVeinDashboard()
    app.mainloop()