import tkinter as tk
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient  # <--- NEW: For sending data
import threading
from collections import deque

# --- 1. NETWORK CONFIGURATION ---
# Incoming from OpenBCI (on this PC)
IN_IP = "127.0.0.1"
IN_PORT = 12345

# Outgoing to the other computer
OUT_IP = "192.168.1.172"
OUT_PORT = 5005 # You can change this to match your receiver
osc_out = SimpleUDPClient(OUT_IP, OUT_PORT)

# --- 2. GLOBAL VARIABLES & BUFFERS ---
latest_thetas = [0.0] * 4
latest_alphas = [0.0] * 4
latest_betas = [0.0] * 4

BUFFER_SIZE = 500
relaxed_buffer = deque(maxlen=BUFFER_SIZE)
attention_buffer = deque(maxlen=BUFFER_SIZE)
nasa_buffer = deque(maxlen=BUFFER_SIZE)

smoothed_relaxed = 0.0
smoothed_attention = 0.0
smoothed_nasa = 0.0

# --- 3. OSC LISTENER & PUBLISHER LOGIC ---
def bandpower_handler(address, *args):
    global smoothed_relaxed, smoothed_attention, smoothed_nasa
    
    parts = address.split('/')
    if len(parts) >= 3:
        try:
            ch_num = int(parts[-1])
            if 0 <= ch_num <= 3:
                latest_thetas[ch_num] = args[1]
                latest_alphas[ch_num] = args[2]
                latest_betas[ch_num]  = args[3]
                
                if ch_num == 3:
                    avg_theta = sum(latest_thetas) / 4.0
                    avg_alpha = sum(latest_alphas) / 4.0
                    avg_beta = sum(latest_betas) / 4.0
                    
                    # Calculate RAW and add to buffers
                    if avg_beta > 0:
                        relaxed_buffer.append(avg_alpha / avg_beta)
                    if avg_theta > 0:
                        attention_buffer.append(avg_beta / avg_theta)
                    
                    denom = avg_alpha + avg_theta
                    if denom > 0:
                        nasa_buffer.append(avg_beta / denom)
                        
                    # Calculate SMOOTHED averages
                    if len(relaxed_buffer) > 0:
                        smoothed_relaxed = sum(relaxed_buffer) / len(relaxed_buffer)
                    if len(attention_buffer) > 0:
                        smoothed_attention = sum(attention_buffer) / len(attention_buffer)
                    if len(nasa_buffer) > 0:
                        smoothed_nasa = sum(nasa_buffer) / len(nasa_buffer)

                    # This sends the 3 smoothed values to 192.168.1.172
                    osc_out.send_message("/focus/relaxed", smoothed_relaxed)
                    osc_out.send_message("/focus/attention", smoothed_attention)
                    osc_out.send_message("/focus/engagement", smoothed_nasa)
                        
        except ValueError:
            pass

def start_osc_server():
    dispatcher = Dispatcher()
    dispatcher.map("/openbci/band-power/*", bandpower_handler)
    server = BlockingOSCUDPServer((IN_IP, IN_PORT), dispatcher)
    print(f"Listening on {IN_IP}:{IN_PORT}")
    print(f"Publishing to {OUT_IP}:{OUT_PORT}")
    server.serve_forever()

# --- 4. GUI LOGIC ---
def update_gui():
    global smoothed_relaxed, smoothed_attention, smoothed_nasa
    
    lbl_relaxed_val.config(text=f"{smoothed_relaxed:.2f}")
    lbl_attention_val.config(text=f"{smoothed_attention:.2f}")
    lbl_nasa_val.config(text=f"{smoothed_nasa:.2f}")
    
    # Animate bars
    w_rel = min(400, int(smoothed_relaxed * 150))
    canvas.coords(bar_relaxed, 50, 40, 50 + w_rel, 70)
    
    w_att = min(400, int(smoothed_attention * 150))
    canvas.coords(bar_attention, 50, 120, 50 + w_att, 150)
    
    w_nasa = min(400, int(smoothed_nasa * 300))
    canvas.coords(bar_nasa, 50, 200, 50 + w_nasa, 230)
    
    root.after(50, update_gui)

# --- 5. BUILD WINDOW ---
osc_thread = threading.Thread(target=start_osc_server, daemon=True)
osc_thread.start()

root = tk.Tk()
root.title("Neuro-Broadcaster")
root.geometry("500x480")
root.configure(bg="#1e272e")

tk.Label(root, text="Brain Data Publisher", font=("Helvetica", 18, "bold"), bg="#1e272e", fg="white").pack(pady=10)
tk.Label(root, text=f"Target: {OUT_IP}", font=("Courier", 10), bg="#1e272e", fg="#0be881").pack()

frame_text = tk.Frame(root, bg="#1e272e")
frame_text.pack(fill="x", padx=50, pady=10)

tk.Label(frame_text, text="Relaxed Focus:", bg="#1e272e", fg="#34e7e4").grid(row=0, column=0, sticky="w")
lbl_relaxed_val = tk.Label(frame_text, text="0.0", bg="#1e272e", fg="white")
lbl_relaxed_val.grid(row=0, column=1, sticky="e", padx=20)

tk.Label(frame_text, text="Active Attention:", bg="#1e272e", fg="#ffdd59").grid(row=1, column=0, sticky="w", pady=60)
lbl_attention_val = tk.Label(frame_text, text="0.0", bg="#1e272e", fg="white")
lbl_attention_val.grid(row=1, column=1, sticky="e", padx=20)

tk.Label(frame_text, text="NASA Engagement:", bg="#1e272e", fg="#0be881").grid(row=2, column=0, sticky="w")
lbl_nasa_val = tk.Label(frame_text, text="0.0", bg="#1e272e", fg="white")
lbl_nasa_val.grid(row=2, column=1, sticky="e", padx=20)

canvas = tk.Canvas(root, width=500, height=280, bg="#1e272e", highlightthickness=0)
canvas.place(y=160)
canvas.create_rectangle(50, 40, 450, 70, fill="#485460", outline="")
canvas.create_rectangle(50, 120, 450, 150, fill="#485460", outline="")
canvas.create_rectangle(50, 200, 450, 230, fill="#485460", outline="")

bar_relaxed = canvas.create_rectangle(50, 40, 50, 70, fill="#34e7e4", outline="")
bar_attention = canvas.create_rectangle(50, 120, 50, 150, fill="#ffdd59", outline="")
bar_nasa = canvas.create_rectangle(50, 200, 50, 230, fill="#0be881", outline="")

root.after(50, update_gui)
root.mainloop()