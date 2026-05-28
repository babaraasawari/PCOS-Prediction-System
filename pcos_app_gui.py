import customtkinter as ctk
from tkinter import filedialog
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pytesseract
import cv2

# =========================
# TESSERACT PATH
# =========================

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# =========================
# LOAD MODEL
# =========================

model = joblib.load("artifacts/pcos_hybrid_model.joblib")
feature_cols = joblib.load("artifacts/feature_cols.joblib")

# =========================
# GUI SETUP
# =========================

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.geometry("1200x900")
app.title("PCOS Prediction System")

# =========================
# TITLE
# =========================

title = ctk.CTkLabel(
    app,
    text="⚕ PCOS Prediction System",
    font=("Arial", 38, "bold")
)
title.pack(pady=20)

subtitle = ctk.CTkLabel(
    app,
    text="Hybrid Model (Random Forest + XGBoost) | Accuracy ~92%",
    font=("Arial", 18)
)
subtitle.pack()

# =========================
# MAIN FRAME
# =========================

main_frame = ctk.CTkFrame(app, corner_radius=20)
main_frame.pack(pady=20, padx=20, fill="x")

left_frame = ctk.CTkFrame(main_frame)
left_frame.pack(side="left", padx=40, pady=20)

right_frame = ctk.CTkFrame(main_frame)
right_frame.pack(side="right", padx=40, pady=20)

# =========================
# DROPDOWNS
# =========================

def dropdown(parent, text):

    frame = ctk.CTkFrame(parent, fg_color="transparent")
    frame.pack(pady=10, anchor="w")

    label = ctk.CTkLabel(frame, text=text, font=("Arial", 16))
    label.pack(side="left", padx=10)

    combo = ctk.CTkComboBox(
        frame,
        values=["0", "1"],
        width=120
    )
    combo.set("0")
    combo.pack(side="right")

    return combo

skin_combo = dropdown(left_frame, "Skin Darkening")
hair_combo = dropdown(left_frame, "Hair Growth")
weight_gain_combo = dropdown(left_frame, "Weight Gain")
cycle_combo = dropdown(left_frame, "Cycle (0=I,1=R)")
fastfood_combo = dropdown(left_frame, "Fast Food")
pimples_combo = dropdown(left_frame, "Pimples")

pregnant_combo = dropdown(right_frame, "Pregnant")

# =========================
# SLIDERS
# =========================

def slider(parent, text, from_, to):

    frame = ctk.CTkFrame(parent, fg_color="transparent")
    frame.pack(pady=15, anchor="w")

    label = ctk.CTkLabel(frame, text=f"{text}: 0", font=("Arial", 16))
    label.pack(anchor="w")

    slider = ctk.CTkSlider(
        frame,
        from_=from_,
        to=to,
        width=300
    )
    slider.pack()

    def update_value(value):
        label.configure(text=f"{text}: {round(value,1)}")

    slider.configure(command=update_value)

    return slider

follicle_left_slider = slider(right_frame, "Follicle Left", 0, 30)
follicle_right_slider = slider(right_frame, "Follicle Right", 0, 30)
weight_slider = slider(right_frame, "Weight (Kg)", 30, 120)
abortions_slider = slider(right_frame, "Abortions", 0, 10)

lhfsh_slider = slider(left_frame, "LH/FSH Ratio", 0, 5)
insulin_slider = slider(left_frame, "Insulin Level", 0, 40)
testosterone_slider = slider(left_frame, "Testosterone Level", 0, 5)
total_follicle_slider = slider(right_frame, "Total Follicle Count", 0, 50)

# =========================
# RESULT FRAME
# =========================

result_frame = ctk.CTkFrame(app, corner_radius=20)
result_frame.pack(pady=20, padx=20, fill="both", expand=True)

result_label = ctk.CTkLabel(
    result_frame,
    text="Prediction Result",
    font=("Arial", 24, "bold")
)
result_label.pack(pady=20)

# =========================
# OCR FUNCTION
# =========================

def upload_report():

    file_path = filedialog.askopenfilename(
        filetypes=[
            ("Image Files", "*.png *.jpg *.jpeg"),
            ("PDF Files", "*.pdf")
        ]
    )

    if not file_path:
        return

    image = cv2.imread(file_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    text = pytesseract.image_to_string(gray)

    text = text.lower()

    print(text)

    if "multiple cysts" in text:

        follicle_left_slider.set(18)
        follicle_right_slider.set(18)
        total_follicle_slider.set(36)

    if "cyst" in text:

        lhfsh_slider.set(2.5)
        insulin_slider.set(22)
        testosterone_slider.set(2.1)

# =========================
# PREDICT FUNCTION
# =========================

def predict():

    data = {

        "Skin darkening (Y/N)": int(skin_combo.get()),
        "hair growth(Y/N)": int(hair_combo.get()),
        "Weight gain(Y/N)": int(weight_gain_combo.get()),
        "Cycle(R/I)": int(cycle_combo.get()),
        "Fast food (Y/N)": int(fastfood_combo.get()),
        "Pimples(Y/N)": int(pimples_combo.get()),

        "Follicle No. (L)": follicle_left_slider.get(),
        "Follicle No. (R)": follicle_right_slider.get(),

        "Weight (Kg)": weight_slider.get(),

        "Pregnant(Y/N)": int(pregnant_combo.get()),

        "No. of abortions": abortions_slider.get(),

        "LH/FSH Ratio": lhfsh_slider.get(),
        "Insulin Level": insulin_slider.get(),
        "Testosterone Level": testosterone_slider.get(),
        "Total Follicle Count": total_follicle_slider.get()
    }

    input_df = pd.DataFrame([data])

    input_df = input_df.reindex(columns=feature_cols, fill_value=0)

    pred = model.predict(input_df)[0]

    prob = model.predict_proba(input_df)[0][1]

    probability = round(prob * 100, 2)

    if probability >= 70:
        result = f"PCOS POSITIVE\nProbability: {probability}%"
    else:
        result = f"PCOS NEGATIVE\nProbability: {probability}%"

    result_label.configure(text=result)

    # =====================
    # GRAPH
    # =====================

    for widget in result_frame.winfo_children():

        if widget != result_label:
            widget.destroy()

    result_label.pack(pady=20)

    fig, ax = plt.subplots(figsize=(5,4))

    ax.bar(
        ["No PCOS", "PCOS"],
        [1-prob, prob]
    )

    ax.set_title("Prediction Probability")

    canvas = FigureCanvasTkAgg(fig, master=result_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# =========================
# BUTTONS
# =========================

btn_frame = ctk.CTkFrame(app, fg_color="transparent")
btn_frame.pack(pady=20)

predict_btn = ctk.CTkButton(
    btn_frame,
    text="Predict",
    command=predict,
    width=220,
    height=50,
    font=("Arial", 20, "bold")
)

predict_btn.pack(side="left", padx=20)

browse_btn = ctk.CTkButton(
    btn_frame,
    text="Browse Medical Report",
    command=upload_report,
    width=260,
    height=50,
    font=("Arial", 20, "bold"),
    fg_color="#2E8B57",
    hover_color="#256d46"
)

browse_btn.pack(side="right", padx=20)

# =========================
# RUN APP
# =========================

app.mainloop()