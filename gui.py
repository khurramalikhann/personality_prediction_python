"""
gui.py
Desktop GUI for Personality Prediction — Introvert / Extrovert
Run: python gui.py

Requires models to be trained first (via main.py), or use the
built-in Train tab to train directly from the GUI.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ── colour palette ────────────────────────────────────────────────────────────
BG          = "#F7F8FC"
SURFACE     = "#FFFFFF"
BORDER      = "#E2E8F0"
PRIMARY     = "#4F46E5"          # indigo
PRIMARY_HVR = "#4338CA"
INTROVERT   = "#2563EB"          # blue
EXTROVERT   = "#EA580C"          # orange
SUCCESS     = "#16A34A"
DANGER      = "#DC2626"
TEXT        = "#1E293B"
MUTED       = "#64748B"
LIGHT_TEXT  = "#94A3B8"
CARD_BG     = "#FFFFFF"
HEADER_BG   = "#4F46E5"
HEADER_TEXT = "#FFFFFF"
BAR_BG      = "#E2E8F0"

FONT_FAMILY = "Segoe UI" if sys.platform == "win32" else "Helvetica"


# ── helpers ───────────────────────────────────────────────────────────────────

def models_exist() -> bool:
    needed = ["models/tfidf.pkl", "models/lr_model.pkl",
              "models/nb_model.pkl", "models/label_encoder.pkl"]
    return all(os.path.exists(p) for p in needed)


def make_rounded_button(parent, text, command, bg=PRIMARY, fg="white",
                         hover=PRIMARY_HVR, font_size=10, padx=20, pady=8):
    btn = tk.Button(
        parent, text=text, command=command,
        bg=bg, fg=fg, activebackground=hover, activeforeground=fg,
        font=(FONT_FAMILY, font_size, "bold"),
        relief="flat", cursor="hand2",
        padx=padx, pady=pady, bd=0
    )
    btn.bind("<Enter>", lambda e: btn.config(bg=hover))
    btn.bind("<Leave>", lambda e: btn.config(bg=bg))
    return btn


# ── main application ──────────────────────────────────────────────────────────

class PersonalityApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Personality Prediction — Introvert / Extrovert")
        self.geometry("860x680")
        self.minsize(720, 580)
        self.configure(bg=BG)
        self.resizable(True, True)

        # State
        self._tfidf = None
        self._lr    = None
        self._le    = None

        self._build_header()
        self._build_tabs()
        self._build_statusbar()

        # Load models silently if already trained
        if models_exist():
            self.after(200, self._load_models_silent)

    # ── header ────────────────────────────────────────────────────────────────

    def _build_header(self):
        hdr = tk.Frame(self, bg=HEADER_BG, pady=14)
        hdr.pack(fill="x")

        tk.Label(
            hdr, text="🧠  Personality Prediction",
            font=(FONT_FAMILY, 18, "bold"),
            bg=HEADER_BG, fg=HEADER_TEXT
        ).pack()

        tk.Label(
            hdr, text="Introvert / Extrovert  ·  NLP + Machine Learning",
            font=(FONT_FAMILY, 10),
            bg=HEADER_BG, fg="#C7D2FE"
        ).pack(pady=(2, 0))

    # ── tabs ──────────────────────────────────────────────────────────────────

    def _build_tabs(self):
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook",              background=BG, borderwidth=0)
        style.configure("TNotebook.Tab",          background=BORDER, foreground=MUTED,
                        font=(FONT_FAMILY, 10, "bold"), padding=[18, 8])
        style.map("TNotebook.Tab",
                  background=[("selected", SURFACE)],
                  foreground=[("selected", PRIMARY)])

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=16, pady=12)

        self._predict_tab = tk.Frame(nb, bg=BG)
        self._train_tab   = tk.Frame(nb, bg=BG)

        nb.add(self._predict_tab, text="  Predict  ")
        nb.add(self._train_tab,   text="  Train Model  ")

        self._build_predict_tab()
        self._build_train_tab()

    # ── predict tab ───────────────────────────────────────────────────────────

    def _build_predict_tab(self):
        tab = self._predict_tab

        # Model status banner
        self._status_frame = tk.Frame(tab, bg=BG)
        self._status_frame.pack(fill="x", padx=4, pady=(0, 8))

        self._model_banner = tk.Label(
            self._status_frame,
            text="⚠  Models not found. Go to the Train tab to train the model first.",
            font=(FONT_FAMILY, 9), bg="#FEF3C7", fg="#92400E",
            relief="flat", anchor="w", padx=12, pady=8
        )
        self._model_banner.pack(fill="x")

        # Input card
        input_card = tk.Frame(tab, bg=CARD_BG, relief="flat",
                              highlightbackground=BORDER, highlightthickness=1)
        input_card.pack(fill="both", expand=True, padx=4, pady=(0, 10))

        tk.Label(
            input_card,
            text="Enter text to analyse:",
            font=(FONT_FAMILY, 10, "bold"), bg=CARD_BG, fg=TEXT, anchor="w"
        ).pack(fill="x", padx=16, pady=(14, 4))

        tk.Label(
            input_card,
            text="Paste a social media post, journal entry, or any personal text.",
            font=(FONT_FAMILY, 9), bg=CARD_BG, fg=MUTED, anchor="w"
        ).pack(fill="x", padx=16, pady=(0, 6))

        text_frame = tk.Frame(input_card, bg=CARD_BG)
        text_frame.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        scrollbar = tk.Scrollbar(text_frame, cursor="arrow")
        scrollbar.pack(side="right", fill="y")

        self._text_input = tk.Text(
            text_frame,
            height=7, wrap="word",
            font=(FONT_FAMILY, 10),
            bg="#F8FAFC", fg=TEXT,
            insertbackground=PRIMARY,
            relief="flat",
            highlightbackground=BORDER,
            highlightthickness=1,
            selectbackground=PRIMARY,
            selectforeground="white",
            padx=10, pady=8,
            yscrollcommand=scrollbar.set
        )
        self._text_input.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self._text_input.yview)

        # Placeholder behaviour
        self._placeholder = "Start typing or paste text here..."
        self._text_input.insert("1.0", self._placeholder)
        self._text_input.config(fg=LIGHT_TEXT)
        self._text_input.bind("<FocusIn>",  self._clear_placeholder)
        self._text_input.bind("<FocusOut>", self._restore_placeholder)

        # Buttons row
        btn_row = tk.Frame(input_card, bg=CARD_BG)
        btn_row.pack(fill="x", padx=16, pady=(0, 14))

        self._predict_btn = make_rounded_button(
            btn_row, "  Predict Personality", self._run_predict,
            font_size=11, padx=24, pady=10
        )
        self._predict_btn.pack(side="left")

        make_rounded_button(
            btn_row, "Clear", self._clear_text,
            bg="#F1F5F9", fg=MUTED, hover="#E2E8F0",
            font_size=10, padx=16, pady=10
        ).pack(side="left", padx=(10, 0))

        # Result card
        self._result_card = tk.Frame(tab, bg=CARD_BG, relief="flat",
                                     highlightbackground=BORDER, highlightthickness=1)
        self._result_card.pack(fill="x", padx=4, pady=(0, 4))

        self._build_result_area()

    def _build_result_area(self):
        card = self._result_card

        self._result_top = tk.Frame(card, bg=CARD_BG)
        self._result_top.pack(fill="x", padx=16, pady=14)

        self._result_emoji = tk.Label(
            self._result_top, text="—",
            font=(FONT_FAMILY, 28), bg=CARD_BG, fg=LIGHT_TEXT
        )
        self._result_emoji.pack(side="left")

        right = tk.Frame(self._result_top, bg=CARD_BG)
        right.pack(side="left", padx=14, fill="x", expand=True)

        self._result_label = tk.Label(
            right, text="Awaiting input",
            font=(FONT_FAMILY, 14, "bold"), bg=CARD_BG, fg=MUTED, anchor="w"
        )
        self._result_label.pack(fill="x")

        self._result_sub = tk.Label(
            right, text="Enter text above and click Predict.",
            font=(FONT_FAMILY, 9), bg=CARD_BG, fg=LIGHT_TEXT, anchor="w"
        )
        self._result_sub.pack(fill="x", pady=(2, 6))

        # Confidence bar container
        bar_row = tk.Frame(right, bg=CARD_BG)
        bar_row.pack(fill="x")

        tk.Label(bar_row, text="Confidence", font=(FONT_FAMILY, 8),
                 bg=CARD_BG, fg=MUTED).pack(side="left")
        self._conf_pct_label = tk.Label(
            bar_row, text="", font=(FONT_FAMILY, 8, "bold"),
            bg=CARD_BG, fg=MUTED
        )
        self._conf_pct_label.pack(side="right")

        self._bar_bg = tk.Frame(right, bg=BAR_BG, height=8)
        self._bar_bg.pack(fill="x", pady=(3, 0))
        self._bar_bg.pack_propagate(False)

        self._bar_fill = tk.Frame(self._bar_bg, bg=LIGHT_TEXT, height=8, width=0)
        self._bar_fill.place(x=0, y=0, relheight=1.0)

    # ── train tab ─────────────────────────────────────────────────────────────

    def _build_train_tab(self):
        tab = self._train_tab

        # Dataset card
        ds_card = tk.Frame(tab, bg=CARD_BG, relief="flat",
                           highlightbackground=BORDER, highlightthickness=1)
        ds_card.pack(fill="x", padx=4, pady=(4, 10))

        tk.Label(ds_card, text="Dataset",
                 font=(FONT_FAMILY, 11, "bold"), bg=CARD_BG, fg=TEXT, anchor="w"
                 ).pack(fill="x", padx=16, pady=(14, 2))

        tk.Label(
            ds_card,
            text="Download mbti_1.csv from kaggle.com/datasets/datasnaek/mbti-type "
                 "and select it below.",
            font=(FONT_FAMILY, 9), bg=CARD_BG, fg=MUTED,
            anchor="w", wraplength=700, justify="left"
        ).pack(fill="x", padx=16, pady=(0, 8))

        path_row = tk.Frame(ds_card, bg=CARD_BG)
        path_row.pack(fill="x", padx=16, pady=(0, 14))

        self._ds_path_var = tk.StringVar(value=os.path.abspath("data/mbti_1.csv"))
        path_entry = tk.Entry(
            path_row, textvariable=self._ds_path_var,
            font=(FONT_FAMILY, 9), bg="#F8FAFC", fg=TEXT,
            relief="flat", highlightbackground=BORDER, highlightthickness=1,
            readonlybackground="#F8FAFC"
        )
        path_entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))

        make_rounded_button(
            path_row, "Browse", self._browse_dataset,
            bg="#F1F5F9", fg=TEXT, hover=BORDER,
            font_size=9, padx=12, pady=6
        ).pack(side="left")

        # Train card
        tr_card = tk.Frame(tab, bg=CARD_BG, relief="flat",
                           highlightbackground=BORDER, highlightthickness=1)
        tr_card.pack(fill="x", padx=4, pady=(0, 10))

        tk.Label(tr_card, text="Training",
                 font=(FONT_FAMILY, 11, "bold"), bg=CARD_BG, fg=TEXT, anchor="w"
                 ).pack(fill="x", padx=16, pady=(14, 4))

        self._train_btn = make_rounded_button(
            tr_card, "  Start Training", self._run_train,
            font_size=11, padx=24, pady=10
        )
        self._train_btn.pack(anchor="w", padx=16, pady=(0, 14))

        # Progress
        prog_frame = tk.Frame(tab, bg=CARD_BG, relief="flat",
                              highlightbackground=BORDER, highlightthickness=1)
        prog_frame.pack(fill="x", padx=4, pady=(0, 10))

        tk.Label(prog_frame, text="Progress",
                 font=(FONT_FAMILY, 11, "bold"), bg=CARD_BG, fg=TEXT, anchor="w"
                 ).pack(fill="x", padx=16, pady=(14, 4))

        self._progress_var = tk.DoubleVar()
        style = ttk.Style()
        style.configure("Custom.Horizontal.TProgressbar",
                        troughcolor=BAR_BG, background=PRIMARY,
                        thickness=12, borderwidth=0)
        self._progress_bar = ttk.Progressbar(
            prog_frame, variable=self._progress_var,
            maximum=100, style="Custom.Horizontal.TProgressbar"
        )
        self._progress_bar.pack(fill="x", padx=16, pady=(0, 8))

        self._train_log = tk.Text(
            prog_frame, height=10, state="disabled",
            font=("Courier New" if sys.platform == "win32" else "Monaco", 9),
            bg="#1E293B", fg="#94A3B8",
            insertbackground=PRIMARY,
            relief="flat", padx=10, pady=8,
            wrap="word"
        )
        self._train_log.pack(fill="both", expand=True, padx=16, pady=(0, 14))

    # ── status bar ────────────────────────────────────────────────────────────

    def _build_statusbar(self):
        bar = tk.Frame(self, bg=BORDER, height=1)
        bar.pack(fill="x")

        sb = tk.Frame(self, bg="#F1F5F9", pady=5)
        sb.pack(fill="x")

        self._status_var = tk.StringVar(value="Ready")
        tk.Label(sb, textvariable=self._status_var,
                 font=(FONT_FAMILY, 8), bg="#F1F5F9", fg=MUTED
                 ).pack(side="left", padx=12)

        self._model_status_var = tk.StringVar()
        self._model_status_lbl = tk.Label(
            sb, textvariable=self._model_status_var,
            font=(FONT_FAMILY, 8, "bold"), bg="#F1F5F9"
        )
        self._model_status_lbl.pack(side="right", padx=12)

        self._update_model_status()

    # ── model loading ─────────────────────────────────────────────────────────

    def _load_models_silent(self):
        try:
            from predict import load_models
            self._tfidf, self._lr, self._le = load_models()
            self._update_model_status(loaded=True)
            self._model_banner.config(
                text="✔  Models loaded and ready. Enter text to predict.",
                bg="#DCFCE7", fg="#166534"
            )
            self._status_var.set("Models loaded successfully.")
        except Exception as e:
            self._update_model_status(loaded=False)
            self._status_var.set(f"Model load error: {e}")

    def _update_model_status(self, loaded: bool = None):
        if loaded is None:
            loaded = models_exist() and (self._lr is not None)
        if loaded:
            self._model_status_var.set("● Models Ready")
            self._model_status_lbl.config(fg=SUCCESS)
        else:
            self._model_status_var.set("● Models Not Trained")
            self._model_status_lbl.config(fg=DANGER)

    # ── prediction logic ──────────────────────────────────────────────────────

    def _clear_placeholder(self, event=None):
        if self._text_input.get("1.0", "end-1c") == self._placeholder:
            self._text_input.delete("1.0", "end")
            self._text_input.config(fg=TEXT)

    def _restore_placeholder(self, event=None):
        if not self._text_input.get("1.0", "end-1c").strip():
            self._text_input.insert("1.0", self._placeholder)
            self._text_input.config(fg=LIGHT_TEXT)

    def _clear_text(self):
        self._text_input.delete("1.0", "end")
        self._restore_placeholder()
        self._reset_result()

    def _reset_result(self):
        self._result_emoji.config(text="—", fg=LIGHT_TEXT)
        self._result_label.config(text="Awaiting input", fg=MUTED)
        self._result_sub.config(text="Enter text above and click Predict.")
        self._conf_pct_label.config(text="", fg=MUTED)
        self._bar_fill.config(bg=LIGHT_TEXT, width=0)

    def _run_predict(self):
        if self._lr is None:
            messagebox.showwarning(
                "Models Not Loaded",
                "Please train the model first using the Train tab."
            )
            return

        raw = self._text_input.get("1.0", "end-1c").strip()
        if not raw or raw == self._placeholder:
            messagebox.showinfo("No Input", "Please enter some text to analyse.")
            return

        self._status_var.set("Predicting...")
        self._predict_btn.config(state="disabled", text="Predicting...")
        self.update_idletasks()

        def _predict():
            try:
                from predict import predict_personality
                label, conf = predict_personality(raw, self._tfidf, self._lr, self._le)
                self.after(0, lambda: self._show_result(label, conf))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Prediction Error", str(e)))
            finally:
                self.after(0, lambda: self._predict_btn.config(
                    state="normal", text="  Predict Personality"))

        threading.Thread(target=_predict, daemon=True).start()

    def _show_result(self, label: str, confidence: float):
        is_intro = (label == "Introvert")
        colour   = INTROVERT if is_intro else EXTROVERT
        emoji    = "📚" if is_intro else "🎉"
        desc     = (
            "Tends to be reflective, reserved, and energised by solitude."
            if is_intro else
            "Tends to be outgoing, expressive, and energised by social interaction."
        )

        self._result_emoji.config(text=emoji, fg=colour)
        self._result_label.config(text=label, fg=colour)
        self._result_sub.config(text=desc, fg=MUTED)
        self._conf_pct_label.config(text=f"{confidence:.1f}%", fg=colour)

        # Animate confidence bar
        self._bar_fill.config(bg=colour)
        self._animate_bar(0, confidence)
        self._status_var.set(f"Result: {label}  |  Confidence: {confidence:.1f}%")

    def _animate_bar(self, current: float, target: float):
        if current >= target:
            return
        step = max(1, (target - current) / 8)
        nxt  = min(current + step, target)
        total_width = self._bar_bg.winfo_width() or 400
        pix = int(total_width * nxt / 100)
        self._bar_fill.place(x=0, y=0, relheight=1.0, width=pix)
        self.after(20, lambda: self._animate_bar(nxt, target))

    # ── training logic ────────────────────────────────────────────────────────

    def _browse_dataset(self):
        path = filedialog.askopenfilename(
            title="Select MBTI Dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self._ds_path_var.set(path)

    def _log(self, msg: str, colour: str = "#94A3B8"):
        self._train_log.config(state="normal")
        self._train_log.insert("end", msg + "\n")
        self._train_log.see("end")
        self._train_log.config(state="disabled")
        self.update_idletasks()

    def _run_train(self):
        dataset = self._ds_path_var.get().strip()
        if not os.path.exists(dataset):
            messagebox.showerror(
                "File Not Found",
                f"Dataset not found:\n{dataset}\n\n"
                "Download mbti_1.csv from Kaggle and select it using Browse."
            )
            return

        self._train_btn.config(state="disabled", text="Training…")
        self._train_log.config(state="normal")
        self._train_log.delete("1.0", "end")
        self._train_log.config(state="disabled")
        self._progress_var.set(0)

        def _train():
            try:
                self.after(0, lambda: self._log("[1/4] Loading and preprocessing data…"))
                self.after(0, lambda: self._progress_var.set(10))

                # Import fresh each time without reload side-effects
                import sys as _sys
                for _m in ['preprocess', 'model', 'evaluate']:
                    _sys.modules.pop(_m, None)

                import preprocess as pp, model as md, evaluate as ev

                df = pp.preprocess(dataset)
                total = len(df)
                intro = (df['label'] == 'Introvert').sum()
                extro = (df['label'] == 'Extrovert').sum()

                self.after(0, lambda: self._log(
                    f"    Samples: {total}  |  Introvert: {intro}  |  Extrovert: {extro}"))
                self.after(0, lambda: self._progress_var.set(35))

                self.after(0, lambda: self._log("\n[2/4] Training models…"))
                lr, nb, tfidf, le, X_test, y_test = md.train_models(df)
                self.after(0, lambda: self._progress_var.set(65))

                self.after(0, lambda: self._log("\n[3/4] Evaluating models…"))
                from sklearn.metrics import accuracy_score
                lr_acc = accuracy_score(y_test, lr.predict(X_test))
                nb_acc = accuracy_score(y_test, nb.predict(X_test))
                self.after(0, lambda: self._log(
                    f"    Logistic Regression accuracy : {lr_acc * 100:.2f}%"))
                self.after(0, lambda: self._log(
                    f"    Naive Bayes accuracy         : {nb_acc * 100:.2f}%"))
                self.after(0, lambda: self._progress_var.set(85))

                self.after(0, lambda: self._log("\n[4/4] Saving evaluation plots…"))
                os.makedirs("outputs", exist_ok=True)
                ev.evaluate_model(lr, X_test, y_test, "Logistic Regression", le)
                ev.evaluate_model(nb, X_test, y_test, "Naive Bayes", le)

                self._tfidf = tfidf
                self._lr    = lr
                self._le    = le

                self.after(0, lambda: self._progress_var.set(100))
                self.after(0, lambda: self._log(
                    "\n✔  Training complete. Switch to the Predict tab.", "#4ADE80"))
                self.after(0, lambda: self._update_model_status(loaded=True))
                self.after(0, lambda: self._model_banner.config(
                    text="✔  Models loaded and ready. Enter text to predict.",
                    bg="#DCFCE7", fg="#166534"))
                self.after(0, lambda: self._status_var.set("Training complete."))

            except Exception as e:
                self.after(0, lambda: self._log(f"\n✖  Error: {e}", "#F87171"))
                self.after(0, lambda: messagebox.showerror("Training Error", str(e)))
            finally:
                self.after(0, lambda: self._train_btn.config(
                    state="normal", text="  Start Training"))

        threading.Thread(target=_train, daemon=True).start()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = PersonalityApp()
    app.mainloop()
