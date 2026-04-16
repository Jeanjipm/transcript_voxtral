"""
Fenêtre de paramètres Voxtral Dictée — tkinter (stdlib, zéro install).

6 onglets : Modèle, Langue, Raccourci, Sons, Avancé, À propos.

Lancée en sous-processus depuis app.py (cf. commentaire dans app.py).
Sauvegarde dans ~/.voxtral/config.yaml ; un redémarrage de l'app menu
bar peut être nécessaire pour appliquer certains changements (ex. modèle).
"""

from __future__ import annotations

import tkinter as tk
import webbrowser
from tkinter import messagebox, ttk
from typing import Callable

from config import Config, load_config, save_config
from hotkey_manager import _parse_key, display_combo
from model_manager import (
    AVAILABLE_MODELS,
    is_downloaded,
)


def _validate_combo(combo: str) -> str | None:
    """Retourne None si le combo est valide, sinon un message d'erreur."""
    combo = combo.strip()
    if not combo:
        return "Raccourci vide."
    tokens = [t.strip() for t in combo.split("+")]
    if any(not t for t in tokens):
        return f"Jetons vides dans '{combo}'."
    try:
        for t in tokens:
            _parse_key(t)
    except ValueError as exc:
        return str(exc)
    return None


# Raccourcis système macOS connus → warning de conflit dans l'UI
KNOWN_SYSTEM_CONFLICTS: set[str] = {
    # Système
    "cmd+space",       # Spotlight
    "cmd+h",           # Masquer
    "cmd+option+h",    # Masquer les autres
    "cmd+option+d",    # Toggle Dock
    "cmd+shift+3",     # Capture écran
    "cmd+shift+4",     # Capture sélection
    "cmd+shift+5",     # Outils capture
    "cmd+ctrl+space",  # Emoji picker
    "cmd+shift+space", # Spotlight Réseau
    "cmd+tab",         # Switch app
    "cmd+q",           # Quitter
    # Finder — tous ces cmd+shift+LETTRE ouvrent un dossier Finder,
    # donc volent le focus et font que le paste se fait dans Finder
    "cmd+shift+h",     # Dossier Départ
    "cmd+shift+d",     # Bureau
    "cmd+shift+a",     # Applications
    "cmd+shift+u",     # Utilitaires
    "cmd+shift+o",     # Documents
    "cmd+shift+c",     # Ordinateur
    "cmd+shift+i",     # iCloud Drive
    "cmd+shift+f",     # Récents
    "cmd+shift+g",     # Aller au dossier
    "cmd+shift+k",     # Réseau
}


# Touches "single-key tenue" proposées dans le sélecteur
SINGLE_KEY_OPTIONS: list[tuple[str, str]] = [
    ("alt_r", "⌥ Option droite (recommandé — talkie-walkie)"),
    ("cmd_r", "⌘ Cmd droite"),
    ("ctrl_r", "⌃ Ctrl droite"),
    ("shift_r", "⇧ Shift droite"),
    ("f13", "F13"),
    ("f14", "F14"),
    ("f15", "F15"),
    ("f16", "F16"),
    ("f17", "F17"),
    ("f18", "F18"),
    ("f19", "F19"),
]


# Langues exposées dans l'UI (reflète Voxtral § F3)
LANGUAGE_OPTIONS: list[tuple[str, str]] = [
    ("auto", "Auto-détection"),
    ("fr", "Français"),
    ("en", "English"),
    ("de", "Deutsch"),
    ("es", "Español"),
    ("it", "Italiano"),
    ("pt", "Português"),
    ("nl", "Nederlands"),
    ("hi", "हिन्दी"),
]


class SettingsWindow:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Voxtral — Préférences")
        self.root.geometry("560x440")
        self.root.minsize(520, 400)

        self.config: Config = load_config()

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self._build_model_tab()
        self._build_language_tab()
        self._build_hotkey_tab()
        self._build_sounds_tab()
        self._build_advanced_tab()
        self._build_about_tab()

        # Boutons globaux
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(btn_frame, text="Annuler", command=self.root.destroy).pack(
            side=tk.RIGHT, padx=(5, 0)
        )
        ttk.Button(btn_frame, text="Enregistrer", command=self._save).pack(
            side=tk.RIGHT
        )

    # ------------------------------------------------------------------
    # Onglets
    # ------------------------------------------------------------------

    def _build_model_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(frame, text="Modèle")

        ttk.Label(frame, text="Modèle de transcription :").grid(
            row=0, column=0, sticky="w", pady=(0, 5)
        )

        self.model_var = tk.StringVar(value=self.config.model.name)
        for i, m in enumerate(AVAILABLE_MODELS, start=1):
            downloaded = is_downloaded(m.repo_id, self.config.model.resolved_path)
            tag = " ✓ téléchargé" if downloaded else ""
            label = f"{m.label} — {m.size_gb:.1f} Go{tag}\n   {m.description}"
            ttk.Radiobutton(
                frame,
                text=label,
                variable=self.model_var,
                value=m.repo_id,
            ).grid(row=i, column=0, sticky="w", pady=3)

        ttk.Label(
            frame,
            text=(
                "Le téléchargement se fait via le bouton 'Mettre à jour le "
                "modèle' du menu, ou via :\n  python download_model.py --model NOM"
            ),
            foreground="gray",
            justify=tk.LEFT,
        ).grid(row=len(AVAILABLE_MODELS) + 1, column=0, sticky="w", pady=(15, 0))

    def _build_language_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(frame, text="Langue")

        ttk.Label(frame, text="Langue de transcription :").grid(
            row=0, column=0, sticky="w"
        )
        self.lang_var = tk.StringVar(value=self.config.transcription.language)
        lang_combo = ttk.Combobox(
            frame,
            textvariable=self.lang_var,
            values=[code for code, _ in LANGUAGE_OPTIONS],
            state="readonly",
            width=20,
        )
        lang_combo.grid(row=0, column=1, sticky="w", padx=(10, 0))

        ttk.Label(frame, text="Tâche :").grid(
            row=1, column=0, sticky="w", pady=(15, 0)
        )
        self.task_var = tk.StringVar(value=self.config.transcription.task)
        ttk.Radiobutton(
            frame,
            text="Transcription (texte dans la langue parlée)",
            variable=self.task_var,
            value="transcribe",
        ).grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(
            frame,
            text="Traduction (vers anglais)",
            variable=self.task_var,
            value="translate",
        ).grid(row=3, column=0, columnspan=2, sticky="w")

    def _build_hotkey_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(frame, text="Raccourci")

        ttk.Label(frame, text="Type de raccourci :").grid(
            row=0, column=0, sticky="w"
        )

        self.hotkey_type_var = tk.StringVar(
            value="single" if "+" not in self.config.hotkey.combo else "combo"
        )
        ttk.Radiobutton(
            frame,
            text="Touche unique tenue (mode talkie-walkie)",
            variable=self.hotkey_type_var,
            value="single",
            command=self._on_hotkey_type_change,
        ).grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(
            frame,
            text="Combinaison de touches",
            variable=self.hotkey_type_var,
            value="combo",
            command=self._on_hotkey_type_change,
        ).grid(row=2, column=0, columnspan=2, sticky="w")

        # Sélecteur single-key
        ttk.Label(frame, text="Touche :").grid(
            row=3, column=0, sticky="w", pady=(15, 0)
        )
        single_default = (
            self.config.hotkey.combo
            if "+" not in self.config.hotkey.combo
            else "alt_r"
        )
        self.single_key_var = tk.StringVar(value=single_default)
        self.single_combo = ttk.Combobox(
            frame,
            textvariable=self.single_key_var,
            values=[code for code, _ in SINGLE_KEY_OPTIONS],
            state="readonly",
            width=20,
        )
        self.single_combo.grid(row=3, column=1, sticky="w", padx=(10, 0), pady=(15, 0))

        # Champ combinaison libre
        ttk.Label(frame, text="Combinaison :").grid(
            row=4, column=0, sticky="w", pady=(10, 0)
        )
        # Défaut proposé pour le champ combo : ctrl+option+space. Choisi
        # pour ne pas voler le focus (contrairement à cmd+shift+h qui ouvre
        # le dossier Départ dans Finder et casse le paste automatique).
        combo_default = (
            self.config.hotkey.combo
            if "+" in self.config.hotkey.combo
            else "ctrl+option+space"
        )
        self.combo_var = tk.StringVar(value=combo_default)
        self.combo_entry = ttk.Entry(frame, textvariable=self.combo_var, width=22)
        self.combo_entry.grid(
            row=4, column=1, sticky="w", padx=(10, 0), pady=(10, 0)
        )
        ttk.Label(
            frame,
            text="Format : 'ctrl+option+space' (recommandé), 'ctrl+alt+d'…",
            foreground="gray",
        ).grid(row=5, column=1, sticky="w", padx=(10, 0))

        # Mode
        ttk.Label(frame, text="Mode :").grid(
            row=6, column=0, sticky="w", pady=(15, 0)
        )
        self.mode_var = tk.StringVar(value=self.config.hotkey.mode)
        self.mode_radio_ptt = ttk.Radiobutton(
            frame,
            text="Push-to-talk (maintenir = enregistrer)",
            variable=self.mode_var,
            value="push_to_talk",
        )
        self.mode_radio_ptt.grid(row=7, column=0, columnspan=2, sticky="w")
        self.mode_radio_toggle = ttk.Radiobutton(
            frame,
            text="Toggle (appuyer pour démarrer / arrêter)",
            variable=self.mode_var,
            value="toggle",
        )
        self.mode_radio_toggle.grid(row=8, column=0, columnspan=2, sticky="w")

        self.hotkey_warning = ttk.Label(frame, text="", foreground="orange")
        self.hotkey_warning.grid(
            row=9, column=0, columnspan=2, sticky="w", pady=(15, 0)
        )

        # Watchers
        self.combo_var.trace_add("write", lambda *_: self._update_hotkey_warning())
        self.single_key_var.trace_add("write", lambda *_: self._update_hotkey_warning())
        self.hotkey_type_var.trace_add("write", lambda *_: self._update_hotkey_warning())

        self._on_hotkey_type_change()
        self._update_hotkey_warning()

    def _on_hotkey_type_change(self) -> None:
        is_single = self.hotkey_type_var.get() == "single"
        if is_single:
            self.single_combo.configure(state="readonly")
            self.combo_entry.configure(state="disabled")
            # Single-key force push-to-talk
            self.mode_var.set("push_to_talk")
            self.mode_radio_toggle.configure(state="disabled")
        else:
            self.single_combo.configure(state="disabled")
            self.combo_entry.configure(state="normal")
            self.mode_radio_toggle.configure(state="normal")

    def _update_hotkey_warning(self) -> None:
        combo = self._current_combo()
        if combo.lower() in KNOWN_SYSTEM_CONFLICTS:
            self.hotkey_warning.configure(
                text=f"⚠ Conflit système connu : {display_combo(combo)} est "
                "déjà pris par macOS."
            )
        else:
            self.hotkey_warning.configure(text="")

    def _current_combo(self) -> str:
        if self.hotkey_type_var.get() == "single":
            return self.single_key_var.get().strip()
        return self.combo_var.get().strip()

    def _build_sounds_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(frame, text="Sons")

        self.sounds_enabled_var = tk.BooleanVar(value=self.config.sounds.enabled)
        ttk.Checkbutton(
            frame,
            text="Activer les sons d'activation / désactivation",
            variable=self.sounds_enabled_var,
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(frame, text="Volume :").grid(
            row=1, column=0, sticky="w", pady=(15, 0)
        )
        self.volume_var = tk.DoubleVar(value=self.config.sounds.volume * 100)
        ttk.Scale(
            frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.volume_var,
            length=300,
        ).grid(row=1, column=1, sticky="w", padx=(10, 0), pady=(15, 0))

        ttk.Label(frame, text="Thème :").grid(
            row=2, column=0, sticky="w", pady=(15, 0)
        )
        self.theme_var = tk.StringVar(value=self.config.sounds.theme)
        ttk.Combobox(
            frame,
            textvariable=self.theme_var,
            values=["system", "soft", "subtle", "custom"],
            state="readonly",
            width=15,
        ).grid(row=2, column=1, sticky="w", padx=(10, 0), pady=(15, 0))

        ttk.Label(
            frame,
            text="Thème 'system' = sons macOS (Tink/Pop). Recommandé.",
            foreground="gray",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(5, 0))

    def _build_advanced_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(frame, text="Avancé")

        ttk.Label(frame, text="Température de génération :").grid(
            row=0, column=0, sticky="w"
        )
        self.temp_var = tk.DoubleVar(value=self.config.transcription.temperature)
        ttk.Spinbox(
            frame,
            from_=0.0,
            to=1.0,
            increment=0.1,
            textvariable=self.temp_var,
            width=8,
        ).grid(row=0, column=1, sticky="w", padx=(10, 0))
        ttk.Label(
            frame,
            text="0.0 = transcription la plus fidèle. > 0 déconseillé.",
            foreground="gray",
        ).grid(row=1, column=0, columnspan=2, sticky="w")

        ttk.Label(frame, text="max_new_tokens :").grid(
            row=2, column=0, sticky="w", pady=(15, 0)
        )
        self.tokens_var = tk.IntVar(value=self.config.transcription.max_new_tokens)
        ttk.Spinbox(
            frame,
            from_=128,
            to=4096,
            increment=128,
            textvariable=self.tokens_var,
            width=8,
        ).grid(row=2, column=1, sticky="w", padx=(10, 0), pady=(15, 0))

        self.streaming_var = tk.BooleanVar(value=self.config.transcription.streaming)
        ttk.Checkbutton(
            frame,
            text="Mode streaming (audio > 30 s)",
            variable=self.streaming_var,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(15, 0))

        self.notif_var = tk.BooleanVar(value=self.config.ui.notification_on_paste)
        ttk.Checkbutton(
            frame,
            text="Notification après collage",
            variable=self.notif_var,
        ).grid(row=4, column=0, columnspan=2, sticky="w")

        self.autopaste_var = tk.BooleanVar(value=self.config.ui.auto_paste)
        ttk.Checkbutton(
            frame,
            text="Coller automatiquement à la position du curseur",
            variable=self.autopaste_var,
        ).grid(row=5, column=0, columnspan=2, sticky="w")

    def _build_about_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(frame, text="À propos")

        ttk.Label(
            frame,
            text="Voxtral Dictée",
            font=("Helvetica", 16, "bold"),
        ).pack(anchor="w")
        ttk.Label(frame, text="Version 0.1.0").pack(anchor="w", pady=(5, 0))
        ttk.Label(
            frame,
            text=(
                "Dictée vocale 100% locale via MLX sur Apple Silicon.\n"
                "Aucune donnée ne quitte votre Mac."
            ),
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(15, 0))

        link_frame = ttk.Frame(frame)
        link_frame.pack(anchor="w", pady=(20, 0))
        self._link_button(
            link_frame,
            "GitHub",
            "https://github.com/Jeanjipm/transcript_voxtral",
        ).pack(side=tk.LEFT)

    def _link_button(self, parent: tk.Widget, text: str, url: str) -> ttk.Button:
        return ttk.Button(parent, text=text, command=lambda: webbrowser.open(url))

    # ------------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------------

    def _save(self) -> None:
        # Valide le raccourci AVANT tout : un combo invalide sauvegardé
        # ferait planter l'app au prochain démarrage dans _parse_key.
        combo = self._current_combo()
        error = _validate_combo(combo)
        if error is not None:
            messagebox.showerror(
                "Raccourci invalide",
                f"{error}\n\nFormat attendu : 'alt_r', 'cmd+shift+h', "
                "'ctrl+alt+space'. Jetons valides : cmd, alt, ctrl, shift, "
                "space, enter, tab, esc, f13-f19, ou une lettre.",
            )
            self.notebook.select(2)  # onglet Raccourci
            return

        # Reconstitue Config depuis les widgets
        cfg = self.config
        cfg.model.name = self.model_var.get()
        cfg.transcription.language = self.lang_var.get()
        cfg.transcription.task = self.task_var.get()
        cfg.transcription.temperature = float(self.temp_var.get())
        cfg.transcription.max_new_tokens = int(self.tokens_var.get())
        cfg.transcription.streaming = bool(self.streaming_var.get())
        cfg.hotkey.combo = combo
        cfg.hotkey.mode = self.mode_var.get()
        cfg.sounds.enabled = bool(self.sounds_enabled_var.get())
        cfg.sounds.volume = float(self.volume_var.get()) / 100.0
        cfg.sounds.theme = self.theme_var.get()
        cfg.ui.notification_on_paste = bool(self.notif_var.get())
        cfg.ui.auto_paste = bool(self.autopaste_var.get())

        save_config(cfg)
        messagebox.showinfo(
            "Préférences enregistrées",
            "Les paramètres ont été sauvegardés. Ils seront appliqués "
            "automatiquement d'ici 2-3 secondes — pas besoin de redémarrer.",
        )
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    SettingsWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
