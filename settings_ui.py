"""
Fenêtre de paramètres Voxtral Dictée — tkinter (stdlib, zéro install).

Lancée en sous-processus depuis app.py (cf. commentaire dans app.py).
Sauvegarde dans ~/.voxtral/config.yaml ; l'app menu bar recharge d'elle-même
en 2-3 s.

## Organisation

Les onglets suivent ce que l'utilisateur VEUT FAIRE, pas le type technique du
réglage. L'ancienne version avait huit onglets (Modèle, Langue, Raccourci,
Sons, Fichiers, Stockage, Avancé, À propos) qui éclataient une même tâche à
plusieurs endroits : régler sa dictée demandait de visiter trois onglets.

    Dictée   — tout le trajet raccourci → parole → texte collé
    Fichiers — transcrire un enregistrement existant
    Modèles  — quel modèle pour quoi, et la place qu'ils prennent
    Avancé   — réglages fins dont personne n'a besoin au quotidien
    À propos

## Deux règles suivies partout

1. **Jamais de code technique affiché.** Les listes montrent « Français » et
   « ⌥ Option droite », pas `fr` et `alt_r`. Les libellés existaient déjà dans
   le code mais n'étaient pas utilisés — on voyait les codes bruts.
2. **Rien ne se passe en silence.** Choisir un modèle absent propose son
   téléchargement avec une progression réelle, plutôt que de laisser l'app
   figée plusieurs minutes au premier usage sans explication.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
import webbrowser
from tkinter import filedialog, messagebox, ttk

from config import Config, load_config, save_config
from hotkey_capture import HotkeyCapture, Outcome
from hotkey_manager import display_combo, generic_combo, validate_combo
from model_manager import (
    USAGE_DICTATION,
    USAGE_FILES,
    ModelInfo,
    cached_size_bytes,
    delete_cached_model,
    download_model,
    find_model,
    format_size,
    is_downloaded,
    list_available_models,
    models_in_use,
    scan_cached_models,
)


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


# Durée maximale d'une capture de raccourci restée sans appui. Un listener
# clavier global qui traîne parce que l'utilisateur a cliqué « Modifier… »
# puis est parti déjeuner n'a aucune raison d'exister.
CAPTURE_TIMEOUT_MS = 15_000
CAPTURE_POLL_MS = 40


REPO_URL = "https://github.com/Jeanjipm/transcript_voxtral"


LANGUAGE_OPTIONS: list[tuple[str, str]] = [
    ("auto", "Détection automatique"),
    ("fr", "Français"),
    ("en", "Anglais"),
    ("de", "Allemand"),
    ("es", "Espagnol"),
    ("it", "Italien"),
    ("pt", "Portugais"),
    ("nl", "Néerlandais"),
    ("hi", "Hindi"),
]


class LabelledChoice:
    """Combobox qui affiche un libellé lisible et retourne un code technique.

    Sans ça, la liste des langues affichait « fr » et celle des raccourcis
    « alt_r » — les libellés lisibles existaient dans le code mais n'étaient
    jamais montrés.
    """

    def __init__(
        self,
        parent: tk.Widget,
        options: list[tuple[str, str]],
        current: str,
        width: int = 30,
    ) -> None:
        self._options = options
        self._by_label = {label: code for code, label in options}
        self._by_code = {code: label for code, label in options}
        # Une valeur de config hors liste (éditée à la main) ne doit pas
        # disparaître silencieusement : on l'affiche telle quelle.
        initial = self._by_code.get(current, current)
        self.var = tk.StringVar(value=initial)
        self.widget = ttk.Combobox(
            parent,
            textvariable=self.var,
            values=[label for _, label in options],
            state="readonly",
            width=width,
        )

    def grid(self, **kwargs) -> "LabelledChoice":  # noqa: ANN003
        self.widget.grid(**kwargs)
        return self

    def get_code(self) -> str:
        return self._by_label.get(self.var.get(), self.var.get())

    def set_code(self, code: str) -> None:
        self.var.set(self._by_code.get(code, code))


class DownloadDialog:
    """Fenêtre modale de téléchargement, avec progression réelle.

    La progression vient de la taille du dossier sur le disque, comparée au
    poids annoncé du modèle. C'est approximatif au pourcent près, mais c'est
    une vraie mesure — et surtout ça ne dépend pas des barres tqdm internes
    de huggingface_hub, qu'on ne peut pas capter proprement.
    """

    POLL_MS = 400

    def __init__(self, parent: tk.Tk, model: ModelInfo) -> None:
        self.model = model
        self.expected = int(model.size_gb * 1_000_000_000)
        self.error: Exception | None = None
        self._done = threading.Event()

        self.win = tk.Toplevel(parent)
        self.win.title("Téléchargement du modèle")
        self.win.geometry("420x160")
        self.win.transient(parent)
        self.win.grab_set()
        # Pas de fermeture par la croix : interrompre un snapshot_download en
        # cours laisserait un cache partiel.
        self.win.protocol("WM_DELETE_WINDOW", lambda: None)

        ttk.Label(self.win, text=model.label, font=("", 13, "bold")).pack(
            pady=(18, 4)
        )
        self.status = ttk.Label(
            self.win, text=f"Préparation… ({model.size_gb:.1f} Go à récupérer)"
        )
        self.status.pack()

        self.bar = ttk.Progressbar(
            self.win, orient=tk.HORIZONTAL, length=340, mode="determinate",
            maximum=100,
        )
        self.bar.pack(pady=14)

        ttk.Label(
            self.win,
            text="Tu peux laisser cette fenêtre, le téléchargement continue.",
            foreground="gray",
        ).pack()

    def run(self) -> bool:
        """Lance le téléchargement et attend. True si réussi."""
        thread = threading.Thread(target=self._worker, daemon=True)
        thread.start()
        self._poll()
        self.win.wait_window()
        return self.error is None

    def _worker(self) -> None:
        try:
            download_model(self.model.repo_id)
        except Exception as exc:  # noqa: BLE001
            self.error = exc
        finally:
            self._done.set()

    def _poll(self) -> None:
        if self._done.is_set():
            self.win.grab_release()
            self.win.destroy()
            return

        size = cached_size_bytes(self.model.repo_id)
        pct = min(99, int(100 * size / self.expected)) if self.expected else 0
        self.bar["value"] = pct
        self.status.configure(
            text=f"{format_size(size)} sur {self.model.size_gb:.1f} Go — {pct} %"
        )
        self.win.after(self.POLL_MS, self._poll)


class SettingsWindow:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Voxtral — Préférences")
        # Plus grand qu'avant : l'onglet Modèles liste plusieurs modèles avec
        # leur description, et l'ancienne taille les tronquait.
        self.root.geometry("640x600")
        self.root.minsize(600, 520)

        self.config: Config = load_config()

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self._build_dictation_tab()
        self._build_files_tab()
        self._build_models_tab()
        self._build_advanced_tab()
        self._build_about_tab()

        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        # Ordre macOS : l'action principale est la plus à droite. L'ancienne
        # version les avait inversés.
        ttk.Button(btn_frame, text="Enregistrer", command=self._save).pack(
            side=tk.RIGHT
        )
        ttk.Button(btn_frame, text="Annuler", command=self._close).pack(
            side=tk.RIGHT, padx=(0, 8)
        )
        # Fermer par la croix doit passer par le même chemin : sinon une
        # capture en cours laisserait un écouteur clavier global derrière elle.
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _close(self) -> None:
        self._end_capture(None)
        self.root.destroy()

    # ------------------------------------------------------------------
    # Onglet Dictée — tout le trajet raccourci → parole → texte collé
    # ------------------------------------------------------------------

    def _build_dictation_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(frame, text="Dictée")

        # --- Raccourci ---
        ttk.Label(frame, text="Raccourci", font=("", 12, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(
            frame,
            text="Maintiens la touche pour parler, relâche pour transcrire.",
            foreground="gray",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 8))

        # Le raccourci ne se tape plus, il s'enregistre. L'ancien champ de
        # texte libre demandait de connaître le vocabulaire interne (`alt_r`,
        # `cmd+shift+h`) et acceptait des raccourcis syntaxiquement valides
        # mais introuvables au clavier.
        self._combo: str = self.config.hotkey.combo
        self._capture: HotkeyCapture | None = None
        self._capture_timeout_id: str | None = None
        self._capture_saw_a_key = False

        hotkey_row = ttk.Frame(frame)
        hotkey_row.grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 0))

        # takefocus : pendant la capture, ce libellé prend le focus clavier.
        # Sans ça, appuyer sur Espace ou Entrée pour l'enregistrer activerait
        # le bouton qui a le focus au lieu d'être capturé.
        self.hotkey_display = ttk.Label(
            hotkey_row, width=26, anchor="center", relief="groove",
            padding=(8, 7), takefocus=True,
        )
        self.hotkey_display.pack(side=tk.LEFT)
        self.hotkey_display.bind("<KeyPress>", self._swallow_key)
        self.hotkey_display.bind("<KeyRelease>", self._swallow_key)

        self.capture_button = ttk.Button(
            hotkey_row, text="Modifier…", command=self._toggle_capture
        )
        self.capture_button.pack(side=tk.LEFT, padx=(8, 0))

        self.reset_hotkey_button = ttk.Button(
            hotkey_row, text="Par défaut", command=self._reset_combo
        )
        self.reset_hotkey_button.pack(side=tk.LEFT, padx=(6, 0))

        self.hotkey_hint = ttk.Label(frame, text="", foreground="gray")
        self.hotkey_hint.grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

        self.hotkey_warning = ttk.Label(frame, text="", foreground="#c25e00")
        self.hotkey_warning.grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(2, 0)
        )

        ttk.Label(
            frame,
            text="La dictée est en pause tant que cette fenêtre est ouverte.",
            foreground="gray",
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 14))

        # --- Langue ---
        ttk.Label(frame, text="Langue", font=("", 12, "bold")).grid(
            row=6, column=0, columnspan=2, sticky="w"
        )
        self.lang = LabelledChoice(
            frame, LANGUAGE_OPTIONS, self.config.transcription.language, width=28
        ).grid(row=7, column=0, sticky="w", pady=(4, 4))

        self.task_var = tk.StringVar(value=self.config.transcription.task)
        ttk.Radiobutton(
            frame, text="Écrire dans la langue parlée", variable=self.task_var,
            value="transcribe",
        ).grid(row=8, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(
            frame, text="Traduire vers l'anglais", variable=self.task_var,
            value="translate",
        ).grid(row=9, column=0, columnspan=2, sticky="w", pady=(0, 14))

        # --- Résultat ---
        ttk.Label(frame, text="Résultat", font=("", 12, "bold")).grid(
            row=10, column=0, columnspan=2, sticky="w"
        )
        self.autopaste_var = tk.BooleanVar(value=self.config.ui.auto_paste)
        ttk.Checkbutton(
            frame,
            text="Coller directement à la position du curseur",
            variable=self.autopaste_var,
        ).grid(row=11, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Label(
            frame,
            text="Décoché, le texte est seulement copié dans le presse-papier.",
            foreground="gray",
        ).grid(row=12, column=0, columnspan=2, sticky="w", pady=(0, 14))

        # --- Sons ---
        ttk.Label(frame, text="Sons", font=("", 12, "bold")).grid(
            row=13, column=0, columnspan=2, sticky="w"
        )
        self.sounds_enabled_var = tk.BooleanVar(value=self.config.sounds.enabled)
        ttk.Checkbutton(
            frame, text="Signal sonore au début et à la fin",
            variable=self.sounds_enabled_var,
        ).grid(row=14, column=0, columnspan=2, sticky="w", pady=(4, 0))

        volume_row = ttk.Frame(frame)
        volume_row.grid(row=15, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Label(volume_row, text="Volume").pack(side=tk.LEFT)
        self.volume_var = tk.DoubleVar(value=self.config.sounds.volume * 100)
        ttk.Scale(
            volume_row, from_=0, to=100, orient=tk.HORIZONTAL,
            variable=self.volume_var, length=200,
            command=lambda _v: self._update_volume_label(),
        ).pack(side=tk.LEFT, padx=(10, 8))
        self.volume_label = ttk.Label(volume_row, text="", width=5)
        self.volume_label.pack(side=tk.LEFT)
        ttk.Button(volume_row, text="Écouter", command=self._preview_sound).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        self._update_volume_label()

        self._refresh_hotkey_display()

    def _update_volume_label(self) -> None:
        self.volume_label.configure(text=f"{int(self.volume_var.get())} %")

    def _preview_sound(self) -> None:
        """Joue le son au volume choisi. Sans ça on règle un volume à l'aveugle."""
        try:
            from audio_feedback import AudioFeedback

            cfg = load_config()
            cfg.sounds.enabled = True
            cfg.sounds.volume = float(self.volume_var.get()) / 100.0
            AudioFeedback(cfg).play_start()
        except Exception as exc:  # noqa: BLE001
            messagebox.showwarning("Son indisponible", str(exc))

    # --- Capture du raccourci ---------------------------------------------
    #
    # L'écoute clavier vit dans `hotkey_capture` et parle par une file
    # d'événements. Tout ce qui suit tourne sur le thread de tkinter, jamais
    # dans le callback pynput : Tk n'est pas thread-safe.

    def _current_combo(self) -> str:
        return self._combo.strip()

    def _swallow_key(self, _event: tk.Event) -> str | None:
        """Absorbe les touches pendant la capture.

        Sans ça, enregistrer Espace « cliquerait » le bouton qui a le focus,
        et enregistrer une lettre l'écrirait dans le champ voisin.
        """
        return "break" if self._capture is not None else None

    def _toggle_capture(self) -> None:
        if self._capture is not None:
            self._end_capture(None)
        else:
            self._start_capture()

    def _start_capture(self) -> None:
        try:
            capture = HotkeyCapture()
            capture.start()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror(
                "Capture impossible",
                f"{exc}\n\nVoxtral a besoin de l'autorisation « Saisie au "
                f"clavier » dans Réglages Système → Confidentialité et "
                f"sécurité.",
            )
            return

        self._capture = capture
        self._capture_saw_a_key = False
        self.capture_button.configure(text="Annuler")
        self.reset_hotkey_button.configure(state="disabled")
        self.hotkey_display.configure(text="Appuie sur les touches…")
        self.hotkey_hint.configure(
            text="Tiens la touche (ou la combinaison) puis relâche. Échap annule."
        )
        self.hotkey_warning.configure(text="")
        self.hotkey_display.focus_set()

        self._capture_timeout_id = self.root.after(
            CAPTURE_TIMEOUT_MS, self._on_capture_timeout
        )
        self.root.after(CAPTURE_POLL_MS, self._poll_capture)

    def _poll_capture(self) -> None:
        capture = self._capture
        if capture is None:
            return

        while True:
            try:
                event = capture.events.get_nowait()
            except queue.Empty:
                break

            self._capture_saw_a_key = True

            if event.outcome is Outcome.DONE:
                self._end_capture(event.combo)
                return
            if event.outcome is Outcome.CANCELLED:
                self._end_capture(None)
                return
            if event.outcome is Outcome.UNSUPPORTED:
                self.hotkey_warning.configure(
                    text="⚠ Cette touche ne peut pas servir de raccourci."
                )
            elif event.combo:
                self.hotkey_display.configure(
                    text=display_combo(event.combo, verbose=True)
                )

        self.root.after(CAPTURE_POLL_MS, self._poll_capture)

    def _on_capture_timeout(self) -> None:
        self._capture_timeout_id = None
        if self._capture is None:
            return

        # Aucun événement reçu du tout : le cas bénin est « l'utilisateur est
        # parti », mais c'est aussi la signature d'une autorisation « Saisie
        # au clavier » manquante — pynput n'échoue pas bruyamment dans ce cas,
        # il ne remonte simplement jamais rien. On nomme les deux.
        silent = not self._capture_saw_a_key
        self._end_capture(None)
        self.hotkey_hint.configure(
            text=(
                "Aucune touche reçue. Vérifie l'autorisation « Saisie au "
                "clavier » de Voxtral dans Réglages Système."
                if silent
                else "Capture annulée : trop de temps sans relâcher."
            )
        )

    def _end_capture(self, combo: str | None) -> None:
        """Arrête l'écoute et applique le raccourci (None = on garde l'ancien)."""
        if self._capture is not None:
            self._capture.stop()
            self._capture = None
        if self._capture_timeout_id is not None:
            self.root.after_cancel(self._capture_timeout_id)
            self._capture_timeout_id = None

        self.capture_button.configure(text="Modifier…")
        self.reset_hotkey_button.configure(state="normal")
        self.hotkey_hint.configure(text="")

        error = validate_combo(combo) if combo else None
        if combo and error is None:
            self._combo = combo

        self._refresh_hotkey_display()
        if error is not None:
            # Ne devrait pas arriver : la capture ne produit que des jetons
            # issus du vocabulaire. On le dit plutôt que de sauvegarder un
            # raccourci que le listener ne saura pas lire. Après le refresh,
            # sinon celui-ci écraserait l'avertissement.
            self.hotkey_warning.configure(text=f"⚠ {error}")

    def _reset_combo(self) -> None:
        """Revient au raccourci livré par défaut, sans avoir à le presser.

        Utile si le raccourci enregistré est devenu inaccessible — clavier
        externe débranché, touche cassée.
        """
        self._combo = Config().hotkey.combo
        self._refresh_hotkey_display()

    def _refresh_hotkey_display(self) -> None:
        self.hotkey_display.configure(
            text=display_combo(self._combo, verbose=True) or "aucun"
        )
        # La comparaison se fait sur la forme générique : macOS ne distingue
        # pas les côtés pour ses propres raccourcis, donc cmd_r+space entre en
        # conflit avec Spotlight tout autant que cmd+space.
        if generic_combo(self._combo) in KNOWN_SYSTEM_CONFLICTS:
            self.hotkey_warning.configure(
                text=f"⚠ {display_combo(self._combo)} est déjà utilisé par macOS."
            )
        else:
            self.hotkey_warning.configure(text="")

    # ------------------------------------------------------------------
    # Onglet Fichiers
    # ------------------------------------------------------------------

    def _build_files_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(frame, text="Fichiers")
        cfg = self.config.file_transcription

        ttk.Label(
            frame,
            text="Réglages de « Transcrire un fichier audio… » dans le menu.",
            foreground="gray",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

        ttk.Label(frame, text="Dossier des transcriptions", font=("", 12, "bold")).grid(
            row=1, column=0, columnspan=2, sticky="w"
        )
        self.output_dir_var = tk.StringVar(value=cfg.output_dir)
        ttk.Entry(frame, textvariable=self.output_dir_var, width=38).grid(
            row=2, column=0, sticky="w", pady=(4, 0)
        )
        ttk.Button(frame, text="Choisir…", command=self._pick_output_dir).grid(
            row=2, column=1, sticky="w", padx=(8, 0), pady=(4, 0)
        )
        ttk.Button(frame, text="Ouvrir", command=self._open_output_dir).grid(
            row=3, column=1, sticky="w", padx=(8, 0), pady=(4, 14)
        )

        self.timestamps_var = tk.BooleanVar(value=cfg.include_timestamps)
        ttk.Checkbutton(
            frame, text="Indiquer l'heure de chaque paragraphe — [00:12:34]",
            variable=self.timestamps_var,
        ).grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Label(
            frame,
            text="Pratique pour retrouver un passage dans un long enregistrement.",
            foreground="gray",
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(0, 14))

        self.diarization_var = tk.BooleanVar(value=cfg.diarization)
        ttk.Checkbutton(
            frame,
            text="Identifier qui parle — « Locuteur 1 : … »",
            variable=self.diarization_var,
            command=self._update_diarization_note,
        ).grid(row=6, column=0, columnspan=2, sticky="w")
        self.diarization_note = ttk.Label(frame, foreground="gray", justify=tk.LEFT)
        self.diarization_note.grid(
            row=7, column=0, columnspan=2, sticky="w", pady=(0, 14)
        )
        self._update_diarization_note()

        ttk.Label(frame, text="Refuser les fichiers de plus de").grid(
            row=8, column=0, sticky="w"
        )
        self.file_max_hours_var = tk.DoubleVar(value=cfg.max_duration_s / 3600.0)
        limit_row = ttk.Frame(frame)
        limit_row.grid(row=9, column=0, sticky="w", pady=(4, 0))
        ttk.Spinbox(
            limit_row, from_=0.5, to=12.0, increment=0.5,
            textvariable=self.file_max_hours_var, width=6,
        ).pack(side=tk.LEFT)
        ttk.Label(limit_row, text="heures").pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(
            frame,
            text="Garde-fou : évite de lancer un long traitement par erreur.",
            foreground="gray",
        ).grid(row=10, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def _update_diarization_note(self) -> None:
        """Dit ce que coûte l'option, et ce qui manque le cas échéant.

        L'identification des locuteurs repose sur un paquet optionnel et sur
        un modèle qui n'est pas livré avec l'app. Cocher la case sans rien
        dire, puis voir un .txt sans étiquettes, serait le pire des scénarios.
        """
        if not self.diarization_var.get():
            self.diarization_note.configure(text="")
            return

        import diarizer

        if not diarizer.is_available():
            self.diarization_note.configure(
                text="⚠ Nécessite un composant supplémentaire. Dans le Terminal :\n"
                     "   ~/.voxtral/venv/bin/pip install mlx-audio",
                foreground="#c25e00",
            )
            return

        self.diarization_note.configure(
            text="Jusqu'à 4 personnes. Premier usage : 225 Mo à télécharger.\n"
                 "Fonctionne bien sur des voix distinctes ; à vérifier sur tes\n"
                 "propres enregistrements avant de compter dessus.",
            foreground="gray",
        )

    def _pick_output_dir(self) -> None:
        chosen = filedialog.askdirectory(
            title="Dossier des transcriptions",
            initialdir=str(self.config.file_transcription.resolved_output_dir),
        )
        if chosen:
            self.output_dir_var.set(chosen)

    def _open_output_dir(self) -> None:
        """Ouvre le dossier dans le Finder — « où sont mes fichiers ? »."""
        import subprocess
        from pathlib import Path

        target = Path(self.output_dir_var.get()).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["open", str(target)])

    # ------------------------------------------------------------------
    # Onglet Modèles — choix + place occupée, au même endroit
    # ------------------------------------------------------------------

    def _build_models_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(frame, text="Modèles")

        self.model_var = tk.StringVar(value=self.config.model.name)
        self.file_model_var = tk.StringVar(
            value=self.config.file_transcription.model
        )

        row = self._build_model_section(
            frame, row=0, title="Pour la dictée",
            variable=self.model_var,
            models=list_available_models(USAGE_DICTATION),
        )
        row = self._build_model_section(
            frame, row=row, title="Pour les fichiers audio",
            variable=self.file_model_var,
            models=list_available_models(USAGE_FILES),
            note="Seuls ces modèles indiquent à quel moment chaque phrase est dite.",
        )

        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(
            row=row, column=0, sticky="ew", pady=12
        )
        self._storage_frame = ttk.Frame(frame)
        self._storage_frame.grid(row=row + 1, column=0, sticky="w")
        self._render_storage()

    def _build_model_section(
        self,
        parent: tk.Widget,
        row: int,
        title: str,
        variable: tk.StringVar,
        models: list[ModelInfo],
        note: str | None = None,
    ) -> int:
        ttk.Label(parent, text=title, font=("", 12, "bold")).grid(
            row=row, column=0, sticky="w", pady=(0, 4)
        )
        row += 1

        for m in models:
            downloaded = is_downloaded(m.repo_id, self.config.model.resolved_path)
            mark = "✓ sur ton Mac" if downloaded else f"⤓ {m.size_gb:.1f} Go à télécharger"
            ttk.Radiobutton(
                parent,
                text=f"{m.label}   —   {mark}\n     {m.description}",
                variable=variable,
                value=m.repo_id,
                command=self._render_storage,
            ).grid(row=row, column=0, sticky="w", pady=2)
            row += 1

        if note:
            ttk.Label(parent, text=note, foreground="gray").grid(
                row=row, column=0, sticky="w", pady=(0, 12)
            )
            row += 1
        return row

    def _render_storage(self) -> None:
        """(Re)dessine le bloc « place occupée ».

        Rappelé à chaque changement de sélection : un modèle qu'on vient de
        désélectionner devient supprimable immédiatement, sans avoir à fermer
        et rouvrir la fenêtre.
        """
        frame = self._storage_frame
        for child in frame.winfo_children():
            child.destroy()

        cached = scan_cached_models()
        in_use = models_in_use(
            self.model_var.get(),
            self.file_model_var.get(),
            diarization=bool(self.diarization_var.get()),
        )
        total = sum(c.size_bytes for c in cached)

        ttk.Label(
            frame, text=f"Place occupée sur le disque : {format_size(total)}",
            font=("", 12, "bold"),
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))

        if not cached:
            ttk.Label(
                frame, text="Aucun modèle téléchargé pour l'instant.",
                foreground="gray",
            ).grid(row=1, column=0, sticky="w")
            return

        for i, entry in enumerate(cached, start=1):
            used = entry.repo_id in in_use
            ttk.Label(frame, text=entry.size_str, width=9).grid(
                row=i, column=0, sticky="e", padx=(0, 10), pady=1
            )
            ttk.Label(
                frame, text=entry.label + ("   (en service)" if used else "")
            ).grid(row=i, column=1, sticky="w", pady=1)

            if used:
                ttk.Label(frame, text="", width=10).grid(row=i, column=2)
            else:
                ttk.Button(
                    frame, text="Supprimer", width=10,
                    command=lambda e=entry: self._delete_model(e),
                ).grid(row=i, column=2, padx=(15, 0))

        ttk.Label(
            frame,
            text=(
                "Un modèle « en service » sert à la dictée, aux fichiers ou à\n"
                "la traduction. Supprimer les autres est sans risque : ils se\n"
                "re-téléchargent si tu les resélectionnes."
            ),
            foreground="gray",
            justify=tk.LEFT,
        ).grid(row=len(cached) + 1, column=0, columnspan=3, sticky="w", pady=(8, 0))

    def _delete_model(self, entry) -> None:  # noqa: ANN001
        if not messagebox.askyesno(
            "Supprimer ce modèle ?",
            f"{entry.label}\n\n{entry.size_str} seront libérés. Le modèle se "
            f"re-téléchargera si tu le resélectionnes plus tard.",
        ):
            return
        try:
            freed = delete_cached_model(entry.repo_id)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Suppression impossible", str(exc))
            return
        messagebox.showinfo("Modèle supprimé", f"{format_size(freed)} libérés.")
        self._render_storage()

    # ------------------------------------------------------------------
    # Onglet Avancé
    # ------------------------------------------------------------------

    def _build_advanced_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(frame, text="Avancé")

        ttk.Label(
            frame,
            text="Ces réglages conviennent tels quels dans la quasi-totalité\n"
                 "des cas. À ne toucher qu'en cas de souci précis.",
            foreground="gray",
            justify=tk.LEFT,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 14))

        ttk.Label(frame, text="Fin de phrase conservée (millisecondes) :").grid(
            row=1, column=0, sticky="w"
        )
        self.tail_var = tk.IntVar(value=self.config.recording.tail_padding_ms)
        ttk.Spinbox(
            frame, from_=0, to=1500, increment=50,
            textvariable=self.tail_var, width=8,
        ).grid(row=1, column=1, sticky="w", padx=(10, 0))
        ttk.Label(
            frame,
            text="On continue d'enregistrer ce délai après que tu as relâché la\n"
                 "touche. Augmente si tes fins de phrase sont encore coupées.",
            foreground="gray",
            justify=tk.LEFT,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 14))

        ttk.Label(frame, text="Durée maximale d'une dictée (secondes) :").grid(
            row=3, column=0, sticky="w"
        )
        self.max_rec_var = tk.IntVar(value=self.config.recording.max_duration_s)
        ttk.Spinbox(
            frame, from_=30, to=1800, increment=30,
            textvariable=self.max_rec_var, width=8,
        ).grid(row=3, column=1, sticky="w", padx=(10, 0))
        ttk.Label(
            frame,
            text="Coupe-circuit si le relâchement de touche n'est jamais reçu.\n"
                 "L'audio est conservé dans ~/.voxtral/recordings/.",
            foreground="gray",
            justify=tk.LEFT,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(2, 14))

        ttk.Label(frame, text="Longueur maximale du texte (jetons) :").grid(
            row=5, column=0, sticky="w"
        )
        self.tokens_var = tk.IntVar(value=self.config.transcription.max_new_tokens)
        ttk.Spinbox(
            frame, from_=128, to=4096, increment=128,
            textvariable=self.tokens_var, width=8,
        ).grid(row=5, column=1, sticky="w", padx=(10, 0))
        ttk.Label(
            frame,
            text="1024 ≈ 5 minutes de texte. Pour un enregistrement long,\n"
                 "utilise « Transcrire un fichier audio… » plutôt que la dictée.",
            foreground="gray",
            justify=tk.LEFT,
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(2, 14))

        self.autocheck_var = tk.BooleanVar(value=self.config.updates.auto_check)
        ttk.Checkbutton(
            frame, text="Vérifier les mises à jour au démarrage",
            variable=self.autocheck_var,
        ).grid(row=7, column=0, columnspan=2, sticky="w")

    # ------------------------------------------------------------------
    # Onglet À propos
    # ------------------------------------------------------------------

    def _build_about_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(frame, text="À propos")

        ttk.Label(frame, text="Voxtral Dictée", font=("", 16, "bold")).pack(
            anchor="w"
        )
        ttk.Label(
            frame,
            text="Dictée vocale 100 % locale sur Apple Silicon.\n"
                 "Aucune donnée ne quitte ton Mac.",
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(6, 14))

        ttk.Label(frame, text="Aide-mémoire", font=("", 12, "bold")).pack(anchor="w")
        ttk.Label(
            frame,
            text=(
                "• Maintiens le raccourci, parle, relâche — le texte s'écrit.\n"
                "• Pour un enregistrement déjà fait : menu → Transcrire un\n"
                "  fichier audio…\n"
                "• Les réglages s'appliquent seuls en quelques secondes."
            ),
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(4, 14))

        self._link_button(frame, "Code source et documentation", REPO_URL)

    def _link_button(self, parent: tk.Widget, text: str, url: str) -> ttk.Button:
        button = ttk.Button(parent, text=text, command=lambda: webbrowser.open(url))
        button.pack(anchor="w", pady=2)
        return button

    # ------------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------------

    def _save(self) -> None:
        # Une capture encore ouverte ferait enregistrer l'ancien raccourci
        # sans le dire, et laisserait l'écouteur clavier en vie.
        self._end_capture(None)

        combo = self._current_combo()
        error = validate_combo(combo)
        if error is not None:
            messagebox.showerror(
                "Raccourci invalide",
                f"{error}\n\nExemples valides : alt+space, cmd+shift+h, f13, "
                f"ou une seule touche tenue.",
            )
            self.notebook.select(0)
            return

        # Proposer le téléchargement AVANT d'enregistrer : découvrir au premier
        # usage que l'app se fige plusieurs minutes sans explication est la
        # pire des surprises.
        for repo in (self.model_var.get(), self.file_model_var.get()):
            if not self._ensure_downloaded(repo):
                return

        cfg = self.config
        cfg.model.name = self.model_var.get()
        cfg.transcription.language = self.lang.get_code()
        cfg.transcription.task = self.task_var.get()
        cfg.transcription.max_new_tokens = int(self.tokens_var.get())
        cfg.hotkey.combo = combo
        cfg.sounds.enabled = bool(self.sounds_enabled_var.get())
        cfg.sounds.volume = float(self.volume_var.get()) / 100.0
        cfg.ui.auto_paste = bool(self.autopaste_var.get())
        cfg.updates.auto_check = bool(self.autocheck_var.get())
        cfg.recording.tail_padding_ms = int(self.tail_var.get())
        cfg.recording.max_duration_s = int(self.max_rec_var.get())
        cfg.file_transcription.model = self.file_model_var.get()
        cfg.file_transcription.output_dir = self.output_dir_var.get().strip()
        cfg.file_transcription.include_timestamps = bool(self.timestamps_var.get())
        cfg.file_transcription.diarization = bool(self.diarization_var.get())
        cfg.file_transcription.max_duration_s = int(
            float(self.file_max_hours_var.get()) * 3600
        )

        save_config(cfg)
        self.root.destroy()

    def _ensure_downloaded(self, repo_id: str) -> bool:
        """Propose de télécharger un modèle absent. False = annuler la sauvegarde.

        Répondre « Plus tard » enregistre quand même : le modèle se
        téléchargera au premier usage, simplement sans progression visible.
        """
        if is_downloaded(repo_id, self.config.model.resolved_path):
            return True
        model = find_model(repo_id)
        if model is None:
            return True

        answer = messagebox.askyesnocancel(
            "Télécharger ce modèle ?",
            f"{model.label} n'est pas encore sur ton Mac "
            f"({model.size_gb:.1f} Go).\n\n"
            f"Le télécharger maintenant ? Sinon il sera récupéré à sa première "
            f"utilisation, ce qui rendra cette première fois très lente.",
        )
        if answer is None:  # Annuler
            return False
        if not answer:  # Plus tard
            return True

        dialog = DownloadDialog(self.root, model)
        if not dialog.run():
            messagebox.showerror(
                "Téléchargement échoué",
                f"{dialog.error}\n\nVérifie ta connexion internet. Le réglage "
                f"n'a pas été enregistré.",
            )
            return False
        return True


def main() -> None:
    root = tk.Tk()
    SettingsWindow(root)
    # L'app menu bar est en mode « accessory » : sans ça la fenêtre s'ouvre
    # derrière celle qu'on était en train d'utiliser. `topmost` est relâché
    # aussitôt, pour ne pas coller la fenêtre par-dessus tout le reste.
    root.lift()
    root.attributes("-topmost", True)
    root.after(200, lambda: root.attributes("-topmost", False))
    root.focus_force()
    root.mainloop()


if __name__ == "__main__":
    main()
