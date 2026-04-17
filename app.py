"""
Point d'entrée Voxtral Dictée — app menu bar macOS via rumps.

Orchestration :
    raccourci pressé   → AudioFeedback.play_start + AudioRecorder.start
    raccourci relâché  → AudioRecorder.stop → Transcriber.transcribe
                       → AudioFeedback.play_stop → clipboard.paste_text

L'enregistrement et la transcription tournent dans des threads séparés
pour ne pas geler la menu bar.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import rumps
from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSColor,
    NSImage,
    NSImageSymbolConfiguration,
)

from audio_capture import AudioRecorder
from audio_feedback import AudioFeedback
from clipboard import paste_text
from config import (
    USER_CONFIG_PATH,
    Config,
    ensure_user_config_exists,
    load_config,
)
from hotkey_manager import HotkeyManager, display_combo
from model_manager import find_model
from transcriber import Transcriber, make_transcriber


# Menu bar only, pas de Dock. Équivalent pyobjc de LSUIElement=true dans
# Info.plist, mais qui marche AUSSI hors bundle (terminal, LaunchAgent,
# ou bundle dont l'identité est perdue après exec Python).
NSApplication.sharedApplication().setActivationPolicy_(
    NSApplicationActivationPolicyAccessory
)


APP_NAME = "Voxtral"
APP_VERSION = "0.1.0"

# SF Symbols affichés dans la menu bar (rendus via NSImage sur le
# NSStatusItem de rumps). Template monochrome pour idle/transcribing
# (teinte auto light/dark). Recording = rond rouge fixe, signal fort.
SYMBOL_IDLE = "mic.fill"
SYMBOL_RECORDING = "circle.fill"
SYMBOL_TRANSCRIBING = "hourglass"

# Fallback emoji par état : utilisé comme title rumps si le NSStatusItem
# ne peut pas afficher le SF Symbol (button pas prêt, image nil, etc.).
# Garantit qu'on voit TOUJOURS quelque chose dans la menu bar, au pire.
_SYMBOL_TO_EMOJI = {
    SYMBOL_IDLE: "🎤",
    SYMBOL_RECORDING: "🔴",
    SYMBOL_TRANSCRIBING: "⏳",
}


class VoxtralApp(rumps.App):
    def __init__(self) -> None:
        # title=emoji micro à l'init : le NSStatusItem de rumps est créé avec
        # NSVariableStatusItemLength (-1) — si ni title ni image à la
        # création, la largeur calculée reste 0 et l'item est INVISIBLE
        # (reproduit en cold-start bundle .app). L'emoji garantit width > 0
        # dès le début ; il sera remplacé par le SF Symbol dans _on_first_tick
        # dès que le button du status item est prêt.
        super().__init__(APP_NAME, title="🎤", quit_button=None)

        # 1) Config
        ensure_user_config_exists()
        self.config: Config = load_config()

        # 2) Composants audio + transcription
        self.recorder = AudioRecorder()
        self.feedback = AudioFeedback(self.config)
        self.transcriber: Transcriber = make_transcriber(self.config)

        # 3) Menu
        self.status_item = rumps.MenuItem("État : prêt")
        self.hotkey_item = rumps.MenuItem(
            f"Raccourci : {display_combo(self.config.hotkey.combo)}"
        )
        self.lang_item = rumps.MenuItem(
            f"Langue : {self._language_label()}"
        )
        self.model_item = rumps.MenuItem(f"Modèle : {self._model_label()}")

        self.menu = [
            self.status_item,
            self.hotkey_item,
            self.lang_item,
            self.model_item,
            None,  # séparateur
            rumps.MenuItem("Préférences…", callback=self.open_preferences),
            rumps.MenuItem(
                "Mettre à jour le modèle", callback=self.update_model
            ),
            None,
            rumps.MenuItem("À propos", callback=self.about),
            rumps.MenuItem("Quitter", callback=self.quit_app),
        ]

        # Désactiver la sélection des items purement informatifs
        for item in (self.status_item, self.hotkey_item, self.lang_item, self.model_item):
            item.set_callback(None)

        # 4) Raccourci global
        self.hotkey = HotkeyManager(
            combo=self.config.hotkey.combo,
            mode=self.config.hotkey.mode,
            on_start=self._on_hotkey_start,
            on_stop=self._on_hotkey_stop,
        )
        self.hotkey.start()

        # Flag "occupé" pour éviter qu'une 2e dictée démarre pendant qu'une
        # transcription tourne encore. On utilise un bool + Lock plutôt que
        # threading.Lock tout seul : release() d'un Lock non détenu lève
        # RuntimeError, ce qui est dangereux quand start/stop/transcribe
        # tournent potentiellement sur 3 threads différents.
        self._busy_lock = threading.Lock()
        self._busy = False

        # 5) Watcher hot-reload : quand settings_ui.py écrit une nouvelle
        # config utilisateur, on l'applique sans redémarrer l'app.
        # On utilise rumps.Timer (callback sur le thread principal) plutôt
        # qu'un threading.Thread : AppKit exige que les mises à jour de
        # menu se fassent sur le main thread, sinon crash silencieux.
        self._config_mtime = (
            USER_CONFIG_PATH.stat().st_mtime if USER_CONFIG_PATH.exists() else 0.0
        )
        self._config_timer = rumps.Timer(self._check_config_change, 2.0)
        self._config_timer.start()

        # 6) Timer one-shot pour poser l'icône initiale. On ne peut pas la
        # setter dans __init__ car rumps ne crée le NSStatusItem qu'à l'entrée
        # dans .run(). Un Timer qui tire 0.1s après le démarrage du main loop
        # garantit que self._nsapp.nsstatusitem existe.
        self._init_icon_timer = rumps.Timer(self._on_first_tick, 0.1)
        self._init_icon_timer.start()

    # ------------------------------------------------------------------
    # Hot-reload config
    # ------------------------------------------------------------------

    def _check_config_change(self, _sender: "rumps.Timer | None" = None) -> None:
        """Tick du timer : vérifie si ~/.voxtral/config.yaml a été modifié
        et déclenche un reload si oui. Tourne sur le main thread."""
        try:
            if not USER_CONFIG_PATH.exists():
                return
            mtime = USER_CONFIG_PATH.stat().st_mtime
            if mtime == self._config_mtime:
                return
            self._config_mtime = mtime
            self._reload_config()
        except Exception:
            import traceback
            traceback.print_exc()

    def _reload_config(self) -> None:
        """Relit la config et applique les changements nécessaires au
        runtime (hotkey, modèle, labels menu). Les autres champs (langue,
        sons, volume, temperature, etc.) sont lus à la demande via self.config
        donc il suffit de remplacer la référence."""
        print("[config] rechargement de ~/.voxtral/config.yaml", file=sys.stderr)
        new_config = load_config()

        # Raccourci : relancer le listener si combo ou mode a changé.
        if (
            new_config.hotkey.combo != self.config.hotkey.combo
            or new_config.hotkey.mode != self.config.hotkey.mode
        ):
            self.hotkey.update_binding(
                new_config.hotkey.combo, new_config.hotkey.mode
            )
            self.hotkey_item.title = (
                f"Raccourci : {display_combo(new_config.hotkey.combo)}"
            )

        # Modèle : recréer le transcriber (il rechargera le modèle au
        # prochain transcribe, pas tout de suite — pas de pause pour
        # l'utilisateur tant qu'il ne dicte pas).
        if new_config.model.name != self.config.model.name:
            self.transcriber = make_transcriber(new_config)
            self.model_item.title = (
                f"Modèle : {find_model(new_config.model.name).label if find_model(new_config.model.name) else new_config.model.name}"
            )

        # Feedback audio : recréer pour que le volume / theme / enabled
        # soient pris en compte immédiatement.
        self.feedback = AudioFeedback(new_config)

        self.config = new_config
        self.lang_item.title = f"Langue : {self._language_label()}"

    # ------------------------------------------------------------------
    # Gestion du flag "occupé"
    # ------------------------------------------------------------------

    def _try_begin_busy(self) -> bool:
        """Passe busy=True de façon atomique. Retourne False si déjà busy."""
        with self._busy_lock:
            if self._busy:
                return False
            self._busy = True
            return True

    def _end_busy(self) -> None:
        with self._busy_lock:
            self._busy = False

    # ------------------------------------------------------------------
    # Callbacks raccourci clavier
    # ------------------------------------------------------------------

    def _on_hotkey_start(self) -> None:
        if not self._try_begin_busy():
            return
        try:
            self.feedback.play_start()
            self.recorder.start()
            self._set_state(SYMBOL_RECORDING, "État : écoute en cours…", red=True)
        except Exception:
            self._end_busy()
            raise

    def _on_hotkey_stop(self) -> None:
        # Log diagnostic temporaire pour tracer le flow release → transcription.
        print(
            f"[hotkey] stop fired, is_recording={self.recorder.is_recording}",
            file=sys.stderr,
        )
        if not self.recorder.is_recording:
            # on_stop sans on_start effectif : soit busy était déjà pris
            # par une transcription en cours (ce press a été ignoré), soit
            # start() a levé et a déjà libéré busy. Dans les deux cas on
            # ne possède pas le flag — ne pas le libérer, sinon on écrase
            # une session concurrente.
            return

        try:
            wav_path = self.recorder.stop()
            print(f"[hotkey] wav written to {wav_path}", file=sys.stderr)

            # Skip les appuis quasi-instantanés : en-dessous d'un demi-seconde,
            # l'audio contient surtout le son Tink de feedback + silence, et
            # mlx-voxtral hallucine une phrase typique (ex. "Thank you"). On
            # supprime silencieusement — pas de Pop, pas de notification.
            import soundfile as sf
            duration = sf.info(str(wav_path)).duration
            if duration < 0.5:
                print(
                    f"[hotkey] wav trop court ({duration:.2f}s), skip transcription",
                    file=sys.stderr,
                )
                try:
                    wav_path.unlink(missing_ok=True)
                except OSError:
                    pass
                self._set_state(SYMBOL_IDLE, "État : prêt")
                self._end_busy()
                return

            self.feedback.play_stop()
            self._set_state(SYMBOL_TRANSCRIBING, "État : transcription…")
        except Exception:
            self._set_state(SYMBOL_IDLE, "État : prêt")
            self._end_busy()
            raise

        # Transcription dans un thread pour ne pas geler la menu bar
        thread = threading.Thread(
            target=self._transcribe_and_paste,
            args=(wav_path,),
            daemon=True,
        )
        thread.start()

    def _transcribe_and_paste(self, wav_path: Path) -> None:
        try:
            text = self.transcriber.transcribe(
                wav_path,
                language=self.config.transcription.language,
                task=self.config.transcription.task,
                temperature=self.config.transcription.temperature,
                max_new_tokens=self.config.transcription.max_new_tokens,
            )
            paste_text(text, auto_paste=self.config.ui.auto_paste)

            if self.config.ui.notification_on_paste:
                rumps.notification(
                    title=APP_NAME,
                    subtitle="Transcription collée",
                    message=text[:80] + ("…" if len(text) > 80 else ""),
                )
        except Exception as exc:
            # Log diagnostic temporaire : le popup rumps seul masquait la
            # cause des échecs silencieux. À retirer une fois le flow
            # dictée stable et les erreurs attendues bien gérées.
            import traceback
            traceback.print_exc()
            rumps.notification(
                title=APP_NAME,
                subtitle="Erreur de transcription",
                message=str(exc)[:200],
            )
        finally:
            try:
                wav_path.unlink(missing_ok=True)
            except OSError:
                pass
            self._set_state(SYMBOL_IDLE, "État : prêt")
            self._end_busy()

    # ------------------------------------------------------------------
    # Items de menu
    # ------------------------------------------------------------------

    def open_preferences(self, _sender: rumps.MenuItem) -> None:
        # On lance settings_ui.py dans un sous-processus : tkinter ne
        # cohabite pas bien avec la mainloop rumps (deux event loops Cocoa).
        # Sub-process = isolation simple et robuste.
        subprocess.Popen(
            [sys.executable, str(Path(__file__).parent / "settings_ui.py")],
        )

    def update_model(self, _sender: rumps.MenuItem) -> None:
        # Idem : on délègue à download_model.py en sous-processus pour
        # ne pas bloquer la menu bar pendant un téléchargement de 3 Go.
        subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).parent / "download_model.py"),
                "--model",
                self.config.model.name,
            ],
        )
        rumps.notification(
            title=APP_NAME,
            subtitle="Mise à jour du modèle",
            message=f"Téléchargement de {self.config.model.name} en cours…",
        )

    def about(self, _sender: rumps.MenuItem) -> None:
        rumps.alert(
            title=f"{APP_NAME} {APP_VERSION}",
            message=(
                "Dictée vocale 100% locale via MLX.\n"
                f"Modèle : {self.config.model.name}\n"
                f"Raccourci : {display_combo(self.config.hotkey.combo)}\n"
                f"Mode : {self.config.hotkey.mode}\n\n"
                "Aucune donnée ne quitte votre Mac."
            ),
        )

    def quit_app(self, _sender: rumps.MenuItem) -> None:
        self.hotkey.stop()
        rumps.quit_application()

    # ------------------------------------------------------------------
    # Helpers UI
    # ------------------------------------------------------------------

    def _set_state(
        self, symbol: str, status_text: str, red: bool = False
    ) -> None:
        # Tente le SF Symbol ; si échec, fallback sur l'emoji correspondant.
        # Sans ce fallback, on se retrouve avec rien dans la menu bar quand
        # le bundle démarre à froid (bug observé).
        ok = self._set_status_icon(symbol, red=red)
        if ok:
            self.title = ""
        else:
            self.title = _SYMBOL_TO_EMOJI.get(symbol, "🎤")
        self.status_item.title = status_text

    def _set_status_icon(self, symbol_name: str, red: bool = False) -> bool:
        """Pose un SF Symbol dans la menu bar via le NSStatusItem de rumps.

        Retourne True si l'image a bien été posée, False sinon (l'appelant
        doit alors afficher un fallback, ex. emoji dans self.title).

        red=True : teinte rouge fixe (non-template). Utilisé pour l'état
        recording — garde un rond rouge visible en light et dark mode.
        red=False : image template → macOS teinte auto (blanc en dark mode,
        noir en light mode)."""
        nsapp = getattr(self, "_nsapp", None)
        if nsapp is None:
            # Main loop rumps pas encore démarré (appel depuis __init__ avant
            # .run()). Icône initiale posée plus tard par _on_first_tick.
            return False
        img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(
            symbol_name, None
        )
        if img is None:
            print(
                f"[statusbar] SF Symbol '{symbol_name}' nil — fallback emoji",
                file=sys.stderr, flush=True,
            )
            return False
        if red:
            config = NSImageSymbolConfiguration.configurationWithPaletteColors_(
                [NSColor.systemRedColor()]
            )
            img = img.imageWithSymbolConfiguration_(config)
            img.setTemplate_(False)
        else:
            img.setTemplate_(True)
        btn = nsapp.nsstatusitem.button()
        if btn is None:
            return False
        btn.setImage_(img)
        # Force l'item à être visible — contourne le cas bundle où l'item
        # existe mais a été masqué/collapsed par AppKit au cold-start.
        try:
            nsapp.nsstatusitem.setVisible_(True)
        except Exception:
            pass
        return True

    def _on_first_tick(self, _sender: "rumps.Timer | None" = None) -> None:
        """Premier tick du timer one-shot : pose l'icône initiale. Si le
        button du NSStatusItem n'est pas encore prêt (cold-start bundle),
        on replanifie une fois 0.3s plus tard."""
        self._init_icon_timer.stop()
        nsapp = getattr(self, "_nsapp", None)
        si = nsapp.nsstatusitem if nsapp else None
        btn = si.button() if si else None
        print(
            f"[statusbar] first_tick nsapp={nsapp is not None} "
            f"statusitem={si is not None} button={btn is not None} "
            f"length={si.length() if si else 'n/a'}",
            file=sys.stderr, flush=True,
        )
        if btn is None:
            # Button pas prêt — replanifie une seule fois 0.3s plus tard.
            if getattr(self, "_icon_retry_done", False):
                print("[statusbar] button still None after retry — emoji fallback", file=sys.stderr, flush=True)
                return
            self._icon_retry_done = True
            self._init_icon_timer = rumps.Timer(self._on_first_tick, 0.3)
            self._init_icon_timer.start()
            return
        # Button prêt : tenter le SF Symbol. Si succès, clear emoji title.
        if self._set_status_icon(SYMBOL_IDLE):
            self.title = ""

    def _language_label(self) -> str:
        lang = self.config.transcription.language
        labels = {
            "auto": "Auto",
            "fr": "🇫🇷 Français",
            "en": "🇬🇧 English",
            "de": "🇩🇪 Deutsch",
            "es": "🇪🇸 Español",
            "it": "🇮🇹 Italiano",
            "pt": "🇵🇹 Português",
            "nl": "🇳🇱 Nederlands",
        }
        return labels.get(lang, lang)

    def _model_label(self) -> str:
        info = find_model(self.config.model.name)
        return info.label if info else self.config.model.name


def main() -> None:
    VoxtralApp().run()


if __name__ == "__main__":
    main()
