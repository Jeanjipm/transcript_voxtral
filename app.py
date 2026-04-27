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

import faulthandler
import signal
import subprocess
import sys
import threading
import traceback
import warnings
from pathlib import Path

# Filtres warnings cosmétiques émis à chaque démarrage / shutdown :
# - huggingface_hub : "Please set a HF_TOKEN" — sans bénéfice quand le modèle
#   est en cache local (notre cas après la 1re install).
# - multiprocessing.resource_tracker : "leaked semaphore" — bug connu de
#   Python 3.13 quand des libs ML utilisent multiprocessing, sans impact
#   runtime. Polluait voxtral.log de plusieurs lignes par session.
warnings.filterwarnings("ignore", message=r".*HF_TOKEN.*")
warnings.filterwarnings(
    "ignore", message=r".*resource_tracker.*leaked semaphore.*"
)

import rumps
import soundfile as sf
from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSColor,
    NSImage,
    NSImageSymbolConfiguration,
    NSMakeSize,
)
from PyObjCTools import AppHelper

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


# Capture les crashs natifs (segfault MLX/pyobjc, OOM soft, etc.) en écrivant
# la stack C/Python dans stderr (→ voxtral.log). faulthandler est actif aussi
# longtemps que le process tourne.
faulthandler.enable(sys.stderr)


def _log_exception(exc_type, exc_value, exc_tb) -> None:
    print("[crash] uncaught exception:", file=sys.stderr)
    traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)
    sys.stderr.flush()


sys.excepthook = _log_exception
# threading.excepthook (Python 3.8+) attrape les exceptions dans les threads
# worker (recorder, transcriber) qui sinon disparaissent silencieusement.
threading.excepthook = lambda args: _log_exception(
    args.exc_type, args.exc_value, args.exc_traceback
)


def _log_signal(signum, frame) -> None:
    print(f"[crash] signal {signum} reçu, stack:", file=sys.stderr)
    traceback.print_stack(frame, file=sys.stderr)
    sys.stderr.flush()
    sys.exit(128 + signum)


# SIGKILL ne se catch pas (kernel), mais SIGTERM/SIGHUP (arrêt propre,
# logout, rotation logs) si. On logue avant de mourir.
for _sig in (signal.SIGTERM, signal.SIGHUP):
    signal.signal(_sig, _log_signal)


# Menu bar only, pas de Dock.
NSApplication.sharedApplication().setActivationPolicy_(
    NSApplicationActivationPolicyAccessory
)


APP_NAME = "Voxtral"
APP_VERSION = "0.1.0"

SYMBOL_IDLE = "mic.fill"
SYMBOL_RECORDING = "circle.fill"
# Sablier animé : alterne entre les 2 frames → sable qui coule.
SYMBOL_TRANSCRIBING_FRAMES = ("hourglass.tophalf.filled", "hourglass.bottomhalf.filled")
# Téléchargement : alternance de 2 SF Symbols de formes très distinctes
# (box+arrow ↔ arrow seule). Le rendu template macOS en menu bar aplatit
# les variantes `.fill` vs outline du même symbole → blink invisible.
# Prendre 2 glyphes à silhouette nettement différente rend l'alternance
# clairement lisible.
SYMBOL_DOWNLOADING_FRAMES = ("square.and.arrow.down.fill", "arrow.down")


class VoxtralApp(rumps.App):
    def __init__(self) -> None:
        # title=🎤 garantit une largeur > 0 au NSStatusItem au moment de sa
        # création par rumps. L'emoji est remplacé par le SF Symbol dans
        # _on_first_tick, une fois que self._nsapp existe.
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
            on_start=self._on_hotkey_start,
            on_stop=self._on_hotkey_stop,
        )
        self.hotkey.start()

        # bool + Lock pour check-and-set atomique sur 3 threads (start/stop/
        # transcribe) : un threading.Lock seul ne marche pas car release()
        # sans acquire lève RuntimeError.
        self._busy_lock = threading.Lock()
        self._busy = False

        # Animation icône menu bar (sablier, download). rumps.Timer tourne
        # sur le main thread → safe pour muter l'icône NSStatusItem.
        self._anim_timer: "rumps.Timer | None" = None
        self._anim_frames: tuple[str, ...] = ()
        self._anim_idx = 0

        # Hot-reload config : rumps.Timer exige le main thread pour toute
        # mutation de menu — un threading.Thread crasherait silencieusement.
        self._config_mtime = (
            USER_CONFIG_PATH.stat().st_mtime if USER_CONFIG_PATH.exists() else 0.0
        )
        self._config_timer = rumps.Timer(self._check_config_change, 2.0)
        self._config_timer.start()

        # Icône initiale posée après .run() (_nsapp n'existe pas avant).
        self._init_icon_timer = rumps.Timer(self._on_first_tick, 0.1)
        self._init_icon_timer.start()

    # ------------------------------------------------------------------
    # Hot-reload config
    # ------------------------------------------------------------------

    def _check_config_change(self, _sender: "rumps.Timer | None" = None) -> None:
        try:
            mtime = USER_CONFIG_PATH.stat().st_mtime
        except FileNotFoundError:
            return
        if mtime == self._config_mtime:
            return
        self._config_mtime = mtime
        # YAML invalide ou modèle introuvable : on veut logger mais pas tuer
        # le timer — sinon le hot-reload reste muet jusqu'au prochain redémarrage.
        try:
            self._reload_config()
        except Exception:
            traceback.print_exc()

    def _reload_config(self) -> None:
        old = self.config
        new_config = load_config()

        # Ops qui peuvent lever (model load, audio resources) d'abord, dans
        # des variables temporaires. Si une d'elles échoue, self.config reste
        # old et l'état global stable (pas de config/transcriber incohérents).
        if new_config.sounds != old.sounds:
            new_feedback = AudioFeedback(new_config)
        else:
            # Réutiliser l'instance préserve le cache NSSound pré-chargé.
            new_feedback = self.feedback
        if new_config.model.name != old.model.name:
            new_transcriber = make_transcriber(new_config)
        else:
            new_transcriber = self.transcriber

        # Swap atomique une fois que tout a réussi.
        self.config = new_config
        self.feedback = new_feedback
        self.transcriber = new_transcriber

        if new_config.hotkey.combo != old.hotkey.combo:
            self.hotkey.update_binding(new_config.hotkey.combo)
            self.hotkey_item.title = (
                f"Raccourci : {display_combo(new_config.hotkey.combo)}"
            )

        if new_config.model.name != old.model.name:
            self.model_item.title = f"Modèle : {self._model_label()}"

        if new_config.transcription.language != old.transcription.language:
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

    def _reset_idle(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            AppHelper.callAfter(self._reset_idle)
            return
        self._stop_animation()
        self._set_state(SYMBOL_IDLE, "État : prêt")
        self._end_busy()

    # ------------------------------------------------------------------
    # Animation icône menu bar
    # ------------------------------------------------------------------

    def _start_animation(self, frames: tuple[str, ...], interval: float) -> None:
        """Alterne l'icône de la menu bar entre `frames` toutes `interval` s.

        Idempotent sur les frames identiques (on compare à l'animation
        courante). On stoppe toute animation précédente pour éviter les
        timers orphelins.
        """
        if threading.current_thread() is not threading.main_thread():
            AppHelper.callAfter(self._start_animation, frames, interval)
            return
        if self._anim_frames == frames and self._anim_timer is not None:
            return
        self._stop_animation()
        self._anim_frames = frames
        self._anim_idx = 0
        # Pose la 1re frame immédiatement (sinon on voit l'icône idle
        # pendant `interval` avant que le timer ne tick).
        self._set_status_icon(frames[0])
        self._anim_timer = rumps.Timer(self._on_anim_tick, interval)
        self._anim_timer.start()

    def _stop_animation(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            AppHelper.callAfter(self._stop_animation)
            return
        if self._anim_timer is not None:
            self._anim_timer.stop()
            self._anim_timer = None
        self._anim_frames = ()
        self._anim_idx = 0

    def _on_anim_tick(self, _sender: "rumps.Timer | None" = None) -> None:
        if not self._anim_frames:
            return
        self._anim_idx = (self._anim_idx + 1) % len(self._anim_frames)
        self._set_status_icon(self._anim_frames[self._anim_idx])

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
        if not self.recorder.is_recording:
            # on_stop sans on_start effectif : busy était déjà pris par une
            # transcription concurrente, ou start() a levé et libéré busy.
            # Dans les deux cas on ne possède pas le flag — ne pas libérer.
            return

        try:
            wav_path = self.recorder.stop()

            # En dessous de 0.5s l'audio est surtout du silence + son Tink,
            # et mlx-voxtral hallucine une phrase ("Thank you"). Skip sans
            # notification.
            duration = sf.info(str(wav_path)).duration
            if duration < 0.5:
                try:
                    wav_path.unlink(missing_ok=True)
                except OSError:
                    pass
                self._reset_idle()
                return

            self.feedback.play_stop()
            self._start_animation(SYMBOL_TRANSCRIBING_FRAMES, 0.4)
            self._set_status_title("État : transcription…")
        except Exception:
            self._reset_idle()
            raise

        # Transcription dans un thread pour ne pas geler la menu bar
        thread = threading.Thread(
            target=self._transcribe_and_paste,
            args=(wav_path,),
            daemon=True,
        )
        thread.start()

    def _model_needs_download(self) -> bool:
        """True si le modèle courant n'est PAS dans le cache HF local.

        Permet d'afficher l'icône de téléchargement avant que
        `from_pretrained` / `mlx_whisper` ne bloquent pendant plusieurs
        minutes. Best-effort : si huggingface_hub est trop vieux pour
        exposer l'API, on suppose cached (pas d'animation → pas de faux
        signal).
        """
        try:
            from huggingface_hub import try_to_load_from_cache
        except ImportError:
            return False
        # `config.json` est présent dans à peu près tous les repos MLX /
        # transformers — sentinelle fiable pour "le repo est en cache".
        result = try_to_load_from_cache(
            repo_id=self.config.model.name, filename="config.json"
        )
        return not isinstance(result, (str, bytes))

    def _transcribe_and_paste(self, wav_path: Path) -> None:
        try:
            if self._model_needs_download():
                self._start_animation(SYMBOL_DOWNLOADING_FRAMES, 0.5)
                self._set_status_title("État : téléchargement du modèle…")
            text = self.transcriber.transcribe(
                wav_path,
                language=self.config.transcription.language,
                task=self.config.transcription.task,
                max_new_tokens=self.config.transcription.max_new_tokens,
            )
            paste_text(text, auto_paste=self.config.ui.auto_paste)
        except Exception as exc:
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
            self._reset_idle()

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
                "Dictée vocale 100 % locale via MLX sur Apple Silicon.\n"
                "Développé par Jeanjipm.\n\n"
                f"Modèle : {self.config.model.name}\n"
                f"Raccourci : {display_combo(self.config.hotkey.combo)}\n\n"
                "Aucune donnée ne quitte votre Mac."
            ),
        )

    def quit_app(self, _sender: rumps.MenuItem) -> None:
        self.hotkey.stop()
        # Libère proprement le stream micro (kept-warm entre les dictées,
        # cf. AudioRecorder.start). Sans ça on laisse fuiter le device
        # CoreAudio jusqu'à ce que macOS le récupère à terme.
        self.recorder.shutdown()
        rumps.quit_application()

    # ------------------------------------------------------------------
    # Helpers UI
    # ------------------------------------------------------------------

    def _set_state(self, symbol: str, status_text: str, red: bool = False) -> None:
        if threading.current_thread() is not threading.main_thread():
            AppHelper.callAfter(self._set_state, symbol, status_text, red)
            return
        # Poser une icône fixe annule toute animation en cours (sinon le
        # prochain tick du timer écraserait l'icône qu'on vient de poser).
        self._stop_animation()
        self._set_status_icon(symbol, red=red)
        self.status_item.title = status_text

    def _set_status_title(self, text: str) -> None:
        """Met à jour le titre de l'item 'État' depuis n'importe quel thread."""
        if threading.current_thread() is not threading.main_thread():
            AppHelper.callAfter(self._set_status_title, text)
            return
        self.status_item.title = text

    def _set_status_icon(self, symbol_name: str, red: bool = False) -> None:
        """Pose un SF Symbol sur le NSStatusItem de rumps.

        red=True : rouge fixe (non-template), visible en light et dark mode —
        utilisé pour l'état recording comme signal visuel fort.
        red=False : template, teinté auto par macOS selon le thème.
        """
        if threading.current_thread() is not threading.main_thread():
            AppHelper.callAfter(self._set_status_icon, symbol_name, red)
            return
        nsapp = getattr(self, "_nsapp", None)
        if nsapp is None:
            return
        img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(
            symbol_name, None
        )
        if img is None:
            return
        # pointSize 16 + setSize_(18×18) = métriques des icônes natives
        # (wifi, son, etc.). Sans ça, la NSImage d'un SF Symbol peut être
        # rendue à 0×0 et ne rien afficher.
        img = img.imageWithSymbolConfiguration_(
            NSImageSymbolConfiguration.configurationWithPointSize_weight_(16.0, 5)
        )
        if red:
            img = img.imageWithSymbolConfiguration_(
                NSImageSymbolConfiguration.configurationWithPaletteColors_(
                    [NSColor.systemRedColor()]
                )
            )
            img.setTemplate_(False)
        else:
            img.setTemplate_(True)
        img.setSize_(NSMakeSize(18, 18))

        # Padding horizontal : sans ça l'icône colle aux voisines de la
        # menu bar (heure, batterie…). On dessine l'icône dans un canvas
        # plus large avec 8 px transparents de chaque côté — largeur
        # retenue après itération UX.
        pad = 8
        canvas = NSImage.alloc().initWithSize_(NSMakeSize(18 + 2 * pad, 18))
        canvas.lockFocus()
        img.drawInRect_(((pad, 0), (18, 18)))
        canvas.unlockFocus()
        canvas.setTemplate_(img.isTemplate())

        btn = nsapp.nsstatusitem.button()
        if btn is not None:
            btn.setImage_(canvas)

    def _on_first_tick(self, _sender: "rumps.Timer | None" = None) -> None:
        self._init_icon_timer.stop()
        self._set_status_icon(SYMBOL_IDLE)
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
