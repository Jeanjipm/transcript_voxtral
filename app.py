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
import os
import signal
import subprocess
import sys
import threading
import time
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

import hf_offline

# Doit être appelé AVANT tout import qui tire huggingface_hub
# (model_manager, transcriber) : la constante HF_HUB_OFFLINE de hf_hub est
# figée à son import, donc poser la variable d'environnement après coup
# serait sans effet sur cette constante.
hf_offline.install_early()

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
import file_picker
import file_transcriber
from dictation_controller import (
    DictationCallbacks,
    DictationController,
    State as DictationState,
)
from file_job import FileJob, FileJobCallbacks, JobResult, JobState
from hotkey_manager import HotkeyManager, display_combo
from inference_worker import (
    PRIORITY_DICTATION,
    PRIORITY_FILE,
    PRIORITY_MODEL,
    InferenceWorker,
)
from model_manager import find_model
from transcriber import Transcriber, make_transcriber
import updater


# Capture les crashs natifs (segfault MLX/pyobjc, OOM soft, etc.) en écrivant
# la stack C/Python dans stderr (→ voxtral.log). faulthandler est actif aussi
# longtemps que le process tourne.
try:
    faulthandler.enable(sys.stderr)
except (RuntimeError, ValueError, AttributeError):
    # Idem plus bas : sans fileno() réel sur stderr, faulthandler refuse.
    # Pas bloquant, on perd juste les stacks natives.
    pass


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


# SIGTERM/SIGHUP : on veut logger la stack avant de mourir, MAIS sans jamais
# empêcher le process de mourir.
#
# Pourquoi pas signal.signal() (ce qu'on faisait avant) : un handler Python
# ne s'exécute qu'à une frontière de bytecode DU THREAD PRINCIPAL. Or notre
# main thread passe sa vie dans NSApp().run(), bloqué dans mach_msg2_trap :
# le handler ne tournait donc jamais, alors qu'il avait déjà remplacé
# l'action par défaut du signal. Résultat : `kill` et « Quitter » étaient
# sans effet, seul « Forcer à quitter » (SIGKILL) fonctionnait.
#
# faulthandler.register pose un handler en C : il écrit immédiatement la
# stack de tous les threads sans attendre l'interpréteur, puis chain=True
# rétablit l'action précédente (ici SIG_DFL) et se re-signale → le process
# meurt vraiment. On garde le diagnostic, on retrouve la tuabilité.
for _sig in (signal.SIGTERM, signal.SIGHUP):
    try:
        faulthandler.register(_sig, file=sys.stderr, all_threads=True, chain=True)
    except (RuntimeError, ValueError, AttributeError):
        # stderr sans fileno() réel (lancement sans redirection) : on laisse
        # l'action par défaut, qui tue le process — c'est le comportement sûr.
        pass


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
SYMBOL_ERROR = "exclamationmark.triangle.fill"
# Transcription de fichier : silhouettes nettement différentes du sablier de
# dictée, pour qu'on distingue les deux d'un coup d'œil.
SYMBOL_FILE_FRAMES = ("doc.text", "doc.text.fill")

FILE_MENU_LABEL = "Transcrire un fichier audio…"
FILE_CANCEL_LABEL = "Annuler la transcription"

# Libellés de l'item d'erreur. Une chaîne vide rendrait l'item invisible mais
# laisserait un séparateur bancal ; un tiret discret indique « rien à voir ».
ERROR_LABEL_NONE = "—"
ERROR_LABEL_PRESENT = "⚠︎ Dernière erreur…"

# Durée du flash rouge de l'icône quand une erreur survient. Assez long pour
# être vu, assez court pour ne pas laisser un état alarmant en permanence.
ERROR_FLASH_S = 4.0

# Labels du menu item "mises à jour" — on bascule entre les 2 selon
# qu'un check au démarrage a détecté une MAJ ou non. macOS bloque les
# rumps.notification pour les apps non-signées, donc le seul moyen de
# signaler la MAJ est le menu lui-même.
UPDATES_LABEL_DEFAULT = "Vérifier les mises à jour…"
UPDATES_LABEL_AVAILABLE = "Mises à jour disponibles…"
UPDATES_LABEL_CHECKING = "Vérification en cours…"

# Délai avant le check au démarrage. Laisse à l'app le temps de finir
# son init (chargement modèle, prewarm audio en threads) avant d'ajouter
# une charge réseau supplémentaire.
UPDATE_CHECK_STARTUP_DELAY = 10.0


class VoxtralApp(rumps.App):
    def __init__(self) -> None:
        # title=🎤 garantit une largeur > 0 au NSStatusItem au moment de sa
        # création par rumps. L'emoji est remplacé par le SF Symbol dans
        # _on_first_tick, une fois que self._nsapp existe.
        super().__init__(APP_NAME, title="🎤", quit_button=None)

        # 1) Config
        ensure_user_config_exists()
        self.config: Config = load_config()

        # 1b) Mode hors-ligne HuggingFace, maintenant qu'on connaît le modèle.
        # Si le modèle est en cache, plus aucune requête réseau ne part au
        # chargement ; sinon on laisse le réseau pour le téléchargement.
        hf_offline.refresh(
            self.config.model.name, prefer_offline=self.config.offline.prefer_offline
        )

        # 2) Composants audio + transcription
        self.recorder = AudioRecorder(
            start_retries=self.config.recording.start_retries
        )
        self.feedback = AudioFeedback(self.config)
        self.transcriber: Transcriber = make_transcriber(self.config)

        # 2b) Les deux workers permanents. Cf. dictation_controller.py pour le
        # contrat de threads complet ; en résumé :
        # - inference-worker : SEUL propriétaire des modèles MLX
        # - dictation-worker : SEUL pilote du micro, et destinataire des
        #   commandes déposées par le callback de l'event tap clavier
        self.inference = InferenceWorker()
        self.inference.start()

        self.dictation = DictationController(
            recorder=self.recorder,
            feedback=self.feedback,
            callbacks=DictationCallbacks(
                on_state_change=self._on_dictation_state,
                submit_transcription=self._submit_transcription,
                on_error=self._show_error,
                on_rearm_needed=self._rearm_hotkey,
                on_recording_kept=self._on_recording_kept,
            ),
            max_duration_s=self.config.recording.max_duration_s,
        )
        self.dictation.start()

        # 2c) Pré-chauffage, pour que la 1re dictée soit aussi rapide que les
        # suivantes. Les deux passent par les files : ouvrir le micro coûte
        # jusqu'à ~4 s device froid, et charger le modèle plusieurs secondes.
        # Aucun thread ad hoc — c'est justement ce qui créait des courses avec
        # la dictée (swap de transcriber, double chargement du modèle).
        self.dictation.request_prewarm()
        self.inference.submit(
            self._preload_model, priority=PRIORITY_MODEL, label="preload"
        )

        # 2c) State pour le système de mise à jour. _update_info contient
        # la dernière UpdateInfo connue (None si pas de MAJ ou pas encore
        # checké). _update_lock évite les checks concurrents (double-clic
        # sur le menu).
        self._update_info: "updater.UpdateInfo | None" = None
        self._update_check_in_progress = False
        self._update_lock = threading.Lock()
        if self.config.updates.auto_check:
            threading.Thread(
                target=self._safe_check_for_update_startup,
                daemon=True,
                name="update-check-startup",
            ).start()

        # 3) Menu
        self.status_item = rumps.MenuItem("État : prêt")
        self.hotkey_item = rumps.MenuItem(
            f"Raccourci : {display_combo(self.config.hotkey.combo)}"
        )
        self.lang_item = rumps.MenuItem(
            f"Langue : {self._language_label()}"
        )
        self.model_item = rumps.MenuItem(f"Modèle : {self._model_label()}")
        # Garde la référence : on modifie le titre quand une MAJ est
        # détectée (cf. _mark_update_available).
        self.updates_item = rumps.MenuItem(
            UPDATES_LABEL_DEFAULT, callback=self.check_for_updates_manual
        )
        # Item d'erreur : masqué (titre vide + pas de callback) tant qu'il n'y
        # a rien à signaler. macOS n'affiche pas les bannières de notification
        # d'une app non signée, donc le menu est le seul canal fiable — c'est
        # le même constat que pour les mises à jour.
        self.error_item = rumps.MenuItem(ERROR_LABEL_NONE)
        self._last_error: tuple[str, str] | None = None

        # Transcription de fichiers. Les items restent en place en permanence
        # plutôt que d'être ajoutés et retirés à chaud : muter le menu rumps
        # pendant l'exécution est plus fragile que changer un titre, et un menu
        # de hauteur stable est moins déroutant.
        #
        # La progression s'affiche DANS cet item plutôt que dans une ligne
        # dédiée — attention, rumps indexe les items par leur titre, donc deux
        # items partageant un même titre de repos s'écrasent silencieusement.
        self.file_item = rumps.MenuItem(
            FILE_MENU_LABEL, callback=self.transcribe_file_dialog
        )
        self.file_cancel_item = rumps.MenuItem(FILE_CANCEL_LABEL)

        self.menu = [
            self.status_item,
            self.hotkey_item,
            self.lang_item,
            self.model_item,
            self.error_item,
            None,  # séparateur
            self.file_item,
            self.file_cancel_item,
            None,
            rumps.MenuItem("Préférences…", callback=self.open_preferences),
            self.updates_item,
            None,
            rumps.MenuItem("À propos", callback=self.about),
            rumps.MenuItem("Quitter", callback=self.quit_app),
        ]

        # Désactiver la sélection des items purement informatifs
        for item in (
            self.status_item, self.hotkey_item, self.lang_item,
            self.model_item, self.error_item, self.file_cancel_item,
        ):
            item.set_callback(None)

        self.file_job = FileJob(
            callbacks=FileJobCallbacks(
                on_progress=self._on_file_progress,
                on_done=self._on_file_done,
            ),
            run_transcription=self._run_file_transcription,
        )
        # Transcriber dédié aux fichiers, construit à la demande : il charge un
        # second modèle (~3 Go) qu'on ne veut pas payer si la fonctionnalité
        # n'est jamais utilisée.
        self._file_transcriber: "Transcriber | None" = None

        # 4) Raccourci global. on_start/on_stop tournent DANS le callback de
        # l'event tap macOS : ils ne font qu'une mise en file (cf. la docstring
        # de dictation_controller).
        self.hotkey = HotkeyManager(
            combo=self.config.hotkey.combo,
            on_start=self.dictation.on_hotkey_start,
            on_stop=self.dictation.on_hotkey_stop,
        )
        self.hotkey.start()

        # Animation icône menu bar (sablier, download). rumps.Timer tourne
        # sur le main thread → safe pour muter l'icône NSStatusItem.
        self._anim_timer: "rumps.Timer | None" = None
        self._anim_frames: tuple[str, ...] = ()
        self._anim_idx = 0
        self._error_flash_timer: "rumps.Timer | None" = None

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
    # Tâches exécutées sur l'inference-worker
    # ------------------------------------------------------------------

    def _preload_model(self) -> None:
        """Charge le modèle. Tourne sur l'inference-worker.

        Un échec est loggué mais pas remonté à l'utilisateur : la dictée
        suivante retentera le chargement et affichera l'erreur à ce
        moment-là, quand elle est actionnable.
        """
        try:
            self.transcriber.preload()
        except Exception:
            traceback.print_exc()

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
            # Ré-évaluer AVANT de construire le transcriber : si l'utilisateur
            # vient de choisir un modèle pas encore téléchargé, il faut
            # re-autoriser le réseau, sinon le téléchargement est bloqué.
            hf_offline.refresh(
                new_config.model.name,
                prefer_offline=new_config.offline.prefer_offline,
            )
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
            # Pré-charge le nouveau modèle pour que la 1re dictée
            # post-changement soit instantanée. Via l'inference-worker : c'est
            # lui qui possède les modèles, donc aucun risque de charger 5 Go
            # deux fois ni de doubler l'usage mémoire.
            self.inference.submit(
                self._preload_model,
                priority=PRIORITY_MODEL,
                label="preload-after-reload",
            )

        if new_config.transcription.language != old.transcription.language:
            self.lang_item.title = f"Langue : {self._language_label()}"

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
    # Réactions aux événements de la dictée
    #
    # Tous ces callbacks sont invoqués depuis `dictation-worker`, jamais
    # depuis le callback de l'event tap clavier — c'est tout l'intérêt du
    # DictationController. Ceux qui touchent Cocoa repassent sur le main
    # thread via le garde habituel.
    # ------------------------------------------------------------------

    def _on_dictation_state(self, state: DictationState, label: str) -> None:
        """Reflète l'état de la dictée dans la menu bar."""
        if state is DictationState.RECORDING:
            self._set_state(SYMBOL_RECORDING, label, red=True)
        elif state is DictationState.PENDING:
            self._start_animation(SYMBOL_TRANSCRIBING_FRAMES, 0.4)
            self._set_status_title(label)
        else:
            self._set_state(SYMBOL_IDLE, label)

    def _submit_transcription(self, wav_path: Path) -> None:
        """Met la transcription en file sur l'inference-worker (priorité haute)."""
        self.inference.submit(
            lambda: self._transcribe_and_paste(wav_path),
            priority=PRIORITY_DICTATION,
            label="dictation",
        )

    def _rearm_hotkey(self) -> None:
        """Reconstruit le listener clavier après un blocage suspecté.

        Appelé depuis `dictation-worker`, jamais depuis le thread du listener
        (`HotkeyManager.stop` refuserait, cf. son garde).
        """
        try:
            self.hotkey.rearm()
        except Exception:
            traceback.print_exc()

    def _on_recording_kept(self, wav_path: Path) -> None:
        """Un enregistrement coupé pour dépassement a été conservé.

        On se contente de le tracer : l'utilisateur a déjà l'information et le
        chemin via `_show_error`, et ouvrir une fenêtre à ce moment-là serait
        intrusif (il est probablement en train de taper).
        """
        print(f"[app] enregistrement conservé : {wav_path}", file=sys.stderr)

    def _model_needs_download(self) -> bool:
        """True si le modèle courant n'est PAS dans le cache HF local.

        Permet d'afficher l'icône de téléchargement avant que
        `from_pretrained` / `mlx_whisper` ne bloquent pendant plusieurs
        minutes. Best-effort : en cas de doute on suppose « en cache »
        (pas d'animation → pas de faux signal).
        """
        return not hf_offline.is_model_cached(self.config.model.name)

    def _transcribe_and_paste(self, wav_path: Path) -> None:
        """Transcrit puis colle. Tourne sur l'inference-worker.

        Ce thread possède le modèle MLX ; il peut aussi appeler `paste_text`,
        qui contient 0,5 s de pauses délibérées et injecte des CGEvents —
        deux choses qu'on ne veut surtout pas sur le main thread.
        """
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
            traceback.print_exc()
            self._show_error("Erreur de transcription", str(exc)[:400])
        finally:
            try:
                wav_path.unlink(missing_ok=True)
            except OSError:
                pass
            # Rend la main à la machine à états, qui remet l'icône au repos.
            self.dictation.notify_transcription_done()

    # ------------------------------------------------------------------
    # Transcription d'un fichier audio
    # ------------------------------------------------------------------

    def transcribe_file_dialog(self, _sender: rumps.MenuItem) -> None:
        """Callback du menu. Main thread (NSOpenPanel l'exige)."""
        if self.file_job.is_running:
            rumps.alert(
                title=APP_NAME,
                message=(
                    "Une transcription de fichier est déjà en cours.\n\n"
                    "Attends la fin, ou annule-la depuis le menu."
                ),
            )
            return

        source = file_picker.choose_audio_file()
        if source is None:
            return

        cfg = self.config.file_transcription
        started = self.file_job.submit(
            source=source,
            transcriber=self._get_file_transcriber(),
            model_name=cfg.model,
            output_dir=cfg.resolved_output_dir,
            block_duration_s=cfg.block_duration_s,
            max_duration_s=cfg.max_duration_s,
            language=self.config.transcription.language,
            task=self.config.transcription.task,
            include_timestamps=cfg.include_timestamps,
        )
        if not started:
            return

        # Pendant le job, l'item devient l'affichage de progression et cesse
        # d'être cliquable : on ne peut pas lancer un 2e job de toute façon.
        self.file_item.set_callback(None)
        self.file_item.title = f"Transcription : {source.name}"
        self.file_cancel_item.set_callback(self.cancel_file_transcription)
        self._start_animation(SYMBOL_FILE_FRAMES, 0.6)

    def cancel_file_transcription(self, _sender: rumps.MenuItem) -> None:
        """Annule le job. Le texte déjà transcrit sera conservé."""
        self.file_job.cancel()
        self.file_item.title = "Annulation en cours…"

    def _get_file_transcriber(self) -> Transcriber:
        """Transcriber dédié aux fichiers, créé au premier usage.

        Si le modèle configuré pour les fichiers est le même que celui des
        dictées, on réutilise l'instance : inutile de charger deux fois les
        mêmes poids.
        """
        cfg = self.config.file_transcription
        if cfg.model == self.config.model.name:
            return self.transcriber
        if self._file_transcriber is None:
            hf_offline.refresh(
                cfg.model, prefer_offline=self.config.offline.prefer_offline
            )
            file_config = load_config()
            file_config.model.name = cfg.model
            self._file_transcriber = make_transcriber(file_config)
        return self._file_transcriber

    def _run_file_transcription(self, work):  # noqa: ANN001, ANN202
        """Exécute la transcription sur l'inference-worker, en priorité basse.

        Appelé depuis le thread `file-job`, qui attend ici le résultat. La
        priorité basse est ce qui permet à une dictée de passer devant : elle
        n'attend au pire que la fin du bloc en cours.
        """
        handle = self.inference.submit(work, priority=PRIORITY_FILE, label="file")
        # Pas de timeout : un fichier de 4 h peut légitimement prendre
        # longtemps, et l'annulation est le mécanisme prévu pour l'interrompre.
        return handle.result()

    def _on_file_progress(self, current_s: float, total_s: float) -> None:
        """Progression. Appelé depuis le thread `file-job`."""
        pct = int(100 * current_s / total_s) if total_s > 0 else 0
        label = (
            f"Transcription : {pct} % — "
            f"{file_transcriber.format_duration(current_s)} / "
            f"{file_transcriber.format_duration(total_s)}"
        )
        self._set_menu_title(self.file_item, label)

    def _on_file_done(self, result: JobResult) -> None:
        """Fin du job. Appelé depuis le thread `file-job`."""
        if threading.current_thread() is not threading.main_thread():
            AppHelper.callAfter(self._on_file_done, result)
            return

        self._stop_animation()
        self._set_status_icon(SYMBOL_IDLE)
        self.file_item.title = FILE_MENU_LABEL
        self.file_item.set_callback(self.transcribe_file_dialog)
        self.file_cancel_item.set_callback(None)

        if result.state is JobState.FAILED:
            # Action lancée par l'utilisateur : une fenêtre est attendue ici,
            # contrairement aux erreurs de dictée.
            self._show_error("Transcription du fichier échouée", result.error or "")
            rumps.alert(
                title="Transcription impossible",
                message=result.error or "Erreur inconnue.",
            )
            return

        if result.output_path is None:
            return

        if result.state is JobState.CANCELLED:
            rumps.alert(
                title="Transcription interrompue",
                message=(
                    f"Le texte déjà transcrit a été conservé dans :\n"
                    f"{result.output_path.name}"
                ),
            )
        else:
            rumps.alert(
                title="Transcription terminée",
                message=f"Fichier écrit :\n{result.output_path.name}",
            )
        file_picker.reveal_in_finder(result.output_path)

    def _set_menu_title(self, item: rumps.MenuItem, title: str) -> None:
        """Change le titre d'un item de menu depuis n'importe quel thread."""
        if threading.current_thread() is not threading.main_thread():
            AppHelper.callAfter(self._set_menu_title, item, title)
            return
        item.title = title

    # ------------------------------------------------------------------
    # Erreurs visibles (cause 5)
    # ------------------------------------------------------------------

    def _show_error(self, subtitle: str, message: str) -> None:
        """Rend une erreur visible dans la menu bar.

        Pourquoi pas `rumps.notification` : macOS n'affiche pas les bannières
        des apps non signées, donc l'ancien appel ne montrait RIEN — toute
        erreur de transcription était totalement silencieuse. Et il était fait
        depuis un thread secondaire, ce qui touche Cocoa hors main thread
        (même famille de bug que la PR #8).

        Pas de fenêtre modale ici : une erreur de dictée survient pendant que
        l'utilisateur tape, et lui voler le focus serait pire que le problème.
        L'item de menu reste consultable à son rythme.
        """
        if threading.current_thread() is not threading.main_thread():
            AppHelper.callAfter(self._show_error, subtitle, message)
            return

        self._last_error = (subtitle, message)
        self.error_item.title = ERROR_LABEL_PRESENT
        self.error_item.set_callback(self._show_last_error)

        # Flash rouge : signale l'erreur même menu fermé, puis on revient à
        # l'icône au repos pour ne pas laisser un état alarmant en permanence.
        self._set_status_icon(SYMBOL_ERROR, red=True)
        if self._error_flash_timer is not None:
            self._error_flash_timer.stop()
        self._error_flash_timer = rumps.Timer(self._end_error_flash, ERROR_FLASH_S)
        self._error_flash_timer.start()

    def _end_error_flash(self, _sender: "rumps.Timer | None" = None) -> None:
        if self._error_flash_timer is not None:
            self._error_flash_timer.stop()
            self._error_flash_timer = None
        # Ne pas écraser une animation en cours (une nouvelle dictée a pu
        # démarrer pendant le flash).
        if self._anim_timer is None:
            self._set_status_icon(SYMBOL_IDLE)

    def _show_last_error(self, _sender: rumps.MenuItem) -> None:
        """Affiche le détail. Déclenché par un clic, donc la modale est voulue."""
        if self._last_error is None:
            return
        subtitle, message = self._last_error
        rumps.alert(title=f"{APP_NAME} — {subtitle}", message=message)
        # Consultée : on remet l'item en veille.
        self._last_error = None
        self.error_item.title = ERROR_LABEL_NONE
        self.error_item.set_callback(None)

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

    # ------------------------------------------------------------------
    # Mises à jour de l'app (cf. updater.py)
    # ------------------------------------------------------------------

    def _safe_check_for_update_startup(self) -> None:
        """Check silencieux au démarrage. Tourne dans un thread daemon.

        Délai initial pour ne pas concurrencer le boot de l'app
        (chargement modèle, prewarm audio…). Si MAJ détectée, on remonte
        sur le main thread pour modifier le label du menu — la mutation
        UI Cocoa exige le main thread (cf. PR #8).
        """
        try:
            time.sleep(UPDATE_CHECK_STARTUP_DELAY)
            info = updater.check_for_update()
            if info is not None:
                AppHelper.callAfter(self._mark_update_available, info)
        except Exception:
            traceback.print_exc()

    def _mark_update_available(self, info: "updater.UpdateInfo") -> None:
        """Sur main thread : signale la MAJ via le label du menu.

        macOS bloque rumps.notification pour les apps non-signées (cf.
        commit 2e24576), donc on ne peut pas montrer une bannière
        système — le label menu reste le seul canal de feedback fiable.
        """
        self._update_info = info
        self.updates_item.title = UPDATES_LABEL_AVAILABLE

    def check_for_updates_manual(self, _sender: rumps.MenuItem) -> None:
        """Callback du menu : force un check + propose si dispo.

        Lance le check dans un thread daemon (l'API GitHub peut prendre
        jusqu'à 5s, on ne veut pas geler la menu bar). Pendant le check,
        on affiche un label "Vérification en cours…".
        """
        with self._update_lock:
            if self._update_check_in_progress:
                return
            self._update_check_in_progress = True

        self.updates_item.title = UPDATES_LABEL_CHECKING
        threading.Thread(
            target=self._safe_check_for_update_manual,
            daemon=True,
            name="update-check-manual",
        ).start()

    def _safe_check_for_update_manual(self) -> None:
        try:
            info = updater.check_for_update()
        except Exception:
            traceback.print_exc()
            info = None
        AppHelper.callAfter(self._on_manual_check_done, info)

    def _on_manual_check_done(
        self, info: "updater.UpdateInfo | None"
    ) -> None:
        """Sur main thread : montre le résultat à l'user."""
        with self._update_lock:
            self._update_check_in_progress = False

        if info is None:
            # Soit on est à jour, soit pas de réseau — on ne distingue
            # pas, l'utilisateur ne peut rien faire de plus dans les
            # 2 cas. Message explicite pour ne pas laisser planer le doute.
            self._update_info = None
            self.updates_item.title = UPDATES_LABEL_DEFAULT
            rumps.alert(
                title=APP_NAME,
                message=(
                    "Voxtral est à jour, ou la connexion à GitHub est "
                    "indisponible (mode hors-ligne)."
                ),
            )
            return

        self._update_info = info
        self.updates_item.title = UPDATES_LABEL_AVAILABLE
        self._offer_update(info)

    def _offer_update(self, info: "updater.UpdateInfo") -> None:
        """Sur main thread : alerte modale qui propose la MAJ.

        Si requirements.txt ou install.sh ont changé, on refuse de
        mettre à jour automatiquement — un git pull seul laisserait l'app
        dans un état incohérent (deps Python obsolètes, launchers cassés).
        """
        if info.requires_manual_action:
            files = ", ".join(info.risky_files)
            rumps.alert(
                title="Mise à jour majeure disponible",
                message=(
                    f"{info.commits_behind} commit(s) avec changements "
                    f"importants ({files}).\n\n"
                    f"Cette mise à jour nécessite une réinstallation. "
                    f"Contacte l'auteur ou re-lance install.sh manuellement."
                ),
            )
            return

        commits_label = (
            "1 nouveau commit"
            if info.commits_behind == 1
            else f"{info.commits_behind} nouveaux commits"
        )
        response = rumps.alert(
            title="Mise à jour disponible",
            message=(
                f"{commits_label} sur Voxtral.\n\n"
                f"Dernier : {info.head_message}\n\n"
                f"Mettre à jour maintenant ?"
            ),
            ok="Mettre à jour",
            cancel="Plus tard",
        )
        if response != 1:  # Cancel ou close
            return

        threading.Thread(
            target=self._safe_apply_update,
            daemon=True,
            name="update-apply",
        ).start()

    def _safe_apply_update(self) -> None:
        try:
            result = updater.apply_update()
        except Exception as exc:
            traceback.print_exc()
            result = updater.ApplyResult(
                success=False,
                message=f"Erreur inattendue : {exc}",
            )
        AppHelper.callAfter(self._on_apply_done, result)

    def _on_apply_done(self, result: "updater.ApplyResult") -> None:
        """Sur main thread : montre le résultat de la mise à jour.

        Si succès + redémarrage requis : on propose à l'user de quitter
        l'app pour pouvoir la relancer (le code Python en RAM ne
        reflète pas le nouveau code on-disk).
        """
        if not result.success:
            rumps.alert(
                title="Échec de la mise à jour",
                message=result.message,
            )
            return

        self._update_info = None
        self.updates_item.title = UPDATES_LABEL_DEFAULT

        if result.requires_restart:
            response = rumps.alert(
                title="Mise à jour appliquée",
                message=result.message,
                ok="Quitter Voxtral",
                cancel="Plus tard",
            )
            if response == 1:
                self.quit_app(None)
        else:
            rumps.alert(title="Mise à jour appliquée", message=result.message)

    # ------------------------------------------------------------------
    # Items de menu (suite)
    # ------------------------------------------------------------------

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

    def quit_app(self, _sender: "rumps.MenuItem | None") -> None:
        """Quitte l'app sans jamais bloquer le main thread.

        Les deux libérations d'ici peuvent bloquer plusieurs secondes :
        `listener.stop()` attend jusqu'à 1 s la fin du tour de CFRunLoop de
        pynput, et `Pa_CloseStream` peut se coincer si le périphérique audio
        a disparu. Les faire sur le main thread, c'est exactement le
        « Python ne répond plus » qu'on cherche à supprimer — on les délègue
        donc à un thread, et on ne les attend pas.
        """
        # Les workers reçoivent leur ordre d'arrêt sans qu'on l'attende : une
        # inférence MLX en cours peut durer plusieurs secondes.
        self.file_job.cancel()
        self.dictation.shutdown(wait=False)
        self.inference.shutdown(wait=False)

        threading.Thread(
            target=self._safe_release_resources, daemon=True, name="quit-release"
        ).start()

        # Filet de sécurité : si la descente de NSApp se coince (thread MLX
        # dans Metal, CoreAudio bloqué), on tue le process à la main. Sans
        # ça l'utilisateur retombe sur « Forcer à quitter ».
        killer = threading.Timer(3.0, lambda: os._exit(0))
        killer.daemon = True
        killer.start()

        rumps.quit_application()

    def _safe_release_resources(self) -> None:
        """Libère raccourci + micro, hors main thread, sans jamais lever."""
        try:
            self.hotkey.stop()
        except Exception:
            traceback.print_exc()
        try:
            self.recorder.shutdown()
        except Exception:
            traceback.print_exc()

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
