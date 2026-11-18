"""Tests d'app.py — l'orchestration de la menu bar, sans Cocoa.

Tout ce qui touche au matériel ou au système est remplacé : micro,
raccourci, workers, modèles, fenêtres. On vérifie ce qu'app.py DÉCIDE :
quel composant il reconstruit quand la config change, quel transcriber il
donne aux fichiers, ce que le menu affiche après une erreur ou une fin de
job, ce qu'il transmet à la transcription d'une dictée.

Les tests tournent sur le main thread, donc les gardes
`AppHelper.callAfter` d'app.py ne sont jamais empruntés : les callbacks
s'exécutent en direct.
"""

from __future__ import annotations

import faulthandler
import signal
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

import app as app_module
import config as config_module
from app import (
    ERROR_LABEL_NONE,
    ERROR_LABEL_PRESENT,
    FILE_MENU_LABEL,
    READ_MENU_LABEL,
    READ_STOP_LABEL,
    STATUS_SPEECH_DOWNLOAD,
    SYMBOL_SPEECH_FRAMES,
    SYMBOL_DOWNLOADING_FRAMES,
    SYMBOL_IDLE,
    SYMBOL_RECORDING,
    SYMBOL_TRANSCRIBING_FRAMES,
    UPDATES_LABEL_AVAILABLE,
    UPDATES_LABEL_DEFAULT,
    VoxtralApp,
)
from audio_device import DeviceInfo
from dictation_controller import State as DictationState
from file_job import JobResult, JobState
from inference_worker import PRIORITY_DICTATION, PRIORITY_FILE, PRIORITY_MODEL
import rumps
import updater


# app.py pose faulthandler et des handlers de signaux sur stderr à l'import :
# utiles en production, mais ici stderr est l'objet de capture de pytest,
# qui sera fermé. On rend faulthandler à la vraie sortie d'erreur (comme
# pytest le fait lui-même) et on retire les handlers de signaux.
import sys  # noqa: E402

faulthandler.enable(file=sys.__stderr__)
for _sig in (signal.SIGTERM, signal.SIGHUP):
    faulthandler.unregister(_sig)


VOXTRAL = "mzbac/voxtral-mini-3b-4bit-mixed"
TURBO = "mlx-community/whisper-large-v3-turbo"
LARGE_V3 = "mlx-community/whisper-large-v3-mlx"


def _write_user_config(path: Path, **sections) -> None:
    """Écrit un ~/.voxtral/config.yaml de test. `updates.auto_check` est
    toujours coupé : sinon l'app lance un thread qui attend 10 s puis appelle
    GitHub."""
    data: dict = {"updates": {"auto_check": False}}
    for section, values in sections.items():
        data.setdefault(section, {}).update(values)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


@pytest.fixture
def user_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "config.yaml"
    _write_user_config(path)
    monkeypatch.setattr(app_module, "USER_CONFIG_PATH", path)
    monkeypatch.setattr(app_module, "ensure_user_config_exists", lambda: None)
    monkeypatch.setattr(
        app_module, "load_config",
        lambda: config_module.load_config(user_path=path),
    )
    return path


@pytest.fixture
def deps(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Tous les composants qu'app.py construit, remplacés par des mocks."""
    mocks = {
        "AudioRecorder": MagicMock(name="AudioRecorder"),
        # Une instance NEUVE par construction : on vérifie que le hot-reload
        # reconstruit le feedback seulement quand la section sons change.
        "AudioFeedback": MagicMock(
            name="AudioFeedback", side_effect=lambda cfg: MagicMock(name="feedback")
        ),
        "InferenceWorker": MagicMock(name="InferenceWorker"),
        "DictationController": MagicMock(name="DictationController"),
        "HotkeyManager": MagicMock(name="HotkeyManager"),
        "paste_text": MagicMock(name="paste_text"),
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(app_module, name, mock)

    def fake_make_transcriber(cfg):
        t = MagicMock(name=f"transcriber[{cfg.model.name}]")
        t.model_repo = cfg.model.name
        t.repo_for.side_effect = lambda task: cfg.model.name
        return t

    mocks["make_transcriber"] = MagicMock(side_effect=fake_make_transcriber)
    monkeypatch.setattr(app_module, "make_transcriber", mocks["make_transcriber"])

    mocks["hf_refresh"] = MagicMock(name="hf_offline.refresh", return_value=True)
    monkeypatch.setattr(app_module.hf_offline, "refresh", mocks["hf_refresh"])
    mocks["is_model_cached"] = MagicMock(return_value=True)
    monkeypatch.setattr(
        app_module.hf_offline, "is_model_cached", mocks["is_model_cached"]
    )
    class FakeDetector:
        """Détecteur de parole simulé : présent, et qui entend une voix."""

        def __init__(self) -> None:
            self.is_available = True
            self.verdict: bool | None = True
            self.seen: list[Path] = []
            self.ensure_downloaded = MagicMock(name="ensure_downloaded")

        def has_speech(self, wav_path: Path) -> bool | None:
            self.seen.append(wav_path)
            return self.verdict

    mocks["SpeechDetector"] = FakeDetector
    monkeypatch.setattr(app_module.speech_detector, "SpeechDetector", FakeDetector)

    mocks["housekeeping"] = MagicMock(name="run_startup_housekeeping")
    monkeypatch.setattr(
        app_module.housekeeping, "run_startup_housekeeping", mocks["housekeeping"]
    )
    mocks["reveal_in_finder"] = MagicMock(name="reveal_in_finder")
    monkeypatch.setattr(
        app_module.file_picker, "reveal_in_finder", mocks["reveal_in_finder"]
    )
    mocks["choose_audio_file"] = MagicMock(name="choose_audio_file", return_value=None)
    monkeypatch.setattr(
        app_module.file_picker, "choose_audio_file", mocks["choose_audio_file"]
    )
    rumps.alert.reset_mock()
    rumps.alert.return_value = 1
    rumps.quit_application.reset_mock()
    return mocks


@pytest.fixture
def voxtral_app(user_config: Path, deps: dict) -> VoxtralApp:
    return VoxtralApp()


def _reload(voxtral_app: VoxtralApp, user_config: Path, **sections) -> None:
    """Réécrit la config utilisateur puis déclenche le rechargement."""
    _write_user_config(user_config, **sections)
    voxtral_app._reload_config()


# ---- Construction ----


def test_init_builds_menu_and_starts_workers(voxtral_app: VoxtralApp, deps):
    """L'app démarre ses deux workers, pré-chauffe le micro, pré-charge le
    modèle via l'inference-worker, et arme le raccourci."""
    assert voxtral_app.status_item.title == "État : prêt"
    assert voxtral_app.hotkey_item.title.startswith("Raccourci : ")
    assert voxtral_app.error_item.title == ERROR_LABEL_NONE
    assert voxtral_app.file_item.title == FILE_MENU_LABEL

    voxtral_app.inference.start.assert_called_once()
    voxtral_app.dictation.start.assert_called_once()
    voxtral_app.dictation.request_prewarm.assert_called_once()
    preload = voxtral_app.inference.submit.call_args
    assert preload.kwargs["priority"] == PRIORITY_MODEL
    assert preload.kwargs["label"] == "preload"
    voxtral_app.hotkey.start.assert_called_once()


def test_init_passes_recording_settings_to_components(
    user_config: Path, deps
):
    _write_user_config(
        user_config,
        recording={
            "max_duration_s": 120, "tail_padding_ms": 500,
            "silence_padding_ms": 100, "start_retries": 2,
            "silence_threshold_dbfs": -55, "min_capture_ratio": 0.5,
        },
    )
    VoxtralApp()
    deps["AudioRecorder"].assert_called_once_with(
        start_retries=2, silence_padding_ms=100
    )
    ctrl_kwargs = deps["DictationController"].call_args.kwargs
    assert ctrl_kwargs["max_duration_s"] == 120
    assert ctrl_kwargs["tail_padding_ms"] == 500
    assert ctrl_kwargs["silence_threshold_dbfs"] == -55
    assert ctrl_kwargs["silence_padding_ms"] == 100
    assert ctrl_kwargs["min_capture_ratio"] == 0.5


def test_init_refreshes_offline_mode_for_configured_model(
    user_config: Path, deps
):
    _write_user_config(user_config, model={"name": TURBO})
    VoxtralApp()
    deps["hf_refresh"].assert_called_once_with(TURBO, prefer_offline=True)


def test_app_version_is_the_installed_commit(monkeypatch: pytest.MonkeyPatch):
    """Sprint 6c (P3) : plus de constante figée à 0.1.0, la version est le
    commit installé ; « dev » si git ne répond pas."""
    fake = MagicMock(return_value=MagicMock(returncode=0, stdout="abc1234 (2026-11-04)\n"))
    monkeypatch.setattr(app_module.subprocess, "run", fake)
    assert app_module.app_version() == "abc1234 (2026-11-04)"
    assert fake.call_args.args[0][:3] == ["git", "-C", str(app_module.PROJECT_ROOT)]

    monkeypatch.setattr(
        app_module.subprocess, "run", MagicMock(side_effect=OSError("pas de git"))
    )
    assert app_module.app_version() == "dev"

    monkeypatch.setattr(
        app_module.subprocess, "run",
        MagicMock(return_value=MagicMock(returncode=128, stdout="")),
    )
    assert app_module.app_version() == "dev"


def test_init_runs_housekeeping_with_configured_limits(user_config: Path, deps):
    """Sprint 6c : l'entretien (journal, vieilles dictées) tourne au démarrage
    avec les seuils de la config — et avant tout le reste, voir __init__."""
    _write_user_config(
        user_config, maintenance={"log_max_mb": 3, "recordings_max_age_days": 7}
    )
    VoxtralApp()
    deps["housekeeping"].assert_called_once()
    cfg = deps["housekeeping"].call_args.args[0]
    assert (cfg.log_max_mb, cfg.recordings_max_age_days) == (3, 7)


def test_informational_items_are_not_clickable(voxtral_app: VoxtralApp):
    for item in (
        voxtral_app.status_item, voxtral_app.hotkey_item, voxtral_app.lang_item,
        voxtral_app.model_item, voxtral_app.error_item,
        voxtral_app.file_cancel_item,
    ):
        assert item.callback is None


# ---- Hot-reload de la config ----


def test_reload_hotkey_change_rebinds_and_relabels(
    voxtral_app: VoxtralApp, user_config: Path
):
    _reload(voxtral_app, user_config, hotkey={"combo": "f13"})
    voxtral_app.hotkey.update_binding.assert_called_once_with("f13")
    assert voxtral_app.hotkey_item.title == "Raccourci : F13"
    assert voxtral_app.config.hotkey.combo == "f13"


def test_reload_model_change_builds_new_transcriber_and_preloads(
    voxtral_app: VoxtralApp, user_config: Path, deps
):
    old_transcriber = voxtral_app.transcriber
    deps["make_transcriber"].reset_mock()
    deps["hf_refresh"].reset_mock()
    voxtral_app.inference.submit.reset_mock()

    _reload(voxtral_app, user_config, model={"name": TURBO})

    # Le réseau est ré-évalué AVANT la construction (un modèle pas encore
    # téléchargé doit pouvoir l'être).
    deps["hf_refresh"].assert_called_once_with(TURBO, prefer_offline=True)
    deps["make_transcriber"].assert_called_once()
    assert voxtral_app.transcriber is not old_transcriber
    assert voxtral_app.transcriber.model_repo == TURBO
    assert voxtral_app.model_item.title.startswith("Modèle : ")
    preload = voxtral_app.inference.submit.call_args
    assert preload.kwargs["label"] == "preload-after-reload"
    assert preload.kwargs["priority"] == PRIORITY_MODEL


def test_reload_without_model_change_keeps_transcriber(
    voxtral_app: VoxtralApp, user_config: Path, deps
):
    old_transcriber = voxtral_app.transcriber
    deps["make_transcriber"].reset_mock()
    _reload(voxtral_app, user_config, transcription={"language": "fr"})
    deps["make_transcriber"].assert_not_called()
    assert voxtral_app.transcriber is old_transcriber


def test_reload_language_change_relabels(voxtral_app: VoxtralApp, user_config: Path):
    _reload(voxtral_app, user_config, transcription={"language": "fr"})
    assert voxtral_app.lang_item.title == "Langue : 🇫🇷 Français"
    _reload(voxtral_app, user_config, transcription={"language": "xx"})
    assert voxtral_app.lang_item.title == "Langue : xx"  # code inconnu : tel quel


def test_reload_sounds_change_rebuilds_feedback_only_when_needed(
    voxtral_app: VoxtralApp, user_config: Path, deps
):
    old_feedback = voxtral_app.feedback
    _reload(voxtral_app, user_config, transcription={"language": "fr"})
    assert voxtral_app.feedback is old_feedback  # cache NSSound préservé

    _reload(voxtral_app, user_config, sounds={"volume": 0.9})
    assert voxtral_app.feedback is not old_feedback
    assert voxtral_app.config.sounds.volume == 0.9


def test_check_config_change_ignores_unchanged_mtime(
    voxtral_app: VoxtralApp, monkeypatch: pytest.MonkeyPatch
):
    reload = MagicMock(name="_reload_config")
    monkeypatch.setattr(voxtral_app, "_reload_config", reload)
    voxtral_app._check_config_change()
    reload.assert_not_called()


def test_check_config_change_reloads_when_file_changes(
    voxtral_app: VoxtralApp, user_config: Path, monkeypatch: pytest.MonkeyPatch
):
    reload = MagicMock(name="_reload_config")
    monkeypatch.setattr(voxtral_app, "_reload_config", reload)
    voxtral_app._config_mtime = 0.0  # comme si le fichier venait de bouger
    voxtral_app._check_config_change()
    reload.assert_called_once()


def test_check_config_change_survives_invalid_yaml(
    voxtral_app: VoxtralApp, user_config: Path, capsys
):
    """Une config cassée est journalisée, mais ne tue pas le timer ni ne
    laisse un état incohérent : l'ancienne config reste en place."""
    before = voxtral_app.config
    user_config.write_text("model: [pas: {du: yaml", encoding="utf-8")
    voxtral_app._config_mtime = 0.0
    voxtral_app._check_config_change()  # ne lève pas
    assert voxtral_app.config is before
    assert "Traceback" in capsys.readouterr().err


def test_check_config_change_missing_file_is_silent(
    voxtral_app: VoxtralApp, user_config: Path
):
    user_config.unlink()
    voxtral_app._check_config_change()  # ne lève pas


# ---- Transcriber des fichiers ----


def test_file_transcriber_reuses_dictation_transcriber_for_same_model(
    user_config: Path, deps
):
    """Même modèle pour les dictées et les fichiers : inutile de charger deux
    fois les mêmes poids."""
    _write_user_config(user_config, model={"name": TURBO})
    voxtral_app = VoxtralApp()
    deps["make_transcriber"].reset_mock()
    assert voxtral_app._get_file_transcriber() is voxtral_app.transcriber
    deps["make_transcriber"].assert_not_called()


def test_file_transcriber_is_built_once_and_cached(
    voxtral_app: VoxtralApp, deps
):
    deps["make_transcriber"].reset_mock()
    first = voxtral_app._get_file_transcriber()
    second = voxtral_app._get_file_transcriber()
    assert first is second
    assert first is not voxtral_app.transcriber
    deps["make_transcriber"].assert_called_once()
    built_with = deps["make_transcriber"].call_args.args[0]
    assert built_with.model.name == TURBO


def test_file_transcriber_does_not_toggle_global_offline_mode(
    voxtral_app: VoxtralApp, deps
):
    """Issue 12 : construire le transcriber fichiers ne doit pas rebasculer
    le mode hors-ligne du process. Il est indexé sur le modèle de dictée ;
    le transcriber ouvre lui-même le réseau, localement, si son modèle manque
    (hf_offline.network_if_missing)."""
    deps["hf_refresh"].reset_mock()
    deps["is_model_cached"].return_value = False  # modèle fichiers absent

    voxtral_app._get_file_transcriber()

    deps["hf_refresh"].assert_not_called()


def test_file_transcriber_rebuilt_when_file_model_changes(
    voxtral_app: VoxtralApp, user_config: Path, deps
):
    first = voxtral_app._get_file_transcriber()
    _reload(voxtral_app, user_config, file_transcription={"model": LARGE_V3})
    second = voxtral_app._get_file_transcriber()
    assert second is not first
    assert second.model_repo == LARGE_V3


def test_file_transcriber_turbo_is_replaced_for_translation(
    voxtral_app: VoxtralApp, user_config: Path, deps
):
    """Turbo ne sait pas traduire : en traduction, le transcriber fichiers
    est construit sur large-v3, sans que l'utilisateur ait à le savoir."""
    _reload(voxtral_app, user_config, transcription={"task": "translate"})
    t = voxtral_app._get_file_transcriber()
    assert t.model_repo == LARGE_V3


# ---- Fin d'un job fichier ----


def _done(state: JobState, output: Path | None, error: str | None = None):
    return JobResult(state=state, output_path=output, error=error)


def test_file_done_success_alerts_and_reveals(
    voxtral_app: VoxtralApp, deps, tmp_path: Path
):
    out = tmp_path / "x.txt"
    voxtral_app.file_item.title = "Transcription : x.m4a"
    voxtral_app.file_item.set_callback(None)
    voxtral_app.file_cancel_item.set_callback(voxtral_app.cancel_file_transcription)

    voxtral_app._on_file_done(_done(JobState.DONE, out))

    assert "terminée" in rumps.alert.call_args.kwargs["title"]
    deps["reveal_in_finder"].assert_called_once_with(out)
    # Menu remis en place.
    assert voxtral_app.file_item.title == FILE_MENU_LABEL
    assert voxtral_app.file_item.callback == voxtral_app.transcribe_file_dialog
    assert voxtral_app.file_cancel_item.callback is None


def test_file_done_cancelled_keeps_partial_text(
    voxtral_app: VoxtralApp, deps, tmp_path: Path
):
    out = tmp_path / "partiel.txt"
    voxtral_app._on_file_done(_done(JobState.CANCELLED, out))
    assert "interrompue" in rumps.alert.call_args.kwargs["title"]
    assert "partiel.txt" in rumps.alert.call_args.kwargs["message"]
    deps["reveal_in_finder"].assert_called_once_with(out)


def test_file_done_failed_shows_error_item_and_alert(
    voxtral_app: VoxtralApp, deps
):
    voxtral_app._on_file_done(_done(JobState.FAILED, None, error="disque plein"))
    assert voxtral_app.error_item.title == ERROR_LABEL_PRESENT
    assert voxtral_app._last_error == ("Transcription du fichier échouée", "disque plein")
    assert rumps.alert.call_args.kwargs["message"] == "disque plein"
    deps["reveal_in_finder"].assert_not_called()


def test_file_done_without_output_is_silent(voxtral_app: VoxtralApp, deps):
    voxtral_app._on_file_done(_done(JobState.CANCELLED, None))
    rumps.alert.assert_not_called()
    deps["reveal_in_finder"].assert_not_called()


def test_file_progress_updates_item_title(voxtral_app: VoxtralApp):
    voxtral_app._on_file_progress(90.0, 3600.0)
    assert voxtral_app.file_item.title == "Transcription : 2 % — 00:01:30 / 01:00:00"


# ---- Lancement d'un job fichier ----


def test_file_dialog_refuses_when_job_running(voxtral_app: VoxtralApp, deps):
    voxtral_app.file_job = MagicMock(is_running=True)
    voxtral_app.transcribe_file_dialog(None)
    assert "déjà en cours" in rumps.alert.call_args.kwargs["message"]
    deps["choose_audio_file"].assert_not_called()


def test_file_dialog_cancelled_picker_does_nothing(voxtral_app: VoxtralApp, deps):
    voxtral_app.file_job = MagicMock(is_running=False)
    voxtral_app.transcribe_file_dialog(None)
    voxtral_app.file_job.submit.assert_not_called()


def test_file_dialog_submits_with_config_and_switches_menu(
    voxtral_app: VoxtralApp, deps, user_config: Path, tmp_path: Path
):
    _reload(voxtral_app, user_config, transcription={"language": "fr"})
    source = tmp_path / "reunion.m4a"
    deps["choose_audio_file"].return_value = source
    voxtral_app.file_job = MagicMock(is_running=False)
    voxtral_app.file_job.submit.return_value = True

    voxtral_app.transcribe_file_dialog(None)

    kwargs = voxtral_app.file_job.submit.call_args.kwargs
    assert kwargs["source"] == source
    assert kwargs["language"] == "fr"
    assert kwargs["model_name"] == TURBO
    assert kwargs["block_duration_s"] == 300
    assert voxtral_app.file_item.title == "Transcription : reunion.m4a"
    assert voxtral_app.file_item.callback is None
    assert voxtral_app.file_cancel_item.callback == voxtral_app.cancel_file_transcription
    assert voxtral_app._anim_timer is not None


# ---- Erreurs visibles ----


def test_show_error_marks_menu_and_flashes(
    voxtral_app: VoxtralApp, monkeypatch: pytest.MonkeyPatch
):
    icon = MagicMock(name="_set_status_icon")
    monkeypatch.setattr(voxtral_app, "_set_status_icon", icon)

    voxtral_app._show_error("Micro indisponible", "détail")

    assert voxtral_app.error_item.title == ERROR_LABEL_PRESENT
    assert voxtral_app.error_item.callback == voxtral_app._show_last_error
    assert voxtral_app._last_error == ("Micro indisponible", "détail")
    icon.assert_called_with(app_module.SYMBOL_ERROR, red=True)
    assert voxtral_app._error_flash_timer is not None
    assert voxtral_app._error_flash_timer.running is True


def test_show_last_error_alerts_then_resets(voxtral_app: VoxtralApp):
    voxtral_app._show_error("Sous-titre", "message")
    voxtral_app._show_last_error(None)
    assert "Sous-titre" in rumps.alert.call_args.kwargs["title"]
    assert rumps.alert.call_args.kwargs["message"] == "message"
    assert voxtral_app._last_error is None
    assert voxtral_app.error_item.title == ERROR_LABEL_NONE
    assert voxtral_app.error_item.callback is None


def test_show_last_error_without_error_is_noop(voxtral_app: VoxtralApp):
    voxtral_app._show_last_error(None)
    rumps.alert.assert_not_called()


def test_error_flash_end_restores_idle_icon(
    voxtral_app: VoxtralApp, monkeypatch: pytest.MonkeyPatch
):
    icon = MagicMock(name="_set_status_icon")
    monkeypatch.setattr(voxtral_app, "_set_status_icon", icon)
    voxtral_app._show_error("x", "y")
    voxtral_app._end_error_flash()
    icon.assert_called_with(SYMBOL_IDLE)
    assert voxtral_app._error_flash_timer is None


def test_error_flash_end_does_not_override_running_animation(
    voxtral_app: VoxtralApp, monkeypatch: pytest.MonkeyPatch
):
    """Une dictée a pu démarrer pendant le flash : le sablier reste."""
    icon = MagicMock(name="_set_status_icon")
    monkeypatch.setattr(voxtral_app, "_set_status_icon", icon)
    voxtral_app._show_error("x", "y")
    voxtral_app._start_animation(SYMBOL_TRANSCRIBING_FRAMES, 0.4)
    icon.reset_mock()
    voxtral_app._end_error_flash()
    icon.assert_not_called()


# ---- État de la dictée dans la menu bar ----


def test_dictation_state_recording_is_red_icon(
    voxtral_app: VoxtralApp, monkeypatch: pytest.MonkeyPatch
):
    icon = MagicMock(name="_set_status_icon")
    monkeypatch.setattr(voxtral_app, "_set_status_icon", icon)
    voxtral_app._on_dictation_state(DictationState.RECORDING, "État : écoute en cours…")
    icon.assert_called_with(SYMBOL_RECORDING, red=True)
    assert voxtral_app.status_item.title == "État : écoute en cours…"
    assert voxtral_app._anim_timer is None


def test_dictation_state_pending_animates_hourglass(voxtral_app: VoxtralApp):
    voxtral_app._on_dictation_state(DictationState.PENDING, "État : transcription…")
    assert voxtral_app._anim_frames == SYMBOL_TRANSCRIBING_FRAMES
    assert voxtral_app._anim_timer.running is True
    assert voxtral_app.status_item.title == "État : transcription…"


def test_dictation_state_idle_stops_animation(voxtral_app: VoxtralApp):
    voxtral_app._on_dictation_state(DictationState.PENDING, "…")
    voxtral_app._on_dictation_state(DictationState.IDLE, "État : prêt")
    assert voxtral_app._anim_timer is None
    assert voxtral_app.status_item.title == "État : prêt"


def test_animation_is_idempotent_on_same_frames(voxtral_app: VoxtralApp):
    voxtral_app._start_animation(SYMBOL_TRANSCRIBING_FRAMES, 0.4)
    timer = voxtral_app._anim_timer
    voxtral_app._start_animation(SYMBOL_TRANSCRIBING_FRAMES, 0.4)
    assert voxtral_app._anim_timer is timer  # pas de timer orphelin


def test_animation_tick_cycles_frames(
    voxtral_app: VoxtralApp, monkeypatch: pytest.MonkeyPatch
):
    icon = MagicMock(name="_set_status_icon")
    monkeypatch.setattr(voxtral_app, "_set_status_icon", icon)
    voxtral_app._start_animation(SYMBOL_TRANSCRIBING_FRAMES, 0.4)
    voxtral_app._anim_timer.fire()
    voxtral_app._anim_timer.fire()
    assert [c.args[0] for c in icon.call_args_list] == [
        SYMBOL_TRANSCRIBING_FRAMES[0],
        SYMBOL_TRANSCRIBING_FRAMES[1],
        SYMBOL_TRANSCRIBING_FRAMES[0],
    ]


# ---- Transcription d'une dictée ----


def test_submit_transcription_uses_dictation_priority(voxtral_app: VoxtralApp):
    voxtral_app.inference.submit.reset_mock()
    voxtral_app._submit_transcription(Path("/tmp/x.wav"))
    kwargs = voxtral_app.inference.submit.call_args.kwargs
    assert kwargs["priority"] == PRIORITY_DICTATION
    assert kwargs["label"] == "dictation"


def test_transcribe_and_paste_passes_config_and_cleans_up(
    voxtral_app: VoxtralApp, deps, user_config: Path, tmp_path: Path
):
    _reload(
        voxtral_app, user_config,
        transcription={"language": "fr", "max_new_tokens": 2048},
        ui={"auto_paste": False, "clipboard_restore_delay_ms": 800},
    )
    wav = tmp_path / "d.wav"
    wav.touch()
    voxtral_app.transcriber.transcribe.return_value = "bonjour"

    voxtral_app._transcribe_and_paste(wav)

    voxtral_app.transcriber.transcribe.assert_called_once_with(
        wav, language="fr", task="transcribe", max_new_tokens=2048,
        initial_prompt=None,
    )
    deps["paste_text"].assert_called_once_with(
        "bonjour", auto_paste=False, restore_delay_s=0.8
    )
    assert not wav.exists()
    voxtral_app.dictation.notify_transcription_done.assert_called_once()
    assert voxtral_app._last_error is None


def test_transcribe_and_paste_reports_error_and_still_notifies(
    voxtral_app: VoxtralApp, deps, tmp_path: Path, capsys
):
    """Une transcription qui lève : erreur visible dans le menu, WAV
    supprimé, et la machine à états est TOUJOURS prévenue — sinon l'icône
    resterait sur le sablier pour toujours."""
    wav = tmp_path / "d.wav"
    wav.touch()
    voxtral_app.transcriber.transcribe.side_effect = RuntimeError("modèle KO")

    voxtral_app._transcribe_and_paste(wav)

    deps["paste_text"].assert_not_called()
    assert voxtral_app._last_error == ("Erreur de transcription", "modèle KO")
    assert not wav.exists()
    voxtral_app.dictation.notify_transcription_done.assert_called_once()


def test_transcribe_shows_download_animation_when_model_missing(
    voxtral_app: VoxtralApp, deps, tmp_path: Path
):
    """Le modèle sollicité (celui de la tâche, pas forcément le configuré)
    n'est pas en cache : on annonce le téléchargement avant que
    from_pretrained ne bloque plusieurs minutes."""
    deps["is_model_cached"].return_value = False
    wav = tmp_path / "d.wav"
    wav.touch()
    voxtral_app.transcriber.transcribe.return_value = ""

    voxtral_app._transcribe_and_paste(wav)

    assert voxtral_app._anim_frames == SYMBOL_DOWNLOADING_FRAMES
    assert voxtral_app.status_item.title == "État : téléchargement du modèle…"
    deps["is_model_cached"].assert_called_once_with(VOXTRAL)


def test_model_needs_download_asks_transcriber_for_task_repo(
    voxtral_app: VoxtralApp, deps, user_config: Path
):
    """En traduction c'est Whisper large-v3 qui travaille : c'est LUI qu'il
    faut chercher dans le cache, pas le modèle configuré."""
    _reload(voxtral_app, user_config, transcription={"task": "translate"})
    voxtral_app.transcriber.repo_for.side_effect = None
    voxtral_app.transcriber.repo_for.return_value = LARGE_V3
    voxtral_app._model_needs_download()
    voxtral_app.transcriber.repo_for.assert_called_with("translate")
    deps["is_model_cached"].assert_called_with(LARGE_V3)


# ---- Suivi du micro par défaut ----


def test_device_change_is_forwarded_to_dictation_worker(
    voxtral_app: VoxtralApp, monkeypatch: pytest.MonkeyPatch, capsys
):
    """Le timer sonde ; un changement réel est journalisé et mis en file sur
    le dictation-worker (reconstruire un flux coûte ~4 s : pas ici)."""
    readings = iter([
        DeviceInfo(1, "MacBook Pro Microphone"),
        DeviceInfo(1, "MacBook Pro Microphone"),
        DeviceInfo(2, "AirPods"),
    ])
    monkeypatch.setattr(voxtral_app._device_watcher, "_read", lambda: next(readings))

    voxtral_app._device_timer.fire()
    voxtral_app._device_timer.fire()
    voxtral_app.dictation.request_device_refresh.assert_not_called()

    voxtral_app._device_timer.fire()
    voxtral_app.dictation.request_device_refresh.assert_called_once()
    err = capsys.readouterr().err
    assert "MacBook Pro Microphone → AirPods" in err


def test_device_timer_is_armed_at_startup(voxtral_app: VoxtralApp):
    assert voxtral_app._device_timer.running is True
    assert voxtral_app._device_timer.interval == app_module.DEVICE_POLL_INTERVAL_S


# ---- Mises à jour ----


def _info(risky: list[str] | None = None) -> updater.UpdateInfo:
    return updater.UpdateInfo(
        local_sha="a", remote_sha="b", commits_behind=2,
        head_message="fix: x", risky_files=risky or [],
    )


def test_mark_update_available_changes_menu_label(voxtral_app: VoxtralApp):
    voxtral_app._mark_update_available(_info())
    assert voxtral_app.updates_item.title == UPDATES_LABEL_AVAILABLE


def test_manual_check_without_update_resets_label_and_explains(
    voxtral_app: VoxtralApp,
):
    voxtral_app._update_check_in_progress = True
    voxtral_app._on_manual_check_done(None)
    assert voxtral_app.updates_item.title == UPDATES_LABEL_DEFAULT
    assert voxtral_app._update_check_in_progress is False
    assert "à jour" in rumps.alert.call_args.kwargs["message"]


def test_offer_update_with_risky_files_refuses_auto_update(voxtral_app: VoxtralApp):
    """install.sh ou requirements modifiés : un git pull laisserait l'app
    incohérente, on demande une réinstallation au lieu de proposer la MAJ."""
    voxtral_app._offer_update(_info(["requirements.lock"]))
    assert "réinstallation" in rumps.alert.call_args.kwargs["message"]
    assert "requirements.lock" in rumps.alert.call_args.kwargs["message"]
    assert "ok" not in rumps.alert.call_args.kwargs  # pas de bouton « Mettre à jour »


def test_offer_update_declined_does_nothing(
    voxtral_app: VoxtralApp, monkeypatch: pytest.MonkeyPatch
):
    rumps.alert.return_value = 0  # « Plus tard »
    thread = MagicMock(name="Thread")
    monkeypatch.setattr(app_module.threading, "Thread", thread)
    voxtral_app._offer_update(_info())
    assert rumps.alert.call_args.kwargs["ok"] == "Mettre à jour"
    thread.assert_not_called()


# ---- Quitter ----


def test_quit_stops_workers_without_blocking(
    voxtral_app: VoxtralApp, monkeypatch: pytest.MonkeyPatch
):
    """quit_app ne doit jamais attendre : les workers reçoivent leur ordre
    d'arrêt sans `wait`, et un coupe-circuit de 3 s est armé au cas où la
    descente de NSApp se coince."""
    killer = MagicMock(name="threading.Timer")
    monkeypatch.setattr(app_module.threading, "Timer", killer)
    voxtral_app.file_job = MagicMock()

    voxtral_app.quit_app(None)

    voxtral_app.file_job.cancel.assert_called_once()
    voxtral_app.dictation.shutdown.assert_called_once_with(wait=False)
    voxtral_app.inference.shutdown.assert_called_once_with(wait=False)
    assert killer.call_args.args[0] == 3.0
    killer.return_value.start.assert_called_once()
    rumps.quit_application.assert_called_once()


# ---- Détection de la parole (cf. speech_detector.py) ----


def _voice_wav(tmp_path: Path, amplitude: int = 3000) -> Path:
    """Un WAV réel d'une seconde ; 3000 ≈ −24 dBFS, 30 ≈ −64 dBFS."""
    import numpy as np
    import soundfile as sf

    path = tmp_path / "d.wav"
    t = np.arange(16_000) / 16_000
    audio = (amplitude * np.sin(2 * np.pi * 220 * t)).astype("int16")
    sf.write(path, audio.reshape(-1, 1), 16_000, subtype="PCM_16")
    return path


def test_dictation_without_speech_is_not_transcribed(
    voxtral_app: VoxtralApp, deps, tmp_path: Path, capsys
):
    """Silero n'entend pas de voix : rien n'est envoyé au modèle (qui
    inventerait « Merci. »), rien n'est collé, et la machine à états est
    quand même prévenue."""
    wav = _voice_wav(tmp_path)
    voxtral_app.speech_detector.verdict = False

    voxtral_app._transcribe_and_paste(wav)

    voxtral_app.transcriber.transcribe.assert_not_called()
    deps["paste_text"].assert_not_called()
    assert not wav.exists()
    voxtral_app.dictation.notify_transcription_done.assert_called_once()
    assert voxtral_app._last_error is None
    assert "pas de parole" in capsys.readouterr().err


def test_dictation_with_speech_is_transcribed(
    voxtral_app: VoxtralApp, deps, tmp_path: Path
):
    wav = _voice_wav(tmp_path)
    voxtral_app.transcriber.transcribe.return_value = "bonjour"
    voxtral_app._transcribe_and_paste(wav)
    assert voxtral_app.speech_detector.seen == [wav]
    assert deps["paste_text"].call_args.args[0] == "bonjour"


def test_volume_filter_takes_over_when_detector_cannot_answer(
    voxtral_app: VoxtralApp, deps, tmp_path: Path
):
    """Poids pas encore téléchargés, ou erreur : le seuil de volume réglé
    par l'utilisateur décide, comme avant ce sprint."""
    voxtral_app.speech_detector.verdict = None
    voxtral_app.transcriber.transcribe.return_value = "bonjour"

    voxtral_app._transcribe_and_paste(_voice_wav(tmp_path, amplitude=30))
    voxtral_app.transcriber.transcribe.assert_not_called()

    voxtral_app._transcribe_and_paste(_voice_wav(tmp_path, amplitude=3000))
    voxtral_app.transcriber.transcribe.assert_called_once()


def test_unreadable_recording_is_transcribed_anyway(
    voxtral_app: VoxtralApp, deps, tmp_path: Path
):
    """Détecteur muet ET fichier illisible : perdre une dictée serait pire
    qu'une phrase inventée."""
    voxtral_app.speech_detector.verdict = None
    wav = tmp_path / "d.wav"
    wav.touch()
    voxtral_app._transcribe_and_paste(wav)
    voxtral_app.transcriber.transcribe.assert_called_once()


def test_detection_can_be_switched_off(
    voxtral_app: VoxtralApp, deps, user_config: Path, tmp_path: Path
):
    _reload(voxtral_app, user_config, recording={"speech_detection": False})
    voxtral_app.speech_detector.verdict = False
    voxtral_app._transcribe_and_paste(_voice_wav(tmp_path))
    assert voxtral_app.speech_detector.seen == []
    voxtral_app.transcriber.transcribe.assert_called_once()


def test_silence_threshold_drops_to_dead_silence_with_detection(
    voxtral_app: VoxtralApp, deps, user_config: Path
):
    """Avec Silero, le seuil de volume n'écarte plus que le silence complet :
    une phrase chuchotée doit arriver jusqu'au détecteur. Le réglage est lu
    à chaque dictée, donc suivi sans redémarrer."""
    provider = deps["DictationController"].call_args.kwargs["silence_threshold_provider"]
    assert provider() == app_module.speech_detector.DEAD_SILENCE_DBFS

    _reload(voxtral_app, user_config, recording={"speech_detection": False})
    assert provider() == -45.0

    _reload(voxtral_app, user_config,
            recording={"speech_detection": False, "silence_threshold_dbfs": -30})
    assert provider() == -30.0

    # Un filtre coupé par l'utilisateur reste coupé, détection ou pas.
    _reload(voxtral_app, user_config, recording={"silence_threshold_dbfs": -120})
    assert provider() == -120.0


def test_detector_weights_are_fetched_in_background_when_missing(
    user_config: Path, deps, monkeypatch: pytest.MonkeyPatch
):
    """2 Mo, une fois, sans rien demander ; et jamais si la détection est
    coupée ou si les poids sont déjà là."""
    started: list[str] = []

    class RecordingThread:
        def __init__(self, target=None, name=None, **kwargs):  # noqa: ANN001
            self.name = name

        def start(self) -> None:
            started.append(self.name)

    monkeypatch.setattr(app_module.threading, "Thread", RecordingThread)

    VoxtralApp()
    assert "speech-detector-download" not in started  # déjà sur le disque

    class Missing(deps["SpeechDetector"]):
        def __init__(self) -> None:
            super().__init__()
            self.is_available = False

    monkeypatch.setattr(app_module.speech_detector, "SpeechDetector", Missing)
    VoxtralApp()
    assert started.count("speech-detector-download") == 1

    _write_user_config(user_config, recording={"speech_detection": False})
    VoxtralApp()
    assert started.count("speech-detector-download") == 1


# ---- Lexique personnel (cf. lexique.py) ----


def test_dictation_applies_personal_corrections(
    voxtral_app: VoxtralApp, deps, user_config: Path, tmp_path: Path
):
    _reload(
        voxtral_app, user_config,
        lexicon={"corrections": {"KickSit": "Kixit", "la TV a": "la TVA"}},
    )
    wav = tmp_path / "d.wav"
    wav.touch()
    voxtral_app.transcriber.transcribe.return_value = "Kicksit déclare la TV a."

    voxtral_app._transcribe_and_paste(wav)

    assert deps["paste_text"].call_args.args[0] == "Kixit déclare la TVA."
    # Les mots ne sont PAS soufflés tant que l'option n'est pas cochée.
    assert voxtral_app.transcriber.transcribe.call_args.kwargs["initial_prompt"] is None


def test_dictation_prompts_words_only_when_enabled(
    voxtral_app: VoxtralApp, deps, user_config: Path, tmp_path: Path
):
    _reload(
        voxtral_app, user_config,
        lexicon={"words": ["Kixit", "TVA"], "prompt_words": True},
    )
    wav = tmp_path / "d.wav"
    wav.touch()
    voxtral_app.transcriber.transcribe.return_value = "bonjour Kixit"

    voxtral_app._transcribe_and_paste(wav)

    assert (
        voxtral_app.transcriber.transcribe.call_args.kwargs["initial_prompt"]
        == "Glossaire : Kixit, TVA."
    )
    assert deps["paste_text"].call_args.args[0] == "bonjour Kixit"


def test_dictation_never_pastes_the_prompt_back(
    voxtral_app: VoxtralApp, deps, user_config: Path, tmp_path: Path
):
    """Sur un audio presque vide, Whisper « transcrit » l'amorce : on ne
    colle rien plutôt que « Glossaire : … » au milieu d'un courriel."""
    _reload(
        voxtral_app, user_config,
        lexicon={"words": ["Kixit", "TVA"], "prompt_words": True},
    )
    wav = tmp_path / "d.wav"
    wav.touch()
    voxtral_app.transcriber.transcribe.return_value = "Glossaire: Kixit, TVA"

    voxtral_app._transcribe_and_paste(wav)

    assert deps["paste_text"].call_args.args[0] == ""


# ---- Lecture du presse-papier à voix haute ----


@pytest.fixture
def speech(voxtral_app: VoxtralApp, deps: dict, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Presse-papier, moteur de synthèse et lecture simulés.

    La vraie `Lecture` (threads, file d'attente) a ses propres tests dans
    test_lecture.py ; ici on vérifie ce que l'app lui confie et comment elle
    réagit à ses signaux.
    """
    state: dict = {"clipboard": "Le segment AB mesure 6 cm.", "lectures": []}
    monkeypatch.setattr(
        app_module, "read_clipboard_text", lambda: state["clipboard"]
    )

    class FakeSynth:
        def __init__(self, repo: str) -> None:
            self.repo = repo
            self.sample_rate = 24_000
            self.calls: list[tuple[str, str]] = []

        def lire(self, texte: str, voix: str):
            self.calls.append((texte, voix))
            return "audio"

    class FakeLecture:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            self.__dict__.update(kwargs)
            self.started = False
            self.cancelled = False
            state["lectures"].append(self)

        def start(self) -> None:
            self.started = True

        def cancel(self) -> None:
            self.cancelled = True

    monkeypatch.setattr(app_module.synthese, "Synthetiseur", FakeSynth)
    monkeypatch.setattr(app_module.synthese, "demarrer", MagicMock(name="demarrer"))
    monkeypatch.setattr(app_module.synthese, "attendre", MagicMock(name="attendre"))
    monkeypatch.setattr(app_module.synthese, "arreter", MagicMock(name="arreter"))
    monkeypatch.setattr(app_module.lecture, "Lecture", FakeLecture)
    voxtral_app.inference.submit.reset_mock()
    return state


def test_menu_has_the_read_aloud_item(voxtral_app: VoxtralApp):
    assert voxtral_app.read_item.title == READ_MENU_LABEL
    assert voxtral_app.read_item in voxtral_app.menu
    assert voxtral_app.read_item.callback == voxtral_app.read_clipboard_aloud


def test_read_aloud_with_empty_clipboard_alerts_and_does_nothing(
    voxtral_app: VoxtralApp, speech: dict
):
    speech["clipboard"] = "   "
    voxtral_app.read_clipboard_aloud(None)
    rumps.alert.assert_called_once()
    assert "presse-papier" in rumps.alert.call_args.kwargs["message"]
    assert speech["lectures"] == []
    assert voxtral_app.read_item.title == READ_MENU_LABEL


def test_read_aloud_starts_a_reading_sentence_by_sentence(
    voxtral_app: VoxtralApp, speech: dict
):
    """Le texte part découpé en phrases ; l'item du menu devient le bouton
    d'arrêt, et l'icône s'anime."""
    speech["clipboard"] = (
        "Voici la première phrase du texte copié. Et voici la seconde, tout aussi longue."
    )
    voxtral_app.read_clipboard_aloud(None)
    rumps.alert.assert_not_called()
    (lect,) = speech["lectures"]
    assert lect.started
    assert lect.phrases == [
        "Voici la première phrase du texte copié.",
        "Et voici la seconde, tout aussi longue.",
    ]
    assert voxtral_app.read_item.title == READ_STOP_LABEL
    assert voxtral_app.read_item.callback == voxtral_app.stop_reading
    assert voxtral_app._anim_frames == SYMBOL_SPEECH_FRAMES


def test_reading_synthesizes_on_the_worker_in_low_priority(
    voxtral_app: VoxtralApp, speech: dict, user_config: Path
):
    """Chaque phrase est une tâche de l'inference-worker (seul propriétaire
    des modèles), derrière une dictée éventuelle ; la voix est celle de la
    config, le son est joué à la fréquence du modèle."""
    _reload(voxtral_app, user_config, synthesis={"voice": "fr_male"})
    voxtral_app.read_clipboard_aloud(None)
    (lect,) = speech["lectures"]

    lect.submit("tâche")
    call = voxtral_app.inference.submit.call_args
    assert call.args == ("tâche",)
    assert call.kwargs == {"priority": PRIORITY_FILE, "label": "speech"}

    assert lect.synthesize("Une phrase.") == "audio"
    assert voxtral_app._synthetiseur.calls == [("Une phrase.", "fr_male")]

    lect.start_playback("audio")
    app_module.synthese.demarrer.assert_called_once_with("audio", 24_000)
    assert lect.wait_playback is app_module.synthese.attendre
    assert lect.stop_playback is app_module.synthese.arreter


def test_read_aloud_asks_before_downloading_the_model(
    voxtral_app: VoxtralApp, speech: dict, deps: dict
):
    deps["is_model_cached"].return_value = False
    rumps.alert.return_value = 0  # Annuler
    voxtral_app.read_clipboard_aloud(None)
    assert rumps.alert.call_args.kwargs["ok"] == "Télécharger"
    assert speech["lectures"] == []

    rumps.alert.return_value = 1  # Télécharger
    voxtral_app.read_clipboard_aloud(None)
    assert len(speech["lectures"]) == 1
    assert voxtral_app._anim_frames == SYMBOL_DOWNLOADING_FRAMES
    assert voxtral_app.status_item.title == STATUS_SPEECH_DOWNLOAD

    # La première phrase joue : le téléchargement est fini.
    voxtral_app._on_speech_progress(1, 1)
    assert voxtral_app.status_item.title == "État : prêt"
    assert voxtral_app._anim_frames == SYMBOL_SPEECH_FRAMES


def test_read_aloud_warns_when_text_is_truncated(voxtral_app: VoxtralApp, speech: dict):
    speech["clipboard"] = "mot " * 8000
    voxtral_app.read_clipboard_aloud(None)
    assert "premiers caractères" in rumps.alert.call_args.kwargs["message"]
    (lect,) = speech["lectures"]
    assert sum(len(p) for p in lect.phrases) <= app_module.synthese.MAX_CHARS


def test_a_long_article_is_read_without_truncation(voxtral_app: VoxtralApp, speech: dict):
    """L'ancien plafond de 2 000 caractères n'existe plus."""
    speech["clipboard"] = "Une phrase de taille tout à fait ordinaire. " * 100
    voxtral_app.read_clipboard_aloud(None)
    rumps.alert.assert_not_called()
    assert len(speech["lectures"][0].phrases) == 100


def test_numbers_are_spelled_out_before_reading(
    voxtral_app: VoxtralApp, speech: dict, user_config: Path
):
    """Les voix lisent mal les chiffres (cf. oral.py)."""
    speech["clipboard"] = "Rendez-vous le 12 mars à 14h30 avec un budget de 3500 euros."
    voxtral_app.read_clipboard_aloud(None)
    assert speech["lectures"][0].phrases == [
        "Rendez-vous le douze mars à quatorze heures trente avec un budget "
        "de trois mille cinq cents euros.",
    ]

    voxtral_app._on_speech_done(None)
    _reload(voxtral_app, user_config, synthesis={"spell_numbers": False})
    voxtral_app.read_clipboard_aloud(None)
    assert speech["lectures"][1].phrases == [speech["clipboard"]]


def test_user_pronunciations_are_applied_before_reading(
    voxtral_app: VoxtralApp, speech: dict, user_config: Path
):
    _reload(voxtral_app, user_config,
            synthesis={"pronunciations": {"enneigement": "en-neigement"}})
    speech["clipboard"] = "Les stations voient leur Enneigement reculer d'année en année."
    voxtral_app.read_clipboard_aloud(None)
    assert speech["lectures"][0].phrases == [
        "Les stations voient leur en-neigement reculer d'année en année.",
    ]


def test_menu_item_stops_the_reading(voxtral_app: VoxtralApp, speech: dict):
    voxtral_app.read_clipboard_aloud(None)
    (lect,) = speech["lectures"]
    voxtral_app.stop_reading(None)
    assert lect.cancelled
    # Un second clic arrivé avant la fin n'ouvre pas une deuxième lecture.
    voxtral_app.read_clipboard_aloud(None)
    assert len(speech["lectures"]) == 1


def test_progress_is_shown_in_the_menu_item(voxtral_app: VoxtralApp, speech: dict):
    voxtral_app.read_clipboard_aloud(None)
    voxtral_app._on_speech_progress(2, 5)
    assert voxtral_app.read_item.title == f"{READ_STOP_LABEL} (2/5)"
    # Une seule phrase : pas de compteur.
    voxtral_app._on_speech_progress(1, 1)
    assert voxtral_app.read_item.title == READ_STOP_LABEL


def test_end_of_reading_restores_menu_and_icon(voxtral_app: VoxtralApp, speech: dict):
    voxtral_app.read_clipboard_aloud(None)
    voxtral_app._on_speech_done(None)
    assert voxtral_app.read_item.title == READ_MENU_LABEL
    assert voxtral_app.read_item.callback == voxtral_app.read_clipboard_aloud
    assert voxtral_app._lecture is None
    assert voxtral_app._anim_timer is None
    assert voxtral_app._last_error is None
    # Et une nouvelle lecture est possible.
    voxtral_app.read_clipboard_aloud(None)
    assert len(speech["lectures"]) == 2


def test_reading_failure_shows_error_and_restores_menu(
    voxtral_app: VoxtralApp, speech: dict
):
    voxtral_app.read_clipboard_aloud(None)
    voxtral_app._on_speech_done("voix inconnue")
    assert voxtral_app.error_item.title == ERROR_LABEL_PRESENT
    assert voxtral_app._last_error == ("Lecture impossible", "voix inconnue")
    assert voxtral_app.read_item.title == READ_MENU_LABEL
    assert voxtral_app._lecture is None


def test_starting_a_dictation_silences_the_reading(
    voxtral_app: VoxtralApp, speech: dict, monkeypatch: pytest.MonkeyPatch
):
    """Sinon le micro entend la voix de synthèse. Et la fin de la lecture ne
    doit pas effacer le rond rouge de la dictée qui vient de commencer."""
    voxtral_app.read_clipboard_aloud(None)
    (lect,) = speech["lectures"]

    voxtral_app.dictation.state = DictationState.RECORDING
    voxtral_app._on_dictation_state(DictationState.RECORDING, "État : écoute en cours…")
    assert lect.cancelled

    icon = MagicMock(name="_set_status_icon")
    monkeypatch.setattr(voxtral_app, "_set_status_icon", icon)
    voxtral_app._on_speech_done(None)
    assert voxtral_app.read_item.title == READ_MENU_LABEL
    assert voxtral_app._lecture is None
    icon.assert_not_called()


def test_synthesizer_is_rebuilt_only_when_its_model_changes(
    voxtral_app: VoxtralApp, speech: dict, user_config: Path
):
    first = voxtral_app._get_synthetiseur()
    assert voxtral_app._get_synthetiseur() is first
    _reload(voxtral_app, user_config, synthesis={"voice": "fr_male"})
    assert voxtral_app._get_synthetiseur() is first
    _reload(voxtral_app, user_config, synthesis={"model": "org/autre-tts"})
    second = voxtral_app._get_synthetiseur()
    assert second is not first and second.repo == "org/autre-tts"


def test_quit_stops_any_playback(
    voxtral_app: VoxtralApp, speech: dict, monkeypatch: pytest.MonkeyPatch
):
    voxtral_app.read_clipboard_aloud(None)
    monkeypatch.setattr(app_module.threading, "Thread", MagicMock())
    monkeypatch.setattr(app_module.threading, "Timer", MagicMock())
    voxtral_app.quit_app(None)
    assert speech["lectures"][0].cancelled
    app_module.synthese.arreter.assert_called_once()
