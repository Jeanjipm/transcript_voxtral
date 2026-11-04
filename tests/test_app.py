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
from inference_worker import PRIORITY_DICTATION, PRIORITY_MODEL
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
            "silence_threshold_dbfs": -55,
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


def test_init_refreshes_offline_mode_for_configured_model(
    user_config: Path, deps
):
    _write_user_config(user_config, model={"name": TURBO})
    VoxtralApp()
    deps["hf_refresh"].assert_called_once_with(TURBO, prefer_offline=True)


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
    deps["hf_refresh"].assert_called_with(TURBO, prefer_offline=True)


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
        ui={"auto_paste": False},
    )
    wav = tmp_path / "d.wav"
    wav.touch()
    voxtral_app.transcriber.transcribe.return_value = "bonjour"

    voxtral_app._transcribe_and_paste(wav)

    voxtral_app.transcriber.transcribe.assert_called_once_with(
        wav, language="fr", task="transcribe", max_new_tokens=2048
    )
    deps["paste_text"].assert_called_once_with("bonjour", auto_paste=False)
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
