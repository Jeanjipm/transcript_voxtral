"""Tests de dictation_controller.py — machine à états, cause 1, coupe-circuit.

La plupart des tests pilotent `_handle*` directement (synchrone, déterministe)
plutôt que de passer par la file : la file et le thread sont testés à part.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

from dictation_controller import (
    DictationCallbacks,
    DictationController,
    State,
    _Command,
)


def _make_wav(tmp_path: Path, seconds: float, name: str = "d.wav") -> Path:
    """Écrit un WAV réel de la durée demandée (sf.info est utilisé pour de vrai)."""
    path = tmp_path / name
    frames = max(0, int(16_000 * seconds))
    sf.write(path, np.zeros((frames, 1), dtype="int16"), 16_000, subtype="PCM_16")
    return path


@pytest.fixture
def cb():
    """Callbacks entièrement mockés, inspectables."""
    return DictationCallbacks(
        on_state_change=MagicMock(name="on_state_change"),
        submit_transcription=MagicMock(name="submit_transcription"),
        on_error=MagicMock(name="on_error"),
        on_rearm_needed=MagicMock(name="on_rearm_needed"),
        on_recording_kept=MagicMock(name="on_recording_kept"),
    )


@pytest.fixture
def recorder(tmp_path: Path):
    rec = MagicMock(name="recorder")
    rec.is_recording = False
    rec.stop.return_value = _make_wav(tmp_path, 3.0)
    return rec


@pytest.fixture
def feedback():
    return MagicMock(name="feedback")


@pytest.fixture
def ctrl(recorder, feedback, cb, tmp_path: Path):
    return DictationController(
        recorder=recorder,
        feedback=feedback,
        callbacks=cb,
        max_duration_s=300,
        recordings_dir=tmp_path / "recordings",
    )


# ---- LE test de la cause 1 ----


def test_on_hotkey_start_does_no_work_synchronously(ctrl, recorder, feedback):
    """Régression cause 1 : appelée depuis le callback de l'event tap macOS,
    on_hotkey_start ne doit RIEN faire de coûteux.

    macOS désactive un tap dont le callback dépasse ~1 s, et pynput ne le
    réarme jamais : le relâchement n'arrive alors plus et l'app reste bloquée
    en écoute. Ouvrir le micro peut coûter plusieurs secondes — donc rien de
    tout ça ne doit se produire dans cet appel.
    """
    ctrl.on_hotkey_start()

    assert recorder.start.called is False
    assert recorder.prewarm.called is False
    assert feedback.play_start.called is False
    assert ctrl.state is State.IDLE  # rien n'a bougé, c'est juste en file


def test_on_hotkey_stop_does_no_work_synchronously(ctrl, recorder):
    ctrl.on_hotkey_stop()
    assert recorder.stop.called is False


def test_hotkey_entry_points_are_fast(ctrl):
    """Budget du callback event tap : ~1 s. On doit être des ordres de
    grandeur en dessous."""
    start = time.perf_counter()
    for _ in range(200):
        ctrl.on_hotkey_start()
        ctrl.on_hotkey_stop()
    elapsed = time.perf_counter() - start
    assert elapsed < 0.1, f"400 mises en file ont pris {elapsed:.3f}s"


def test_full_queue_drops_without_raising(ctrl, capsys):
    """File pleine : on abandonne la commande, on ne bloque pas et on ne lève
    pas — bloquer ici gèlerait le callback de l'event tap."""
    for _ in range(50):
        ctrl.on_hotkey_start()  # la file est à maxsize=8
    assert "file pleine" in capsys.readouterr().err


# ---- Transitions nominales ----


def test_start_records_and_reports_state(ctrl, recorder, feedback, cb):
    ctrl._handle_start()

    recorder.start.assert_called_once()
    feedback.play_start.assert_called_once()
    assert ctrl.state is State.RECORDING
    cb.on_state_change.assert_called_with(State.RECORDING, "État : écoute en cours…")


def test_stop_submits_transcription(ctrl, recorder, cb):
    ctrl._handle_start()
    ctrl._handle_stop()

    recorder.stop.assert_called_once()
    cb.submit_transcription.assert_called_once()
    assert ctrl.state is State.PENDING


def test_short_dictation_is_discarded_silently(ctrl, recorder, cb, tmp_path: Path):
    """Sous 0,5 s l'audio n'est que du silence et le son d'activation ; le
    modèle hallucinerait une phrase. On abandonne sans notifier."""
    recorder.stop.return_value = _make_wav(tmp_path, 0.2, "court.wav")

    ctrl._handle_start()
    ctrl._handle_stop()

    assert cb.submit_transcription.called is False
    assert cb.on_error.called is False
    assert ctrl.state is State.IDLE


def test_transcription_done_returns_to_idle(ctrl, cb):
    ctrl._handle_start()
    ctrl._handle_stop()
    assert ctrl.state is State.PENDING

    ctrl._handle_transcription_done()
    assert ctrl.state is State.IDLE
    cb.on_state_change.assert_called_with(State.IDLE, "État : prêt")


def test_notify_transcription_done_goes_through_the_queue(ctrl):
    """L'état ne doit être muté que par le worker : la notification venue du
    thread d'inférence passe par la file, sinon course avec un appui touche."""
    ctrl.on_hotkey_start()
    ctrl._queue.get_nowait()  # vide la file

    ctrl.notify_transcription_done()
    assert ctrl._queue.get_nowait() is _Command.TRANSCRIPTION_DONE


# ---- Transitions illégales : abandonner, jamais lever ----


def test_double_start_records_only_once(ctrl, recorder):
    """Auto-repeat, ou appui reçu juste après un réarmement du tap."""
    ctrl._handle_start()
    ctrl._handle_start()
    recorder.start.assert_called_once()
    assert ctrl.state is State.RECORDING


def test_stop_without_start_is_ignored(ctrl, recorder, capsys):
    ctrl._handle_stop()
    assert recorder.stop.called is False
    assert ctrl.state is State.IDLE
    assert "ignorée en état idle" in capsys.readouterr().err


def test_start_while_pending_starts_a_new_recording(ctrl, recorder):
    """Régression signalée en usage réel : enchaîner deux dictées.

    L'ancienne version refusait de démarrer tant que la transcription
    précédente tournait, et le faisait EN SILENCE — ni son, ni icône rouge.
    On parlait plusieurs secondes dans le vide, ce qui est indistinguable
    d'une app bloquée. Le micro est libre dès que le WAV est écrit ; le
    worker d'inférence sérialise les textes, donc l'ordre est préservé.
    """
    ctrl._handle_start()
    ctrl._handle_stop()
    assert ctrl.state is State.PENDING

    recorder.start.reset_mock()
    ctrl._handle_start()

    assert recorder.start.called is True
    assert ctrl.state is State.RECORDING


def test_double_start_while_recording_is_still_ignored(ctrl, recorder, capsys):
    """L'auto-repeat, lui, doit toujours être jeté : on enregistre déjà."""
    ctrl._handle_start()
    assert ctrl.state is State.RECORDING

    recorder.start.reset_mock()
    ctrl._handle_start()

    assert recorder.start.called is False
    assert "ignorée en état recording" in capsys.readouterr().err


def test_second_dictation_is_submitted_too(ctrl, cb):
    """Les deux dictées doivent arriver au worker, pas seulement la première."""
    for _ in range(2):
        ctrl._handle_start()
        ctrl._handle_stop()

    assert cb.submit_transcription.call_count == 2


def test_finished_transcription_does_not_reset_a_live_recording(ctrl, cb):
    """Le pire scénario du correctif : la transcription précédente se termine
    pendant qu'on enregistre déjà. Elle ne doit pas repasser l'icône au vert
    « prêt » alors que le micro est ouvert."""
    ctrl._handle_start()
    ctrl._handle_stop()          # -> PENDING
    ctrl._handle_start()         # nouvelle dictée pendant la transcription
    assert ctrl.state is State.RECORDING

    cb.on_state_change.reset_mock()
    ctrl._handle_transcription_done()

    assert ctrl.state is State.RECORDING
    assert cb.on_state_change.called is False


def test_state_stays_pending_while_work_remains(ctrl, cb):
    """Deux transcriptions en vol : la première qui finit ne doit pas
    annoncer « prêt » alors que la seconde tourne encore."""
    for _ in range(2):
        ctrl._handle_start()
        ctrl._handle_stop()

    ctrl._handle_transcription_done()
    assert ctrl.state is State.PENDING

    ctrl._handle_transcription_done()
    assert ctrl.state is State.IDLE


def test_extra_done_notifications_do_not_go_negative(ctrl):
    """Robustesse : un notify en trop ne doit pas coincer l'état en PENDING."""
    ctrl._handle_transcription_done()
    ctrl._handle_transcription_done()
    assert ctrl.state is State.IDLE

    ctrl._handle_start()
    ctrl._handle_stop()
    ctrl._handle_transcription_done()
    assert ctrl.state is State.IDLE


def test_stop_during_arming_is_honoured_after_arming(
    recorder, feedback, cb, tmp_path: Path
):
    """Le relâchement reçu pendant l'ouverture du micro (qui peut durer
    plusieurs secondes) doit être honoré dès la fin de l'armement.

    Sans cette mémorisation, un appui bref au lancement laisserait l'app
    bloquée en écoute sans jamais recevoir de stop.
    """
    ctrl = DictationController(
        recorder=recorder, feedback=feedback, callbacks=cb,
        recordings_dir=tmp_path / "rec",
    )

    # Simule un relâchement arrivé pendant que recorder.start() tourne.
    def slow_start() -> None:
        ctrl._handle_stop()  # arrive en état ARMING

    recorder.start.side_effect = slow_start

    ctrl._handle_start()

    assert ctrl.state is State.PENDING, "le stop mémorisé n'a pas été honoré"
    recorder.stop.assert_called_once()
    cb.submit_transcription.assert_called_once()


# ---- Régression cause 2, vue du contrôleur ----


def test_start_failure_returns_to_idle_and_reports(ctrl, recorder, cb):
    """Micro indisponible : retour à IDLE et erreur VISIBLE (pas silencieuse)."""
    recorder.start.side_effect = RuntimeError("device disparu")

    ctrl._handle_start()

    assert ctrl.state is State.IDLE
    cb.on_error.assert_called_once()
    assert "Micro indisponible" in cb.on_error.call_args[0][0]


def test_next_dictation_works_after_a_failed_start(ctrl, recorder, cb):
    """Régression cause 2 : un échec ne doit pas condamner les dictées suivantes."""
    recorder.start.side_effect = RuntimeError("device disparu")
    ctrl._handle_start()
    assert ctrl.state is State.IDLE

    recorder.start.side_effect = None
    ctrl._handle_start()
    assert ctrl.state is State.RECORDING


# ---- Coupe-circuit de durée maximale ----


def test_tick_below_limit_does_nothing(ctrl, recorder):
    ctrl._handle_start()
    ctrl._on_tick()
    assert ctrl.state is State.RECORDING
    assert recorder.stop.called is False


def test_tick_past_limit_cuts_and_keeps_audio(
    recorder, feedback, cb, tmp_path: Path
):
    """Dépassement : on coupe, on NE COLLE RIEN, on conserve l'audio et on
    demande un réarmement du raccourci."""
    ctrl = DictationController(
        recorder=recorder, feedback=feedback, callbacks=cb,
        max_duration_s=1, recordings_dir=tmp_path / "recordings",
    )
    ctrl._handle_start()
    ctrl._recording_started_at = time.monotonic() - 5.0

    ctrl._on_tick()

    recorder.stop.assert_called_once()
    assert ctrl.state is State.IDLE
    assert cb.submit_transcription.called is False, "rien ne doit être collé"
    cb.on_rearm_needed.assert_called_once()
    cb.on_recording_kept.assert_called_once()

    kept = cb.on_recording_kept.call_args[0][0]
    assert kept.exists() and kept.parent.name == "recordings"
    assert kept.name.startswith("dictee-")


def test_tick_past_limit_reports_visible_error(recorder, feedback, cb, tmp_path: Path):
    ctrl = DictationController(
        recorder=recorder, feedback=feedback, callbacks=cb,
        max_duration_s=1, recordings_dir=tmp_path / "recordings",
    )
    ctrl._handle_start()
    ctrl._recording_started_at = time.monotonic() - 5.0
    ctrl._on_tick()

    cb.on_error.assert_called_once()
    assert "trop long" in cb.on_error.call_args[0][0]


def test_tick_ignored_when_not_recording(ctrl, recorder, cb):
    ctrl._on_tick()
    assert recorder.stop.called is False
    assert cb.on_rearm_needed.called is False


# ---- Robustesse : le worker ne doit jamais mourir ----


def test_safe_forces_idle_and_reports_on_exception(ctrl, recorder, cb):
    """Une exception inattendue : log, retour à IDLE, micro libéré, erreur
    visible — et surtout le thread survit."""
    recorder.is_recording = True

    def boom() -> None:
        raise ValueError("imprévu")

    ctrl._safe(boom)

    assert ctrl.state is State.IDLE
    cb.on_error.assert_called_once()
    recorder.stop.assert_called_once()  # micro libéré


def test_worker_thread_survives_a_raising_command(recorder, feedback, cb, tmp_path):
    """Test avec le vrai thread : une commande qui lève ne l'abat pas."""
    ctrl = DictationController(
        recorder=recorder, feedback=feedback, callbacks=cb,
        recordings_dir=tmp_path / "rec",
    )
    recorder.start.side_effect = [RuntimeError("boom"), None]

    ctrl.start()
    try:
        ctrl.on_hotkey_start()  # lève -> gérée
        deadline = time.time() + 2.0
        while cb.on_error.called is False and time.time() < deadline:
            time.sleep(0.01)
        assert cb.on_error.called

        # Le thread doit encore répondre.
        recorder.start.side_effect = None
        ctrl.on_hotkey_start()
        deadline = time.time() + 2.0
        while ctrl.state is not State.RECORDING and time.time() < deadline:
            time.sleep(0.01)
        assert ctrl.state is State.RECORDING
    finally:
        ctrl.shutdown(wait=True)


def test_stop_on_unreadable_wav_treated_as_too_short(
    ctrl, recorder, cb, tmp_path: Path
):
    """Un WAV illisible (disque plein…) ne doit pas remonter en exception :
    durée 0 => traité comme trop court et abandonné."""
    recorder.stop.return_value = tmp_path / "inexistant.wav"

    ctrl._handle_start()
    ctrl._handle_stop()

    assert ctrl.state is State.IDLE
    assert cb.submit_transcription.called is False


# ---- Périphérique changé / prewarm ----


def test_device_changed_rebuilds_stream_when_idle(ctrl, recorder):
    ctrl._handle_device_changed()
    recorder.shutdown.assert_called_once()
    recorder.prewarm.assert_called_once()


def test_device_changed_left_alone_during_recording(ctrl, recorder):
    """En pleine dictée on ne touche à rien : AudioRecorder.stop() sait déjà
    survivre à la disparition du périphérique et conserve l'audio."""
    ctrl._handle_start()
    recorder.shutdown.reset_mock()

    ctrl._handle_device_changed()

    assert recorder.shutdown.called is False


def test_shutdown_releases_recorder(ctrl, recorder):
    ctrl._handle_start()
    ctrl._handle_shutdown()
    recorder.shutdown.assert_called_once()
    assert ctrl.state is State.IDLE


# ---- File et thread ----


def test_commands_are_processed_in_order(recorder, feedback, cb, tmp_path):
    ctrl = DictationController(
        recorder=recorder, feedback=feedback, callbacks=cb,
        recordings_dir=tmp_path / "rec",
    )
    ctrl.start()
    try:
        ctrl.on_hotkey_start()
        ctrl.on_hotkey_stop()
        deadline = time.time() + 2.0
        while not cb.submit_transcription.called and time.time() < deadline:
            time.sleep(0.01)
        cb.submit_transcription.assert_called_once()
    finally:
        ctrl.shutdown(wait=True)


def test_start_is_idempotent(recorder, feedback, cb, tmp_path):
    ctrl = DictationController(
        recorder=recorder, feedback=feedback, callbacks=cb,
        recordings_dir=tmp_path / "rec",
    )
    ctrl.start()
    try:
        thread = ctrl._thread
        ctrl.start()
        assert ctrl._thread is thread
    finally:
        ctrl.shutdown(wait=True)


# ---- Post-roll : la fin de phrase tronquée ----


def test_stop_waits_before_stopping_the_microphone(
    recorder, feedback, cb, tmp_path: Path
):
    """Régression mesurée : le micro s'arrêtait 0,1 ms après le relâchement,
    alors qu'on lâche la touche en finissant de prononcer le dernier mot — la
    fin de phrase était coupée en pleine syllabe."""
    ctrl = DictationController(
        recorder=recorder, feedback=feedback, callbacks=cb,
        recordings_dir=tmp_path / "rec", tail_padding_ms=200,
    )
    ctrl._handle_start()

    started = time.perf_counter()
    ctrl._handle_stop()
    elapsed = time.perf_counter() - started

    assert elapsed >= 0.2, f"le micro s'est arrêté après seulement {elapsed*1000:.0f} ms"
    recorder.stop.assert_called_once()


def test_zero_tail_padding_stops_immediately(
    recorder, feedback, cb, tmp_path: Path
):
    """Réglable à 0 pour qui préfère la latence minimale."""
    ctrl = DictationController(
        recorder=recorder, feedback=feedback, callbacks=cb,
        recordings_dir=tmp_path / "rec", tail_padding_ms=0,
    )
    ctrl._handle_start()

    started = time.perf_counter()
    ctrl._handle_stop()

    assert time.perf_counter() - started < 0.1


def test_tail_padding_not_applied_on_max_duration_cut(
    recorder, feedback, cb, tmp_path: Path
):
    """Coupure pour dépassement : inutile d'attendre encore, l'utilisateur ne
    tient plus la touche depuis longtemps."""
    ctrl = DictationController(
        recorder=recorder, feedback=feedback, callbacks=cb,
        max_duration_s=1, recordings_dir=tmp_path / "recordings",
        tail_padding_ms=2000,
    )
    ctrl._handle_start()
    ctrl._recording_started_at = time.monotonic() - 5.0

    started = time.perf_counter()
    ctrl._on_tick()

    assert time.perf_counter() - started < 0.5
