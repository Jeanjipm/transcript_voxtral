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


def _make_wav(
    tmp_path: Path, seconds: float, name: str = "d.wav", amplitude: int = 3000
) -> Path:
    """Écrit un WAV réel de la durée demandée (sf.info et la mesure de niveau
    sont utilisés pour de vrai).

    Par défaut une sinusoïde à ~-24 dBFS : une « voix », pour que le filtre
    anti-silence (actif par défaut) laisse passer les dictées des tests.
    `amplitude=0` produit du silence numérique.
    """
    path = tmp_path / name
    frames = max(0, int(16_000 * seconds))
    t = np.arange(frames) / 16_000
    tone = (amplitude * np.sin(2 * np.pi * 220 * t)).astype("int16")
    sf.write(path, tone.reshape(-1, 1), 16_000, subtype="PCM_16")
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


# ---- Filtre anti-silence (sprint 6b, issue 6) ----


def test_silent_dictation_is_discarded_and_logged(
    ctrl, recorder, cb, feedback, tmp_path: Path, capsys
):
    """Deux secondes de silence → rien n'est envoyé au modèle (qui inventerait
    « Merci. »), le WAV est supprimé, l'état revient à prêt, et le journal
    dit pourquoi."""
    wav = _make_wav(tmp_path, 2.0, "silence.wav", amplitude=0)
    recorder.stop.return_value = wav
    ctrl._handle_start()
    feedback.reset_mock()

    ctrl._handle_stop()

    cb.submit_transcription.assert_not_called()
    feedback.play_stop.assert_not_called()
    assert not wav.exists()
    assert ctrl.state is State.IDLE
    cb.on_state_change.assert_called_with(State.IDLE, "État : prêt")
    err = capsys.readouterr().err
    assert "silence jeté" in err
    assert "-45 dBFS" in err


def test_quiet_room_noise_is_discarded(ctrl, recorder, cb, tmp_path: Path):
    """Bruit de fond réaliste (−60 dBFS), pas du zéro numérique : jeté aussi."""
    wav = _make_wav(tmp_path, 2.0, "bruit.wav", amplitude=30)  # ≈ −64 dBFS
    recorder.stop.return_value = wav
    ctrl._handle_start()
    ctrl._handle_stop()
    cb.submit_transcription.assert_not_called()


def test_voice_is_submitted(ctrl, recorder, cb, tmp_path: Path):
    """Le défaut de `_make_wav` (≈ −24 dBFS) est une voix : transmise."""
    ctrl._handle_start()
    ctrl._handle_stop()
    cb.submit_transcription.assert_called_once()


def test_brief_word_in_long_silence_is_submitted(
    ctrl, recorder, cb, tmp_path: Path
):
    """Un « oui » de 200 ms après deux secondes de réflexion n'est pas du
    silence : la mesure porte sur la fenêtre la plus forte, pas la moyenne."""
    path = tmp_path / "oui.wav"
    silence = np.zeros(16_000 * 2)
    t = np.arange(int(16_000 * 0.2)) / 16_000
    word = 3000 * np.sin(2 * np.pi * 220 * t)
    audio = np.concatenate([silence, word, np.zeros(16_000)]).astype("int16")
    sf.write(path, audio.reshape(-1, 1), 16_000, subtype="PCM_16")
    recorder.stop.return_value = path

    ctrl._handle_start()
    ctrl._handle_stop()

    cb.submit_transcription.assert_called_once_with(path)


def test_silence_filter_can_be_disabled(recorder, feedback, cb, tmp_path: Path):
    """Seuil à -120 : tout passe, même le silence numérique."""
    ctrl = DictationController(
        recorder=recorder, feedback=feedback, callbacks=cb,
        recordings_dir=tmp_path / "rec", silence_threshold_dbfs=-120.0,
    )
    recorder.stop.return_value = _make_wav(tmp_path, 2.0, "s.wav", amplitude=0)
    ctrl._handle_start()
    ctrl._handle_stop()
    cb.submit_transcription.assert_called_once()


def test_custom_threshold_is_honoured(recorder, feedback, cb, tmp_path: Path):
    """Seuil remonté à -20 : la « voix » de test (−24 dBFS) devient du silence."""
    ctrl = DictationController(
        recorder=recorder, feedback=feedback, callbacks=cb,
        recordings_dir=tmp_path / "rec", silence_threshold_dbfs=-20.0,
    )
    ctrl._handle_start()
    ctrl._handle_stop()
    cb.submit_transcription.assert_not_called()


def test_unreadable_level_does_not_block_transcription(
    ctrl, recorder, cb, monkeypatch: pytest.MonkeyPatch
):
    """Si la mesure échoue, on transcrit quand même : perdre une dictée est
    pire qu'une hallucination."""
    import audio_level

    def boom(path):
        raise OSError("lecture impossible")

    monkeypatch.setattr(audio_level, "measure_peak_rms_dbfs", boom)
    ctrl._handle_start()
    ctrl._handle_stop()
    cb.submit_transcription.assert_called_once()


def test_silent_dictation_leaves_in_flight_untouched(ctrl, recorder, cb, tmp_path):
    """Une dictée jetée pendant qu'une transcription tourne ne doit pas
    remettre l'icône à « prêt »."""
    ctrl._handle_start()
    ctrl._handle_stop()  # voix → en vol = 1
    recorder.stop.return_value = _make_wav(tmp_path, 2.0, "s.wav", amplitude=0)
    ctrl._handle_start()
    ctrl._handle_stop()  # silence
    assert ctrl.state is State.PENDING
    assert ctrl._in_flight == 1


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
    """À l'arrêt : on fait relire la liste des périphériques à PortAudio
    (`reset_device`) PUIS on prépare un flux neuf (`prewarm`). Un simple
    `shutdown` + `prewarm` rebâtirait le flux sur l'ancien micro."""
    ctrl._handle_device_changed()
    recorder.reset_device.assert_called_once()
    recorder.prewarm.assert_called_once()
    assert recorder.mock_calls.index(("reset_device", (), {})) < (
        recorder.mock_calls.index(("prewarm", (), {}))
    )


def test_device_changed_left_alone_during_recording(ctrl, recorder):
    """En pleine dictée on ne touche à rien : AudioRecorder.stop() sait déjà
    survivre à la disparition du périphérique et conserve l'audio."""
    ctrl._handle_start()
    recorder.reset_device.reset_mock()
    recorder.prewarm.reset_mock()

    ctrl._handle_device_changed()

    assert recorder.reset_device.called is False
    assert recorder.prewarm.called is False


def test_device_changed_during_recording_is_applied_after_stop(ctrl, recorder, cb):
    """Le changement n'est pas perdu : il est appliqué à la fin de la dictée,
    APRÈS la soumission de la transcription (reconstruire un flux coûte
    plusieurs secondes que la transcription n'a pas à attendre)."""
    ctrl._handle_start()
    ctrl._handle_device_changed()
    recorder.reset_device.assert_not_called()

    ctrl._handle_stop()

    recorder.reset_device.assert_called_once()
    recorder.prewarm.assert_called_once()
    cb.submit_transcription.assert_called_once()
    # La soumission précède la reconstruction.
    manager_order = [c[0] for c in recorder.mock_calls]
    assert manager_order.index("stop") < manager_order.index("reset_device")


def test_device_changed_during_recording_is_applied_even_if_too_short(
    ctrl, recorder, tmp_path: Path
):
    recorder.stop.return_value = _make_wav(tmp_path, 0.2, "court.wav")
    ctrl._handle_start()
    ctrl._handle_device_changed()

    ctrl._handle_stop()

    recorder.reset_device.assert_called_once()


def test_device_refresh_is_applied_once(ctrl, recorder):
    """Deux changements reçus pendant une même dictée → une seule
    reconstruction, et plus rien à la dictée suivante."""
    ctrl._handle_start()
    ctrl._handle_device_changed()
    ctrl._handle_device_changed()
    ctrl._handle_stop()
    assert recorder.reset_device.call_count == 1

    ctrl._handle_transcription_done()
    ctrl._handle_start()
    ctrl._handle_stop()
    assert recorder.reset_device.call_count == 1


def test_device_changed_during_arming_is_deferred(ctrl, recorder):
    """ARMING = micro en cours d'ouverture : même traitement que RECORDING."""
    ctrl._state = State.ARMING
    ctrl._handle_device_changed()
    recorder.reset_device.assert_not_called()
    assert ctrl._device_refresh_pending is True


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


# ---- Régression : l'ordre d'arrêt jeté quand la file est pleine ----
#
# Constaté dans voxtral.log : « [dictation] file pleine, commande shutdown
# abandonnée ». Le worker était occupé (micro qui tarde à répondre), huit
# commandes s'étaient accumulées, et l'arrêt a subi le même sort qu'un appui de
# trop. Le micro n'était pas libéré ; seul le os._exit de secours a conclu.


def test_shutdown_is_honoured_even_when_the_queue_is_full(
    recorder, feedback, cb, tmp_path, capsys,
):
    """Reproduit l'incident : worker bloqué, file saturée, puis shutdown."""
    import threading

    from dictation_controller import _QUEUE_MAX

    unblock = threading.Event()
    recorder.prewarm.side_effect = lambda: unblock.wait(timeout=5.0)

    ctrl = DictationController(
        recorder=recorder, feedback=feedback, callbacks=cb,
        recordings_dir=tmp_path / "rec",
    )
    ctrl.start()
    try:
        ctrl.request_prewarm()               # le worker part et se bloque
        deadline = time.time() + 2.0
        while recorder.prewarm.called is False and time.time() < deadline:
            time.sleep(0.01)
        assert recorder.prewarm.called

        for _ in range(_QUEUE_MAX):          # la file se remplit derrière lui
            ctrl.request_prewarm()
        assert ctrl._queue.full()

        thread = ctrl._thread
        ctrl.shutdown(wait=False)            # la commande en file est perdue…
        assert "file pleine" in capsys.readouterr().err

        unblock.set()                        # …mais le drapeau, lui, est vu.
        thread.join(timeout=3.0)
        assert thread.is_alive() is False, "le worker doit s'arrêter malgré la file pleine"
        recorder.shutdown.assert_called()
    finally:
        unblock.set()


def test_shutdown_releases_the_microphone_when_recording(recorder, feedback, cb, tmp_path):
    """Chemin normal avec le vrai thread : arrêt pendant un enregistrement →
    stop() puis shutdown() du micro, et le thread se termine."""
    ctrl = DictationController(
        recorder=recorder, feedback=feedback, callbacks=cb,
        recordings_dir=tmp_path / "rec",
    )
    ctrl.start()
    ctrl.on_hotkey_start()
    deadline = time.time() + 2.0
    while ctrl.state is not State.RECORDING and time.time() < deadline:
        time.sleep(0.01)
    assert ctrl.state is State.RECORDING

    thread = ctrl._thread
    ctrl.shutdown(wait=True)
    assert thread.is_alive() is False
    recorder.stop.assert_called()
    recorder.shutdown.assert_called_once()
