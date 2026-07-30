"""Tests d'audio_capture.py — cycle de vie du stream + idempotence + buffer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import audio_capture
from audio_capture import AudioRecorder


@pytest.fixture
def mock_inputstream(monkeypatch: pytest.MonkeyPatch):
    """Remplace `sd.InputStream` par un MagicMock instanciable.

    On retourne le MagicMock fabrique pour que les tests puissent
    inspecter combien d'instances ont été créées et leurs méthodes.
    """
    factory = MagicMock(name="InputStreamFactory")
    factory.return_value = MagicMock(name="InputStreamInstance")
    monkeypatch.setattr("audio_capture.sd.InputStream", factory)
    return factory


@pytest.fixture
def rec():
    return AudioRecorder()


# ---- start / stop : cycle simple ----


def test_start_then_stop_writes_wav(
    rec: AudioRecorder, mock_inputstream, tmp_path
):
    """Cycle nominal : start → simule des chunks → stop écrit un WAV."""
    rec.start()
    assert rec.is_recording is True

    # Simule un sample reçu via le callback.
    fake_chunk = np.ones((100, 1), dtype="int16")
    rec._on_audio(fake_chunk, 100, MagicMock(), MagicMock())

    wav_path = rec.stop()
    assert rec.is_recording is False
    assert wav_path.exists()
    assert wav_path.suffix == ".wav"
    # Cleanup
    wav_path.unlink()


def test_start_creates_stream_lazily(
    rec: AudioRecorder, mock_inputstream
):
    """Le stream n'est créé qu'au 1er start, pas dans __init__."""
    assert mock_inputstream.call_count == 0
    rec.start()
    assert mock_inputstream.call_count == 1


def test_start_reuses_stream_across_calls(
    rec: AudioRecorder, mock_inputstream
):
    """2 cycles start/stop → stream créé une seule fois (le keep-warm
    de la PR #9 : on ne veut PAS de re-création)."""
    rec.start()
    wav1 = rec.stop()
    rec.start()
    wav2 = rec.stop()

    # Stream créé une seule fois.
    assert mock_inputstream.call_count == 1
    # 2 starts sur le même stream.
    assert rec._stream.start.call_count == 2

    wav1.unlink(missing_ok=True)
    wav2.unlink(missing_ok=True)


def test_start_is_idempotent_when_already_recording(
    rec: AudioRecorder, mock_inputstream
):
    """Un 2e start pendant un enregistrement en cours est ignoré."""
    rec.start()
    stream_after_first_start = rec._stream
    rec.start()  # devrait être un no-op

    assert rec._stream is stream_after_first_start
    assert rec._stream.start.call_count == 1
    rec.stop()


def test_stop_without_start_raises(
    rec: AudioRecorder, mock_inputstream
):
    """stop() sans start() préalable doit lever explicitement."""
    with pytest.raises(RuntimeError, match="stop"):
        rec.stop()


def test_stop_does_not_close_stream(
    rec: AudioRecorder, mock_inputstream
):
    """Régression PR #9 : stop() arrête mais ne ferme PAS le stream
    (sinon on perd le keep-warm hardware)."""
    rec.start()
    rec.stop()

    # stream.stop() appelé, mais pas stream.close()
    assert rec._stream.stop.called
    assert not rec._stream.close.called


# ---- shutdown ----


def test_shutdown_closes_stream(
    rec: AudioRecorder, mock_inputstream
):
    """shutdown() au quit de l'app doit close pour libérer CoreAudio."""
    rec.start()
    rec.stop()
    stream = rec._stream

    rec.shutdown()
    assert stream.close.called
    assert rec._stream is None


def test_shutdown_safe_when_stream_never_created(
    rec: AudioRecorder, mock_inputstream
):
    """shutdown() avant tout start() ne doit pas crasher."""
    rec.shutdown()  # pas d'exception
    assert rec._stream is None


def test_shutdown_swallows_already_stopped_error(
    rec: AudioRecorder, mock_inputstream
):
    """Si stream.stop() lève PortAudioError ('déjà stoppé'), shutdown
    ne doit pas remonter — on est en train de fermer de toute façon."""
    import sounddevice as sd

    rec.start()
    rec._stream.stop.side_effect = sd.PortAudioError("already stopped")
    rec.shutdown()  # pas d'exception
    assert rec._stream is None


# ---- prewarm ----


def test_prewarm_creates_stream(
    rec: AudioRecorder, mock_inputstream
):
    """prewarm() au démarrage crée le stream + start/stop pour amortir
    le coût d'init CoreAudio."""
    rec.prewarm()
    assert mock_inputstream.call_count == 1
    assert rec._stream.start.called
    assert rec._stream.stop.called


def test_prewarm_does_not_set_recording_flag(
    rec: AudioRecorder, mock_inputstream
):
    """prewarm() ne touche PAS au flag _recording (sinon il bloquerait
    un hotkey concurrent qui fait start()/stop())."""
    rec.prewarm()
    assert rec.is_recording is False


def test_prewarm_skipped_when_already_recording(
    rec: AudioRecorder, mock_inputstream
):
    """Si l'utilisateur a déjà appuyé sur le hotkey avant le prewarm,
    le stream est déjà chaud — prewarm est un no-op."""
    rec.start()  # simule hotkey concurrent
    stream_before = rec._stream
    starts_before = rec._stream.start.call_count

    rec.prewarm()  # devrait être ignoré

    # Pas de nouvelle création de stream, pas de start/stop additionnel.
    assert rec._stream is stream_before
    assert rec._stream.start.call_count == starts_before

    rec.stop()


def test_prewarm_swallows_portaudio_error(
    rec: AudioRecorder, mock_inputstream
):
    """Si stream.start() lève PortAudioError, prewarm le swallow."""
    import sounddevice as sd

    mock_inputstream.return_value.start.side_effect = sd.PortAudioError("oops")
    rec.prewarm()  # pas d'exception


# ---- _on_audio : le callback CoreAudio ----


def test_on_audio_appends_chunk_when_recording(rec: AudioRecorder):
    """Le callback ajoute les samples au buffer pendant l'enregistrement."""
    rec._recording = True
    chunk = np.array([[1], [2], [3]], dtype="int16")
    rec._on_audio(chunk, 3, MagicMock(), MagicMock())

    assert len(rec._chunks) == 1
    assert np.array_equal(rec._chunks[0], chunk)


def test_on_audio_ignores_chunk_when_not_recording(rec: AudioRecorder):
    """Garde-fou de la PR #10 : si _recording=False (entre les dictées
    ou pendant prewarm), on ignore les samples résiduels."""
    rec._recording = False
    chunk = np.array([[1], [2], [3]], dtype="int16")
    rec._on_audio(chunk, 3, MagicMock(), MagicMock())

    assert rec._chunks == []


def test_on_audio_copies_indata(rec: AudioRecorder):
    """Le callback doit copier le buffer indata — sounddevice peut le
    réutiliser pour le prochain chunk, on ne veut pas de race."""
    rec._recording = True
    indata = np.ones((10, 1), dtype="int16")
    rec._on_audio(indata, 10, MagicMock(), MagicMock())

    # Modification de indata APRÈS le callback ne doit pas affecter le buffer
    indata[0] = 999
    assert rec._chunks[0][0][0] == 1


# ---- stop : écriture WAV correcte ----


def test_stop_concatenates_chunks(
    rec: AudioRecorder, mock_inputstream, tmp_path
):
    """Plusieurs chunks → un seul WAV avec tous les samples concaténés."""
    rec.start()
    rec._on_audio(np.array([[1]] * 100, dtype="int16"), 100, None, None)
    rec._on_audio(np.array([[2]] * 50, dtype="int16"), 50, None, None)

    wav_path = rec.stop()
    import soundfile as sf
    audio, sr = sf.read(str(wav_path))
    assert sr == 16_000
    assert len(audio) == 150
    wav_path.unlink()


def test_stop_writes_empty_wav_when_no_chunks(
    rec: AudioRecorder, mock_inputstream
):
    """User relâche le hotkey instantanément → 0 chunks → on écrit
    quand même un WAV vide pour ne pas casser le pipeline aval."""
    rec.start()
    wav_path = rec.stop()
    assert wav_path.exists()
    import soundfile as sf
    audio, _ = sf.read(str(wav_path))
    assert len(audio) == 0
    wav_path.unlink()


def test_stop_clears_chunks_for_next_session(
    rec: AudioRecorder, mock_inputstream
):
    """Après stop(), les chunks doivent être vidés pour ne pas polluer
    la session suivante."""
    rec.start()
    rec._on_audio(np.ones((10, 1), dtype="int16"), 10, None, None)
    wav = rec.stop()
    assert rec._chunks == []
    wav.unlink()


# ---- is_recording (thread-safety) ----


def test_is_recording_returns_current_state(
    rec: AudioRecorder, mock_inputstream
):
    assert rec.is_recording is False
    rec.start()
    assert rec.is_recording is True
    rec.stop()
    assert rec.is_recording is False


# ---- Régression sprint 5 / cause 2 : le drapeau _recording bloqué à True ----


def test_start_resets_recording_flag_when_stream_start_fails(
    rec: AudioRecorder, mock_inputstream
):
    """Régression cause 2 : si stream.start() lève, _recording NE DOIT PAS
    rester à True.

    L'ancienne version posait le drapeau avant de démarrer le stream et ne
    le rattrapait jamais : l'app affichait ensuite l'icône « enregistrement »
    et ne captait plus rien, définitivement.
    """
    rec.start_retries = 0  # pas de reprise, on veut l'échec sec
    mock_inputstream.return_value.start.side_effect = (
        audio_capture.sd.PortAudioError("device disparu")
    )

    with pytest.raises(audio_capture.sd.PortAudioError):
        rec.start()

    assert rec.is_recording is False


def test_start_works_again_after_a_failed_start(rec: AudioRecorder):
    """Régression cause 2 : une dictée doit redevenir possible après un échec."""
    factory = MagicMock(name="InputStreamFactory")
    failing = MagicMock(name="failing")
    failing.start.side_effect = audio_capture.sd.PortAudioError("boom")
    healthy = MagicMock(name="healthy")
    factory.side_effect = [failing, healthy]

    with patch("audio_capture.sd.InputStream", factory):
        rec.start_retries = 0
        with pytest.raises(audio_capture.sd.PortAudioError):
            rec.start()
        # 2e essai : nouveau stream, sain
        rec.start()
        assert rec.is_recording is True
        healthy.start.assert_called_once()


def test_start_retry_rebuilds_stream_after_device_change(
    rec: AudioRecorder, monkeypatch: pytest.MonkeyPatch
):
    """Changement de périphérique : le 1er start échoue, on reconstruit un
    stream neuf et le 2e réussit."""
    factory = MagicMock(name="InputStreamFactory")
    failing = MagicMock(name="failing")
    failing.start.side_effect = audio_capture.sd.PortAudioError("stale device")
    healthy = MagicMock(name="healthy")
    factory.side_effect = [failing, healthy]
    monkeypatch.setattr("audio_capture.sd.InputStream", factory)

    reinit = MagicMock(name="reinit")
    monkeypatch.setattr(AudioRecorder, "_reinit_portaudio", staticmethod(reinit))

    rec.start()

    assert rec.is_recording is True
    assert factory.call_count == 2  # stream reconstruit
    reinit.assert_called_once()  # cache de périphériques purgé
    healthy.start.assert_called_once()


def test_start_raises_after_exhausting_retries(
    rec: AudioRecorder, mock_inputstream, monkeypatch: pytest.MonkeyPatch
):
    """Deux échecs = micro réellement indisponible : ça doit remonter en
    erreur visible, pas boucler."""
    monkeypatch.setattr(
        AudioRecorder, "_reinit_portaudio", staticmethod(MagicMock())
    )
    mock_inputstream.return_value.start.side_effect = (
        audio_capture.sd.PortAudioError("pas de micro")
    )

    with pytest.raises(audio_capture.sd.PortAudioError):
        rec.start()
    assert rec.is_recording is False


# ---- Régression sprint 5 / cause 3 : verrous et thread temps-réel ----


def test_on_audio_takes_no_lock(rec: AudioRecorder, mock_inputstream):
    """Régression cause 3 : _on_audio tourne dans le thread temps-réel
    CoreAudio et ne doit prendre AUCUN verrou.

    On tient le verrou d'état, puis on appelle _on_audio depuis un autre
    thread : s'il tentait de l'acquérir, il resterait bloqué.
    """
    import threading

    rec.start()
    done = threading.Event()

    def feed() -> None:
        rec._on_audio(np.ones((10, 1), dtype="int16"), 10, None, None)
        done.set()

    with rec._state_lock:
        t = threading.Thread(target=feed, daemon=True)
        t.start()
        assert done.wait(timeout=1.0), "_on_audio a bloqué sur le verrou"

    rec.stop().unlink()


def test_prewarm_does_not_hold_lock_during_construction(
    rec: AudioRecorder, monkeypatch: pytest.MonkeyPatch
):
    """Régression cause 3 : la construction du stream (~4,1 s) ne doit pas se
    faire sous verrou, sinon un appui sur le raccourci s'y bloque — et le
    callback clavier de macOS n'a qu'environ 1 seconde de budget."""
    observed: dict[str, bool] = {}

    def slow_factory(**_kwargs):
        # Pendant la « construction », le verrou doit être libre.
        acquired = rec._state_lock.acquire(blocking=False)
        observed["lock_free"] = acquired
        if acquired:
            rec._state_lock.release()
        return MagicMock(name="stream")

    monkeypatch.setattr("audio_capture.sd.InputStream", slow_factory)
    rec.prewarm()

    assert observed.get("lock_free") is True


def test_start_rebinds_chunks_instead_of_clearing(
    rec: AudioRecorder, mock_inputstream
):
    """start() doit RÉASSIGNER _chunks, pas le vider : un callback temps-réel
    retardataire garde une référence à l'ancienne liste et doit écrire
    dedans, sans polluer la nouvelle dictée."""
    rec.start()
    old_list = rec._chunks
    rec.stop().unlink()

    rec.start()
    assert rec._chunks is not old_list

    # Le retardataire écrit dans l'ancienne liste : la nouvelle reste intacte.
    old_list.append(np.ones((5, 1), dtype="int16"))
    assert rec._chunks == []
    rec.stop().unlink()


def test_stop_keeps_audio_when_device_vanished(
    rec: AudioRecorder, mock_inputstream
):
    """Casque débranché en pleine dictée : stream.stop() lève, mais on doit
    quand même récupérer ce qui a été capté avant la coupure."""
    rec.start()
    rec._on_audio(np.ones((100, 1), dtype="int16"), 100, None, None)
    mock_inputstream.return_value.stop.side_effect = (
        audio_capture.sd.PortAudioError("device disparu")
    )

    wav_path = rec.stop()

    import soundfile as _sf

    assert _sf.info(str(wav_path)).frames == 100  # l'audio capté est préservé
    assert rec.is_recording is False
    assert rec._stream is None  # stream jeté, le prochain start reconstruira
    wav_path.unlink()


# ---- Rembourrage de silence (fins de phrase tronquées) ----


def test_silence_padding_extends_both_ends(mock_inputstream):
    """Les modèles perdent des phonèmes quand la parole commence ou finit pile
    au bord du fichier. On rembourre des deux côtés."""
    import soundfile as _sf

    rec = AudioRecorder(silence_padding_ms=250)
    rec.start()
    rec._on_audio(np.ones((16_000, 1), dtype="int16"), 16_000, None, None)
    wav = rec.stop()

    # 1 s de signal + 2 x 250 ms de silence
    assert _sf.info(str(wav)).frames == 16_000 + 2 * 4_000
    wav.unlink()


def test_no_padding_when_disabled(mock_inputstream):
    import soundfile as _sf

    rec = AudioRecorder(silence_padding_ms=0)
    rec.start()
    rec._on_audio(np.ones((16_000, 1), dtype="int16"), 16_000, None, None)
    wav = rec.stop()

    assert _sf.info(str(wav)).frames == 16_000
    wav.unlink()


def test_padding_skipped_on_empty_recording(mock_inputstream):
    """Un enregistrement vide doit rester vide : rembourrer du néant
    produirait un WAV de silence pur que le pipeline prendrait pour de
    l'audio à transcrire."""
    import soundfile as _sf

    rec = AudioRecorder(silence_padding_ms=250)
    rec.start()
    wav = rec.stop()

    assert _sf.info(str(wav)).frames == 0
    wav.unlink()
