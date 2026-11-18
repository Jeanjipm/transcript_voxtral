"""
Y a-t-il de la parole dans cette dictée ? Silero VAD répond.

Jusqu'ici la seule garde contre une dictée sans voix était un seuil de volume
(`audio_level.py`). Mesuré le 18 novembre 2026 (BENCH-VOIX-2026-11.md), il se
trompe dans les deux sens : un bruit sans parole passe (et le modèle de
transcription invente « Merci. »), une phrase chuchotée est jetée. Silero, un
petit réseau entraîné à reconnaître la voix, a répondu juste 21 fois sur 21
là où le seuil s'arrêtait à 18.

Il est déjà fourni par le `mlx-audio` de l'app (`mlx_audio.vad`) : aucune
dépendance à ajouter, seulement ses poids (2 Mo), téléchargés une fois en
arrière-plan au démarrage (`ensure_downloaded`).

Deux règles d'usage :

- **Le modèle est un modèle MLX** : `has_speech` ne doit être appelé que
  depuis l'inference-worker, seul propriétaire des modèles (cf.
  inference_worker.py). C'est là que l'app l'appelle, juste avant de
  transcrire.
- **Jamais bloquant** : modèle pas encore téléchargé, bibliothèque absente,
  fichier illisible… `has_speech` rend alors `None` (« je ne sais pas ») et
  l'appelant retombe sur le seuil de volume. Perdre une dictée serait pire
  qu'une phrase inventée de temps en temps.
"""

from __future__ import annotations

import threading
import traceback
from pathlib import Path
from typing import Any

import hf_offline


REPO = "mlx-community/silero-vad"
SAMPLE_RATE = 16_000

# Durée de parole la plus courte qu'on accepte de reconnaître. Le défaut de
# Silero est 250 ms : mesuré, un « Oui. » dit seul (280 ms de voix) passait
# alors pour du silence, et la dictée aurait été jetée. À 100 ms il est
# reconnu, et le bruit du banc reste rejeté.
MIN_SPEECH_MS = 100

# Quand le détecteur est en service, le seuil de volume ne garde qu'un rôle :
# écarter le silence complet sans réveiller de modèle. Une phrase chuchotée
# (−62 dBFS au banc) doit le franchir pour arriver jusqu'à Silero.
DEAD_SILENCE_DBFS = -70.0


class SpeechDetector:
    """Silero VAD, chargé à la première dictée."""

    def __init__(self, repo: str = REPO) -> None:
        self.repo = repo
        self._model: Any = None
        self._download_lock = threading.Lock()

    @property
    def is_available(self) -> bool:
        """True si les poids sont sur le disque."""
        return hf_offline.is_model_cached(self.repo)

    def ensure_downloaded(self) -> bool:
        """Télécharge les poids s'ils manquent. Rend True s'ils sont là.

        Fait pour un thread d'arrière-plan : 2 Mo, une fois. Sans réseau, on
        rend simplement False et l'app continue avec le seuil de volume ; la
        tentative sera refaite au prochain lancement.
        """
        if self.is_available:
            return True
        with self._download_lock:
            if self.is_available:
                return True
            try:
                from huggingface_hub import snapshot_download

                with hf_offline.allow_network():
                    snapshot_download(repo_id=self.repo)
            except Exception as exc:  # noqa: BLE001 — hors-ligne, proxy, disque plein…
                print(f"[speech_detector] téléchargement impossible : {exc}")
                return False
        return self.is_available

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Import retardé : mlx_audio tire mlx, absent de l'environnement de test.
        from mlx_audio.vad import load  # type: ignore[import-not-found]

        # Le dossier local, pas l'identifiant du dépôt : aucun appel réseau
        # au chargement (cf. hf_offline.resolve_local_path).
        self._model = load(hf_offline.resolve_local_path(self.repo))

    def has_speech(self, wav_path: Path) -> bool | None:
        """True s'il y a de la parole, False s'il n'y en a pas, None si on ne
        peut pas le dire. À appeler depuis l'inference-worker uniquement."""
        if not self.is_available:
            return None
        try:
            import soundfile as sf

            audio, sample_rate = sf.read(str(wav_path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if audio.size == 0:
                return False
            self._ensure_loaded()
            stamps = self._model.get_speech_timestamps(
                audio,
                sample_rate=int(sample_rate),
                return_seconds=True,
                min_speech_duration_ms=MIN_SPEECH_MS,
            )
            return bool(stamps)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            return None
