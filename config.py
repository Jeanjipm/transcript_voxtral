"""
Lecture / écriture de la configuration Voxtral Dictée.

Charge les valeurs par défaut depuis ./config.yaml (livré avec l'app),
puis fusionne avec les overrides utilisateur de ~/.voxtral/config.yaml.

L'utilisateur ne touche jamais ./config.yaml ; toutes ses modifications
(via l'UI Préférences) sont écrites dans ~/.voxtral/config.yaml.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields as dc_fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


T = TypeVar("T")


def _build(cls: type[T], data: dict[str, Any]) -> T:
    """Instancie une dataclass en ignorant les clés inconnues du dict.

    Tolère les anciens config.yaml qui contiennent des champs retirés
    depuis — sans ça, TypeError bloque le démarrage de l'app.
    """
    valid = {f.name for f in dc_fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in valid})


# Emplacements canoniques
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
USER_CONFIG_DIR = Path.home() / ".voxtral"
USER_CONFIG_PATH = USER_CONFIG_DIR / "config.yaml"


@dataclass
class ModelConfig:
    name: str = "mzbac/voxtral-mini-3b-4bit-mixed"
    path: str = "~/.voxtral/models/"

    @property
    def resolved_path(self) -> Path:
        return Path(self.path).expanduser()


@dataclass
class HotkeyConfig:
    combo: str = "alt_r"


@dataclass
class TranscriptionConfig:
    language: str = "auto"
    task: str = "transcribe"  # ou "translate"
    max_new_tokens: int = 1024


@dataclass
class SoundsConfig:
    enabled: bool = True
    volume: float = 0.5


@dataclass
class UIConfig:
    auto_paste: bool = True


@dataclass
class UpdatesConfig:
    # Au démarrage, un thread daemon interroge GitHub pour détecter une
    # MAJ. Désactivable via ~/.voxtral/config.yaml pour les users qui
    # ne veulent pas du tout d'appel réseau (rare en pratique).
    auto_check: bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    sounds: SoundsConfig = field(default_factory=SoundsConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    updates: UpdatesConfig = field(default_factory=UpdatesConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Fusion récursive : `override` écrase `base` clé par clé."""
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _dict_to_config(data: dict[str, Any]) -> Config:
    """Convertit un dict YAML en `Config` typé. Tolère les clés manquantes
    ET les clés obsolètes (cf. `_build`)."""
    return Config(
        model=_build(ModelConfig, data.get("model", {})),
        hotkey=_build(HotkeyConfig, data.get("hotkey", {})),
        transcription=_build(TranscriptionConfig, data.get("transcription", {})),
        sounds=_build(SoundsConfig, data.get("sounds", {})),
        ui=_build(UIConfig, data.get("ui", {})),
        updates=_build(UpdatesConfig, data.get("updates", {})),
    )


def load_config(
    user_path: Path = USER_CONFIG_PATH,
    default_path: Path = DEFAULT_CONFIG_PATH,
) -> Config:
    """
    Charge la config en fusionnant defaults projet + overrides utilisateur.

    Si le fichier utilisateur n'existe pas, retourne les defaults.
    """
    with open(default_path, "r", encoding="utf-8") as f:
        defaults = yaml.safe_load(f) or {}

    if user_path.exists():
        with open(user_path, "r", encoding="utf-8") as f:
            user_overrides = yaml.safe_load(f) or {}
        merged = _deep_merge(defaults, user_overrides)
    else:
        merged = defaults

    return _dict_to_config(merged)


def save_config(cfg: Config, user_path: Path = USER_CONFIG_PATH) -> None:
    """Écrit la config dans ~/.voxtral/config.yaml (crée le dossier au besoin)."""
    user_path.parent.mkdir(parents=True, exist_ok=True)
    with open(user_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            cfg.to_dict(),
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


def ensure_user_config_exists() -> Path:
    """
    Crée ~/.voxtral/config.yaml depuis les defaults s'il n'existe pas.
    Retourne le chemin du fichier utilisateur.
    """
    if not USER_CONFIG_PATH.exists():
        USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        cfg = load_config()
        save_config(cfg)
    return USER_CONFIG_PATH
