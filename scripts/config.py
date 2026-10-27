"""Carrega e valida o manifesto de um experimento GNSS + ML.

O preset padrão reproduz as seções 5 e 6 do Estudo Dirigido. Caminhos e
recortes alternativos entram por outro arquivo JSON ou por opções de linha de
comando; os scripts científicos continuam consumindo um único contrato.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = SCRIPT_DIRECTORY.parent
DEFAULT_EXPERIMENT_CONFIG = DEFAULT_PROJECT_ROOT / "configs" / "estudo_dirigido.json"


def _bootstrap_cli() -> None:
    """Converte opções comuns em ambiente antes de resolver a configuração."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config")
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--ismr-dir", action="append")
    parser.add_argument("--space-weather", action="append")
    parser.add_argument("--stations")
    parser.add_argument("--station")
    options, _ = parser.parse_known_args(sys.argv[1:])

    if options.config:
        os.environ["GNSS_EXPERIMENT_CONFIG"] = str(Path(options.config).resolve())
    if options.data_root:
        os.environ["GNSS_DATA_ROOT"] = str(Path(options.data_root).resolve())
    if options.output_root:
        os.environ["GNSS_OUTPUT_ROOT"] = str(Path(options.output_root).resolve())
    if options.ismr_dir:
        os.environ["GNSS_ISMR_DIRECTORIES"] = os.pathsep.join(
            str(Path(item).resolve()) for item in options.ismr_dir
        )
    if options.space_weather:
        os.environ["GNSS_SPACE_WEATHER_FILES"] = os.pathsep.join(
            str(Path(item).resolve()) for item in options.space_weather
        )
    if options.stations:
        os.environ["GNSS_STATIONS"] = options.stations
    if options.station:
        os.environ["GNSS_STATION"] = options.station


_bootstrap_cli()


PROJECT_ROOT = Path(
    os.environ.get("GNSS_PROJECT_ROOT", DEFAULT_PROJECT_ROOT)
).resolve()
EXPERIMENT_CONFIG_PATH = Path(
    os.environ.get("GNSS_EXPERIMENT_CONFIG", DEFAULT_EXPERIMENT_CONFIG)
).resolve()


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuração do experimento não encontrada: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"JSON inválido em {path}, linha {error.lineno}, coluna {error.colno}."
        ) from error
    if not isinstance(manifest, dict):
        raise ValueError("A configuração do experimento precisa ser um objeto JSON.")
    if manifest.get("schema_version") != 1:
        raise ValueError("schema_version precisa ser 1.")
    return manifest


MANIFEST = _load_manifest(EXPERIMENT_CONFIG_PATH)


def _section(name: str) -> dict[str, Any]:
    value = MANIFEST.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"A seção '{name}' da configuração precisa ser um objeto.")
    return value


def _resolve(value: str | os.PathLike[str], base: Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _environment_paths(*names: str) -> tuple[Path, ...]:
    raw = next((os.environ.get(name) for name in names if os.environ.get(name)), None)
    if not raw:
        return ()
    return tuple(
        Path(item.strip()).resolve()
        for item in raw.split(os.pathsep)
        if item.strip()
    )


def _integer(section: dict[str, Any], key: str, default: int) -> int:
    value = section.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"'{key}' precisa ser inteiro.")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"'{key}' precisa ser inteiro.") from error


def _float(section: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(section.get(key, default))
    except (TypeError, ValueError) as error:
        raise ValueError(f"'{key}' precisa ser numérico.") from error


def _unique_strings(values: Iterable[Any], field: str) -> tuple[str, ...]:
    result = tuple(str(item).strip() for item in values if str(item).strip())
    if not result:
        raise ValueError(f"'{field}' precisa conter ao menos um valor.")
    if len(result) != len(set(result)):
        raise ValueError(f"'{field}' contém valores duplicados.")
    return result


INPUTS = _section("inputs")
OUTPUTS = _section("outputs")
PERIOD = _section("period")
PREPROCESSING = _section("preprocessing")
CLASSIFICATION = _section("classification")
XGBOOST = CLASSIFICATION.get("xgboost", {})
REGRESSION = _section("regression")
VALIDATION = _section("validation")
ALLOW_ISMR_GAPS = VALIDATION.get("allow_ismr_gaps", False)
if not isinstance(ALLOW_ISMR_GAPS, bool):
    raise ValueError("validation.allow_ismr_gaps precisa ser booleano.")
ALLOW_PARTIAL_TEST_OBSERVATIONS = VALIDATION.get("allow_partial_test_observations", False)
if not isinstance(ALLOW_PARTIAL_TEST_OBSERVATIONS, bool):
    raise ValueError("validation.allow_partial_test_observations precisa ser booleano.")
MINIMUM_TEST_OBSERVATIONS = _integer(VALIDATION, "minimum_test_observations", 1)
if MINIMUM_TEST_OBSERVATIONS < 1:
    raise ValueError("validation.minimum_test_observations precisa ser ao menos 1.")
if not isinstance(XGBOOST, dict):
    raise ValueError("A seção 'classification.xgboost' precisa ser um objeto.")

EXPERIMENT_NAME = str(MANIFEST.get("name", EXPERIMENT_CONFIG_PATH.stem)).strip()
EXPERIMENT_DECLARED_MODE = str(MANIFEST.get("mode", "derivado")).strip().lower()
EXPERIMENT_MODE = EXPERIMENT_DECLARED_MODE
EXPERIMENT_REFERENCE = str(MANIFEST.get("reference", "")).strip()
if not EXPERIMENT_NAME:
    raise ValueError("'name' não pode ser vazio.")

DATA_ROOT = _resolve(
    os.environ.get("GNSS_DATA_ROOT", INPUTS.get("data_root", "../Dados")),
    PROJECT_ROOT,
)
_ismr_environment = _environment_paths("GNSS_ISMR_DIRECTORIES")
if _ismr_environment:
    ISMR_DIRECTORIES = _ismr_environment
else:
    configured_ismr = INPUTS.get("ismr_directories", ["Dados ismr 2024"])
    if not isinstance(configured_ismr, list):
        raise ValueError("'inputs.ismr_directories' precisa ser uma lista.")
    ISMR_DIRECTORIES = tuple(_resolve(item, DATA_ROOT) for item in configured_ismr)

ISMR_FILE_EXTENSIONS = (".ismr", ".ismr.gz")
ISM_FILE_EXTENSIONS = ISMR_FILE_EXTENSIONS

STUDY_STATIONS = tuple(
    str(item).strip().upper() for item in MANIFEST.get("stations", ())
)
STUDY_STATIONS = _unique_strings(STUDY_STATIONS, "stations")
STATIONS = tuple(
    station.strip().upper()
    for station in os.environ.get(
        "GNSS_STATIONS", ",".join(STUDY_STATIONS)
    ).split(",")
    if station.strip()
)
STATIONS = _unique_strings(STATIONS, "GNSS_STATIONS")
STATION = os.environ.get("GNSS_STATION", STATIONS[0]).strip().upper()
if not STATION:
    raise ValueError("GNSS_STATION não pode ser vazio.")
if STATION not in STATIONS:
    raise ValueError(f"GNSS_STATION={STATION} não pertence a GNSS_STATIONS.")

_output_root = os.environ.get("GNSS_OUTPUT_ROOT")
if _output_root:
    output_root = Path(_output_root).resolve()
    PROCESSED_DATA = output_root / "dados_processados"
    MODELS_DIRECTORY = output_root / "modelos"
    RESULTS_DIRECTORY = output_root / "resultados"
else:
    PROCESSED_DATA = _resolve(
        OUTPUTS.get("processed_data", "dados_processados"), PROJECT_ROOT
    )
    MODELS_DIRECTORY = _resolve(OUTPUTS.get("models", "modelos"), PROJECT_ROOT)
    RESULTS_DIRECTORY = _resolve(OUTPUTS.get("results", "resultados"), PROJECT_ROOT)
EXTERNAL_DATA = _resolve(OUTPUTS.get("external_data", "dados_externos"), PROJECT_ROOT)

_weather_environment = _environment_paths(
    "GNSS_SPACE_WEATHER_FILES", "GNSS_SPACE_WEATHER_FILE"
)
if _weather_environment:
    SPACE_WEATHER_FILES = _weather_environment
else:
    configured_weather = INPUTS.get("space_weather_files", [])
    if not isinstance(configured_weather, list):
        raise ValueError("'inputs.space_weather_files' precisa ser uma lista.")
    SPACE_WEATHER_FILES = tuple(
        _resolve(item, PROJECT_ROOT) for item in configured_weather
    )
    if not SPACE_WEATHER_FILES and REGRESSION.get("space_weather_features", ()): 
        default_weather = EXTERNAL_DATA / "omniweb.csv"
        weekly_weather = tuple(sorted(EXTERNAL_DATA.glob("omniweb_*.csv")))
        legacy_weather = (
            PROJECT_ROOT / "Geomagnéticos" / "dados_solar_geomag_2024.txt"
        )
        if default_weather.is_file():
            SPACE_WEATHER_FILES = (default_weather.resolve(),)
        elif weekly_weather:
            SPACE_WEATHER_FILES = tuple(path.resolve() for path in weekly_weather)
        else:
            SPACE_WEATHER_FILES = (legacy_weather.resolve(),)
SPACE_WEATHER_FILE = SPACE_WEATHER_FILES[0] if SPACE_WEATHER_FILES else None

for folder in (PROCESSED_DATA, MODELS_DIRECTORY, RESULTS_DIRECTORY, EXTERNAL_DATA):
    folder.mkdir(parents=True, exist_ok=True)


# Recorte temporal ---------------------------------------------------------

STUDY_START = os.environ.get(
    "GNSS_STUDY_START", str(PERIOD.get("study_start", "2024-01-01 00:00:00"))
)
TRAIN_END = os.environ.get(
    "GNSS_TRAIN_END", str(PERIOD.get("train_end", "2024-03-26 23:59:59"))
)
TEST_START = os.environ.get(
    "GNSS_TEST_START", str(PERIOD.get("test_start", "2024-03-28 00:00:00"))
)
TEST_END = os.environ.get(
    "GNSS_TEST_END", str(PERIOD.get("test_end", "2024-03-28 23:59:59"))
)
try:
    _study_start_dt = datetime.fromisoformat(STUDY_START)
    _train_end_dt = datetime.fromisoformat(TRAIN_END)
    _test_start_dt = datetime.fromisoformat(TEST_START)
    _test_end_dt = datetime.fromisoformat(TEST_END)
except ValueError as error:
    raise ValueError("As datas devem usar o formato ISO AAAA-MM-DD HH:MM:SS.") from error
if not _study_start_dt <= _train_end_dt < _test_start_dt <= _test_end_dt:
    raise ValueError(
        "Período inválido: use STUDY_START <= TRAIN_END < TEST_START <= TEST_END."
    )


# Pré-processamento --------------------------------------------------------

MIN_ELEVATION = _float(PREPROCESSING, "minimum_elevation", 15.0)
MAX_ELEVATION = _float(PREPROCESSING, "maximum_elevation", 90.0)
MIN_S4 = _float(PREPROCESSING, "minimum_s4", 0.0)
MAX_S4 = _float(PREPROCESSING, "maximum_s4", 2.0)
MIN_VTEC = _float(PREPROCESSING, "minimum_vtec", 0.0)
MAX_VTEC = _float(PREPROCESSING, "maximum_vtec", 250.0)
EARTH_RADIUS_KM = _float(PREPROCESSING, "earth_radius_km", 6371.0)
IONOSPHERE_HEIGHT_KM = _float(PREPROCESSING, "ionosphere_height_km", 450.0)
IONOSPHERE_HEIGHT = IONOSPHERE_HEIGHT_KM
USE_CORRECTED_S4 = bool(PREPROCESSING.get("use_corrected_s4", True))
TIME_RESOLUTION = str(PREPROCESSING.get("time_resolution", "30min"))
FORECAST_MINUTES = _integer(PREPROCESSING, "forecast_minutes", 30)
FORECAST_HORIZON_STEPS = _integer(PREPROCESSING, "forecast_horizon_steps", 1)
ROLLING_WINDOW_STEPS = _integer(PREPROCESSING, "rolling_window_steps", 6)
DAILY_LAG_STEPS = _integer(PREPROCESSING, "daily_lag_steps", 48)
try:
    CSV_CHUNK_SIZE = int(os.environ.get("GNSS_CSV_CHUNK_SIZE", "250000"))
except ValueError as error:
    raise ValueError("GNSS_CSV_CHUNK_SIZE deve ser inteiro.") from error
if CSV_CHUNK_SIZE <= 0:
    raise ValueError("GNSS_CSV_CHUNK_SIZE deve ser positivo.")


# Núcleo 1 ----------------------------------------------------------------

RANDOM_STATE = _integer(CLASSIFICATION, "random_state", 42)
TSCV_SPLITS = _integer(CLASSIFICATION, "temporal_folds", 5)
TSCV_GAP = str(CLASSIFICATION.get("temporal_gap", "24h"))
S4_THRESHOLD = _float(CLASSIFICATION, "target_threshold_s4", 0.5)
S4_PROBABILITY_THRESHOLD = _float(
    CLASSIFICATION, "filter_probability_threshold", 0.75
)
XGB_N_ESTIMATORS = int(
    os.environ.get(
        "GNSS_XGB_N_ESTIMATORS", _integer(XGBOOST, "n_estimators", 150)
    )
)
XGB_MAX_DEPTH = _integer(XGBOOST, "max_depth", 6)
XGB_LEARNING_RATE = _float(XGBOOST, "learning_rate", 0.05)
XGB_SCALE_POS_WEIGHT = _float(XGBOOST, "scale_pos_weight", 1.0)
XGB_SUBSAMPLE = _float(XGBOOST, "subsample", 0.8)
XGB_COLSAMPLE_BYTREE = _float(XGBOOST, "colsample_bytree", 0.8)
CLASSIFICATION_TUNING = CLASSIFICATION.get("tune_hyperparameters", False)
if not isinstance(CLASSIFICATION_TUNING, bool):
    raise ValueError("classification.tune_hyperparameters precisa ser booleano.")

_configured_weather_columns = REGRESSION.get(
    "space_weather_features", ("Kp", "Dst", "F107", "Ap", "SSN")
)
if not isinstance(_configured_weather_columns, list | tuple):
    raise ValueError("'regression.space_weather_features' precisa ser uma lista.")
SPACE_WEATHER_COLUMNS = tuple(str(item).strip() for item in _configured_weather_columns if str(item).strip())
CLASSIFICATION_FEATURES = _unique_strings(
    CLASSIFICATION.get("features", ()), "classification.features"
)
if "S4" in CLASSIFICATION_FEATURES or "S4_future" in CLASSIFICATION_FEATURES:
    raise ValueError("O valor-alvo de S4 não pode entrar em classification.features.")


# Núcleo 2 ----------------------------------------------------------------

TRAIN_PERCENT = _float(REGRESSION, "train_fraction", 0.8)
LOOK_BACK = _integer(REGRESSION, "look_back", 48)
PIECEWISE_LOOK_BACK = _integer(REGRESSION, "piecewise_look_back_days", 15)
LSTM_UNITS = _integer(REGRESSION, "lstm_units", 64)
LSTM_DROPOUT = _float(REGRESSION, "dropout", 0.2)
LSTM_OPTIMIZER = str(REGRESSION.get("optimizer", "adam"))
LSTM_LOSS = str(REGRESSION.get("loss", "mse"))
try:
    LSTM_EPOCHS = int(
        os.environ.get("GNSS_LSTM_EPOCHS", _integer(REGRESSION, "epochs", 50))
    )
    LSTM_PATIENCE = int(
        os.environ.get(
            "GNSS_LSTM_PATIENCE",
            _integer(REGRESSION, "early_stopping_patience", 5),
        )
    )
except ValueError as error:
    raise ValueError("GNSS_LSTM_EPOCHS e GNSS_LSTM_PATIENCE devem ser inteiros.") from error
LSTM_BATCH_SIZE = _integer(REGRESSION, "batch_size", 400)
LSTM_LAYERS = _integer(REGRESSION, "lstm_layers", 2)
LSTM_LEARNING_RATE = _float(REGRESSION, "learning_rate", 0.001)
LSTM_LEARNING_RATE_CANDIDATES = tuple(
    float(value) for value in REGRESSION.get("learning_rate_candidates", [])
)
LSTM_OUTPUT_ACTIVATION = str(REGRESSION.get("output_activation", "linear"))
LSTM_LEAKY_RELU_SLOPE = _float(REGRESSION, "leaky_relu_slope", 0.3)
LOSS_MASKING = bool(REGRESSION.get("loss_masking", False))
if LSTM_LAYERS < 1 or LSTM_LEARNING_RATE <= 0 or any(
    not 0 < value < 1 for value in LSTM_LEARNING_RATE_CANDIDATES
):
    raise ValueError("Camadas e taxas LSTM inválidas.")
if LSTM_OUTPUT_ACTIVATION not in {"linear", "leaky_relu"}:
    raise ValueError("output_activation deve ser linear ou leaky_relu.")
if not 0 <= LSTM_LEAKY_RELU_SLOPE < 1:
    raise ValueError("leaky_relu_slope deve estar entre zero (inclusive) e um.")


@dataclass(frozen=True)
class LSTMExperiment:
    config_id: str
    organization: str
    attention: bool
    multivariate: bool

    @property
    def label(self) -> str:
        organization = (
            "Convencional" if self.organization == "convencional" else "Piecewise"
        )
        architecture = "LSTM-Attention" if self.attention else "LSTM"
        inputs = "multivariada" if self.multivariate else "univariada"
        return f"{self.config_id} — {organization}, {architecture}, {inputs}"


def _load_architectures() -> tuple[LSTMExperiment, ...]:
    raw = REGRESSION.get("architectures", ())
    if not isinstance(raw, list) or not raw:
        raise ValueError("'regression.architectures' precisa ser uma lista não vazia.")
    result: list[LSTMExperiment] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"A arquitetura {index} precisa ser um objeto.")
        config_id = str(item.get("id", "")).strip().upper()
        organization = str(item.get("organization", "")).strip().lower()
        if not config_id:
            raise ValueError(f"A arquitetura {index} não possui 'id'.")
        if organization not in {"convencional", "piecewise"}:
            raise ValueError(
                f"Arquitetura {config_id}: organization deve ser convencional ou piecewise."
            )
        for field in ("attention", "multivariate"):
            if not isinstance(item.get(field), bool):
                raise ValueError(
                    f"Arquitetura {config_id}: '{field}' deve ser booleano."
                )
        result.append(
            LSTMExperiment(
                config_id=config_id,
                organization=organization,
                attention=item["attention"],
                multivariate=item["multivariate"],
            )
        )
    ids = [item.config_id for item in result]
    combinations = [
        (item.organization, item.attention, item.multivariate) for item in result
    ]
    if len(ids) != len(set(ids)):
        raise ValueError("Os ids em regression.architectures precisam ser únicos.")
    if len(combinations) != len(set(combinations)):
        raise ValueError("Há combinações duplicadas em regression.architectures.")
    return tuple(result)


def _environment_options(name: str, allowed: set[str]) -> set[str] | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    values = {item.strip().lower() for item in raw.split(",") if item.strip()}
    invalid = values.difference(allowed)
    if not values or invalid:
        raise ValueError(
            f"{name} contém opções inválidas: "
            f"{', '.join(sorted(invalid)) or 'vazio'}."
        )
    return values


ALL_LSTM_EXPERIMENTS = _load_architectures()
_organizations = _environment_options(
    "GNSS_LSTM_ORGANIZATIONS", {"convencional", "piecewise"}
)
_attention = _environment_options("GNSS_LSTM_ATTENTION", {"false", "true"})
_multivariate = _environment_options(
    "GNSS_LSTM_MULTIVARIATE", {"false", "true"}
)
LSTM_EXPERIMENTS = tuple(
    item
    for item in ALL_LSTM_EXPERIMENTS
    if (_organizations is None or item.organization in _organizations)
    and (_attention is None or str(item.attention).lower() in _attention)
    and (_multivariate is None or str(item.multivariate).lower() in _multivariate)
)
if not LSTM_EXPERIMENTS:
    raise ValueError("Os filtros GNSS_LSTM_* removeram todas as arquiteturas.")
LSTM_ORGANIZATIONS = tuple(
    dict.fromkeys(item.organization for item in LSTM_EXPERIMENTS)
)
LSTM_ATTENTION_OPTIONS = tuple(
    dict.fromkeys(item.attention for item in LSTM_EXPERIMENTS)
)
LSTM_MULTIVARIATE_OPTIONS = tuple(
    dict.fromkeys(item.multivariate for item in LSTM_EXPERIMENTS)
)


# Validações globais -------------------------------------------------------

if not 0.0 < TRAIN_PERCENT < 1.0:
    raise ValueError("train_fraction deve estar entre zero e um.")
if not 0.0 <= MIN_ELEVATION <= MAX_ELEVATION <= 90.0:
    raise ValueError("Faixa de elevação inválida.")
if not MIN_S4 <= S4_THRESHOLD <= MAX_S4:
    raise ValueError("target_threshold_s4 deve pertencer à faixa de S4.")
if not 0.0 < S4_PROBABILITY_THRESHOLD < 1.0:
    raise ValueError("filter_probability_threshold deve estar entre zero e um.")
if any(
    value <= 0
    for value in (
        TSCV_SPLITS,
        FORECAST_MINUTES,
        LOOK_BACK,
        PIECEWISE_LOOK_BACK,
        LSTM_UNITS,
        LSTM_EPOCHS,
        LSTM_BATCH_SIZE,
    )
):
    raise ValueError("Contagens científicas e hiperparâmetros devem ser positivos.")
if LSTM_PATIENCE < 0:
    raise ValueError("early_stopping_patience não pode ser negativo.")


def _reproduction_deviations() -> list[str]:
    """Lista diferenças metodológicas em relação às seções 5 e 6 do relatório."""

    expected_architectures = tuple(
        (f"M{index}", organization, attention, multivariate)
        for index, (organization, attention, multivariate) in enumerate(
            (
                ("convencional", False, False),
                ("convencional", False, True),
                ("convencional", True, False),
                ("convencional", True, True),
                ("piecewise", False, False),
                ("piecewise", False, True),
                ("piecewise", True, False),
                ("piecewise", True, True),
            ),
            start=1,
        )
    )
    actual_architectures = tuple(
        (
            item.config_id,
            item.organization,
            item.attention,
            item.multivariate,
        )
        for item in LSTM_EXPERIMENTS
    )
    checks = {
        "cobertura_ismr_obrigatoria": not ALLOW_ISMR_GAPS,
        "periodo": (
            STUDY_START,
            TRAIN_END,
            TEST_START,
            TEST_END,
        )
        == (
            "2024-01-01 00:00:00",
            "2024-03-26 23:59:59",
            "2024-03-28 00:00:00",
            "2024-03-28 23:59:59",
        ),
        "estacoes": STATIONS == ("FRTZ", "PALM", "PRU2", "POAL"),
        "reamostragem_e_horizonte": (
            TIME_RESOLUTION,
            FORECAST_MINUTES,
            FORECAST_HORIZON_STEPS,
            ROLLING_WINDOW_STEPS,
            DAILY_LAG_STEPS,
        )
        == ("30min", 30, 1, 6, 48),
        "xgboost": (
            TSCV_SPLITS,
            TSCV_GAP,
            S4_THRESHOLD,
            S4_PROBABILITY_THRESHOLD,
            XGB_N_ESTIMATORS,
            XGB_MAX_DEPTH,
            XGB_LEARNING_RATE,
            XGB_SCALE_POS_WEIGHT,
        )
        == (5, "24h", 0.5, 0.75, 150, 6, 0.05, 1.0) and not CLASSIFICATION_TUNING,
        "lstm": (
            TRAIN_PERCENT,
            LOOK_BACK,
            PIECEWISE_LOOK_BACK,
            LSTM_UNITS,
            LSTM_DROPOUT,
            LSTM_OPTIMIZER.lower(),
            LSTM_LOSS.lower(),
            LSTM_EPOCHS,
            LSTM_BATCH_SIZE,
            LSTM_PATIENCE,
        )
        == (0.8, 48, 15, 64, 0.2, "adam", "mse", 50, 400, 5)
        and (LSTM_LAYERS, LSTM_LEARNING_RATE, LSTM_OUTPUT_ACTIVATION) == (2, 0.001, "linear")
        and not LSTM_LEARNING_RATE_CANDIDATES,
        "matriz_m1_m8": actual_architectures == expected_architectures,
    }
    return [name for name, matches in checks.items() if not matches]


REPRODUCTION_DEVIATIONS = _reproduction_deviations()
if EXPERIMENT_DECLARED_MODE == "reproducao" and REPRODUCTION_DEVIATIONS:
    EXPERIMENT_MODE = "derivado"


def resolved_experiment() -> dict[str, Any]:
    """Retorna o contrato efetivamente usado, incluindo overrides operacionais."""

    return {
        "schema_version": 1,
        "name": EXPERIMENT_NAME,
        "mode": EXPERIMENT_MODE,
        "declared_mode": EXPERIMENT_DECLARED_MODE,
        "reproduction_deviations": REPRODUCTION_DEVIATIONS,
        "reference": EXPERIMENT_REFERENCE,
        "source_config": str(EXPERIMENT_CONFIG_PATH),
        "project_root": str(PROJECT_ROOT),
        "inputs": {
            "data_root": str(DATA_ROOT),
            "ismr_directories": [str(path) for path in ISMR_DIRECTORIES],
            "space_weather_files": [str(path) for path in SPACE_WEATHER_FILES],
        },
        "outputs": {
            "processed_data": str(PROCESSED_DATA),
            "models": str(MODELS_DIRECTORY),
            "results": str(RESULTS_DIRECTORY),
        },
        "period": {
            "study_start": STUDY_START,
            "train_end": TRAIN_END,
            "test_start": TEST_START,
            "test_end": TEST_END,
        },
        "stations": list(STATIONS),
        "validation": {"allow_ismr_gaps": ALLOW_ISMR_GAPS},
        "classification": {
            "tune_hyperparameters": CLASSIFICATION_TUNING,
            "features": list(CLASSIFICATION_FEATURES),
            "target_threshold_s4": S4_THRESHOLD,
            "filter_probability_threshold": S4_PROBABILITY_THRESHOLD,
            "temporal_folds": TSCV_SPLITS,
            "temporal_gap": TSCV_GAP,
            "xgboost": {
                "n_estimators": XGB_N_ESTIMATORS,
                "max_depth": XGB_MAX_DEPTH,
                "learning_rate": XGB_LEARNING_RATE,
                "scale_pos_weight": XGB_SCALE_POS_WEIGHT,
                "subsample": XGB_SUBSAMPLE,
                "colsample_bytree": XGB_COLSAMPLE_BYTREE,
            },
        },
        "regression": {
            "lstm_layers": LSTM_LAYERS,
            "learning_rate": LSTM_LEARNING_RATE,
            "learning_rate_candidates": list(LSTM_LEARNING_RATE_CANDIDATES),
            "output_activation": LSTM_OUTPUT_ACTIVATION,
            "leaky_relu_slope": LSTM_LEAKY_RELU_SLOPE,
            "train_fraction": TRAIN_PERCENT,
            "look_back": LOOK_BACK,
            "piecewise_look_back_days": PIECEWISE_LOOK_BACK,
            "lstm_units": LSTM_UNITS,
            "dropout": LSTM_DROPOUT,
            "optimizer": LSTM_OPTIMIZER,
            "loss": LSTM_LOSS,
            "epochs": LSTM_EPOCHS,
            "batch_size": LSTM_BATCH_SIZE,
            "early_stopping_patience": LSTM_PATIENCE,
            "space_weather_features": list(SPACE_WEATHER_COLUMNS),
            "loss_masking": LOSS_MASKING,
            "architectures": [
                asdict(item) | {"label": item.label} for item in LSTM_EXPERIMENTS
            ],
        },
    }


def experiment_digest() -> str:
    payload = json.dumps(
        resolved_experiment(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def show_config() -> None:
    print("=" * 70)
    print("EXPERIMENTO GNSS + MACHINE LEARNING")
    print("Configuração:", EXPERIMENT_CONFIG_PATH)
    print("Nome/modo:", EXPERIMENT_NAME, "/", EXPERIMENT_MODE)
    print("Estação ativa:", STATION)
    print("Estações configuradas:", ", ".join(STATIONS))
    print("Treino até:", TRAIN_END)
    print("Teste:", TEST_START, "a", TEST_END)
    print("Resolução:", TIME_RESOLUTION)
    print("ISMR:", "; ".join(map(str, ISMR_DIRECTORIES)))
    print("Clima espacial:", "; ".join(map(str, SPACE_WEATHER_FILES)))
    print("Arquiteturas:", ", ".join(item.config_id for item in LSTM_EXPERIMENTS))
    print("Digest:", experiment_digest())
    print("=" * 70)


if __name__ == "__main__":
    show_config()
