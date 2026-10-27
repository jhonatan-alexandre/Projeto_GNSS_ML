"""Reaggregate immutable minute observations and retrain the scintillation core.

Nominal keys name the *start* of each interval. Features become available at
its end; the next interval aggregate becomes observable 15 minutes later.
Interpolation is retrospective and isolated across development/context/test.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType

import joblib
import numpy as np
import pandas as pd

from .protocolo import Protocol

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
WEATHER_COLUMNS = ("Kp", "Dst", "F107", "Ap", "SSN")
CLASSIFICATION_FEATURES = ("elev", "azim_sin", "azim_cos", "hour_sin", "hour_cos",
    "month_sin", "month_cos", "S4_mean_3h", "S4_std_3h", "S4_lag_24h", *WEATHER_COLUMNS)


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _scope_values(protocol, station="FRTZ", output_root=None, classification=None):
    p = protocol
    c = classification or {}
    x = c.get("xgboost", {})
    folder = Path(output_root or ROOT)
    return dict(CSV_CHUNK_SIZE=250000, DAILY_LAG_STEPS=p.daily_lag_steps,
        PROCESSED_DATA=str(folder/"dados_processados"), ROLLING_WINDOW_STEPS=p.rolling_window_steps,
        SPACE_WEATHER_COLUMNS=WEATHER_COLUMNS, SPACE_WEATHER_FILES=(), STATION=station, STUDY_START=p.study_start,
        TEST_END=p.test_end, TIME_RESOLUTION=p.time_resolution,
        CLASSIFICATION_FEATURES=tuple(c.get("features", CLASSIFICATION_FEATURES)),
        FORECAST_HORIZON_STEPS=p.forecast_horizon_steps, FORECAST_MINUTES=p.forecast_minutes,
        S4_THRESHOLD=p.s4_threshold, CLASSIFICATION_TUNING=True,
        MODELS_DIRECTORY=str(folder/"modelos"), RESULTS_DIRECTORY=str(folder/"resultados"),
        RANDOM_STATE=c.get("random_state", 42), S4_PROBABILITY_THRESHOLD=.75,
        TEST_START=p.test_start, TRAIN_END=p.train_end,
        # The inherited validator compares nominal target keys. Add one bin so
        # its 24-hour purge also includes the time needed to observe that label.
        TSCV_GAP=str(pd.Timedelta(p.temporal_gap)+pd.Timedelta(p.time_resolution)),
        TSCV_SPLITS=c.get("temporal_folds", 5), XGB_LEARNING_RATE=x.get("learning_rate", .05),
        XGB_MAX_DEPTH=x.get("max_depth", 6), XGB_N_ESTIMATORS=x.get("n_estimators", 150),
        XGB_SCALE_POS_WEIGHT=1., XGB_SUBSAMPLE=x.get("subsample", .8),
        XGB_COLSAMPLE_BYTREE=x.get("colsample_bytree", .8))


@contextmanager
def _legacy_modules(protocol, station="FRTZ", output_root=None, classification=None):
    """Load independent copies with a scoped config; restore legacy import state."""
    saved_config = sys.modules.get("config")
    saved_path = sys.path.copy()
    config = ModuleType("config")
    config.__dict__.update(_scope_values(protocol, station, output_root, classification))
    sys.modules["config"] = config
    modules = {}
    names = []
    try:
        for key, filename in (("features", "06_feature_engineering.py"),
                ("classification", "07_dataset_classificacao.py"),
                ("classifier", "08_train_xgboost.py"), ("interpolation", "interpolacao_vtec.py"),
                ("weather", "05_integrar_clima_espacial.py")):
            name = "_duplo_15min_" + key
            spec = importlib.util.spec_from_file_location(name, SCRIPTS/filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            names.append(name)
            spec.loader.exec_module(module)
            modules[key] = module
        yield modules
    finally:
        for name in names:
            sys.modules.pop(name, None)
        if saved_config is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = saved_config
        sys.path[:] = saved_path


def audit_clean_source(path, protocol: Protocol, chunk_size=250000):
    """Check physical quality and measured epoch resolution without copying sources."""
    path = Path(path)
    count, min_delta, first, last, previous = 0, np.inf, None, None, None
    last_epoch_svids = set()
    with pd.read_csv(path, usecols=["datetime", "svid", "elev", "azim", "S4", "VTEC"],
                     chunksize=chunk_size) as reader:
        for chunk in reader:
            chunk["datetime"] = pd.to_datetime(chunk["datetime"], errors="raise")
            if chunk.datetime.dt.second.ne(0).any():
                raise ValueError("Clean source is not a minute-epoch CSV.")
            if not chunk.datetime.is_monotonic_increasing:
                raise ValueError("Clean source epochs must be sorted.")
            if previous is not None and chunk.datetime.iloc[0] < previous:
                raise ValueError("Clean source chunks must be sorted.")
            selected = chunk.loc[chunk.datetime.between(protocol.study_start, protocol.test_end)]
            if not selected.empty:
                numeric = selected[["svid", "elev", "S4", "VTEC"]].apply(pd.to_numeric, errors="raise")
                valid = (np.isfinite(numeric[["svid", "elev", "S4"]]).all(axis=1)
                    & numeric.svid.ge(1) & numeric.svid.eq(np.floor(numeric.svid))
                    & numeric.elev.between(15,90) & numeric.S4.between(0,2)
                    & (numeric.VTEC.isna() | (np.isfinite(numeric.VTEC)&numeric.VTEC.between(0,250))))
                if not valid.all() or selected.duplicated(["datetime", "svid"]).any():
                    raise ValueError("Clean source violates historical physical quality criteria.")
                epochs = selected.datetime.drop_duplicates()
                previous_epoch = pd.Timestamp(last) if last is not None else None
                if previous_epoch == epochs.iloc[0]:
                    boundary_svids = set(numeric.loc[selected.datetime.eq(previous_epoch), "svid"].astype("int64"))
                    if last_epoch_svids.intersection(boundary_svids):
                        raise ValueError("Clean source has duplicate measurements across chunk boundaries.")
                deltas = epochs.diff().dt.total_seconds()
                if last is not None:
                    boundary_delta = (epochs.iloc[0]-previous_epoch).total_seconds()
                    if boundary_delta > 0:
                        min_delta = min(min_delta, float(boundary_delta))
                positive = deltas[deltas.gt(0)]
                if len(positive):
                    min_delta = min(min_delta, float(positive.min()))
                final_epoch = epochs.iloc[-1]
                final_svids = set(numeric.loc[selected.datetime.eq(final_epoch), "svid"].astype("int64"))
                if final_epoch == previous_epoch:
                    last_epoch_svids.update(final_svids)
                else:
                    last_epoch_svids = final_svids
                first = first or str(epochs.iloc[0])
                last = str(final_epoch)
                count += len(selected)
            previous = chunk.datetime.iloc[-1]
    if count == 0 or min_delta >= protocol.resolution_minutes*60:
        raise ValueError("Source resolution is not finer than 15min; do not reuse 30min aggregates.")
    return dict(path=str(path.resolve()), sha256=_sha256(path), rows_in_period=count,
        first_epoch=first, last_epoch=last, minimum_positive_epoch_delta_seconds=min_delta,
        quality_verified=True, source_kind="historical_minute_clean_csv")


def build_features(aggregated, weather, protocol: Protocol, station):
    """Reuse circular azimuth/time features and rolling windows on a regular grid."""
    with _legacy_modules(protocol, station) as legacy:
        result = legacy["features"].criar_features(aggregated, weather)
    delta = pd.Timedelta(protocol.time_resolution)
    result["station"] = station
    result["window_start"] = result.datetime
    result["window_end"] = result.datetime + delta
    result["input_available_at"] = result.window_end
    result["emission_datetime"] = result.window_end
    if "weather_available_at" in weather:
        availability = weather[["datetime", "weather_available_at"]].copy()
        result = result.merge(availability, on="datetime", how="left", validate="many_to_one")
        if (result.weather_available_at > result.emission_datetime).any():
            raise ValueError("Weather inputs are unavailable at emission.")
    result["partition"] = result.datetime.map(protocol.partition)
    return result


def build_classification_dataset(features, protocol: Protocol):
    """Keep inference rows when future S4 is absent; labels never gate inference."""
    with _legacy_modules(protocol) as legacy:
        labels = legacy["classification"].criar_alvo(features)
    data = features.sort_values(["svid", "datetime"]).copy()
    data["target_datetime"] = data.datetime + pd.Timedelta(protocol.time_resolution)
    data = data.loc[data.target_datetime.le(pd.Timestamp(protocol.test_end))]
    data = data.merge(labels[["datetime", "svid", "S4_future", "classe_future"]],
                      on=["datetime", "svid"], how="left", validate="one_to_one")
    data["target_available_at"] = data.target_datetime + pd.Timedelta(protocol.time_resolution)
    data["target_partition"] = data.target_datetime.map(protocol.partition)
    return data.sort_values(["target_datetime", "svid"]).reset_index(drop=True)


def build_regression_dataset(features, predictions, protocol: Protocol):
    """Remove every predicted alert; observed labels do not participate in the mask."""
    keys = ["station", "svid", "datetime"]
    features = features.copy()
    features["datetime"] = pd.to_datetime(features.datetime)
    predictions = predictions.copy().rename(columns={"datetime": "feature_datetime", "target_datetime": "datetime"})
    predictions["datetime"] = pd.to_datetime(predictions.datetime)
    if predictions.duplicated(keys).any():
        raise ValueError("Duplicate station/SVID/target prediction keys.")
    columns = [*keys, "probabilidade_severa", "limiar_probabilidade"]
    for name in ("emission_datetime", "input_available_at", "target_available_at", "origem_probabilidade"):
        if name in predictions:
            columns.append(name)
    mask = features[keys+["VTEC"]].merge(predictions[columns], on=keys, how="left", validate="one_to_one")
    mask["probabilidade_severa"] = pd.to_numeric(mask.probabilidade_severa, errors="raise")
    mask["limiar_probabilidade"] = pd.to_numeric(mask.limiar_probabilidade, errors="raise")
    prob = mask.probabilidade_severa
    if ((prob.notna()) & (~np.isfinite(prob) | ~prob.between(0,1))).any():
        raise ValueError("Invalid classifier probabilities.")
    mask["prediction_available"] = prob.notna()
    mask["sinalizar_filtro"] = prob.ge(mask.limiar_probabilidade) & mask.prediction_available
    mask["removed_observed_measurement"] = mask.sinalizar_filtro & mask.VTEC.notna()
    masked = features.merge(mask[keys+["prediction_available", "sinalizar_filtro", "removed_observed_measurement"]], on=keys)
    masked["VTEC_filtered"] = masked.VTEC.mask(masked.sinalizar_filtro)
    masked["observed_svid"] = masked.svid.where(masked.VTEC.notna())
    aggregation = dict(VTEC=("VTEC", "median"), VTEC_filtered=("VTEC_filtered", "median"),
        n_svid_observados=("observed_svid", "nunique"),
        n_svid_filtrados=("removed_observed_measurement", "sum"),
        n_predictions_available=("prediction_available", "sum"),
        **{c:(c,"first") for c in WEATHER_COLUMNS})
    series = masked.groupby(["station", "datetime"], as_index=False).agg(**aggregation)
    results, references = [], []
    with _legacy_modules(protocol) as legacy:
        interpolate = legacy["interpolation"].interpolar_por_periodo
        for station, group in series.groupby("station"):
            grid = pd.date_range(protocol.study_start, pd.Timestamp(protocol.test_end).floor(protocol.time_resolution), freq=protocol.time_resolution)
            group = group.set_index("datetime").reindex(grid).rename_axis("datetime")
            group["station"] = station
            group["VTEC_observed_reference"] = group.VTEC
            group["observed_reference_available"] = group.VTEC.notna()
            group["VTEC_original"] = group.pop("VTEC")
            group["partition"] = [protocol.partition(t) for t in group.index]
            group["target_available_at"] = group.index + pd.Timedelta(protocol.time_resolution)
            group["emission_datetime"] = group.index
            references.append(group[["station", "VTEC_observed_reference", "observed_reference_available", "target_available_at", "partition"]].reset_index())
            for name in ("VTEC_original", "VTEC_filtered"):
                # A stricter separation than the inherited history/test boundary:
                # context observations cannot fill development targets either.
                for partition in ("development", "context", "test"):
                    selected = group.partition.eq(partition)
                    group.loc[selected, name] = interpolate(group.loc[selected, name], None)
            for name in ("n_svid_observados", "n_svid_filtrados", "n_predictions_available"):
                group[name] = group[name].fillna(0).astype("int64")
            group["observado_teste"] = group.partition.eq("test") & group.observed_reference_available
            group["loss_weight"] = group.VTEC_original.notna().astype(float)
            group["retrospective_interpolation"] = True
            results.append(group.reset_index())
    return pd.concat(results, ignore_index=True), pd.concat(references, ignore_index=True), mask


def build_weather(paths, protocol: Protocol):
    """Read raw hourly weather; hold a value only after its nominal hour ends.

    These are conservative proxy availability times, not measured publication
    timestamps. Missing hours remain missing after a one-hour asof tolerance.
    """
    parts = []
    with _legacy_modules(protocol) as legacy:
        module = legacy["weather"]
        for path in paths:
            raw = module.carregar_clima_espacial(path)
            normalized = {module._normalizar(c):c for c in raw.columns}
            part = pd.DataFrame({"source_datetime": module._criar_datetime(raw, normalized)})
            for name in WEATHER_COLUMNS:
                original = module._encontrar(normalized, module.ALIASES[name])
                if original is None:
                    raise ValueError(f"Missing raw weather column: {name}")
                values = pd.to_numeric(raw[original], errors="coerce")
                values = values.mask(~np.isfinite(values) | values.isin(module.CODIGOS_AUSENTES[name]))
                if name == "Kp" and values.max(skipna=True) > 9:
                    values = values/10.
                part[name] = values
            parts.append(part)
    source = pd.concat(parts, ignore_index=True).dropna(subset=["source_datetime"])
    source = source.sort_values("source_datetime").drop_duplicates("source_datetime",keep="last")
    if not source.source_datetime.eq(source.source_datetime.dt.floor("h")).all():
        raise ValueError("Expected raw hourly weather timestamps.")
    source["weather_available_at"] = source.source_datetime+pd.Timedelta("1h")
    grid = pd.DataFrame({"datetime":pd.date_range(protocol.study_start, protocol.test_end,freq=protocol.time_resolution)})
    grid["emission_datetime"] = grid.datetime+pd.Timedelta(protocol.time_resolution)
    result = pd.merge_asof(grid, source.sort_values("weather_available_at"),
        left_on="emission_datetime",right_on="weather_available_at",direction="backward", tolerance=pd.Timedelta("59min59s"))
    return result[["datetime", *WEATHER_COLUMNS, "weather_available_at", "source_datetime"]]


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False, default=str),encoding="utf-8")


def _space_check(output_root):
    free = shutil.disk_usage(output_root).free
    if free < 5*1024**3:
        raise RuntimeError(f"Training/checkpoint requires 5 GiB free; available: {free/1024**3:.3f} GiB")


def _train_classifier(dataset, protocol, station, output_root, classification):
    """Reuse the four historical candidates, F2 selection and fold-local SMOTE."""
    folder = Path(output_root)/"resultados"/station/"nucleo1"
    model_folder = Path(output_root)/"modelos"/station/"nucleo1"
    folder.mkdir(parents=True,exist_ok=True)
    model_folder.mkdir(parents=True,exist_ok=True)
    known = dataset.loc[dataset.classe_future.notna() & dataset.target_partition.eq("development")].copy()
    # A nominal 16-Feb 23:45 label is only available at 17-Feb 00:00.
    # This end-of-development fitting boundary is explicitly persisted below.
    if (known.target_available_at > protocol.development_available_until).any():
        raise ValueError("Classifier labels extend past development availability boundary.")
    _space_check(output_root)
    with _legacy_modules(protocol, station, output_root, classification) as legacy:
        module = legacy["classifier"]
        deps = module._dependencias()
        params, threshold, candidates, curves = module.selecionar_hiperparametros(known, deps)
        # Validation scoring needs observed future labels; inference does not.
        # Use the same folds/candidate as the inherited search, while predicting
        # all feature rows inside each validation window, including missing y.
        times = np.sort(known.target_datetime.unique())
        oof = pd.Series(np.nan,index=dataset.index,dtype=float)
        fold_rows = []
        for fold, (_, val_indices) in enumerate(deps["TimeSeriesSplit"](n_splits=module.TSCV_SPLITS).split(times),start=1):
            start, end = pd.Timestamp(times[val_indices[0]]), pd.Timestamp(times[val_indices[-1]])
            train = known.loc[known.target_datetime < start-pd.Timedelta(module.TSCV_GAP)]
            infer = dataset.loc[dataset.target_datetime.between(start,end)]
            observed = known.loc[known.target_datetime.between(start,end)]
            if train.empty or observed.empty:
                continue
            fold_imputer, fold_model, fold_counts = module._ajustar_modelo(train,deps,params)
            oof.loc[infer.index] = fold_model.predict_proba(fold_imputer.transform(
                infer[list(module.CLASSIFICATION_FEATURES)].to_numpy()))[:,1]
            fold_rows.append(dict(fold=fold,inicio_validacao=start,fim_validacao=end,
                n_treino=len(train),n_validacao=len(observed),n_inference=len(infer),
                training_latest_label_available_at=str(train.target_available_at.max()),
                first_validation_emission_at=str(start),
                **fold_counts,**module._metricas(observed.classe_future.to_numpy(),oof.loc[observed.index].to_numpy(),deps,threshold)))
        folds = pd.DataFrame(fold_rows)
        imputer, model, counts = module._ajustar_modelo(known, deps, params)
        probability = pd.Series(np.nan, index=dataset.index, dtype=float)
        probability.loc[oof.index] = oof
        posterior = dataset.target_partition.isin(["context", "test"])
        if posterior.any():
            probability.loc[posterior] = model.predict_proba(imputer.transform(
                dataset.loc[posterior, list(module.CLASSIFICATION_FEATURES)].to_numpy()))[:,1]
        cols = ["station", "datetime", "svid", "input_available_at", "emission_datetime",
                "target_datetime", "target_available_at", "target_partition", "S4_future", "classe_future"]
        predictions = dataset[cols].copy()
        predictions["probabilidade_severa"] = probability
        predictions["limiar_probabilidade"] = threshold
        predictions["prediction_available"] = probability.notna()
        predictions["sinalizar_filtro"] = probability.ge(threshold)
        predictions["origem_probabilidade"] = "indisponivel_sem_oof"
        predictions.loc[probability.notna() & ~posterior,"origem_probabilidade"] = "out_of_fold"
        predictions.loc[posterior,"origem_probabilidade"] = "modelo_final"
        predictions["fold"] = pd.Series(pd.NA,index=predictions.index,dtype="Int64")
        for row in folds.itertuples():
            selection = predictions.target_datetime.between(row.inicio_validacao,row.fim_validacao)
            predictions.loc[selection,"fold"] = int(row.fold)
        predictions["used_for_threshold_selection"] = (predictions.fold.lt(module.TSCV_SPLITS).fillna(False)
            & probability.notna() & predictions.classe_future.notna())
        folds["usage"] = np.where(folds.fold.lt(module.TSCV_SPLITS), "selection_descriptive", "held_out_fold")
        folds["availability_label_offset_minutes"] = protocol.forecast_minutes
        metric_rows = []
        for partition, selection in (("held_out_fold", predictions.fold.eq(module.TSCV_SPLITS).fillna(False)),
                                     ("test", predictions.target_partition.eq("test"))):
            selection &= predictions.classe_future.notna() & probability.notna()
            if selection.any():
                metrics = module._metricas(predictions.loc[selection,"classe_future"].to_numpy(),
                    probability.loc[selection].to_numpy(),deps,threshold)
                metrics["false_negative_rate"] = metrics["fn"]/metrics["n_positivos"] if metrics["n_positivos"] else None
                metric_rows.append({"partition":partition, **metrics})
        artifacts = dict(predictions=folder/"predicoes_xgboost.csv", classifier_candidates=folder/"busca_hiperparametros.csv",
            threshold_curves=folder/"curvas_limiar_validacao.csv", classifier_folds=folder/"validacao_temporal.csv",
            classifier_metrics=folder/"metricas.csv", classifier_model=model_folder/"xgboost_s4.joblib",
            classifier_summary=folder/"hiperparametros_xgboost.json")
        for key, frame in (("predictions",predictions),("classifier_candidates",candidates),
                ("threshold_curves",curves),("classifier_folds",folds),("classifier_metrics",pd.DataFrame(metric_rows))):
            frame.to_csv(artifacts[key],index=False,date_format="%Y-%m-%d %H:%M:%S")
        summary = dict(candidate_count=len(candidates), selected_parameters=params, probability_threshold=threshold,
            threshold_objective="F2", candidate_tie_break="precision", selection_folds=list(range(1,module.TSCV_SPLITS)),
            held_out_fold=module.TSCV_SPLITS, test_used_for_selection=False, context_used_for_fit=False,
            train_nominal_until=protocol.train_end, train_labels_available_until=str(protocol.development_available_until),
            latest_train_label_available_at=str(known.target_available_at.max()),
            development_rows=int(dataset.target_partition.eq("development").sum()),
            development_oof_predictions=int((probability.notna()&~posterior).sum()),
            development_unavailable_predictions=int((probability.isna()&~posterior).sum()),
            fit_counts=counts, metrics=metric_rows)
        _write_json(artifacts["classifier_summary"],summary)
        _space_check(output_root)
        joblib.dump(dict(model=model, imputer=imputer, features=list(module.CLASSIFICATION_FEATURES),
            probability_threshold=threshold, protocol=protocol.to_dict()),artifacts["classifier_model"])
    return predictions, summary, artifacts


def prepare_station(config_path, station, output_root, resume=False) -> dict[str,str]:
    """Prepare one station; a completed resume requires identical inputs and code.

    Paths in the return value are absolute. Manifest artifact paths are relative
    to output_root, allowing a temp-staged experiment to be relocated intact.
    Historical clean sources and their provenance are read-only references.
    """
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol = Protocol.from_dict(config.get("protocol", {}))
    station = station.upper()
    if station not in config.get("stations", ["FRTZ","PALM","PRU2","POAL"]):
        raise ValueError(f"Station is absent from configuration: {station}")
    output_root = Path(output_root).resolve()
    input_spec = config["inputs"]["clean_sources"][station]
    if isinstance(input_spec,str):
        input_spec = {"path":input_spec}
    resolve = lambda value: (Path(value) if Path(value).is_absolute() else ROOT/Path(value)).resolve()
    clean_path = resolve(input_spec["path"])
    weather_paths = [resolve(path) for path in config["inputs"]["space_weather_files"]]
    input_files = [clean_path,*weather_paths]
    for key in ("quality_report","ismr_manifest","source_configuration"):
        if input_spec.get(key):
            input_files.append(resolve(input_spec[key]))
    sources = {str(path):_sha256(path) for path in input_files}
    code_paths = [Path(__file__),Path(__file__).with_name("protocolo.py"),
        *[SCRIPTS/name for name in ("05_integrar_clima_espacial.py","06_feature_engineering.py",
            "07_dataset_classificacao.py","08_train_xgboost.py","interpolacao_vtec.py")]]
    code = {str(path.relative_to(ROOT)):_sha256(path) for path in code_paths}
    signature = dict(config_sha256=_sha256(config_path),sources=sources,code=code,protocol=protocol.to_dict(),station=station)
    data_folder = output_root/"dados_processados"/station
    metadata_path = data_folder/"preparation_manifest.json"
    if resume and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") != signature:
            raise ValueError("Preparation resume rejected: configuration/data/code/protocol mismatch.")
        for item in metadata["artifacts"].values():
            path = output_root/item["path"]
            if not path.is_file() or _sha256(path) != item["sha256"]:
                raise ValueError("Preparation resume rejected: artifact mismatch.")
        return {**{key:str(output_root/item["path"]) for key,item in metadata["artifacts"].items()},"metadata":str(metadata_path)}
    if metadata_path.is_file() and not resume:
        raise FileExistsError("Preparation already exists; choose resume or a new experiment root.")
    data_folder.mkdir(parents=True,exist_ok=True)
    _space_check(output_root)
    audit = audit_clean_source(clean_path,protocol)
    audit["provenance"] = {key:str(resolve(input_spec[key])) for key in ("quality_report","ismr_manifest","source_configuration") if input_spec.get(key)}
    if input_spec.get("source_configuration"):
        source_config = json.loads(resolve(input_spec["source_configuration"]).read_text(encoding="utf-8"))
        preprocessing = source_config["preprocessing"]
        expected = dict(minimum_elevation=15.,maximum_elevation=90.,minimum_s4=0.,maximum_s4=2.,
            minimum_vtec=0.,maximum_vtec=250.,use_corrected_s4=True,earth_radius_km=6371.,ionosphere_height_km=450.)
        if any(preprocessing.get(k)!=v for k,v in expected.items()):
            raise ValueError("Clean source extraction/quality provenance differs from the approved rules.")
        audit["extraction_quality_rules"] = expected
    with _legacy_modules(protocol,station,output_root) as legacy:
        aggregated = legacy["features"].reamostrar_ismr(str(clean_path))
    weather = build_weather(weather_paths,protocol)
    features = build_features(aggregated,weather,protocol,station)
    classification = build_classification_dataset(features,protocol)
    artifacts = dict(features=data_folder/"features.csv",classification=data_folder/"dataset_classificacao.csv",
        regression=data_folder/"dataset_regressao.csv",observed_reference=data_folder/"observed_reference.csv",
        mask=data_folder/"automatic_mask.csv",weather=data_folder/"clima_espacial_15min.csv")
    features.to_csv(artifacts["features"],index=False,date_format="%Y-%m-%d %H:%M:%S")
    classification.to_csv(artifacts["classification"],index=False,date_format="%Y-%m-%d %H:%M:%S")
    weather.to_csv(artifacts["weather"],index=False,date_format="%Y-%m-%d %H:%M:%S")
    predictions, classifier_summary, classifier_artifacts = _train_classifier(classification,protocol,station,output_root,config.get("classification",{}))
    regression, reference, mask = build_regression_dataset(features,predictions,protocol)
    test_count = int(reference.loc[reference.partition.eq("test"),"observed_reference_available"].sum())
    if config.get("validation",{}).get("require_complete_test_reference",False) and test_count != protocol.slot_count:
        raise ValueError(f"Test reference has {test_count}/{protocol.slot_count} observed bins before interpolation.")
    for key,frame in (("regression",regression),("observed_reference",reference),("mask",mask)):
        frame.to_csv(artifacts[key],index=False,date_format="%Y-%m-%d %H:%M:%S")
    artifacts.update(classifier_artifacts)
    metadata = dict(schema_version=1,signature=signature,source_audit=audit,protocol=protocol.to_dict(),
        station=station,classifier=classifier_summary,test_observed_bins=test_count,
        retrospective_interpolation=True,interpolation_partitions=["development","context","test"],
        reference_definition="median of observed per-SVID 15min VTEC means before alert masking/interpolation",
        datetime_semantics="UTC nominal bin starts; inputs available and emitted at bin end; future aggregate available at next bin end",
        weather_availability="raw hourly timestamp + 1h conservative proxy; publication latency unknown; no 30min aggregate reused",
        artifact_rows=dict(features=len(features),classification=len(classification),regression=len(regression),mask=len(mask)),
        artifacts={key:dict(path=str(path.relative_to(output_root)).replace('\\','/'),sha256=_sha256(path)) for key,path in artifacts.items()})
    _write_json(metadata_path,metadata)
    return {**{key:str(path) for key,path in artifacts.items()},"metadata":str(metadata_path)}
