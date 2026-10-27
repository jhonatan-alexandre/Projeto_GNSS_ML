"""Treina a matriz LSTM/LSTM-Attention do Núcleo 2 para prever VTEC."""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
import zipfile
from typing import Any

import joblib
import numpy as np
import pandas as pd


SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIRECTORY not in sys.path:
    sys.path.insert(0, SCRIPT_DIRECTORY)

from config import (  # noqa: E402
    LOOK_BACK,
    LSTM_BATCH_SIZE,
    LSTM_DROPOUT,
    LSTM_EPOCHS,
    LSTM_EXPERIMENTS,
    LSTM_LAYERS,
    LSTM_LEAKY_RELU_SLOPE,
    LSTM_LEARNING_RATE,
    LSTM_LEARNING_RATE_CANDIDATES,
    LSTM_LOSS,
    LSTM_OPTIMIZER,
    LSTM_OUTPUT_ACTIVATION,
    LSTM_PATIENCE,
    LSTM_UNITS,
    LOSS_MASKING,
    MINIMUM_TEST_OBSERVATIONS,
    MODELS_DIRECTORY,
    PIECEWISE_LOOK_BACK,
    PROCESSED_DATA,
    RANDOM_STATE,
    RESULTS_DIRECTORY,
    SPACE_WEATHER_COLUMNS,
    STATION,
    TEST_END,
    TEST_START,
    TIME_RESOLUTION,
    TRAIN_END,
    TRAIN_PERCENT,
)


PASTA_ESTACAO = os.path.join(PROCESSED_DATA, STATION)
DATASET = os.path.join(PASTA_ESTACAO, f"{STATION}_dataset_regressao.csv")
PASTA_MODELOS = os.path.join(MODELS_DIRECTORY, STATION, "nucleo2")
PASTA_RESULTADOS = os.path.join(RESULTS_DIRECTORY, STATION, "nucleo2")


def _dependencias() -> dict[str, Any]:
    try:
        import tensorflow as tf
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras import Model, Input
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import Dense, Dot, Dropout, Flatten, LeakyReLU, LSTM, Softmax
    except ModuleNotFoundError as erro:
        raise ModuleNotFoundError(
            "TensorFlow e scikit-learn são necessários; instale requirements.txt."
        ) from erro
    # Redes pequenas e 48 ajustes por ramo: pools grandes competem entre si.
    # Os limites são configuráveis e registrados nos artefatos do experimento.
    intra_threads = int(os.environ.get("GNSS_TF_INTRA_OP_THREADS", "2"))
    inter_threads = int(os.environ.get("GNSS_TF_INTER_OP_THREADS", "1"))
    try:
        tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
        tf.config.threading.set_inter_op_parallelism_threads(inter_threads)
    except RuntimeError:
        # Pode ocorrer quando outra etapa já inicializou o runtime no processo.
        intra_threads = tf.config.threading.get_intra_op_parallelism_threads()
        inter_threads = tf.config.threading.get_inter_op_parallelism_threads()
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    tf.config.experimental.enable_op_determinism()
    return locals()


def criar_sequencias(
    valores: np.ndarray,
    datas: np.ndarray,
    look_back: int,
    alvos: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cria janelas causais e permite informar explicitamente o alvo."""

    alvos = valores[:, 0] if alvos is None else np.asarray(alvos).reshape(-1)
    if len(valores) != len(datas) or len(valores) != len(alvos):
        raise ValueError("Valores, datas e alvos precisam ter o mesmo comprimento.")
    X, y, tempos = [], [], []
    for indice in range(look_back, len(valores)):
        janela = valores[indice - look_back : indice]
        if np.isfinite(janela).all() and np.isfinite(alvos[indice]):
            X.append(janela)
            y.append(alvos[indice])
            tempos.append(datas[indice])
    return np.asarray(X), np.asarray(y), np.asarray(tempos)


def criar_modelo(
    input_shape: tuple[int, int], atencao: bool, deps,
    learning_rate: float | None = None,
):
    entrada = deps["Input"](shape=input_shape)
    sequencia = entrada
    for indice in range(LSTM_LAYERS):
        sequencia = deps["LSTM"](
            LSTM_UNITS, return_sequences=atencao or indice < LSTM_LAYERS - 1
        )(sequencia)
    if atencao:
        escores = deps["Dense"](1, activation="tanh")(sequencia)
        pesos = deps["Softmax"](axis=1)(escores)
        contexto = deps["Flatten"]()(deps["Dot"](axes=1)([pesos, sequencia]))
    else:
        contexto = sequencia
    contexto = deps["Dropout"](LSTM_DROPOUT)(contexto)
    saida = deps["Dense"](
        1, activation="linear" if LSTM_OUTPUT_ACTIVATION == "leaky_relu" else LSTM_OUTPUT_ACTIVATION
    )(contexto)
    if LSTM_OUTPUT_ACTIVATION == "leaky_relu":
        saida = deps["LeakyReLU"](negative_slope=LSTM_LEAKY_RELU_SLOPE)(saida)
    modelo = deps["Model"](entrada, saida)
    otimizador = deps["tf"].keras.optimizers.get(LSTM_OPTIMIZER)
    otimizador.learning_rate = LSTM_LEARNING_RATE if learning_rate is None else learning_rate
    modelo.compile(optimizer=otimizador, loss=LSTM_LOSS)
    return modelo


def _ajustar_escalas(
    df: pd.DataFrame, serie_entrada: str, multivariado: bool, deps,
    look_back: int = 0,
):
    colunas = (
        [serie_entrada, *SPACE_WEATHER_COLUMNS]
        if multivariado
        else [serie_entrada]
    )
    # O corte é calculado nos mesmos alvos válidos usados pelo treinamento.
    # Cortar uma fração das linhas brutas pode incorporar alvos de validação
    # quando janelas são eliminadas por lacunas ou pelo lookback piecewise.
    brutos = df[colunas].to_numpy(dtype=float)
    alvo_bruto = df[serie_entrada].to_numpy(dtype=float)
    indices = np.asarray([
        indice for indice in range(look_back, len(df))
        if np.isfinite(brutos[indice - look_back:indice]).all()
        and np.isfinite(alvo_bruto[indice])
        and df["datetime"].iloc[indice] <= pd.Timestamp(TRAIN_END)
    ], dtype=int)
    if len(indices) < 2:
        raise ValueError("São necessários ao menos dois alvos válidos antes do teste.")
    quantidade_ajuste = min(len(indices) - 1, max(1, int(len(indices) * TRAIN_PERCENT)))
    corte = int(indices[quantidade_ajuste - 1])
    scaler_x = deps["MinMaxScaler"]()
    scaler_y = deps["MinMaxScaler"]()
    treino_colunas = df.iloc[:corte + 1][colunas]
    scaler_x.fit(treino_colunas.loc[np.isfinite(treino_colunas.to_numpy(dtype=float)).all(axis=1)])
    # Cada ramo reproduz o conjunto descrito no relatório: original ou tratado.
    treino_alvo = df.iloc[:corte + 1][[serie_entrada]]
    scaler_y.fit(treino_alvo.loc[np.isfinite(treino_alvo.to_numpy(dtype=float)).all(axis=1)])
    valores = scaler_x.transform(df[colunas]).astype(np.float32)
    alvos = scaler_y.transform(df[[serie_entrada]]).ravel().astype(np.float32)
    return valores, alvos, scaler_x, scaler_y, corte


def _preparar_dados(df, serie_entrada, multivariado, look_back, deps):
    valores, alvos, scaler_x, scaler_y, corte = _ajustar_escalas(
        df, serie_entrada, multivariado, deps, look_back=look_back
    )
    X, y, tempos = criar_sequencias(valores, df["datetime"].to_numpy(), look_back, alvos)
    pesos = (
        df.set_index("datetime")["loss_weight"].reindex(pd.to_datetime(tempos)).fillna(0.0).to_numpy(dtype=np.float32)
        if LOSS_MASKING and "loss_weight" in df
        else np.ones(len(tempos), dtype=np.float32)
    )
    limite = np.datetime64(df["datetime"].iloc[corte])
    ajuste = (tempos <= limite) & (pesos > 0)
    validacao = (tempos > limite) & (tempos <= np.datetime64(TRAIN_END)) & (pesos > 0)
    teste = (tempos >= np.datetime64(TEST_START)) & (tempos <= np.datetime64(TEST_END))
    if not ajuste.any() or not validacao.any():
        raise ValueError("Divisão cronológica de ajuste/validação vazia.")
    escalas = {"entrada": scaler_x, "alvo": scaler_y}
    particao = {
        "ajuste_inicio": str(pd.Timestamp(tempos[ajuste][0])),
        "ajuste_fim": str(pd.Timestamp(tempos[ajuste][-1])),
        "validacao_inicio": str(pd.Timestamp(tempos[validacao][0])),
        "validacao_fim": str(pd.Timestamp(tempos[validacao][-1])),
        "scaler_ajustado_ate": str(df["datetime"].iloc[corte]),
        "n_ajuste": int(ajuste.sum()), "n_validacao": int(validacao.sum()),
    }
    return X, y, tempos, ajuste, validacao, teste, escalas, particao, pesos


def _hiperparametros(learning_rate: float, deps) -> dict[str, Any]:
    return {
        "lstm_layers": LSTM_LAYERS, "lstm_units": LSTM_UNITS,
        "dropout": LSTM_DROPOUT, "optimizer": LSTM_OPTIMIZER,
        "loss": LSTM_LOSS, "learning_rate": float(learning_rate),
        "output_activation": LSTM_OUTPUT_ACTIVATION,
        "leaky_relu_slope": LSTM_LEAKY_RELU_SLOPE,
        "epochs_max": LSTM_EPOCHS, "batch_size": LSTM_BATCH_SIZE,
        "early_stopping_patience": LSTM_PATIENCE,
        "early_stopping_monitor": "val_loss", "restore_best_weights": True,
        "train_fraction": TRAIN_PERCENT, "shuffle": False,
        "random_state": RANDOM_STATE, "deterministic_ops": True,
        "look_back": LOOK_BACK, "piecewise_look_back_days": PIECEWISE_LOOK_BACK,
        "tf_intra_op_threads": deps.get("intra_threads"),
        "tf_inter_op_threads": deps.get("inter_threads"),
        "tf_data_private_threadpool_size": 1,
        "loss_masking": LOSS_MASKING,
    }


def _ajustar_modelo(X, y, ajuste, validacao, atencao, deps, learning_rate, verbose, pesos=None):
    tf = deps["tf"]
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    modelo = criar_modelo(X.shape[1:], atencao, deps, learning_rate=learning_rate)
    batch_size = min(LSTM_BATCH_SIZE, int(ajuste.sum()))
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    def dataset(mask):
        if pesos is None:
            valores = (X[mask], y[mask])
        else:
            valores = (X[mask], y[mask], pesos[mask])
        return tf.data.Dataset.from_tensor_slices(valores).batch(batch_size).with_options(options).prefetch(1)
    parada = deps["EarlyStopping"](
        monitor="val_loss", mode="min", patience=LSTM_PATIENCE, restore_best_weights=True
    )
    historia = modelo.fit(
        dataset(ajuste), validation_data=dataset(validacao), epochs=LSTM_EPOCHS,
        shuffle=False, callbacks=[parada], verbose=verbose,
    )
    historico = {nome: [float(valor) for valor in valores] for nome, valores in historia.history.items()}
    perdas = np.asarray(historico["val_loss"])
    finitas = np.isfinite(perdas)
    melhor = int(np.argmin(np.where(finitas, perdas, np.inf))) if finitas.any() else None
    registro = {
        "hiperparametros": _hiperparametros(learning_rate, deps),
        "batch_size_efetivo": batch_size,
        "epocas_executadas": len(perdas),
        "melhor_epoca": None if melhor is None else melhor + 1,
        "melhor_val_loss": None if melhor is None else float(perdas[melhor]),
        "historico": historico,
    }
    # Keras liga o History de volta ao modelo; quebra-se esse ciclo ao liberar.
    modelo.history = None
    return modelo, registro


def selecionar_taxa_aprendizado(df, deps) -> tuple[float, dict[str, Any]]:
    """Busca uma taxa por estação usando somente desenvolvimento ORIGINAL.

    A rede de referência é convencional, multivariada e sem atenção. A taxa
    é compartilhada entre os 16 experimentos; isso é uma adaptação operacional,
    e não uma otimização independente de cada arquitetura do artigo.
    """
    if not LSTM_LEARNING_RATE_CANDIDATES:
        return LSTM_LEARNING_RATE, {"metodo": "taxa_fixa", "candidatas": [], "learning_rate": LSTM_LEARNING_RATE}
    desenvolvimento = df.loc[df["datetime"] <= pd.Timestamp(TRAIN_END)].reset_index(drop=True)
    X, y, _, ajuste, validacao, _, _, particao, pesos = _preparar_dados(
        desenvolvimento, "VTEC_original", True, LOOK_BACK, deps
    )
    candidatas = []
    for taxa in LSTM_LEARNING_RATE_CANDIDATES:
        print(f"Selecionando learning_rate={taxa:g} na validação ORIGINAL da estação {STATION}", flush=True)
        modelo, registro = _ajustar_modelo(X, y, ajuste, validacao, False, deps, taxa, verbose=2, pesos=pesos if LOSS_MASKING else None)
        candidatas.append({"learning_rate": float(taxa), **registro})
        del modelo
        deps["tf"].keras.backend.clear_session()
        gc.collect()
    validas = [item for item in candidatas if item["melhor_val_loss"] is not None]
    if not validas:
        raise ValueError("Todas as taxas de aprendizado produziram val_loss não finita.")
    escolhida = min(validas, key=lambda item: item["melhor_val_loss"])["learning_rate"]
    return escolhida, {
        "metodo": "menor_val_loss_convencional_multivariada_original_por_estacao",
        "escopo": "Taxa compartilhada entre M1–M8 e ORIGINAL/FILTRADO; adaptação ao artigo.",
        "dados_teste_utilizados": False, "learning_rate": escolhida,
        "candidatas": candidatas, **particao,
    }


def _metricas(real: np.ndarray, previsto: np.ndarray, deps) -> dict[str, float]:
    mse = float(deps["mean_squared_error"](real, previsto))
    return {
        "MAE": float(deps["mean_absolute_error"](real, previsto)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(deps["r2_score"](real, previsto)),
    }


def _avaliar_predicoes(predicoes: pd.DataFrame, deps) -> dict[str, float]:
    avaliaveis = predicoes.loc[predicoes["observado_teste"]]
    if len(avaliaveis) < MINIMUM_TEST_OBSERVATIONS:
        raise ValueError(
            f"Previsões avaliáveis insuficientes: {len(avaliaveis)} de "
            f"{MINIMUM_TEST_OBSERVATIONS} exigidas."
        )
    return {
        **_metricas(
            avaliaveis["observado"].to_numpy(), avaliaveis["predito"].to_numpy(), deps
        ),
        "n_epocas_teste": int(len(predicoes)),
        "n_epocas_avaliadas": int(len(avaliaveis)),
    }


def _marcar_observacoes_teste(df: pd.DataFrame, predicoes: pd.DataFrame) -> pd.DataFrame:
    observados = df.set_index("datetime")["observado_teste"]
    resultado = predicoes.copy()
    resultado["observado_teste"] = (
        pd.to_datetime(resultado["datetime"]).map(observados).fillna(False).astype(bool)
    )
    return resultado


def _epocas_teste_esperadas() -> int:
    return len(
        pd.date_range(
            pd.Timestamp(TEST_START).ceil(TIME_RESOLUTION),
            pd.Timestamp(TEST_END).floor(TIME_RESOLUTION),
            freq=TIME_RESOLUTION,
        )
    )


def treinar_continuo(
    df: pd.DataFrame, serie_entrada: str, multivariado: bool, atencao: bool, deps,
    learning_rate: float = LSTM_LEARNING_RATE,
) -> tuple[dict[str, float], pd.DataFrame, Any, dict[str, Any]]:
    X, y, tempos, ajuste, validacao, teste, escalas, particao, pesos = _preparar_dados(
        df, serie_entrada, multivariado, LOOK_BACK, deps
    )
    if not teste.any():
        raise ValueError("Divisão temporal vazia no modelo contínuo.")
    modelo, registro = _ajustar_modelo(
        X, y, ajuste, validacao, atencao, deps, learning_rate, verbose=2, pesos=pesos if LOSS_MASKING else None
    )
    escalas["treinamento"] = {**registro, **particao}
    scaler_y = escalas["alvo"]
    previsto_norm = np.asarray(modelo(X[teste], training=False)).ravel()
    previsto = scaler_y.inverse_transform(previsto_norm.reshape(-1, 1)).ravel()
    real = scaler_y.inverse_transform(y[teste].reshape(-1, 1)).ravel()
    predicoes = _marcar_observacoes_teste(
        df, pd.DataFrame({"datetime": tempos[teste], "observado": real, "predito": previsto})
    )
    if len(predicoes) != _epocas_teste_esperadas():
        raise ValueError(
            "O teste contínuo não produziu todas as épocas esperadas: "
            f"{len(predicoes)} de {_epocas_teste_esperadas()}."
        )
    return _avaliar_predicoes(predicoes, deps), predicoes, modelo, escalas


def treinar_piecewise(
    df: pd.DataFrame, serie_entrada: str, multivariado: bool, atencao: bool, deps,
    learning_rate: float = LSTM_LEARNING_RATE,
    pasta_modelos: str | None = None,
    artefatos_aprovados: dict[str, str] | None = None,
    proveniencia: dict[str, Any] | None = None,
) -> tuple[dict[str, float], pd.DataFrame, dict[int, Any], dict[int, dict[str, Any]]]:
    """Salva cada slot imediatamente quando pasta_modelos é informada.

    Nesse modo, o dicionário de modelos contém caminhos, liberando cada grafo
    antes do horário seguinte. O modo em memória mantém a interface anterior.
    """
    df = df.copy()
    df["slot"] = df["datetime"].dt.hour * 2 + df["datetime"].dt.minute // 30
    todas_predicoes: list[pd.DataFrame] = []
    modelos: dict[int, Any] = {}
    escalas: dict[int, dict[str, Any]] = {}
    if pasta_modelos is not None:
        os.makedirs(pasta_modelos, exist_ok=True)
    for slot, grupo in df.groupby("slot"):
        print(f"  Treinando slot piecewise {int(slot):02d}/47", flush=True)
        grupo = grupo.reset_index(drop=True)
        X, y, tempos, ajuste, validacao, teste, escalas_slot, particao, pesos = _preparar_dados(
            grupo, serie_entrada, multivariado, PIECEWISE_LOOK_BACK, deps
        )
        if not teste.any():
            continue
        caminho = os.path.join(pasta_modelos, f"slot_{slot:02d}.keras") if pasta_modelos else None
        caminho_historico = os.path.join(pasta_modelos, f"slot_{slot:02d}_treinamento.json") if pasta_modelos else None
        reaproveitado = bool(caminho and _artefato_aprovado(caminho, artefatos_aprovados or {})
                             and _artefato_aprovado(caminho_historico, artefatos_aprovados or {}))
        if reaproveitado:
            registro = _ler_json(caminho_historico)
            _validar_historico(registro, learning_rate, particao)
            print(f"  Recuperando slot {int(slot):02d} sem reajuste", flush=True)
            modelo = deps["tf"].keras.models.load_model(caminho, compile=False)
        else:
            if caminho and os.path.exists(caminho):
                raise ValueError(f"Modelo existente sem checkpoint aprovado: {caminho}")
            modelo, registro = _ajustar_modelo(
                X, y, ajuste, validacao, atencao, deps, learning_rate, verbose=0, pesos=pesos if LOSS_MASKING else None
            )
        escalas_slot["treinamento"] = {**registro, **particao}
        scaler_y = escalas_slot["alvo"]
        previsto = scaler_y.inverse_transform(np.asarray(modelo(X[teste], training=False))).ravel()
        real = scaler_y.inverse_transform(y[teste].reshape(-1, 1)).ravel()
        todas_predicoes.append(_marcar_observacoes_teste(
            grupo, pd.DataFrame({"datetime": tempos[teste], "observado": real, "predito": previsto, "slot": slot})
        ))
        escalas[int(slot)] = escalas_slot
        if pasta_modelos is None:
            modelos[int(slot)] = modelo
        else:
            if not reaproveitado:
                modelo.save(caminho)
                _salvar_json(caminho_historico, escalas_slot["treinamento"])
            modelos[int(slot)] = caminho
            if proveniencia is not None:
                _salvar_checkpoint(os.path.join(pasta_modelos, f"slot_{slot:02d}_checkpoint.json"),
                                   proveniencia, [caminho, caminho_historico])
            del modelo
            deps["tf"].keras.backend.clear_session()
            gc.collect()
    if not todas_predicoes:
        raise ValueError("Nenhum modelo piecewise produziu previsões.")
    slots_ausentes = sorted(set(range(48)).difference(modelos))
    if slots_ausentes:
        raise ValueError(
            "Modelos piecewise ausentes para os slots: "
            + ", ".join(map(str, slots_ausentes))
        )
    predicoes = pd.concat(todas_predicoes).sort_values("datetime")
    if len(predicoes) != _epocas_teste_esperadas():
        raise ValueError(
            "O teste piecewise não produziu todas as épocas esperadas: "
            f"{len(predicoes)} de {_epocas_teste_esperadas()}."
        )
    metricas = _avaliar_predicoes(predicoes, deps)
    return metricas, predicoes, modelos, escalas


def _salvar_json(caminho: str, conteudo: Any) -> None:
    # JSON estrito: perdas divergentes ficam explícitas como null.
    def finitos(valor):
        if isinstance(valor, dict):
            return {chave: finitos(item) for chave, item in valor.items()}
        if isinstance(valor, (list, tuple)):
            return [finitos(item) for item in valor]
        if isinstance(valor, float) and not np.isfinite(valor):
            return None
        return valor
    temporario = caminho + ".tmp"
    with open(temporario, "w", encoding="utf-8") as arquivo:
        json.dump(finitos(conteudo), arquivo, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporario, caminho)


def _ler_json(caminho):
    with open(caminho, encoding="utf-8") as arquivo:
        return json.load(arquivo)


def _sha256(caminho: str) -> str:
    digest = hashlib.sha256()
    with open(caminho, "rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def _proveniencia(learning_rate: float) -> dict[str, Any]:
    """Contrato dos dados e funções científicas, independente do isolamento."""
    funcoes = (criar_sequencias, criar_modelo, _ajustar_escalas, _preparar_dados,
               _hiperparametros, _ajustar_modelo, selecionar_taxa_aprendizado, _metricas)
    fonte = "\n".join(inspect.getsource(funcao) for funcao in funcoes)
    parametros = _hiperparametros(learning_rate, {})
    # Pools são política de execução; arquitetura/partição/semente são ciência.
    parametros = {chave: valor for chave, valor in parametros.items() if not chave.startswith("tf_")}
    return {
        "contrato": "gnss_lstm_xie_v1", "dataset_sha256": _sha256(DATASET),
        "codigo_cientifico_sha256": hashlib.sha256(fonte.encode("utf-8")).hexdigest(),
        "estacao": STATION, "train_end": TRAIN_END, "test_start": TEST_START,
        "test_end": TEST_END, "time_resolution": TIME_RESOLUTION,
        "space_weather_columns": list(SPACE_WEATHER_COLUMNS),
        "learning_rate_candidates": list(LSTM_LEARNING_RATE_CANDIDATES),
        "hiperparametros": parametros,
    }


def _artefato_aprovado(caminho: str, aprovados: dict[str, str]) -> bool:
    normalizado = os.path.abspath(caminho)
    esperado = aprovados.get(normalizado)
    return bool(esperado and os.path.isfile(normalizado) and _sha256(normalizado) == esperado)


def _validar_historico(registro, learning_rate, particao=None):
    esperado = _hiperparametros(learning_rate, {})
    recebido = registro.get("hiperparametros", {})
    for chave, valor in esperado.items():
        if not chave.startswith("tf_") and recebido.get(chave) != valor:
            raise ValueError(f"Checkpoint incompatível: hiperparâmetro {chave}.")
    for chave, valor in (particao or {}).items():
        if registro.get(chave) != valor:
            raise ValueError(f"Checkpoint incompatível: partição {chave}.")
    epocas = registro.get("epocas_executadas", 0)
    perdas = registro.get("historico", {}).get("val_loss", [])
    if not 0 < epocas <= LSTM_EPOCHS or len(perdas) != epocas:
        raise ValueError("Checkpoint sem histórico completo de treinamento.")
    if registro.get("melhor_val_loss") is None or not np.isfinite(registro["melhor_val_loss"]):
        raise ValueError("Checkpoint sem perda de validação finita.")


def _salvar_checkpoint(caminho, proveniencia, artefatos, **extras):
    _salvar_json(caminho, {
        "schema_version": 1, "proveniencia": proveniencia,
        "artefatos": {os.path.abspath(item): _sha256(item) for item in artefatos},
        **extras,
    })


def _aprovar_checkpoint(caminho, proveniencia):
    registro = _ler_json(caminho)
    if registro.get("schema_version") != 1 or registro.get("proveniencia") != proveniencia:
        raise ValueError(f"Proveniência incompatível: {caminho}")
    artefatos = registro.get("artefatos", {})
    if not artefatos or not all(_artefato_aprovado(item, artefatos) for item in artefatos):
        raise ValueError(f"Artefato ausente ou alterado: {caminho}")
    return registro


def _nome_experimento(experiment, filtrado):
    return (f"{experiment.config_id.lower()}_{experiment.organization}_"
            f"{'attention' if experiment.attention else 'lstm'}_"
            f"{'multi' if experiment.multivariate else 'uni'}_"
            f"{'filtrado' if filtrado else 'original'}")


def _matriz():
    return [(experiment, filtrado, _nome_experimento(experiment, filtrado))
            for filtrado in (False, True) for experiment in LSTM_EXPERIMENTS]


def _ler_dados():
    dados = pd.read_csv(DATASET, low_memory=True)
    dados["datetime"] = pd.to_datetime(dados["datetime"], errors="coerce")
    return dados.sort_values("datetime").reset_index(drop=True)


def _validar_predicoes(caminho):
    predicoes = pd.read_csv(caminho, parse_dates=["datetime"])
    grade = pd.date_range(pd.Timestamp(TEST_START).ceil(TIME_RESOLUTION),
                         pd.Timestamp(TEST_END).floor(TIME_RESOLUTION), freq=TIME_RESOLUTION)
    if len(predicoes) != len(grade) or not pd.DatetimeIndex(predicoes["datetime"]).equals(grade):
        raise ValueError(f"Épocas de teste incompletas/duplicadas no checkpoint: {caminho}")
    if not np.isfinite(predicoes[["observado", "predito"]].to_numpy()).all():
        raise ValueError(f"Previsões não finitas no checkpoint: {caminho}")
    return predicoes


def _arquivos_configuracao(experiment, nome):
    arquivos = [os.path.join(PASTA_RESULTADOS, f"predicoes_{nome}.csv"),
                os.path.join(PASTA_RESULTADOS, f"historico_{nome}.json"),
                os.path.join(PASTA_MODELOS, f"escalas_{nome}.joblib")]
    if experiment.organization == "convencional":
        arquivos.append(os.path.join(PASTA_MODELOS, f"{nome}.keras"))
    else:
        arquivos.extend(os.path.join(PASTA_MODELOS, nome, f"slot_{slot:02d}.keras") for slot in range(48))
        arquivos.extend(os.path.join(PASTA_MODELOS, nome, f"slot_{slot:02d}_treinamento.json") for slot in range(48))
    return arquivos


def _validar_configuracao(experiment, nome, learning_rate):
    arquivos = _arquivos_configuracao(experiment, nome)
    _validar_predicoes(arquivos[0])
    historico = _ler_json(arquivos[1])
    registros = [historico] if experiment.organization == "convencional" else list(historico.values())
    if experiment.organization != "convencional" and set(historico) != {str(slot) for slot in range(48)}:
        raise ValueError(f"Histórico piecewise incompleto: {nome}")
    for registro in registros:
        _validar_historico(registro, learning_rate)
    for caminho in arquivos:
        if caminho.endswith(".keras"):
            with zipfile.ZipFile(caminho) as arquivo:
                if not {"config.json", "model.weights.h5"}.issubset(arquivo.namelist()) or arquivo.testzip() is not None:
                    raise ValueError(f"Modelo Keras incompleto: {caminho}")
    return arquivos


def _metrica_previa(nome, aprovados):
    for arquivo in ("metricas_regressao_parciais.csv", "metricas_regressao.csv"):
        caminho = os.path.join(PASTA_RESULTADOS, arquivo)
        if _artefato_aprovado(caminho, aprovados):
            tabela = pd.read_csv(caminho)
            linhas = tabela.loc[tabela["modelo"].eq(nome)]
            if len(linhas) == 1:
                return linhas.iloc[0].to_dict()
    return None


def _recuperar_configuracao(experiment, nome, learning_rate, proveniencia, aprovados):
    caminho = os.path.join(PASTA_RESULTADOS, f"checkpoint_{nome}.json")
    if os.path.isfile(caminho):
        registro = _aprovar_checkpoint(caminho, proveniencia)
        esperados = _arquivos_configuracao(experiment, nome)
        if not all(_artefato_aprovado(item, registro["artefatos"]) for item in esperados):
            raise ValueError(f"Checkpoint não cobre todos os artefatos: {nome}")
        _validar_configuracao(experiment, nome, learning_rate)
        return registro["metricas"]
    arquivos = _arquivos_configuracao(experiment, nome)
    if not all(_artefato_aprovado(item, aprovados) for item in arquivos):
        return None
    metrica = _metrica_previa(nome, aprovados)
    if metrica is None:
        return None
    _validar_configuracao(experiment, nome, learning_rate)
    _salvar_checkpoint(caminho, proveniencia, arquivos, metricas=metrica)
    return metrica


def _aprovados_modelo(nome, proveniencia, aprovados):
    resultado = dict(aprovados)
    pasta = os.path.join(PASTA_MODELOS, nome)
    if os.path.isdir(pasta):
        for slot in range(48):
            caminho = os.path.join(pasta, f"slot_{slot:02d}_checkpoint.json")
            if os.path.isfile(caminho):
                resultado.update(_aprovar_checkpoint(caminho, proveniencia)["artefatos"])
    return resultado


def _carregar_retomada(caminho, proveniencia):
    if not caminho:
        return {}
    registro = _ler_json(caminho)
    if registro.get("schema_version") != 1 or registro.get("proveniencia") != proveniencia:
        raise ValueError(f"Proveniência incompatível: {caminho}")
    # Conferir arquivos no ponto de reuso permite que o CSV agregado avance.
    artefatos = registro.get("artefatos", {})
    if not isinstance(artefatos, dict) or not artefatos:
        raise ValueError(f"Manifesto sem artefatos aprovados: {caminho}")
    return artefatos


def _executar_worker(argumentos):
    # Sem shell intermediário: herda config/estação/paths definidos pelo bootstrap.
    subprocess.run([sys.executable, "-X", "utf8", os.path.abspath(__file__), *argumentos], check=True)


def _selecionar_worker():
    deps = _dependencias()
    learning_rate, selecao = selecionar_taxa_aprendizado(_ler_dados(), deps)
    caminho = os.path.join(PASTA_RESULTADOS, "hiperparametros_lstm.json")
    _salvar_json(caminho, {"estacao": STATION, "hiperparametros": _hiperparametros(learning_rate, deps),
                          "selecao_taxa_aprendizado": selecao})
    _salvar_checkpoint(os.path.join(PASTA_RESULTADOS, "checkpoint_taxa.json"),
                      _proveniencia(learning_rate), [caminho])


def _modelo_worker(nome, manifesto, retomar):
    selecionado = next(((exp, filtrado) for exp, filtrado, item in _matriz() if item == nome), None)
    if selecionado is None:
        raise ValueError(f"Configuração LSTM desconhecida: {nome}")
    experiment, filtrado = selecionado
    learning_rate = _ler_json(os.path.join(PASTA_RESULTADOS, "hiperparametros_lstm.json"))["hiperparametros"]["learning_rate"]
    proveniencia = _proveniencia(learning_rate)
    # O coordenador valida o manifesto de adoção antes de escrever novos CSVs.
    # Workers conferem seus próprios artefatos, permitindo atualizar o CSV
    # agregado sem invalidar a auditoria histórica.
    aprovados = {}
    if manifesto:
        registro = _ler_json(manifesto)
        if registro.get("proveniencia") != proveniencia:
            raise ValueError("Proveniência do worker difere do manifesto.")
        aprovados = registro.get("artefatos", {})
    if retomar:
        aprovados = _aprovados_modelo(nome, proveniencia, aprovados)
    deps = _dependencias()
    dados = _ler_dados()
    inicio = time.monotonic()
    serie_entrada = "VTEC_filtered" if filtrado else "VTEC_original"
    print(f"Processo {os.getpid()}: {nome}", flush=True)
    if experiment.organization == "convencional":
        caminho_modelo = os.path.join(PASTA_MODELOS, f"{nome}.keras")
        if os.path.exists(caminho_modelo):
            raise ValueError(f"Modelo convencional já existe sem checkpoint completo: {nome}")
        metricas, predicoes, modelos, escalas = treinar_continuo(
            dados, serie_entrada, experiment.multivariate, experiment.attention, deps, learning_rate=learning_rate)
        modelos.save(caminho_modelo)
        historico = escalas["treinamento"]
    else:
        metricas, predicoes, modelos, escalas = treinar_piecewise(
            dados, serie_entrada, experiment.multivariate, experiment.attention, deps,
            learning_rate=learning_rate, pasta_modelos=os.path.join(PASTA_MODELOS, nome),
            artefatos_aprovados=aprovados, proveniencia=proveniencia)
        historico = {slot: item["treinamento"] for slot, item in escalas.items()}
    predicoes.to_csv(os.path.join(PASTA_RESULTADOS, f"predicoes_{nome}.csv"), index=False)
    _salvar_json(os.path.join(PASTA_RESULTADOS, f"historico_{nome}.json"), historico)
    joblib.dump(escalas, os.path.join(PASTA_MODELOS, f"escalas_{nome}.joblib"))
    linha = {
        "config_id": experiment.config_id, "configuracao": experiment.label,
        "modelo": nome, "organizacao": experiment.organization,
        "atencao": experiment.attention, "multivariado": experiment.multivariate,
        "filtrado": filtrado, "serie_entrada": serie_entrada, "alvo_avaliacao": serie_entrada,
        "n_predicoes_teste": len(predicoes), "learning_rate": learning_rate,
        "lstm_layers": LSTM_LAYERS, "lstm_units": LSTM_UNITS,
        "output_activation": LSTM_OUTPUT_ACTIVATION,
        "epocas_executadas": (historico["epocas_executadas"] if experiment.organization == "convencional"
                              else sum(item["epocas_executadas"] for item in historico.values())),
        "duracao_segundos": round(time.monotonic() - inicio, 2), **metricas,
    }
    arquivos = _validar_configuracao(experiment, nome, learning_rate)
    _salvar_checkpoint(os.path.join(PASTA_RESULTADOS, f"checkpoint_{nome}.json"),
                      proveniencia, arquivos, metricas=linha)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--select-rate-worker", action="store_true")
    parser.add_argument("--model-worker")
    parser.add_argument("--resume-verified", default=os.environ.get("GNSS_LSTM_RESUME_MANIFEST"))
    args, _ = parser.parse_known_args()
    retomar = bool(args.resume_verified or os.environ.get("GNSS_LSTM_RESUME") == "1")
    if not os.path.isfile(DATASET):
        raise FileNotFoundError(f"Execute primeiro o script 09: {DATASET}")
    os.makedirs(PASTA_MODELOS, exist_ok=True)
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    if args.select_rate_worker:
        _selecionar_worker()
        return
    if args.model_worker:
        _modelo_worker(args.model_worker, args.resume_verified, retomar)
        return
    if args.resume_verified:
        os.environ["GNSS_LSTM_RESUME_MANIFEST"] = os.path.abspath(args.resume_verified)
    if retomar:
        os.environ["GNSS_LSTM_RESUME"] = "1"
    caminho_taxa = os.path.join(PASTA_RESULTADOS, "hiperparametros_lstm.json")
    if not (retomar and os.path.isfile(caminho_taxa)):
        if os.path.exists(caminho_taxa):
            raise ValueError("Seleção já existe; use retomada verificada ou um novo diretório.")
        _executar_worker(["--select-rate-worker"])
    selecao = _ler_json(caminho_taxa)
    learning_rate = selecao["hiperparametros"]["learning_rate"]
    proveniencia = _proveniencia(learning_rate)
    aprovados = _carregar_retomada(args.resume_verified, proveniencia)
    checkpoint_taxa = os.path.join(PASTA_RESULTADOS, "checkpoint_taxa.json")
    if os.path.isfile(checkpoint_taxa):
        registro = _aprovar_checkpoint(checkpoint_taxa, proveniencia)
        if not _artefato_aprovado(caminho_taxa, registro["artefatos"]):
            raise ValueError("Checkpoint não cobre a seleção de taxa.")
    elif _artefato_aprovado(caminho_taxa, aprovados):
        _salvar_checkpoint(checkpoint_taxa, proveniencia, [caminho_taxa])
    else:
        raise ValueError("Seleção de taxa existente sem proveniência aprovada.")
    # Adota todos os resultados legados antes de atualizar o CSV agregado.
    recuperados = {}
    if retomar:
        for experiment, _, nome in _matriz():
            linha = _recuperar_configuracao(experiment, nome, learning_rate, proveniencia, aprovados)
            if linha is not None:
                recuperados[nome] = linha
    resultados = []
    for experiment, _, nome in _matriz():
        if nome in recuperados:
            print(f"Checkpoint verificado, sem reajuste: {nome}", flush=True)
            linha = recuperados[nome]
        else:
            _executar_worker(["--model-worker", nome])
            linha = _recuperar_configuracao(experiment, nome, learning_rate, proveniencia, {})
            if linha is None:
                raise ValueError(f"Worker não produziu checkpoint completo: {nome}")
        resultados.append(linha)
        pd.DataFrame(resultados).to_csv(os.path.join(PASTA_RESULTADOS, "metricas_regressao_parciais.csv"), index=False)
    pd.DataFrame(resultados).to_csv(os.path.join(PASTA_RESULTADOS, "metricas_regressao.csv"), index=False)
    print("Resultados salvos:", PASTA_RESULTADOS, flush=True)


if __name__ == "__main__":
    main()
    if os.name == "nt":
        # Todos os CSV/Keras/joblib já foram fechados por main(). Evita que a
        # finalização do runtime TensorFlow no Windows retenha o subprocesso
        # após salvar a matriz completa, bloqueando a etapa de relatórios.
        # Exceções em main() continuam propagando normalmente com saída de erro.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
