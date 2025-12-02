# dsb_boostings_tabular.py
"""
DSBTabularBridgeGBDT

Schrödinger Bridge для табличных данных с использованием градиентного бустинга
(XGBoost) в качестве аппроксиматоров forward/backward mean map'ов вместо MLP.

Ожидается, что на вход подаются уже подготовленные данные:
- X_train_w: np.ndarray [n_train, d] (например, после StandardScaler + PCA/whiten)
- X_val_w:   np.ndarray [n_val, d]   (та же трансформация, что и для train)

Зависимости:
    pip install numpy scikit-learn xgboost
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Any

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool


# ------------------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def sliced_wasserstein_distance(
    x: np.ndarray,
    y: np.ndarray,
    n_projections: int = 256,
    p: float = 2.0,
    rng: Optional[np.random.RandomState] = None,
) -> float:
    """
    Sliced Wasserstein distance между двумя выборками в R^d.

    x: [n, d], y: [m, d]
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    n, d = x.shape
    m, d2 = y.shape
    assert d == d2, "Dimension mismatch"

    if rng is None:
        rng = np.random.RandomState(0)

    # Выравниваем объём выборок
    n_common = min(n, m)
    if n > n_common:
        idx = rng.choice(n, n_common, replace=False)
        x_use = x[idx]
    else:
        x_use = x

    if m > n_common:
        idx = rng.choice(m, n_common, replace=False)
        y_use = y[idx]
    else:
        y_use = y

    ws = []
    for _ in range(n_projections):
        v = rng.randn(d).astype(np.float32)
        v_norm = np.linalg.norm(v) + 1e-12
        v /= v_norm

        proj_x = x_use @ v  # [n_common]
        proj_y = y_use @ v  # [n_common]

        proj_x = np.sort(proj_x)
        proj_y = np.sort(proj_y)

        diff = proj_x - proj_y
        ws.append((np.mean(np.abs(diff) ** p)) ** (1.0 / p))

    return float(np.mean(ws))


# ------------------------------------------------------------------------
# GBDTMeanMap: бустинг + time embedding
# ------------------------------------------------------------------------

from xgboost.callback import TrainingCallback
from tqdm.auto import tqdm

class TqdmCallback(TrainingCallback):
    """tqdm по boosting rounds для одного XGBRegressor (совместимо с XGBoost>=1.6/2.x)."""
    def __init__(self, total_rounds: int, desc: str = "xgb"):
        self.total = int(total_rounds)
        self.desc = desc
        self._pbar = None
        self._last_iter = -1

    def before_training(self, model):
        # ИНИЦИАЛИЗИРУЕМ ПРОГРЕСС-БАР И ВОЗВРАЩАЕМ model ← ВАЖНО
        self._pbar = tqdm(total=self.total, desc=self.desc, leave=False)
        self._last_iter = -1
        return model

    def after_iteration(self, model, epoch: int, evals_log):
        # epoch идёт с 0; обновляем на +1 за итерацию
        if epoch > self._last_iter:
            self._pbar.update(1)
            self._last_iter = epoch
        return False  # продолжать обучение

    def after_training(self, model):
        # Добиваем и закрываем бар, затем возвращаем model
        if self._pbar is not None:
            remaining = self.total - (self._last_iter + 1)
            if remaining > 0:
                self._pbar.update(remaining)
            self._pbar.close()
            self._pbar = None
        return model


# class GBDTMeanMap:
#     """
#     Бустинг как аппроксиматор mean map: (x, t) -> E[X_next | X_t = x].
#     Обучаем D независимых XGBRegressor — по одному на каждую координату выхода.
#     tqdm:
#       - внешняя полоса: перебор координат (dim)
#       - внутренняя: boosting rounds в XGB через TqdmCallback
#     """

#     def __init__(self, dim, use_fourier_time=True, time_features=16, max_freq=20.0, xgb_params=None, name: str = "F", show_progress: bool = True,
#     ) -> None:
#         self.dim = int(dim)
#         self.use_fourier_time = bool(use_fourier_time)
#         self.time_features = int(time_features)
#         self.max_freq = float(max_freq)
#         self.name = name
#         self.show_progress = show_progress
#         self.models = [None] * self.dim


#         if self.use_fourier_time:
#             self.freq = np.linspace(1.0, self.max_freq, self.time_features, dtype=np.float32)
#         else:
#             self.freq = None

#         # Базовые параметры XGB + возможность переопределить
#         base_params = dict(
#             n_estimators=300,
#             max_depth=6,
#             learning_rate=0.05,
#             subsample=0.9,
#             colsample_bytree=0.9,
#             tree_method="hist",
#             random_state=0,
#             # disable default verbosity (мы показываем прогресс сами)
#             verbosity=0,
#         )
#         if xgb_params:
#             base_params.update(xgb_params)

#         self.xgb_params = base_params
#         self.models: list[XGBRegressor] = [None] * self.dim  # заполняем в fit

#     # ---------- time embedding ----------

#     def _time_embed(self, t: np.ndarray) -> np.ndarray:
#         t = np.asarray(t, dtype=np.float32).reshape(-1, 1)  # [n,1]
#         if self.freq is None:
#             return t  # fallback: просто скаляр t
#         pe = 2.0 * np.pi * t * self.freq.reshape(1, -1)     # [n, F]
#         sin = np.sin(pe)
#         cos = np.cos(pe)
#         return np.concatenate([sin, cos], axis=1)           # [n, 2F]

#     def _make_features(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
#         x = np.asarray(x, dtype=np.float32)
#         t = np.asarray(t, dtype=np.float32).reshape(-1)
#         if self.use_fourier_time:
#             te = self._time_embed(t)                        # [n, 2F]
#             return np.concatenate([x, te], axis=1)
#         else:
#             return np.concatenate([x, t[:, None]], axis=1)

#     # ---------- API ----------

#     def fit(self, x: np.ndarray, t: np.ndarray, y: np.ndarray) -> None:
#         """
#         Обучаем D независимых регрессоров:
#           для каждой координаты j: (x,t) -> y[:, j]
#         tqdm: внешний прогресс по координатам; внутри — по деревьям.
#         """
#         X_feat = self._make_features(x, t)          # [n, d_feat]
#         y = np.asarray(y, dtype=np.float32)         # [n, D]
#         assert y.shape[1] == self.dim, "Target dimension mismatch"

#         rounds = int(self.xgb_params.get("n_estimators", 300))
#         for j in tqdm(range(self.dim), desc="fit outputs", leave=False):
#             reg = XGBRegressor(
#                 **self.xgb_params,
#                 callbacks=[TqdmCallback(rounds, desc=f"xgb[{j}]")],
#             )
#             reg.fit(X_feat, y[:, j])
#             self.models[j] = reg

#     def predict(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
#         """
#         Предсказание: собираем по координатам и склеиваем в [n, D].
#         """
#         X_feat = self._make_features(x, t)
#         preds = [m.predict(X_feat) for m in self.models]
#         return np.stack(preds, axis=1).astype(np.float32)   # [n, D]

class CatBoostMeanMap:
    """
    CatBoost как аппроксиматор mean map: (x, t) -> E[X_next | X_t = x]
    Многомерный выход через loss_function="MultiRMSE".
    Одна модель на весь вектор (никаких циклов по D).
    """

    def __init__(
        self,
        dim: int,
        use_fourier_time: bool = True,
        time_features: int = 16,
        max_freq: float = 20.0,
        cat_params: Optional[Dict[str, Any]] = None,
        verbose = 100,
        name: str = "F",
    ) -> None:
        self.dim = int(dim)
        self.use_fourier_time = bool(use_fourier_time)
        self.time_features = int(time_features)
        self.max_freq = float(max_freq)
        self.verbose = verbose
        self.name = name

        if self.use_fourier_time:
            self.freq = np.linspace(1.0, self.max_freq, self.time_features, dtype=np.float32)
        else:
            self.freq = None

        # Базовые параметры CatBoost c MultiRMSE
        base_params = dict(
            loss_function="MultiRMSE",
            iterations=500,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            random_state=0,
            task_type="GPU",   # включаем CUDA
            devices="0",       
            # CPU по умолчанию; можно указать task_type="GPU", если нужно
        )
        if cat_params:
            base_params.update(cat_params)

        self.params = base_params
        self.model: Optional[CatBoostRegressor] = None

    # ---------- time embedding ----------

    def _time_embed(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.float32).reshape(-1, 1)
        if self.freq is None:
            return t  # fallback: просто столбец t
        pe = 2.0 * np.pi * t * self.freq.reshape(1, -1)
        return np.concatenate([np.sin(pe), np.cos(pe)], axis=1)  # [n, 2F]

    def _make_features(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        t = np.asarray(t, dtype=np.float32).reshape(-1)
        if self.use_fourier_time:
            te = self._time_embed(t)            # [n, 2F]
            return np.concatenate([x, te], axis=1)
        else:
            return np.concatenate([x, t[:, None]], axis=1)

    # ---------- API ----------

    def fit(self, x: np.ndarray, t: np.ndarray, y: np.ndarray) -> None:
        X_feat = self._make_features(x, t)          # [n, d_feat]
        y = np.asarray(y, dtype=np.float32)         # [n, D]
        assert y.shape[1] == self.dim, "Target dimension mismatch for MultiRMSE"

        train_pool = Pool(data=X_feat, label=y)
        self.model = CatBoostRegressor(**self.params)
        # CatBoost сам рисует прогресс; отключить: verbose=False в params или self.verbose
        self.model.fit(train_pool, verbose=self.verbose)

    def predict(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        assert self.model is not None, "CatBoost model is not fitted"
        X_feat = self._make_features(x, t)
        preds = self.model.predict(X_feat)          # [n, D]
        return np.asarray(preds, dtype=np.float32)


# ------------------------------------------------------------------------
# DSBTabularBridgeGBDT
# ------------------------------------------------------------------------


class DSBTabularBridgeGBDT:
    """
    Schrödinger Bridge для табличных данных:

    - Forward процесс F: OU-pretrain + IPF (через target из B).
    - Backward процесс B: IPF (через target из F).
    - В роли аппроксиматоров F и B: GBDTMeanMap (XGBoost).

    Работает в whiten-пространстве; снаружи нужно:
        X -> StandardScaler -> PCA(whiten=True) -> X_w
    """

    def __init__(
        self,
        X_train_w: np.ndarray,
        X_val_w: Optional[np.ndarray] = None,
        T: float = 1.0,
        N: int = 16,
        alpha_ou: float = 1.0,
        time_features: int = 16,
        cat_params: Optional[Dict[str, Any]] = None,   # ← было gbdt_params
        seed: int = 0,
    ) -> None:
        
        from_path_params = cat_params
        X_train_w = np.asarray(X_train_w, dtype=np.float32)
        self.X_train_w = X_train_w

        if X_val_w is not None:
            self.X_val_w = np.asarray(X_val_w, dtype=np.float32)
        else:
            self.X_val_w = None

        self.T = float(T)
        self.N = int(N)
        self.alpha_ou = float(alpha_ou)
        self.D = self.X_train_w.shape[1]

        # шаг по времени (равномерный)
        self.gamma = np.full((self.N,), self.T / self.N, dtype=np.float32)

        # RandomState для контролируемой стохастики
        self.rng = np.random.RandomState(seed)

        # Модели F и B
        self.F = CatBoostMeanMap(
            dim=self.D,
            use_fourier_time=True,
            time_features=time_features,
            cat_params=from_path_params,
            verbose=100,
            name="F",
        )
        self.B = CatBoostMeanMap(
            dim=self.D,
            use_fourier_time=True,
            time_features=time_features,
            cat_params=from_path_params,
            verbose=100,
            name="B",
        )



    # --------------------------------------------------------------------
    # Семплеры p0 и pN
    # --------------------------------------------------------------------

    def sample_data_p0_train(self, n: int, jitter: float = 0.0) -> np.ndarray:
        """
        Семплируем из эмпирического распределения p0 (train),
        с опциональным джиттером.
        """
        n = int(n)
        idx = self.rng.randint(0, self.X_train_w.shape[0], size=n)
        x = self.X_train_w[idx].astype(np.float32)
        if jitter > 0.0:
            x = x + jitter * self.rng.randn(*x.shape).astype(np.float32)
        return x

    def sample_prior_pN(self, n: int, std: float = 1.0) -> np.ndarray:
        """
        Семплируем из prior pN ~ N(0, std^2 I).
        """
        n = int(n)
        x = std * self.rng.randn(n, self.D).astype(np.float32)
        return x

    # --------------------------------------------------------------------
    # OU pretrain для F
    # --------------------------------------------------------------------

    def pretrain_F_ou(self, n_samples: int = 50_000) -> None:
        """
        Предобучение forward-процесса F на OU динамике
        (как в исходном DSB): для каждого (x, t_k, dt) учим
            F(x, t_k) ≈ x + dt * (-alpha_ou * x).
        """
        n_samples = int(n_samples)
        n_data = n_samples // 2
        n_prior = n_samples - n_data

        x_data = self.sample_data_p0_train(n_data, jitter=0.01)
        x_prior = self.sample_prior_pN(n_prior, std=1.0)
        x = np.vstack([x_data, x_prior]).astype(np.float32)

        # случайные временные шаги
        k = self.rng.randint(0, self.N, size=x.shape[0])
        t_k = (k.astype(np.float32) + 0.5) / self.N
        dt = self.gamma[k]

        ou_mean = x + dt[:, None] * (-self.alpha_ou * x)  # таргет для F

        self.F.fit(x, t_k, ou_mean)

    # --------------------------------------------------------------------
    # Forward-пары и датасет для B
    # --------------------------------------------------------------------

    def simulate_forward_pairs_dataset(
        self, n_paths: int, jitter: float = 0.01
    ) -> Tuple[int, list]:
        """
        Генерация forward-пар (k, X_k, X_{k+1}, t_k) при фиксированном F.

        Возвращает:
            (n_paths, pairs), где pairs — список длины N
            с элементами (k, Xk, Xk1, t_k).
        """
        n_paths = int(n_paths)
        Xk = self.sample_data_p0_train(n_paths, jitter=jitter)

        pairs = []

        for k in range(self.N):
            t_k = np.full((n_paths,), (k + 0.5) / self.N, dtype=np.float32)

            mean = self.F.predict(Xk, t_k)  # [n_paths, D]
            noise = (
                self.rng.randn(*Xk.shape).astype(np.float32)
                * math.sqrt(2.0 * float(self.gamma[k]))
            )
            Xk1 = mean + noise

            pairs.append((k, Xk.copy(), Xk1.copy(), t_k.copy()))
            Xk = Xk1

        return n_paths, pairs

    def build_B_dataset(self, n_paths: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Строим регрессионный датасет для обучения B:
            вход:  (X_{k+1}, t_{k+1})
            таргет: X_{k+1} + (F(X_k, t_k) - F(X_{k+1}, t_k))
        как в loss_backward_on_forward_pairs в исходном DSB.
        """
        _, pairs = self.simulate_forward_pairs_dataset(n_paths)

        X_list = []
        t_list = []
        Y_list = []

        for (k, Xk, Xk1, t_k) in pairs:
            # t_{k+1}
            if (k + 1) < self.N:
                t_k1 = np.full(
                    (Xk1.shape[0],),
                    ((k + 1) + 0.5) / self.N,
                    dtype=np.float32,
                )
            else:
                # последний слой — повторяем последний t
                t_k1 = np.full(
                    (Xk1.shape[0],),
                    (self.N - 0.5) / self.N,
                    dtype=np.float32,
                )

            F_Xk = self.F.predict(Xk, t_k)
            F_Xk1k = self.F.predict(Xk1, t_k)

            target = Xk1 + (F_Xk - F_Xk1k)  # как в loss_backward

            X_list.append(Xk1)
            t_list.append(t_k1)
            Y_list.append(target)

        X_train = np.concatenate(X_list, axis=0)
        t_train = np.concatenate(t_list, axis=0)
        Y_train = np.concatenate(Y_list, axis=0)

        return X_train.astype(np.float32), t_train.astype(np.float32), Y_train.astype(
            np.float32
        )

    # --------------------------------------------------------------------
    # Backward-пары и датасет для F
    # --------------------------------------------------------------------

    def simulate_backward_pairs_dataset(
        self, n_paths: int
    ) -> Tuple[int, list]:
        """
        Генерация backward-пар (k, X_k, X_{k+1}) при фиксированном B.

        Стартуем из prior pN и двигаемся назад по времени:
            X_N ~ pN
            X_k ~ B(X_{k+1}, t_k) + sqrt(2 * gamma_k * dt) * N(0,I)

        Возвращает:
            (n_paths, pairs), где pairs — список длины N
            с элементами (k, Xk, Xk1).
        """
        n_paths = int(n_paths)
        Xk = self.sample_prior_pN(n_paths, std=1.0)

        # seq[i] будет соответствовать X_i (после разворота)
        seq = [Xk.copy()]

        # идём от N-1 до 0
        for k in range(self.N - 1, -1, -1):
            t_k = np.full((n_paths,), (k + 0.5) / self.N, dtype=np.float32)
            mean = self.B.predict(Xk, t_k)
            noise = (
                self.rng.randn(*Xk.shape).astype(np.float32)
                * math.sqrt(2.0 * float(self.gamma[k]))
            )
            X_prev = mean + noise
            seq.append(X_prev.copy())
            Xk = X_prev

        # сейчас seq = [X_N, X_{N-1}, ..., X_0]
        seq = seq[::-1]  # [X_0, X_1, ..., X_N]

        pairs = []
        for k in range(self.N):
            Xk = seq[k]
            Xk1 = seq[k + 1]
            pairs.append((k, Xk, Xk1))

        return n_paths, pairs

    def build_F_dataset(self, n_paths: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Строим регрессионный датасет для обучения F:
            вход:  (X_k, t_k)
            таргет: X_k + (B(X_{k+1}, t_{k+1}) - B(X_k, t_{k+1}))
        как в loss_forward_on_backward_pairs в исходном DSB.
        """
        _, pairs = self.simulate_backward_pairs_dataset(n_paths)

        X_list = []
        t_list = []
        Y_list = []

        for (k, Xk, Xk1) in pairs:
            t_k = np.full((Xk.shape[0],), (k + 0.5) / self.N, dtype=np.float32)

            if (k + 1) < self.N:
                t_k1 = np.full(
                    (Xk1.shape[0],),
                    ((k + 1) + 0.5) / self.N,
                    dtype=np.float32,
                )
            else:
                t_k1 = np.full(
                    (Xk1.shape[0],),
                    (self.N - 0.5) / self.N,
                    dtype=np.float32,
                )

            B_Xk1 = self.B.predict(Xk1, t_k1)
            B_Xk1k = self.B.predict(Xk, t_k1)

            target = Xk + (B_Xk1 - B_Xk1k)  # как в loss_forward

            X_list.append(Xk)
            t_list.append(t_k)
            Y_list.append(target)

        X_train = np.concatenate(X_list, axis=0)
        t_train = np.concatenate(t_list, axis=0)
        Y_train = np.concatenate(Y_list, axis=0)

        return X_train.astype(np.float32), t_train.astype(np.float32), Y_train.astype(
            np.float32
        )

    # --------------------------------------------------------------------
    # Обучение (IPF)
    # --------------------------------------------------------------------

    def train(
        self,
        ipf_iters: int = 5,
        n_paths_B: int = 4_000,
        n_paths_F: int = 4_000,
        pretrain_samples: int = 50_000,
        print_swd: bool = True,
        n_swd_projections: int = 256,
    ) -> Dict[str, float]:
        """
        Основной цикл обучения:
            1) OU-pretrain F
            2) IPF:
                a) fix F, тренируем B по forward-парам
                b) fix B, тренируем F по backward-парам
        После каждой итерации — SWD на train (и val, если есть).
        """
        ipf_iters = int(ipf_iters)

        # OU pretrain
        self.pretrain_F_ou(n_samples=pretrain_samples)

        metrics_last: Dict[str, float] = {}

        for it in range(ipf_iters):
            # --- шаг B (фиксируем F) ---
            XB, tB, yB = self.build_B_dataset(n_paths_B)
            self.B.fit(XB, tB, yB)

            # --- шаг F (фиксируем B) ---
            XF, tF, yF = self.build_F_dataset(n_paths_F)
            self.F.fit(XF, tF, yF)

            # --- оценка ---
            metrics = self.evaluate(
                num_samples=min(5_000, self.X_train_w.shape[0]),
                n_swd_projections=n_swd_projections,
            )

            metrics_last = metrics

            if print_swd:
                msg = f"[IPF {it + 1}/{ipf_iters}] "
                msg += f"SWD(train)={metrics['swd_train']:.5f}"
                if metrics.get("swd_val") is not None:
                    msg += f" | SWD(val)={metrics['swd_val']:.5f}"
                print(msg)

        return metrics_last

    # --------------------------------------------------------------------
    # Сэмплирование из моста
    # --------------------------------------------------------------------

    def sample_from_bridge(
        self,
        num: int,
        steps_per_edge: int = 1,
    ) -> np.ndarray:
        """
        Семплирование из моста p_0 (train) через backward-процесс B.

        Стартуем из prior pN и идём назад по времени, по K steps_per_edge
        субшагов на каждый k, используя mean map B.
        """
        num = int(num)
        steps_per_edge = int(steps_per_edge)

        Xk = self.sample_prior_pN(num, std=1.0)

        for k in range(self.N - 1, -1, -1):
            for _ in range(steps_per_edge):
                t_k = np.full((num,), (k + 0.5) / self.N, dtype=np.float32)
                mean = self.B.predict(Xk, t_k)
                noise = (
                    self.rng.randn(*Xk.shape).astype(np.float32)
                    * math.sqrt(2.0 * float(self.gamma[k]) / steps_per_edge)
                )
                Xk = mean + noise

        return Xk.astype(np.float32)

    # --------------------------------------------------------------------
    # Оценка качества
    # --------------------------------------------------------------------

    def evaluate(
        self,
        num_samples: int = 5_000,
        n_swd_projections: int = 256,
        seed: int = 123,
    ) -> Dict[str, float]:
        """
        Простейшая оценка: SWD между синтетикой и train/val.
        Можно расширить SWD/MMD/KS по аналогии с исходным кодом.
        """
        rng_eval = np.random.RandomState(seed)

        n_train = min(num_samples, self.X_train_w.shape[0])
        X_syn = self.sample_from_bridge(n_train)
        swd_train = sliced_wasserstein_distance(
            self.X_train_w[:n_train],
            X_syn,
            n_projections=n_swd_projections,
            rng=rng_eval,
        )

        metrics: Dict[str, float] = {"swd_train": float(swd_train), "swd_val": None}

        if self.X_val_w is not None:
            n_val = min(num_samples, self.X_val_w.shape[0])
            X_syn_val = self.sample_from_bridge(n_val)
            swd_val = sliced_wasserstein_distance(
                self.X_val_w[:n_val],
                X_syn_val,
                n_projections=n_swd_projections,
                rng=rng_eval,
            )
            metrics["swd_val"] = float(swd_val)

        return metrics


# ------------------------------------------------------------------------
# Пример использования (можно удалить/переопределить под свой пайплайн)
# ------------------------------------------------------------------------

if __name__ == "__main__":
    # Пример: заглушка, чтобы проверить, что модуль импортируется и запускается.
    # Здесь X_train/X_val должны быть подготовлены снаружи.
    print("dsb_boostings_tabular imported as script. "
          "Подключай DSBTabularBridgeGBDT из другого файла.")
