import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import MultiTaskLasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix
from sklearn.impute import SimpleImputer
import joblib
import time

from driada import Experiment

# ==========================================
# CONFIGURATION
# ==========================================
# Directory containing .npz files
DATA_DIR = r'LNOF_SynchronizedData'
# Output directory for models
MODELS_DIR = r'models_LNOF_Lasso'

# List of mice to process
TARGET_MICE = ['J30', 'J06', 'J61', 'J25']

# List of sessions to process
TARGET_SESSIONS = ['1D', '2D', '3D', '4D']

# Hyperparameters
C_PARAM = 0.5
TEST_SIZE = 0.25
MAX_ITER = 100


def load_data(path):
    """
    Загружает данные из .npz и готовит их для MultiTaskLasso (X: нейроны, y: [x, y]).
    """
    filename = os.path.basename(path)
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"[ERROR] Could not load {filename}: {e}")
        return None, None

    # Проверяем наличие всех нужных ключей
    required_keys = ['Calcium', 'x', 'y']
    for key in required_keys:
        if key not in data.files:
            print(f"[ERROR] '{key}' not found in {filename}. Skipping.")
            return None, None

    # PREPROCESSING X (Нейронная активность)
    # ---------------------------------------------------------
    # (n_neurons, n_timepoints) -> (n_timepoints, n_neurons)
    X_raw_np = data['Calcium'].T
    n_samples, n_features = X_raw_np.shape

    # PREPROCESSING Y (Координаты мыши)
    # ---------------------------------------------------------
    # Собираем x и y в одну матрицу (n_samples, 2)
    y_np = np.column_stack((data['x'], data['y']))

    # Валидация размерностей
    if X_raw_np.shape[0] != y_np.shape[0]:
        print(f"  > [Error] Mismatch: X has {X_raw_np.shape[0]} samples, y has {y_np.shape[0]}. Skipping.")
        return None, None

    # --- Конвертация в Pandas ---
    feature_names = [f"neuron_{i}" for i in range(n_features)]

    # DataFrame для признаков (нейроны)
    X_raw = pd.DataFrame(X_raw_np, columns=feature_names)
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)

    # DataFrame для таргетов (координаты)
    # Для MultiTaskLasso y должен быть двумерным
    y = pd.DataFrame(y_np, columns=['x', 'y'])
    y = y.replace([np.inf, -np.inf], np.nan)

    print(f"  > Loaded: {n_samples} timepoints, {n_features} neurons, 2D coordinates (x, y)")

    return X_raw, y


def predict(model, imputer, scaler, X):
    """
    Make predictions with custom classification threshold.
    """
    # Imputer expects 2D array. transform() returns numpy array.
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)

    coords = model.predict(X_scaled)
    # predictions = (probabilities > threshold).astype(int)

    return coords


def train_test_split_temporal(X, y, test_size=0.25):
    """
    Splits data into train and test sets chronologically without shuffling.
    75% first part for training, last 25% for testing.
    """
    split_idx = int(len(X) * (1 - test_size))

    # Simple slicing for temporal consistency
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"  > Temporal split at index {split_idx} ({1 - test_size:.0%}/{test_size:.0%})")
    return X_train, X_test, y_train, y_test


def calculate_metrics(y_true, y_pred):
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    # 1. Вычисляем евклидово расстояние для каждой точки: sqrt((x_true-x_pred)^2 + (y_true-y_pred)^2)
    distances = np.linalg.norm(y_true_np - y_pred_np, axis=1)

    # Среднее расстояние (Mean Euclidean Distance)
    mean_dist = np.mean(distances)

    # Медианное расстояние (более устойчиво к выбросам)
    median_dist = np.median(distances)

    # Коэффициент детерминации R2 (усредненный по обеим координатам)
    r2 = r2_score(y_true, y_pred, multioutput='uniform_average')

    return mean_dist, median_dist, r2

def train_lasso(X_train, y_train, X_test, y_test, C=1.0, random_state=42,
                         max_iter=1000):
    """
    Train Logistic Regression with L1 regularization (LASSO for classification)
    """

    # Handle feature names
    if hasattr(X_train, 'columns'):
        feature_names = X_train.columns.tolist()
    else:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    # Imputer and scaler
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)

    # Logistic Regression with L1 regularization
    clf = MultiTaskLasso(
        alpha=C,
        random_state=random_state,
        max_iter=max_iter
    )

    clf.fit(X_train_scaled, y_train)

    # Predictions with custom threshold
    y_train_pred = predict(clf, imputer, scaler, X_train)
    y_test_pred = predict(clf, imputer, scaler, X_test)

    # print("true ", y_test)
    # print("pred ", y_test_pred)

    # Performance evaluation
    train_dist, train_med_dist, train_r2 = calculate_metrics(y_train, y_train_pred)
    test_dist, test_med_dist, test_r2 = calculate_metrics(y_test, y_test_pred)

    print(f"Train MED: {train_dist:.2f}, R2: {train_r2:.2f}")
    print(f"Test MED: {test_dist:.2f}, R2: {test_r2:.2f}")

    # Feature importance analysis (logging only)
    n_nonzero = 0
    if hasattr(clf, 'coef_'):
        n_nonzero = np.sum(clf.coef_[0] != 0)
        print(f"    [Info] Active neurons (non-zero weights): {n_nonzero}")

    metrics = {
        'train_dist': train_dist,
        'train_med_dist': train_med_dist,
        'train_r2': train_r2,
        'test_dist': test_dist,
        'test_dist_dist': test_med_dist,
        'test_r2': test_r2,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_active_features': int(n_nonzero)
    }

    return clf, imputer, scaler, metrics


def main():
    print("Starting ML Pipeline for Neuro-Behavioral Analysis...")

    # 1. Setup Output Directory
    # Ensure the models directory exists inside the current working directory
    output_path = Path(MODELS_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Models will be saved to: {output_path.absolute()}")

    # Loop over specified mice
    # for mouse in TARGET_MICE:
    #     # Loop over specified sessions
    #     for session in TARGET_SESSIONS:

    folder = Path(DATA_DIR)
    files = [f.name for f in folder.iterdir() if f.is_file()]

    for file in files:
        mouse = file[5:8]
        session = file[9:11]
        filename = f"LNOF_{mouse}_{session}_aligned.npz"
        file_path = os.path.join(DATA_DIR, filename)

        print(f"\n{'=' * 60}")
        print(f"PROCESSING: {mouse} | Session: {session}")
        print(f"{'=' * 60}")

        # Check file existence
        if not os.path.exists(file_path):
            print(f"[SKIP] File not found: {file_path}")
            continue

        # 2. Load Data
        X, y = load_data(file_path)
        # print(np.array(y).T)

        if X is None or y is None:
            continue

        # 3. Split Data
        X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, test_size=TEST_SIZE)

        # 4. Train & Evaluate
        clf, imp, scl, metrics = train_lasso(
            X_train, y_train, X_test, y_test,
            C=C_PARAM, max_iter=MAX_ITER
        )

        # 5. Save Model Artifacts
        # Naming convention: H01_1D_object1.pkl
        model_filename = f"{mouse}_{session}.pkl"
        model_save_path = output_path / model_filename

        # Create a comprehensive dictionary (like in the snippet)
        model_payload = {
            'model': clf,
            'imputer': imp,
            'scaler': scl,
            'feature_names': X_train.columns.tolist(),
            'params': {'C': C_PARAM, 'max_iter': MAX_ITER},
            'metrics': metrics,
            'metadata': {
                'mouse': mouse,
                'session': session,
                # 'object': obj,
                'source_file': filename
            }
        }

        # Save using joblib
        joblib.dump(model_payload, model_save_path)
        print(f"    ✓ Model saved: {model_filename}")

    print(f"\nPipeline processing complete. All models saved in {output_path}")
                # Here you could save the model:
                # import joblib
                # joblib.dump(clf, f"models/model_{mouse}_{session}_{obj}.pkl")


def shuffle_calcium_matrix(X, method="roll_based", min_shift=100):
    """
    Независимая функция для перемешивания матрицы кальциевой активности.
    Идеально подходит для создания случайного бейзлайна в ML-пайплайнах.

    Parameters
    ----------
    X : np.ndarray или pd.DataFrame
        Матрица признаков (кальциевая активность) размерности (n_frames, n_cells).
    method : str, default='roll_based'
        Метод перемешивания. 'roll_based' осуществляет циклический сдвиг.
    min_shift : int, default=100
        Минимальный отступ (в фреймах) для сдвига, чтобы избежать слишком
        слабого смещения, при котором корреляция с поведением может сохраниться.

    Returns
    -------
    X_shuffled : np.ndarray или pd.DataFrame
        Перемешанная матрица в том же формате, что и на входе.
    """
    # Определяем тип входных данных для корректного возврата
    is_df = isinstance(X, pd.DataFrame)
    X_vals = X.values if is_df else np.copy(X)

    n_frames, n_cells = X_vals.shape
    X_shuffled = np.zeros_like(X_vals)

    if method == "roll_based":
        # Проверяем, достаточно ли фреймов для заданного минимального сдвига
        if n_frames <= min_shift * 2:
            min_shift = max(1, n_frames // 10)  # Фолбэк для очень коротких сессий

        for cell_idx in range(n_cells):
            # Генерируем уникальный случайный сдвиг для КАЖДОГО нейрона независимо
            shift_val = np.random.randint(min_shift, n_frames - min_shift)

            # Циклический сдвиг (конец массива переходит в начало)
            X_shuffled[:, cell_idx] = np.roll(X_vals[:, cell_idx], shift_val)

    elif method == "random_permutation":
        # Полное разрушение временной структуры (обычно не рекомендуется для кальция,
        # так как спайки распадаются на белый шум, но полезно для строгого ML-контроля)
        for cell_idx in range(n_cells):
            X_shuffled[:, cell_idx] = np.random.permutation(X_vals[:, cell_idx])

    else:
        raise ValueError(f"Метод '{method}' не поддерживается в standalone версии.")

    # Возвращаем данные в исходном формате
    if is_df:
        return pd.DataFrame(X_shuffled, index=X.index, columns=X.columns)

    return X_shuffled


def main_baseline():
    print("Starting ML Pipeline for Neuro-Behavioral Analysis (RANDOM BASELINE)...")

    # 1. Setup Output Directory for Baselines
    # Создаем отдельную папку, чтобы не перезаписать реальные модели
    baseline_out_dir = f"{MODELS_DIR}_shuffled"
    output_path = Path(baseline_out_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Baseline models will be saved to: {output_path.absolute()}")

    folder = Path(DATA_DIR)
    files = [f.name for f in folder.iterdir() if f.is_file()]

    for file in files:
        # Парсинг имени файла (как в оригинале)
        mouse = file[5:8]
        session = file[9:11]
        filename = f"LNOF_{mouse}_{session}_aligned.npz"
        file_path = os.path.join(DATA_DIR, filename)

        print(f"\n{'=' * 60}")
        print(f"PROCESSING BASELINE: {mouse} | Session: {session}")
        print(f"{'=' * 60}")

        # Check file existence
        if not os.path.exists(file_path):
            print(f"[SKIP] File not found: {file_path}")
            continue

        # 2. Load Data
        X, y = load_data(file_path)

        if X is None or y is None:
            continue

        # -------------------------------------------------------------
        # ШАГ РАНДОМИЗАЦИИ: Перемешиваем кальциевую активность
        # -------------------------------------------------------------
        print("  -> Applying continuous roll-based shuffling to calcium activity...")

        # Используем нашу новую standalone функцию
        X_shuffled = shuffle_calcium_matrix(X, method="roll_based")

        # 3. Split Data (Используем ПЕРЕМЕШАННЫЙ X_shuffled)
        X_train, X_test, y_train, y_test = train_test_split_temporal(X_shuffled, y, test_size=TEST_SIZE)

        # 4. Train & Evaluate
        clf, imp, scl, metrics = train_lasso(
            X_train, y_train, X_test, y_test,
            C=C_PARAM, max_iter=MAX_ITER
        )

        # 3. Split Data (Используем ПЕРЕМЕШАННЫЙ X_shuffled)
        X_train, X_test, y_train, y_test = train_test_split_temporal(X_shuffled, y, test_size=TEST_SIZE)

        # 4. Train & Evaluate
        clf, imp, scl, metrics = train_lasso(
            X_train, y_train, X_test, y_test,
            C=C_PARAM, max_iter=MAX_ITER
        )

        print(f"  -> Baseline metrics: {metrics}")

        # 5. Save Model Artifacts
        # Добавляем суффикс _shuffled к имени файла для надежности
        model_filename = f"{mouse}_{session}_shuffled.pkl"
        model_save_path = output_path / model_filename

        model_payload = {
            'model': clf,
            'imputer': imp,
            'scaler': scl,
            'feature_names': X_train.columns.tolist(),
            'params': {'C': C_PARAM, 'max_iter': MAX_ITER},
            'metrics': metrics,
            'metadata': {
                'mouse': mouse,
                'session': session,
                'source_file': filename,
                'is_baseline': True,  # Удобный флаг для последующего анализа
                'shuffling_method': 'driada.experiment.get_multicell_shuffled_calcium'
            }
        }

        # Save using joblib
        joblib.dump(model_payload, model_save_path)
        print(f"    ✓ Baseline Model saved: {model_filename}")

    print(f"\nBaseline Pipeline complete. All models saved in {output_path}")


def train_top_and_rest_models(top_csv='stat_tables/top_neurons_lasso_3.0sigma.csv',
                              data_dir='LNOF_SynchronizedData',
                              base_models_dir='models_LNOF_Lasso',
                              output_dir='models_LNOF_Lasso',
                              metrics_csv='stat_tables/top_vs_rest_metrics_lasso.csv',
                              c_param=C_PARAM, max_iter=MAX_ITER, test_size=TEST_SIZE):
    """
    Обучает изолированные модели только на Топ-нейронах и только на остальных (Rest) активных нейронах.
    Сохраняет модели в отдельные папки и генерирует сравнительную таблицу метрик.
    """
    path_top_csv = Path(top_csv)
    if not path_top_csv.exists():
        print(f"[ERROR] Файл с топовыми нейронами не найден: {path_top_csv}")
        return

    # Создаем директории для новых моделей
    dir_top = Path(output_dir) / 'models_top'
    dir_rest = Path(output_dir) / 'models_rest'
    dir_top.mkdir(parents=True, exist_ok=True)
    dir_rest.mkdir(parents=True, exist_ok=True)

    Path(metrics_csv).parent.mkdir(parents=True, exist_ok=True)

    # Загружаем список топовых нейронов
    df_top_all = pd.read_csv(path_top_csv)
    # Нам нужны только строки с именами нейронов
    df_names = df_top_all[df_top_all['data_type'] == 'Neuron Names'].copy()

    top_cols = [col for col in df_names.columns if col.startswith('top_')]

    results = []

    # Инициализация таймера
    start_time = time.time()
    last_notify_time = start_time
    notify_interval = 1800  # 30 минут в секундах
    models_trained_count = 0

    print(f"\n{'=' * 70}")
    print(f"Запуск изоляционного обучения: Top vs Rest")
    print(f"Таймер запущен. Уведомления каждые 30 минут.")
    print(f"{'=' * 70}")

    for _, row in df_names.iterrows():
        mouse = row['mouse']
        day = row['session']
        model_name = f"{mouse}_{day}"  # Например, J30_1D

        # 1. Извлекаем списки нейронов
        # Собираем топовые нейроны, очищая от NaN
        top_neurons = [row[c] for c in top_cols if pd.notna(row[c])]

        # Загружаем базовую модель, чтобы узнать, какие нейроны вообще были активны (>0)
        base_model_path = Path(base_models_dir) / f"{model_name}.pkl"
        if not base_model_path.exists():
            print(f"[SKIP] Базовая модель {model_name} не найдена.")
            continue

        base_data = joblib.load(base_model_path)
        base_coef = base_data.get('model').coef_
        all_features = np.array(base_data.get('feature_names'))

        # Ищем активные нейроны в базовой модели
        importances = np.linalg.norm(base_coef, axis=0)
        active_features = all_features[importances > 0].tolist()

        # Формируем список "Остальных" (активные минус топовые)
        rest_neurons = [f for f in active_features if f not in top_neurons]

        # 2. Загружаем сырые данные сессии
        data_file = Path(data_dir) / f"LNOF_{model_name}_aligned.npz"
        if not data_file.exists():
            print(f"[SKIP] Файл данных {data_file} не найден.")
            continue

        features_raw, labels = load_data(data_file)
        if features_raw is None or labels is None:
            continue

        X_train_full, X_test_full, y_train, y_test = train_test_split_temporal(features_raw, labels,
                                                                               test_size=test_size)

        row_metrics = {
            'mouse': mouse,
            'session': day,
            'model_name': model_name,
            'base_active_count': len(active_features),
            'top_count': len(top_neurons),
            'rest_count': len(rest_neurons)
        }

        print(f"[{model_name}] Топ: {len(top_neurons)} шт | Rest: {len(rest_neurons)} шт | Запуск обучения...")

        # --- 3. ОБУЧЕНИЕ TOP МОДЕЛИ ---
        if len(top_neurons) > 0:
            X_train_top = X_train_full[top_neurons]
            X_test_top = X_test_full[top_neurons]

            clf_top, imp_top, scl_top, metrics_top = train_lasso(
                X_train_top, y_train, X_test_top, y_test, C=c_param, max_iter=max_iter
            )

            row_metrics['top_r2'] = metrics_top.get('test_r2', np.nan)
            row_metrics['top_med'] = metrics_top.get('test_dist', np.nan)

            # Сохраняем модель
            joblib.dump({
                'model': clf_top, 'scaler': scl_top, 'feature_names': top_neurons, 'metrics': metrics_top
            }, dir_top / f"{model_name}_top.pkl")
            models_trained_count += 1
        else:
            row_metrics['top_r2'], row_metrics['top_med'] = np.nan, np.nan

        # --- 4. ОБУЧЕНИЕ REST МОДЕЛИ ---
        if len(rest_neurons) > 0:
            X_train_rest = X_train_full[rest_neurons]
            X_test_rest = X_test_full[rest_neurons]

            clf_rest, imp_rest, scl_rest, metrics_rest = train_lasso(
                X_train_rest, y_train, X_test_rest, y_test, C=c_param, max_iter=max_iter
            )

            row_metrics['rest_r2'] = metrics_rest.get('test_r2', np.nan)
            row_metrics['rest_med'] = metrics_rest.get('test_dist', np.nan)

            # Сохраняем модель
            joblib.dump({
                'model': clf_rest, 'scaler': scl_rest, 'feature_names': rest_neurons, 'metrics': metrics_rest
            }, dir_rest / f"{model_name}_rest.pkl")
            models_trained_count += 1
        else:
            row_metrics['rest_r2'], row_metrics['rest_med'] = np.nan, np.nan

        results.append(row_metrics)

        # --- 5. ТАЙМЕР И УВЕДОМЛЕНИЯ ---
        current_time = time.time()
        if current_time - last_notify_time >= notify_interval:
            elapsed_mins = int((current_time - start_time) / 60)
            print(f"\n[⏱ ТАЙМЕР] Прошло {elapsed_mins} минут. Успешно обучено {models_trained_count} моделей...\n")
            last_notify_time = current_time

    # --- 6. СОХРАНЕНИЕ ТАБЛИЦЫ ---
    if results:
        df_results = pd.DataFrame(results)

        # Упорядочиваем колонки для читаемости
        cols = ['mouse', 'session', 'model_name',
                'base_active_count', 'top_count', 'rest_count',
                'top_r2', 'rest_r2', 'top_med', 'rest_med']
        df_results = df_results[cols]

        df_results.to_csv(metrics_csv, index=False)

        total_mins = int((time.time() - start_time) / 60)
        print(f"\n{'=' * 70}")
        print(f"✓ Изоляционное обучение завершено за {total_mins} минут!")
        print(f"✓ Всего обучено моделей: {models_trained_count}")
        print(f"✓ Таблица метрик сохранена: {metrics_csv}")
        print(f"{'=' * 70}")
    else:
        print("[WARNING] Не удалось обучить ни одной модели. Проверьте входные данные.")

    return df_results


if __name__ == "__main__":
    
    # train_top_and_rest_models()

    # main()
    main_baseline()
    # file_path = r"C:\Users\admin\PycharmProjects\NOF_analysis\LNOF_SynchronizedData\LNOF_J24_3D_aligned.npz"
    #
    # data = np.load(file_path, allow_pickle=True)['x']
    # data = data[data < 0]
    # print(data)
    # print(data.files)
    # features, labels = load_data(file_path)
    #
    # X_train, X_test, y_train, y_test = train_test_split_temporal(
    #     features, labels, test_size=TEST_SIZE)
    #
    # clf, imp, scl, metrics = train_lasso(
    #     X_train, y_train, X_test, y_test,
    #     C=C_PARAM, max_iter=MAX_ITER
    # )


    # Загрузка архива
    # data = np.load('LNOF_SynchronizedData/LNOF_J01_1D_aligned.npz')
    #
    # # Просмотр списка массивов внутри
    # print(data['Calcium'])
