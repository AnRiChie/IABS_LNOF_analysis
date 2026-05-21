import os
import re
import joblib
import time
import random
from scipy import stats
from scipy.stats import kendalltau, sem, wilcoxon
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import MultiTaskLasso
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from pathlib import Path
from matplotlib.patches import Patch
from linreg_behavior import load_data
from sklearn.metrics import r2_score
from itertools import combinations
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def print_top_models(models_dir='models_LNOF_Lasso', descend=True, n=3):
    """
    Выводит топ-n модели по метрике R2.
    """
    models_path = Path(models_dir)
    results = []

    for file in models_path.glob("*.pkl"):
        if "best" in file.name: continue
        try:
            data = joblib.load(file)
            r2 = data.get('metrics', {}).get('test_r2', -float('inf'))
            results.append((file.name, r2))
        except:
            continue

    # Сортируем по R2 по убыванию (чем ближе к 1, тем лучше)
    top = sorted(results, key=lambda x: x[1], reverse=descend)[:n]

    print("Top 3 Models by R2 Score:")
    for name, score in top:
        print(f"{name}: R2 = {score:.4f}")

    return top


# Анализ важности признаков (нейронов)
def print_feature_importance(mouse_id=None, session=None, obj_name=None,
                             models_dir='models_LNOF_Lasso', importance_threshold=0.1):
    """
    Выводит значимые нейроны. Важность для MultiTaskLasso = L2 норма весов по осям X и Y.
    """
    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"[ERROR] Directory {models_dir} not found.")
        return

    filtered_files = [f for f in models_path.glob("*.pkl") if "best" not in f.name]
    if mouse_id: filtered_files = [f for f in filtered_files if mouse_id in f.name]
    if session: filtered_files = [f for f in filtered_files if session in f.name]
    if obj_name: filtered_files = [f for f in filtered_files if obj_name in f.name]

    if not filtered_files:
        print("No matching models found to analyze features.")
        return

    for model_file in filtered_files:
        try:
            data = joblib.load(model_file)
            model = data.get('model')
            feature_names = data.get('feature_names')
            metadata = data.get('metadata', {})

            if not hasattr(model, 'coef_'):
                continue

            # MultiTaskLasso coef_ имеет форму (2, n_features)
            # Считаем важность нейрона как корень из суммы квадратов весов для x и y
            importance = np.linalg.norm(model.coef_, axis=0)

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            })

            important_features = importance_df[importance_df['importance'] >= importance_threshold].copy()
            important_features = important_features.sort_values(by='importance', ascending=False)

            print(f"\n{'#' * 70}")
            print(f" MODEL: {metadata.get('mouse')} | Session: {metadata.get('session')}")
            print(f" File: {model_file.name}")
            print(f"{'-' * 70}")

            if important_features.empty:
                print(f" No features met the threshold of importance >= {importance_threshold}")
            else:
                print(f"{'Neuron ID':<20} | {'Importance (L2 Norm)':<20}")
                print(f"{'-' * 45}")
                for _, row in important_features.iterrows():
                    print(f"{row['feature']:<20} | {row['importance']:<10.4f}")

                print(f"{'-' * 70}")
                print(f" Total significant neurons: {len(important_features)}")

        except Exception as e:
            print(f"[ERROR] Could not process {model_file.name}: {e}")


# Поиск гиперпараметров (GridSearch)
def perform_lasso_grid_search(mouse_id, session, data_dir='LNOF_SynchronizedData'):
    """
    Grid Search для MultiTaskLasso. Максимизирует метрику R2.
    """
    file_path = os.path.join(data_dir, f"LNOF_{mouse_id}_{session}_aligned.npz")
    X, y = load_data(file_path)

    if X is None or y is None:
        return None

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('regressor', MultiTaskLasso(random_state=42))
    ])

    # Для Lasso используем alpha (чем больше alpha, тем сильнее зануляются веса)
    param_grid = {
        'regressor__alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        'regressor__max_iter': [10, 50, 100, 500, 1000, 2000]
    }

    cv = TimeSeriesSplit(n_splits=5)

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='r2',  # Используем стандартный R2 score для оценки качества регрессии
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    print(f"\n[OK] Best configuration for {mouse_id}_{session}:")
    print(f"     Optimal Alpha: {best_params['regressor__alpha']}")
    print(f"     Optimal Max Iter:  {best_params['regressor__max_iter']}")
    print(f"     Best CV R2 Score: {grid_search.best_score_:.4f}")

    return {k.split('__')[1]: v for k, v in best_params.items()}


def save_neuron_weights(model_path, top_n=None, threshold=None):
    """
    Извлекает веса нейронов из модели MultiTaskLasso, считает их общую важность,
    фильтрует и сохраняет в CSV.

    Параметры:
    - model_path: путь к файлу .pkl с сохраненной моделью.
    - output_csv_name: имя выходного csv файла (будет сохранен рядом с моделью или по указанному пути).
    - top_n: int. Если задан, оставляет N нейронов с самыми большими весами.
    - threshold: float. Если задан, оставляет нейроны, чья важность >= threshold.
      (Внимание: если заданы оба, сначала применяется threshold, затем top_n).
    """
    path = Path(model_path)
    if not path.exists():
        print(f"[ERROR] Модель не найдена по пути: {model_path}")
        return

    # 1. Загрузка модели и данных
    data = joblib.load(path)
    model = data.get('model')
    feature_names = data.get('feature_names')

    if not hasattr(model, 'coef_'):
        print("[ERROR] В файле нет линейной модели с атрибутом coef_.")
        return

    # coef_ в MultiTaskLasso имеет размерность (n_targets, n_features) -> (2, n_neurons)
    weights = model.coef_

    # Извлекаем веса по X (строка 0) и по Y (строка 1)
    weights_x = weights[0, :]
    weights_y = weights[1, :]

    # 2. Считаем общую важность (Евклидова норма L2)
    importance = np.linalg.norm(weights, axis=0)

    # 3. Собираем всё в DataFrame
    df = pd.DataFrame({
        'neuron_id': feature_names,
        'weight_x': weights_x,
        'weight_y': weights_y,
        'importance_l2': importance
    })

    # 4. Фильтрация
    # Сначала убираем полные нули (неактивные нейроны, выжженные Lasso)
    df = df[df['importance_l2'] > 0]

    if threshold is not None:
        df = df[df['importance_l2'] >= threshold]

    if top_n is not None:
        # nlargest автоматически сортирует по убыванию
        df = df.nlargest(top_n, columns='importance_l2')
    else:
        # Если top_n не задан, просто отсортируем всё, что осталось, по убыванию
        df = df.sort_values(by='importance_l2', ascending=False)

    # 5. Сохранение
    # output_path = Path(output_csv_name)
    # # Создаем директорию, если она не существует
    # output_path.parent.mkdir(parents=True, exist_ok=True)
    #
    # df.to_csv(output_path, index=False)
    # print(f"✓ Таблица весов сохранена: {output_path}")
    # print(f"  Осталось нейронов после фильтрации: {len(df)}")

    return df


def csv_match(path, feature: str):
    df_matched = pd.read_csv(path)
    df_feature = df_matched[df_matched['feature'] == feature]

    return df_feature


def prune_model_features(model_path, threshold=None, feature_to_remove=None):
    """
    Удаляет признаки (нейроны) из пайплайна модели (Imputer -> Scaler -> Lasso).

    Параметры:
    - model_path: путь к .pkl файлу.
    - threshold: float. Если задан, удаляет все нейроны, чья L2-норма весов < threshold.
    - feature_to_remove: str. Если задан, удаляет конкретный нейрон по имени (например, 'neuron_5').

    Возвращает:
    - modified_data: словарь с обновленной моделью и метаданными.
    """
    # Загружаем исходный словарь
    data = joblib.load(model_path)

    model = data['model']
    imputer = data['imputer']
    scaler = data['scaler']
    feature_names = np.array(data['feature_names'])

    # 1. Формируем булеву маску для нейронов, которые мы ОСТАВЛЯЕМ
    mask = np.ones(len(feature_names), dtype=bool)

    if feature_to_remove is not None:
        # Убираем конкретный нейрон
        feature_to_remove = ['neuron_' + str(i) for i in feature_to_remove]
        mask &= ~np.isin(feature_names, feature_to_remove)

    if threshold is not None:
        # Считаем L2-норму весов для каждого нейрона (по x и y)
        # model.coef_ имеет форму (2, n_features)
        importance = np.linalg.norm(model.coef_, axis=0)
        mask = mask & (importance >= threshold)

    # Если ничего не изменилось, просто возвращаем исходные данные
    if np.all(mask):
        print("[INFO] Ни один признак не был удален. Маска совпадает с исходной.")
        return data

    new_n_features = np.sum(mask)
    print(f"[INFO] Удалено {len(mask) - new_n_features} признаков. Осталось: {new_n_features}")

    # 2. Модифицируем атрибуты MultiTaskLasso
    model.coef_ = model.coef_[:, mask]
    model.n_features_in_ = new_n_features
    # У Lasso также может быть атрибут feature_names_in_
    if hasattr(model, 'feature_names_in_'):
        model.feature_names_in_ = model.feature_names_in_[mask]

    # 3. Модифицируем атрибуты SimpleImputer
    imputer.statistics_ = imputer.statistics_[mask]
    imputer.n_features_in_ = new_n_features
    if hasattr(imputer, 'feature_names_in_'):
        imputer.feature_names_in_ = imputer.feature_names_in_[mask]

    # 4. Модифицируем атрибуты StandardScaler
    scaler.mean_ = scaler.mean_[mask]
    scaler.var_ = scaler.var_[mask]
    scaler.scale_ = scaler.scale_[mask]
    scaler.n_features_in_ = new_n_features
    if hasattr(scaler, 'feature_names_in_'):
        scaler.feature_names_in_ = scaler.feature_names_in_[mask]

    # 5. Обновляем список имен в самом словаре
    data['feature_names'] = feature_names[mask].tolist()

    # Возвращаем модифицированный словарь "в оперативке"
    # (если нужно, можешь сделать joblib.dump(data, 'new_model.pkl'))
    return data


def load_dataset_for_modified_model(model_dict, data_dir='LNOF_SynchronizedData'):
    """
    Загружает X и y из .npz, оставляя в X только те нейроны, которые есть в модифицированной модели.
    Индексы вытаскиваются прямо из названий (neuron_0 -> индекс 0).
    """
    metadata = model_dict.get('metadata', {})
    mouse = metadata.get('mouse')
    session = metadata.get('session')

    if not mouse or not session:
        print("[ERROR] В модели отсутствуют метаданные (mouse, session) для поиска файла.")
        return None, None

    # Формируем путь к файлу
    filepath = Path(data_dir) / f"LNOF_{mouse}_{session}_aligned.npz"

    if not filepath.exists():
        print(f"[ERROR] Файл данных не найден: {filepath}")
        return None, None

    # Загружаем сырые данные
    npz = np.load(filepath, allow_pickle=True)

    # Calcium.T переводит из (n_neurons, n_timepoints) в (n_timepoints, n_neurons)
    calcium_raw = npz['Calcium'].T
    y_raw = np.column_stack((npz['x'], npz['y']))

    # Извлекаем индексы нужных нейронов из feature_names измененной модели
    # Используем регулярку, чтобы вытащить число из строк вида "neuron_5" или "neuron_128"
    valid_feature_names = model_dict['feature_names']

    indices_to_keep = []
    for name in valid_feature_names:
        match = re.search(r'\d+', name)
        if match:
            indices_to_keep.append(int(match.group()))
        else:
            print(f"[WARNING] Не удалось извлечь индекс из имени признака: {name}")

    # Срезаем матрицу кальциевой активности по нужным столбцам (индексам)
    try:
        X_filtered = calcium_raw[indices_to_keep]
    except IndexError as e:
        print(
            f"[ERROR] Ошибка индексации массива Calcium. Убедитесь, что индексы в диапазоне 0-{calcium_raw.shape[1] - 1}")
        raise e

    # Конвертируем в Pandas
    X_df = pd.DataFrame(X_filtered, columns=valid_feature_names)
    X_df = X_df.replace([np.inf, -np.inf], np.nan)

    y_df = pd.DataFrame(y_raw, columns=['x', 'y'])
    y_df = y_df.replace([np.inf, -np.inf], np.nan)

    print(f"[OK] Данные загружены. Форма X: {X_df.shape}, Форма y: {y_df.shape}")

    return X_df, y_df


def load_dataset_for_modified_model(model_dict, data_dir='LNOF_SynchronizedData'):
    """
    Загружает X и y из .npz, оставляя в X только те нейроны, которые есть в модифицированной модели.
    Индексы вытаскиваются прямо из названий (neuron_0 -> индекс 0).
    """
    metadata = model_dict.get('metadata', {})
    mouse = metadata.get('mouse')
    session = metadata.get('session')

    if not mouse or not session:
        print("[ERROR] В модели отсутствуют метаданные (mouse, session) для поиска файла.")
        return None, None

    # Формируем путь к файлу
    filepath = Path(data_dir) / f"LNOF_{mouse}_{session}_aligned.npz"

    if not filepath.exists():
        print(f"[ERROR] Файл данных не найден: {filepath}")
        return None, None

    # Загружаем сырые данные
    npz = np.load(filepath, allow_pickle=True)

    # Calcium.T переводит из (n_neurons, n_timepoints) в (n_timepoints, n_neurons)
    calcium_raw = npz['Calcium'].T
    y_raw = np.column_stack((npz['x'], npz['y']))

    # Извлекаем индексы нужных нейронов из feature_names измененной модели
    # Используем регулярку, чтобы вытащить число из строк вида "neuron_5" или "neuron_128"
    valid_feature_names = model_dict['feature_names']

    indices_to_keep = []
    for name in valid_feature_names:
        match = re.search(r'\d+', name)
        if match:
            indices_to_keep.append(int(match.group()))
        else:
            print(f"[WARNING] Не удалось извлечь индекс из имени признака: {name}")

    # Срезаем матрицу кальциевой активности по нужным столбцам (индексам)
    try:
        X_filtered = calcium_raw[indices_to_keep]
    except IndexError as e:
        print(
            f"[ERROR] Ошибка индексации массива Calcium. Убедитесь, что индексы в диапазоне 0-{calcium_raw.shape[1] - 1}")
        raise e

    # Конвертируем в Pandas
    X_df = pd.DataFrame(X_filtered, columns=valid_feature_names)
    X_df = X_df.replace([np.inf, -np.inf], np.nan)

    y_df = pd.DataFrame(y_raw, columns=['x', 'y'])
    y_df = y_df.replace([np.inf, -np.inf], np.nan)

    print(f"[OK] Данные загружены. Форма X: {X_df.shape}, Форма y: {y_df.shape}")

    return X_df, y_df


def plot_ablation_r2(model_paths, output_dir='plots'):
    """
    Удаляет по одному значимому нейрону из модели и строит график R2.
    Красная точка - исходный R2, черные точки - R2 после удаления одного нейрона.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []  # Для черных точек (pruned models)
    baselines = []  # Для красных точек (base models)

    model_paths = ['models_LNOF_Lasso/' + i for i in model_paths]

    for m_path in model_paths:
        base_data = joblib.load(m_path)
        model_name = Path(m_path).stem  # Например, 'J30_1D'

        print(f"\n[{model_name}] Загрузка исходных данных...")

        # Загружаем полный датасет для исходной модели
        filepath = Path('LNOF_SynchronizedData') / f"LNOF_{model_name}_aligned.npz"
        X_full, y_full = load_data(filepath)

        if X_full is None:
            continue

        # Воспроизводим временной сплит 75/25, чтобы считать R2 честно на тесте
        split_idx = int(len(X_full) * 0.75)
        X_test = X_full.iloc[split_idx:]
        y_test = y_full.iloc[split_idx:]

        # --- 1. Считаем бейзлайн (Красная точка) ---
        X_imp = base_data['imputer'].transform(X_test)
        X_sc = base_data['scaler'].transform(X_imp)
        y_pred = base_data['model'].predict(X_sc)
        base_r2 = r2_score(y_test, y_pred, multioutput='uniform_average')

        baselines.append({'model': model_name, 'r2': base_r2})

        # --- 2. Ищем значимые нейроны ---
        importance = np.linalg.norm(base_data['model'].coef_, axis=0)
        active_indices = np.where(importance > 0)[0]

        # Достаем их индексы (числа) из названий ('neuron_5' -> 5)
        active_neuron_ids = []
        for idx in active_indices:
            match = re.search(r'\d+', base_data['feature_names'][idx])
            if match:
                active_neuron_ids.append(int(match.group()))

        print(f"[{model_name}] Найдено {len(active_neuron_ids)} значимых нейронов (>0). Начинаем ablation...")

        # --- 3. Удаляем по одному и считаем R2 (Черные точки) ---
        for feat_id in active_neuron_ids:
            # Обрезаем модель (передаем в виде списка [feat_id])
            pruned_dict = prune_model_features(m_path, feature_to_remove=[feat_id])

            # ОПТИМИЗАЦИЯ: Не загружаем .npz заново! Просто берем нужные колонки из X_test
            valid_columns = pruned_dict['feature_names']
            X_test_pruned = X_test[valid_columns]

            # Предсказываем урезанной моделью
            p_imp = pruned_dict['imputer'].transform(X_test_pruned)
            p_sc = pruned_dict['scaler'].transform(p_imp)
            p_pred = pruned_dict['model'].predict(p_sc)

            p_r2 = r2_score(y_test, p_pred, multioutput='uniform_average')

            results.append({
                'model': model_name,
                'r2': p_r2,
                'removed_neuron': feat_id
            })

    if not results:
        print("[WARNING] Нет данных для построения графика.")
        return

    # --- 4. Отрисовка графика ---
    df_results = pd.DataFrame(results)
    df_base = pd.DataFrame(baselines)

    # Ширина графика динамически зависит от количества моделей
    plt.figure(figsize=(max(10, len(model_paths) * 1.5), 7))

    # Рисуем черные точки (Pruned). stripplot добавит небольшой "джиттер" (разброс по X),
    # чтобы точки не слипались в одну сплошную вертикальную линию.
    sns.stripplot(
        data=df_results,
        x='model',
        y='r2',
        color='black',
        alpha=0.6,
        jitter=True,
        size=5,
        label='Ablation (без 1 нейрона)'
    )

    # Рисуем красные точки (Baseline) поверх всего остального (zorder=10)
    sns.scatterplot(
        data=df_base,
        x='model',
        y='r2',
        color='red',
        s=40,
        zorder=10,
        label='Baseline (все нейроны)',
        edgecolor='none'
    )

    plt.title('Ablation Study: Падение R2 при удалении одного значимого нейрона', fontsize=15)
    plt.xlabel('Модель (Сессия)', fontsize=12)
    plt.ylabel(r'Коэффициент детерминации ($R^2$)', fontsize=12)

    # Сетка и угол наклона подписей
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xticks(rotation=45, ha='right')

    # Убираем дубликаты из легенды (stripplot + scatterplot могут их плодить)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')

    plt.tight_layout()
    save_path = output_path / 'pruned_r2_comparison.png'
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✓ График Ablation Study успешно сохранен: {save_path}")


def plot_cumulative_ablation(model_filenames, models_dir='models_LNOF_Lasso', data_dir='LNOF_SynchronizedData',
                             output_dir='plots'):
    """
    Строит график зависимости R2 от последовательного исключения нейронов.
    Красная кривая - удаление от самых значимых к слабым (Top-Down).
    Синяя кривая - удаление от самых слабых к значимым (Bottom-Up).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Подготавливаем пути по твоей логике
    model_paths = [str(Path(models_dir) / m) for m in model_filenames]

    for m_path in model_paths:
        base_data = joblib.load(m_path)
        model_name = Path(m_path).stem  # Например, 'J30_1D'

        print(f"\n{'=' * 50}")
        print(f"[{model_name}] Загрузка исходных данных...")

        filepath = Path(data_dir) / f"LNOF_{model_name}_aligned.npz"
        X_full, y_full = load_data(filepath)  # Используем твой load_data

        if X_full is None or y_full is None:
            print(f"[SKIP] Нет данных для {model_name}")
            continue

        # Временной сплит 75/25
        split_idx = int(len(X_full) * 0.75)
        X_test = X_full.iloc[split_idx:]
        y_test = y_full.iloc[split_idx:]

        # --- 1. Считаем бейзлайн (точка 0) ---
        X_imp = base_data['imputer'].transform(X_test)
        X_sc = base_data['scaler'].transform(X_imp)
        y_pred = base_data['model'].predict(X_sc)
        base_r2 = r2_score(y_test, y_pred, multioutput='uniform_average')

        # --- 2. Ранжируем значимые нейроны ---
        importance = np.linalg.norm(base_data['model'].coef_, axis=0)
        active_indices = np.where(importance > 0)[0]

        # Собираем кортежи (ID нейрона, важность)
        active_neurons = []
        for idx in active_indices:
            match = re.search(r'\d+', base_data['feature_names'][idx])
            if match:
                active_neurons.append((int(match.group()), importance[idx]))

        if not active_neurons:
            print(f"[SKIP] Нет активных нейронов в {model_name}")
            continue

        # Сортируем
        top_down = sorted(active_neurons, key=lambda x: x[1], reverse=True)  # Самые сильные сверху
        bottom_up = sorted(active_neurons, key=lambda x: x[1], reverse=False)  # Самые слабые сверху

        top_down_ids = [n[0] for n in top_down]
        bottom_up_ids = [n[0] for n in bottom_up]

        # --- 3. Кумулятивное удаление: Top-Down (Красная кривая) ---
        print(f"[{model_name}] Считаем Top-Down (от сильных к слабым)...")
        r2_top_down = [base_r2]
        removed_top = []

        for feat_id in top_down_ids:
            removed_top.append(feat_id)  # Накапливаем удаляемые нейроны

            # Обрезаем модель
            pruned_dict = prune_model_features(m_path, feature_to_remove=removed_top)

            # Оставляем в данных только те фичи, которые остались в модели
            X_test_pruned = X_test[pruned_dict['feature_names']]

            # Предсказываем
            p_imp = pruned_dict['imputer'].transform(X_test_pruned)
            p_sc = pruned_dict['scaler'].transform(p_imp)
            p_pred = pruned_dict['model'].predict(p_sc)

            r2_top_down.append(r2_score(y_test, p_pred, multioutput='uniform_average'))

        # --- 4. Кумулятивное удаление: Bottom-Up (Синяя кривая) ---
        print(f"[{model_name}] Считаем Bottom-Up (от слабых к сильным)...")
        r2_bottom_up = [base_r2]
        removed_bottom = []

        for feat_id in bottom_up_ids:
            removed_bottom.append(feat_id)  # Накапливаем удаляемые нейроны

            pruned_dict = prune_model_features(m_path, feature_to_remove=removed_bottom)

            X_test_pruned = X_test[pruned_dict['feature_names']]

            p_imp = pruned_dict['imputer'].transform(X_test_pruned)
            p_sc = pruned_dict['scaler'].transform(p_imp)
            p_pred = pruned_dict['model'].predict(p_sc)

            r2_bottom_up.append(r2_score(y_test, p_pred, multioutput='uniform_average'))

        # --- 5. Отрисовка графиков ---
        plt.figure(figsize=(10, 6))

        # Ось X: от 0 (ничего не удалено) до N (удалены все значимые)
        x_vals = range(len(top_down_ids) + 1)

        plt.plot(x_vals, r2_top_down, color='red', marker='o', markersize=4, linewidth=2,
                 label='Top-Down (сначала удаляем сильные)')
        plt.plot(x_vals, r2_bottom_up, color='blue', marker='o', markersize=4, linewidth=2,
                 label='Bottom-Up (сначала удаляем слабые)')

        plt.title(f'Cumulative Ablation: Падение R2 при удалении нейронов\nМодель: {model_name}', fontsize=14)
        plt.xlabel('Количество исключенных нейронов', fontsize=12)
        plt.ylabel(r'$R^2$ Score (качество декодирования)', fontsize=12)

        # Защита от сильных провалов в минус, чтобы график оставался читаемым
        min_y = min(min(r2_top_down), min(r2_bottom_up))
        if min_y < -1.0:
            plt.ylim(bottom=-1.0)  # Ограничиваем хвост снизу

        plt.axhline(0, color='black', linewidth=1, linestyle='--')  # Линия нулевого R2
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='best')

        save_path = output_path / f'cumulative_ablation_{model_name}.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"✓ График сохранен: {save_path}")


from linreg_behavior import train_test_split_temporal, train_lasso


def plot_seed_robustness(npz_filenames, data_dir='LNOF_SynchronizedData', output_dir='plots',
                         seeds=[42, 123, 999, 2026, 7777],
                         c_param=1.0, max_iter=1000, test_size=0.25):
    """
    Обучает модели на указанных файлах с разными random_state (сидами)
    и строит график дисперсии R2.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []

    for filename in npz_filenames:
        # Извлекаем красивое имя модели (например, 'J30_1D' из 'LNOF_J30_1D_aligned.npz')
        model_name = filename.replace('LNOF_', '').replace('_aligned.npz', '')
        file_path = Path(data_dir) / filename

        if not file_path.exists():
            print(f"[SKIP] Файл не найден: {file_path}")
            continue

        print(f"\n[{model_name}] Загрузка данных...")

        # Загружаем данные (твоя функция)
        features, labels = load_data(file_path)
        if features is None or labels is None:
            continue

        # Хронологический сплит (одинаковый для всех сидов)
        X_train, X_test, y_train, y_test = train_test_split_temporal(
            features, labels, test_size=test_size
        )

        print(f"[{model_name}] Обучение на {len(seeds)} разных seeds...")
        for seed in seeds:
            # Обучаем модель (убедись, что твой train_lasso передает random_state=seed внутрь MultiTaskLasso)
            clf, imp, scl, metrics = train_lasso(
                X_train, y_train, X_test, y_test,
                C=c_param, max_iter=max_iter, random_state=seed
            )

            # Предполагаем, что test_r2 лежит в словаре metrics (как мы делали ранее)
            # Если твой train_lasso возвращает сразу R2, замени эту строчку
            r2 = metrics.get('test_r2', np.nan)

            results.append({
                'model': model_name,
                'seed': seed,
                'r2': r2
            })

    if not results:
        print("[WARNING] Нет данных для построения графика.")
        return

    # --- Подготовка данных для графика ---
    df_results = pd.DataFrame(results)

    # Считаем среднее значение R2 для каждой модели
    df_mean = df_results.groupby('model')['r2'].mean().reset_index()

    # --- Отрисовка ---
    # Ширина графика динамически зависит от количества моделей
    plt.figure(figsize=(max(10, len(npz_filenames) * 1.5), 7))

    # Рисуем все попытки с разными сидами (черные точки)
    sns.stripplot(
        data=df_results,
        x='model',
        y='r2',
        color='black',
        alpha=0.6,
        jitter=True,
        size=5,
        label='Individual Seed'
    )

    # Рисуем среднее значение (маленькая красная точка БЕЗ обводки)
    # edgecolor='none' убирает черное кольцо, s=40 делает точку компактной
    sns.scatterplot(
        data=df_mean,
        x='model',
        y='r2',
        color='red',
        s=40,
        edgecolor='none',
        zorder=10,
        label='Mean $R^2$'
    )

    plt.title('Robustness Check: Дисперсия $R^2$ при разных Random Seeds', fontsize=15)
    plt.xlabel('Модель (Сессия)', fontsize=12)
    plt.ylabel(r'Коэффициент детерминации ($R^2$)', fontsize=12)

    # Сетка и угол наклона подписей
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xticks(rotation=45, ha='right')

    # Убираем дубликаты из легенды
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')

    plt.tight_layout()
    save_path = output_path / 'seed_robustness_variance.png'
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✓ График проверки сидов успешно сохранен: {save_path}")


def calculate_cross_session_overlap(mice_list, threshold, models_dir='models_LNOF_Lasso',
                                    csv_dir='LNOF_SynchronizedData'):
    """
    Извлекает значимые нейроны по порогу и вычисляет долю совпадений (Jaccard index)
    между сессиями, проходя по ребрам полносвязного графа 4 дней.
    """
    models_path = Path(models_dir)
    csv_path = Path(csv_dir)

    # Сдвиг индекса. Если в модели neuron_0 означает 1 в CSV, shift = 1
    # Если neuron_1 означает 1 в CSV, shift = 0
    INDEX_SHIFT = 1

    final_results = {}

    for mouse in mice_list:
        print(f"\n{'=' * 60}")
        print(f"🐁 МЫШЬ {mouse}")
        print(f"{'=' * 60}")

        # --- ШАГ 1: Загрузка карты сопоставлений (CSV) ---
        tracking_file = csv_path / f"LNOF_{mouse}.csv"  # Подстраиваем имя: если J30, ищем LNOF_J30.csv
        # if not tracking_file.exists():
        #     # На случай, если файл называется ровно как мышь
        #     tracking_file = csv_path / f"LNOF_{mouse}.csv"

        # if not tracking_file.exists():
        #     print(f"  [ОШИБКА] CSV файл трекинга не найден для {mouse}. Пропускаем.")
        #     continue

        print(f"  [OK] Загружена карта сопоставления нейронов: {tracking_file.name}")
        df_tracking = pd.read_csv(tracking_file)

        # --- ШАГ 2: Извлечение значимых нейронов для каждого дня ---
        # Словарь, где ключ - день (1-4), значение - множество глобальных индексов (номеров строк в CSV)
        significant_global_cells = {}

        for day in range(1, 5):
            model_file = models_path / f"{mouse}_{day}D.pkl"
            if not model_file.exists():
                print(f"  [ПРЕДУПРЕЖДЕНИЕ] Модель {model_file.name} не найдена. День {day} будет пуст.")
                significant_global_cells[day] = set()
                continue

            print(f"\n  > Обработка дня {day} ({model_file.name})...")

            # Загружаем модель
            data = joblib.load(model_file)
            model = data['model']
            feature_names = data['feature_names']

            # Считаем важность и фильтруем по порогу
            importance = np.linalg.norm(model.coef_, axis=0)
            active_indices = np.where(importance >= threshold)[0]

            # Достаем локальные ID (числа из названий neuron_X)
            local_sig_ids = set()
            for idx in active_indices:
                match = re.search(r'\d+', feature_names[idx])
                if match:
                    # Применяем сдвиг, чтобы совпало с нумерацией в CSV (с 1)
                    local_id = int(match.group()) + INDEX_SHIFT
                    local_sig_ids.add(local_id)

            # Переводим локальные ID в глобальные ID (номера строк CSV)
            # Считаем, что колонки идут по порядку: день 1 (колонка 0), день 2 (колонка 1) и т.д.
            col_idx = day - 1
            col_name = df_tracking.columns[col_idx]

            # Ищем строки, где значение для текущего дня совпадает с нашими значимыми нейронами
            global_rows = set(df_tracking[df_tracking[col_name].isin(local_sig_ids)].index)
            significant_global_cells[day] = global_rows

            print(f"    - Найдено локальных значимых нейронов (>={threshold}): {len(local_sig_ids)}")
            print(f"    - Успешно спроецировано на глобальную карту: {len(global_rows)} клеток")

        # --- ШАГ 3: Обход полносвязного графа (сравнение дней) ---
        edges = list(combinations([1, 2, 3, 4], 2))  # [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
        print(f"\n  [OK] Запуск сравнения по ребрам графа ({len(edges)} связей)...")

        mouse_results = {}
        overlap_fractions = []

        for dayA, dayB in edges:
            setA = significant_global_cells.get(dayA, set())
            setB = significant_global_cells.get(dayB, set())

            # Пересечение: клетки, которые значимы в ОБА дня
            intersection = len(setA & setB)
            # Объединение: клетки, которые значимы ХОТЯ БЫ В ОДИН из дней
            union = len(setA | setB)

            # Считаем долю совпадения
            fraction = intersection / union if union > 0 else 0.0

            edge_name = f"{dayA}-{dayB}"
            mouse_results[edge_name] = fraction
            overlap_fractions.append(fraction)

            print(
                f"    > Ребро {edge_name}: Совпадений = {intersection} | Объединение = {union} | Доля = {fraction:.3f}")

        # --- ШАГ 4: Сборка итогов ---
        mouse_results['general'] = np.mean(overlap_fractions) if overlap_fractions else 0.0
        print(f"\n  [ИТОГ] Средняя стабильность (general) для {mouse}: {mouse_results['general']:.3f}")

        final_results[mouse] = mouse_results

    return final_results


def export_full_ablation_stats(filenames, models_dir='models_LNOF_Lasso', data_dir='LNOF_SynchronizedData',
                               output_csv='plots/full_ablation_stats.csv',
                               test_size=0.25, c_param=0.5, max_iter=100):
    """
    Выполняет Retraining Ablation для ВСЕХ нейронов сессии.
    Сохраняет результаты (baseline, mean, std и R2 без каждого отдельного нейрона) в CSV.
    """
    start_time = time.time()

    # Убедимся, что папка для сохранения существует
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []

    for filename in filenames:
        clean_name = filename.replace('.pkl', '')
        model_name = clean_name

        file_path = Path(data_dir) / f"LNOF_{clean_name}_aligned.npz"
        model_path = Path(models_dir) / filename

        if not file_path.exists():
            print(f"[SKIP] Файл данных не найден: {file_path}")
            continue
        if not model_path.exists():
            print(f"[SKIP] Файл модели не найден: {model_path}")
            continue

        print(f"\n{'=' * 60}")
        print(f"[{model_name}] Загрузка данных и бейзлайна...")

        # 1. Загружаем данные
        features, labels = load_data(file_path)
        if features is None or labels is None:
            continue

        X_train, X_test, y_train, y_test = train_test_split_temporal(features, labels, test_size=test_size)

        # 2. Загружаем бейзлайн
        data = joblib.load(model_path)
        # Извлекаем R2 так же, как в предыдущей отлаженной функции
        base_r2 = data.get('metrics', {}).get('test_r2', np.nan)

        total_neurons = len(features.columns)
        print(f"[{model_name}] Всего нейронов в сессии: {total_neurons}. Запуск полного Ablation...")

        # Инициализируем словарь для текущей строки DataFrame
        row_data = {
            'model_session': model_name,
            'baseline_r2': base_r2
        }

        session_r2_scores = []

        # 3. Полный цикл Retraining Ablation по всем нейронам
        for i, feat in enumerate(features.columns):
            # Прогресс-бар для спокойствия (каждые 50 нейронов)
            if (i + 1) % 50 == 0 or (i + 1) == total_neurons:
                print(f"  > Обработано {i + 1}/{total_neurons} нейронов...")

            # Вырезаем колонку
            X_train_ablated = X_train.drop(columns=[feat])
            X_test_ablated = X_test.drop(columns=[feat])

            # Переобучаем Lasso
            _, _, _, metrics_ablated = train_lasso(
                X_train_ablated, y_train, X_test_ablated, y_test,
                C=c_param, max_iter=max_iter
            )

            abl_r2 = metrics_ablated.get('test_r2', np.nan)

            session_r2_scores.append(abl_r2)
            row_data[f'ablation_{feat}'] = abl_r2

        # 4. Считаем параметры распределения
        row_data['mean_r2'] = np.mean(session_r2_scores)
        row_data['std_r2'] = np.std(session_r2_scores)

        print(f"[{model_name}] Завершено. Mean: {row_data['mean_r2']:.4f} | Std: {row_data['std_r2']:.4f}")
        all_results.append(row_data)

    if not all_results:
        print("[WARNING] Нет данных для сохранения.")
        return

    # 5. Сборка и форматирование DataFrame
    df = pd.DataFrame(all_results)

    # Переупорядочиваем столбцы, чтобы главные метрики были слева
    # Сначала фиксированные колонки, потом все остальные (динамические ablation_neuron_X)
    primary_cols = ['model_session', 'baseline_r2', 'mean_r2', 'std_r2']
    ablation_cols = [col for col in df.columns if col.startswith('ablation_')]

    # Сортируем ablation колонки для красоты (neuron_0, neuron_1, ...)
    # Используем кастомную сортировку, чтобы neuron_2 был перед neuron_10
    import re
    ablation_cols = sorted(ablation_cols,
                           key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)

    final_cols = primary_cols + ablation_cols
    df = df[final_cols]

    # 6. Сохранение
    df.to_csv(output_path, index=False)

    # 7. Замер времени
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    mins, secs = divmod(elapsed_seconds, 60)

    print(f"\n{'=' * 60}")
    print(f"✓ Полный Ablation анализ успешно завершен!")
    print(f"✓ Результаты сохранены в: {output_path}")
    print(f"⏱ Общее время выполнения: {int(mins)} мин {int(secs)} сек")
    print(f"{'=' * 60}")

    return df


def find_negative_ablation_r2(csv_path='stat_tables/full_ablation_stats.csv'):
    """
    Проверяет таблицу ablation и выводит сессии, в которых
    удаление нейрона привело к отрицательному R2.
    """
    path = Path(csv_path)
    if not path.exists():
        print(f"[ERROR] Файл {path} не найден.")
        return

    df = pd.read_csv(path)

    # Находим все колонки, относящиеся к абляции нейронов
    ablation_cols = [col for col in df.columns if col.startswith('ablation_')]

    if not ablation_cols:
        print("[ERROR] Не найдены колонки, начинающиеся с 'ablation_'")
        return

    print(f"\n{'=' * 75}")
    print("Поиск критических якорей (R2 < 0 при удалении нейрона)")
    print(f"{'=' * 75}")

    total_sessions_with_negatives = 0
    total_negative_neurons = 0

    for index, row in df.iterrows():
        model_session = row['model_session']

        # Извлекаем значения только для колонок абляции
        ablation_values = row[ablation_cols]

        # Ищем строго отрицательные значения, игнорируя NaN
        negative_mask = (ablation_values < 0) & (ablation_values.notna())

        if negative_mask.any():
            total_sessions_with_negatives += 1
            negative_cols = ablation_values[negative_mask].index.tolist()

            # Очищаем названия (убираем префикс)
            neuron_names = [col.replace('ablation_', '') for col in negative_cols]
            count = len(neuron_names)
            total_negative_neurons += count

            print(f"\n[{model_session}] Найдено нейронов: {count}")

            # Выводим имя нейрона и точное значение R2 при его удалении
            for col, n_name in zip(negative_cols, neuron_names):
                r2_val = row[col]
                print(f"  > {n_name:<15} | R2 = {r2_val:>.4f}")

    print(f"\n{'=' * 75}")
    if total_sessions_with_negatives > 0:
        print(f"Итог: В {total_sessions_with_negatives} сессиях найдено {total_negative_neurons} критических нейронов.")
    else:
        print("✓ Ни в одной сессии R2 не упал ниже нуля при удалении отдельных нейронов.")
    print(f"{'=' * 75}\n")


def analyze_performance_wilcoxon(
        csv_path='stat_tables/top_vs_rest_metrics_abl.csv',
        col1='r2_full',  # Колонка Условия 1 (например, базовая модель)
        col2='top_r2',  # Колонка Условия 2 (например, модель только на топовых)
        label1='Full Model',
        label2='Top Neurons (>3$\sigma$)',
        metric_name='$R^2$',
        output_dir='plots'):
    """
    Выполняет парный критерий Уилкоксона для двух колонок и строит Tufte Slope Graph.
    Показывает, как изменяется метрика внутри КАЖДОЙ конкретной сессии.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Загрузка данных
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[ERROR] Не найден файл: {csv_path}")
        return

    # 2. Очистка от NaN (оставляем только строгие пары)
    if col1 not in df.columns or col2 not in df.columns:
        print(f"[ERROR] Колонки {col1} или {col2} не найдены в таблице.")
        return

    df_clean = df.dropna(subset=[col1, col2]).copy()

    if df_clean.empty or len(df_clean) < 5:
        print("[ERROR] Недостаточно парных данных для теста Уилкоксона (нужно хотя бы 5 пар).")
        return

    # 3. Статистика: Парный критерий Уилкоксона
    w_stat, pval = wilcoxon(df_clean[col1], df_clean[col2])

    def get_stars(p):
        return r"^{\ast\ast\ast}" if p < 0.001 else r"^{\ast\ast}" if p < 0.01 else r"^{\ast}" if p < 0.05 else "^{ns}"

    # Расчет медиан для отображения
    median1 = df_clean[col1].median()
    median2 = df_clean[col2].median()

    # ==========================================
    # ЭСТЕТИЧЕСКАЯ ОТРИСОВКА (Paired Slope Graph)
    # ==========================================
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    c_text = '#2C3E50'
    c_decrease = '#E74C3C'  # Красный (если метрика падает)
    c_increase = '#27AE60'  # Зеленый (если метрика растет)
    c_neutral = '#BDC3C7'  # Серый (если не изменилась)
    c_median = 'black'  # Цвет глобального тренда

    x_pos = [1, 2]

    # 1. Отрисовка индивидуальных пар (линии + точки)
    for idx, row in df_clean.iterrows():
        val1 = row[col1]
        val2 = row[col2]

        # Определяем цвет в зависимости от знака разности (d_i)
        if val2 < val1:
            line_color = c_decrease
        elif val2 > val1:
            line_color = c_increase
        else:
            line_color = c_neutral

        # Линия
        ax.plot(x_pos, [val1, val2], color=line_color, alpha=0.3, linewidth=2, zorder=1)
        # Точки
        ax.scatter(x_pos, [val1, val2], color=line_color, alpha=0.5, s=40, zorder=2)

    # 2. Отрисовка глобального тренда (Медианы)
    ax.plot(x_pos, [median1, median2], color=c_median, linewidth=4, linestyle='-', zorder=3)
    ax.scatter(x_pos, [median1, median2], color='white', edgecolor=c_median, linewidth=2.5, s=120, zorder=4)

    # 3. Аннотации статистики Уилкоксона
    stat_text = (f"Wilcoxon signed-rank test\n"
                 f"$W = {w_stat:.1f}$\n"
                 f"$p = {pval:.2e}{get_stars(pval)}$\n"
                 f"$N = {len(df_clean)}$ пар")

    ax.text(0.5, 0.95, stat_text, transform=ax.transAxes, ha='center', va='top',
            fontsize=13, color=c_text, weight='bold',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=5), zorder=5)

    # 4. Оформление осей
    ax.set_title(f"Парное сравнение метрики {metric_name}", fontsize=15, pad=20, fontweight='bold', color=c_text)
    ax.set_ylabel(metric_name, fontsize=14, labelpad=10, color=c_text)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([label1, label2], fontsize=13, fontweight='bold', color=c_text)

    # Настройка лимитов (добавляем отступы по краям для красоты)
    ax.set_xlim(0.5, 2.5)
    y_min, y_max = df_clean[[col1, col2]].min().min(), df_clean[[col1, col2]].max().max()
    padding = (y_max - y_min) * 0.1 if y_max != y_min else 0.1
    ax.set_ylim(y_min - padding, y_max + padding * 2)

    # Эстетика рамки
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)  # Нижняя рамка не нужна в Slope Graph
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['left'].set_color(c_text)

    ax.tick_params(axis='y', labelsize=12, width=1.5, colors=c_text)
    ax.tick_params(axis='x', length=0)  # Убираем засечки снизу
    ax.grid(axis='y', linestyle=':', alpha=0.4, color='#BDC3C7', zorder=0)

    plt.tight_layout()

    save_path = output_path / f'wilcoxon_paired_slope_{col1}_vs_{col2}.png'
    plt.savefig(save_path, dpi=1000, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n✓ Успешно! График парного сравнения Уилкоксона сохранен: {save_path}")
    print(f"  > Wilcoxon W: {w_stat:.1f}, p-value: {pval:.2e}")


def recalculate_text_metrics_to_95ci(stat_tables_dir='stat_tables'):
    """
    Пересчитывает все ключевые метрики из таблиц в формат:
    Среднее ± (1.96 * SE), что соответствует 95% доверительному интервалу.
    Улучшен алгоритм поиска колонок и добавлен точный парсинг tracking_metrics_detailed.csv
    """
    tables_path = Path(stat_tables_dir)
    if not tables_path.exists():
        print(f"[ERROR] Папка {stat_tables_dir} не найдена.")
        return

    def get_stat(series, multiplier=1.0, decimals=2):
        s = pd.to_numeric(series, errors='coerce').dropna() * multiplier
        if len(s) == 0:
            return "N/A ± N/A"
        mean_val = s.mean()
        margin = 1.96 * stats.sem(s)  # 1.96 * Standard Error
        return f"{mean_val:.{decimals}f} ± {margin:.{decimals}f}"

    print(f"\n{'=' * 75}")
    print("ПЕРЕСЧЕТ МЕТРИК ДЛЯ СТАТЬИ (Формат: Среднее ± 1.96*SE)")
    print(f"{'=' * 75}\n")

    # =========================================================================
    # РАЗДЕЛ 1: Точность декодирования
    # =========================================================================
    uni_csv = tables_path / 'unified_model_performance_report.csv'
    if uni_csv.exists():
        df_uni = pd.read_csv(uni_csv)

        # Исключение дефектных сессий
        excluded_mice = ['J20', 'J21']
        if 'mouse' in df_uni.columns:
            df_uni = df_uni[~df_uni['mouse'].isin(excluded_mice)].copy()
        elif 'model_name' in df_uni.columns:
            df_uni = df_uni[~df_uni['model_name'].str.startswith(tuple(excluded_mice))].copy()

        r2_col = 'r2' if 'r2' in df_uni.columns else 'test_r2'
        med_col = 'test_dist' if 'test_dist' in df_uni.columns else ('med' if 'med' in df_uni.columns else 'test_mae')

        print("--- РАЗДЕЛ: Точность декодирования (Абзац 1-2) ---")
        print(f"[Исходный текст]: средний R2 составил 0.59 ± 0.15 и MED – 10.12 см ± 1.46 см")
        print(f"[Новый текст]: средний R2 составил {get_stat(df_uni[r2_col])} и MED – {get_stat(df_uni[med_col])} см\n")

        # Фильтрация лучших сессий
        df_filt = df_uni[(df_uni[r2_col] >= 0.23) & (df_uni[med_col] <= 13.61)]
        print(f"[Исходный текст]: средняя точность моделей составит 0.61 ± 0.09 и средняя ошибка – 10.08 см ± 1.26 см")
        print(
            f"[Новый текст]: средняя точность моделей составит {get_stat(df_filt[r2_col])} и средняя ошибка – {get_stat(df_filt[med_col])} см\n")

    # =========================================================================
    # РАЗДЕЛ 2: Информативные vs Важные нейроны (Кенделл и F1)
    # =========================================================================
    real_csv = tables_path / 'LNOF_prec_rec_real_tau.csv'
    base_csv = tables_path / 'LNOF_prec_rec_base_tau.csv'

    if real_csv.exists() and base_csv.exists():
        df_real = pd.read_csv(real_csv)
        df_base = pd.read_csv(base_csv)

        print("--- РАЗДЕЛ: Информативные vs Важные нейроны (Кенделл и F1) ---")

        # Динамический поиск колонки Кенделла
        tau_cols = [c for c in df_real.columns if 'tau' in c.lower() or 'kendall' in c.lower()]
        if tau_cols:
            print(f"[Исходный текст]: По Кенделлу коэффициент корреляции составил ... 0.33 ± 0.05")
            print(
                f"[Новый текст]: По Кенделлу коэффициент корреляции составил в среднем {get_stat(df_real[tau_cols[0]])}\n")

        if 'F1' in df_real.columns and 'F1' in df_base.columns:
            print(f"[Исходный текст]: Метрики F1 составили в среднем 0.55 ± 0.06")
            print(f"[Новый текст]: Метрики F1 составили в среднем {get_stat(df_real['F1'])}\n")

            print(f"[Исходный текст]: на случайно выбранных нейронах ... метрика F1 была равна 0.38 ± 0.09")
            print(
                f"[Новый текст]: на случайно выбранных нейронах ... метрика F1 была равна {get_stat(df_base['F1'])}\n")

            diff = df_real['F1'].mean() - df_base['F1'].mean()
            print(f"[Доп. инфо]: Средняя разница между выборками значений F1 составила {diff:.2f}\n")

    # =========================================================================
    # РАЗДЕЛ 3: Минимально достаточный ансамбль (Абляция)
    # =========================================================================
    mse_csv = tables_path / 'minimal_sufficient_ensembles.csv'
    if mse_csv.exists():
        df_mse = pd.read_csv(mse_csv)
        print("--- РАЗДЕЛ: Минимально достаточный ансамбль (Абляция) ---")
        frac_cols = [c for c in df_mse.columns if 'fraction' in c.lower() or 'frac' in c.lower()]
        if frac_cols:
            print(f"[Исходный текст]: требуется в среднем 25% ± 9% от общего количества нейронов")
            print(
                f"[Новый текст]: требуется в среднем {get_stat(df_mse[frac_cols[0]], multiplier=100, decimals=1)}% от общего количества нейронов\n")

    # =========================================================================
    # РАЗДЕЛ 4: Репрезентационный дрейф (Жаккард)
    # =========================================================================
    track_csv = tables_path / 'tracking_metrics_detailed.csv'
    print("--- РАЗДЕЛ: Репрезентационный дрейф (Жаккард) ---")

    if track_csv.exists():
        df_t = pd.read_csv(track_csv)

        # Умный поиск колонок по ключевым подстрокам (чтобы избежать KeyError)
        abs_all_col = next((c for c in df_t.columns if 'abs' in c.lower() and 'all' in c.lower()), None)
        abs_imp_col = next(
            (c for c in df_t.columns if 'abs' in c.lower() and ('imp' in c.lower() or 'lin' in c.lower())), None)

        jac_all_col = next((c for c in df_t.columns if 'jac' in c.lower() and 'all' in c.lower()), None)
        jac_imp_col = next(
            (c for c in df_t.columns if 'jac' in c.lower() and ('imp' in c.lower() or 'lin' in c.lower())), None)

        norm_jac_imp = next(
            (c for c in df_t.columns if 'norm' in c.lower() and ('imp' in c.lower() or 'lin' in c.lower())), None)

        # Вывод абсолютных переходов
        print(f"[Исходный текст]: переходило 348 ± 189 нейронов, из которых 62 ± 19 были важными")
        if abs_all_col and abs_imp_col:
            print(
                f"[Новый текст]: переходило {get_stat(df_t[abs_all_col], decimals=0)} нейронов, из которых {get_stat(df_t[abs_imp_col], decimals=0)} были важными\n")
        else:
            print("[!]: Колонки абсолютных количеств (abs_all, abs_imp) не найдены в таблице.\n")

        # Вывод чистых Жаккардов
        print(f"[Исходный текст]: что соответствует коэффициентам Жаккара 0.57 ± 0.09 и 0.24 ± 0.07")
        if jac_all_col and jac_imp_col:
            print(
                f"[Новый текст]: что соответствует коэффициентам Жаккара {get_stat(df_t[jac_all_col])} и {get_stat(df_t[jac_imp_col])}\n")
        else:
            print("[!]: Колонки Жаккарда (jac_all, jac_imp) не найдены.\n")

        # Вывод нормированного Жаккарда
        print(f"[Исходный текст]: нормированный коэффициент Жаккара для важных нейронов составил в среднем 0.43 ± 0.15")
        if norm_jac_imp:
            print(
                f"[Новый текст]: нормированный коэффициент Жаккара для важных нейронов составил в среднем {get_stat(df_t[norm_jac_imp])}\n")
        else:
            print("[!]: Колонка нормированного Жаккарда (norm_...) не найдена.\n")

    else:
        print(f"[ERROR] Файл {track_csv.name} не найден в папке {stat_tables_dir}.")

    print(f"{'=' * 75}")
    print("✓ Анализ завершен. Скопируй [Новый текст] в документ.")


def calculate_session_jaccard_stats(
        ablation_csv='stat_tables/full_ablation_stats.csv',
        database_csv='LNOF_SynchronizedData/LNOF_database.csv',
        models_dir='models_LNOF_Lasso'):
    """
    Рассчитывает средний коэффициент Жаккара и 1.96*SE по СЕССИЯМ.
    Сравнивает реальные пулы со случайным бейзлайном.
    Пересечения:
      1. Важные (LR-weights > 0) & Информативные (MI)
      2. Сильные (z-crit > 0) & Информативные (MI)
    """
    excluded_mice = ['J20', 'J21']

    try:
        df_abl = pd.read_csv(ablation_csv)
        df_db = pd.read_csv(database_csv)
    except FileNotFoundError as e:
        print(f"[ERROR] Не найден файл: {e}")
        return

    sessions_to_process = [s for s in df_abl['model_session'].unique() if s[:3] not in excluded_mice]

    print(f"\n{'=' * 85}")
    print(f"Расчет Жаккара и 1.96*SE по сессиям (Всего сессий: {len(sessions_to_process)})")
    print(f"{'=' * 85}")

    # Списки для хранения значений Жаккара каждой сессии
    real_jac_linreg_list = []
    real_jac_zcrit_list = []
    base_jac_linreg_list = []
    base_jac_zcrit_list = []

    # Вспомогательная функция Жаккара
    def calc_jaccard(set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    for sess in sessions_to_process:
        mouse = sess[:3]
        day = sess[4:6]

        model_path = Path(models_dir) / f"{sess}.pkl"
        if not model_path.exists():
            continue

        data = joblib.load(model_path)
        model = data.get('model')
        if not hasattr(model, 'coef_'):
            continue

        # Пул всех доступных нейронов в этой сессии
        feats = np.array(data.get('feature_names'))
        pool_size = len(feats)

        # --- 1. Извлечение реальных пулов для ТЕКУЩЕЙ сессии ---

        # Информативные (MI)
        df_int = df_db[(df_db['mouse'] == mouse) & (df_db['session'] == day) & (df_db['feature'] == 'place')]
        intense_set = set(
            [f"neuron_{int(idx)}" for idx in df_int['neuron_idx'].unique() if f"neuron_{int(idx)}" in feats])

        # Важные (LR-weights > 0)
        weights = np.linalg.norm(model.coef_, axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        linreg_set = set(feats[weights > 0])

        # Сильные (z-crit > 0)
        zcrit_set = set()
        abl_rows = df_abl[df_abl['model_session'] == sess]
        if not abl_rows.empty:
            abl_row = abl_rows.iloc[0]
            base_r2 = abl_row['baseline_r2']
            abl_r2_vals = np.array([abl_row.get(f'ablation_{f}', np.nan) for f in feats])

            crit = base_r2 - abl_r2_vals
            valid_crit = crit[~np.isnan(crit)]

            if len(valid_crit) > 0:
                std_c = np.std(valid_crit)
                mean_c = np.mean(valid_crit)
                if std_c > 0:
                    zcrit = (crit - mean_c) / std_c
                    zcrit_set = set(feats[~np.isnan(zcrit) & (zcrit > 0)])

        # Расчет реального Жаккара для текущей сессии
        real_jac_linreg_list.append(calc_jaccard(linreg_set, intense_set))
        real_jac_zcrit_list.append(calc_jaccard(zcrit_set, intense_set))

        # --- 2. Генерация случайного бейзлайна для ТЕКУЩЕЙ сессии ---
        # Выбираем случайные нейроны в том же количестве, что и реальные
        rand_linreg = set(np.random.choice(feats, size=min(len(linreg_set), pool_size), replace=False))
        rand_zcrit = set(np.random.choice(feats, size=min(len(zcrit_set), pool_size), replace=False))

        # Расчет базового Жаккара для текущей сессии
        base_jac_linreg_list.append(calc_jaccard(rand_linreg, intense_set))
        base_jac_zcrit_list.append(calc_jaccard(rand_zcrit, intense_set))

    if not real_jac_linreg_list:
        print("[ERROR] Не удалось собрать данные. Проверьте файлы.")
        return

    # --- 3. Расчет статистик (Среднее и 1.96*SE по выборке сессий) ---
    def get_stats(arr):
        arr = np.array(arr)
        return np.mean(arr), 1.96 * sem(arr)

    r_lin_mean, r_lin_ci = get_stats(real_jac_linreg_list)
    b_lin_mean, b_lin_ci = get_stats(base_jac_linreg_list)

    r_zcr_mean, r_zcr_ci = get_stats(real_jac_zcrit_list)
    b_zcr_mean, b_zcr_ci = get_stats(base_jac_zcrit_list)

    # --- 4. Красивый вывод результатов ---
    print("\nРЕЗУЛЬТАТЫ ПЕРЕСЕЧЕНИЯ С ИНФОРМАТИВНЫМИ (MI) НЕЙРОНАМИ")
    print(f"{'-' * 85}")

    print("1. Важные нейроны (LR-weights > 0) ∩ Информативные:")
    print(f"   Реальный Жаккард:   {r_lin_mean:.4f} ± {r_lin_ci:.4f}")
    print(f"   Случайный бейзлайн: {b_lin_mean:.4f} ± {b_lin_ci:.4f}")

    diff_linreg = r_lin_mean - b_lin_mean
    print(f"   Среднее превосходство: +{diff_linreg:.4f}\n")

    print("2. Сильные нейроны (z-crit > 0) ∩ Информативные:")
    print(f"   Реальный Жаккард:   {r_zcr_mean:.4f} ± {r_zcr_ci:.4f}")
    print(f"   Случайный бейзлайн: {b_zcr_mean:.4f} ± {b_zcr_ci:.4f}")

    diff_zcrit = r_zcr_mean - b_zcr_mean
    print(f"   Среднее превосходство: +{diff_zcrit:.4f}")
    print(f"{'=' * 85}\n")


if __name__ == "__main__":

    # recalculate_text_metrics_to_95ci()
    calculate_session_jaccard_stats()

    # find_negative_ablation_r2()
    # analyze_performance_wilcoxon()

    # models_dir = "models_LNOF_Lasso"
    # models_path = Path(models_dir)
    # filtered_files = [f.name for f in models_path.glob("*.pkl")]
    # print(filtered_files)
    # export_full_ablation_stats(filtered_files)

    # mice_to_analyze = ['J24', 'J54', 'J18', 'J59', 'J52']
    # threshold_value = 0.5
    #
    # results_dict = analyze_neuron_tracking_graph(
    #     mice_list=mice_to_analyze,
    #     threshold=threshold_value,
    #     models_dir='models_LNOF_Lasso',
    #     csv_dir='MatchData'
    # )
    #
    # print(results_dict)

    # random.seed(42)
    # csv_path = 'LNOF_SynchronizedData/LNOF_database.csv'
    # df_place = csv_match(csv_path, 'place')
    #
    # metrics = ['mouse', 'session', 'precision', 'recall', 'F1', 'TN', 'FP', 'FN', 'TP', 'linreg_val', 'intense_val', 'overall']
    # precall_df = pd.DataFrame(columns=metrics)
    #
    # for model in filtered_files:
    #     name = model.name
    #     # name = 'J61_4D.pkl'
    #     mouse = name[:3]
    #     day = name[4:6]
    #
    #     file_path = str(model)
    #     # file_path = 'models_LNOF_Lasso/J61_4D.pkl'
    #
    #     df_intense = df_place[(df_place['mouse'] == mouse) & (df_place['session'] == day)]
    #     count = len(df_intense)
    #
    #     df_linreg = save_neuron_weights(
    #         model_path=file_path,
    #         top_n=count
    #     )
    #     linreg_count = len(df_linreg)
    #
    #     data = joblib.load(file_path)
    #     all_features = range(len(data.get('feature_names')))
    #
    #     # neurons_linreg = df_linreg.index.tolist()
    #     neurons_linreg = random.sample(all_features, k=linreg_count)
    #     linreg_val = len(neurons_linreg)
    #     print(np.sort(neurons_linreg), len(neurons_linreg))
    #     neurons_intense = df_intense['neuron_idx'].values
    #     intense_val = len(neurons_intense)
    #     print(neurons_intense, len(neurons_intense))
    #
    #     all_features = range(len(data.get('feature_names')))
    #     overall = len(all_features)
    #     print(all_features, type(all_features[0]))
    #
    #     # Создаем массивы нулей и единиц (1 - нейрон есть в множестве, 0 - нет)
    #     y_true = [1 if f in neurons_intense else 0 for f in all_features]
    #     print(y_true, len(y_true))
    #     y_pred = [1 if f in neurons_linreg else 0 for f in all_features]
    #     print(y_pred, len(y_pred))
    #
    #     prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, zero_division=np.nan, average='binary')
    #     # try:
    #     tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist()
    #     # except ValueError:
    #     #     tn, fp, fn, tp = (confusion_matrix(y_true, y_pred).ravel().tolist() for _ in range(4))
    #
    #     new_row = pd.DataFrame([[mouse, day, prec, rec,
    #                             f1, tn, fp, fn, tp,
    #                             linreg_val, intense_val, overall]],
    #                             columns=precall_df.columns)
    #     precall_df = pd.concat([precall_df, new_row], ignore_index=True)
    #
    # precall_df.to_csv('LNOF_SynchronizedData/LNOF_prec_rec_baseline.csv', index=False)
    # print('saved')

    # seeds = range(5)
    # models = print_top_models(n=5)
    # best_models = ['LNOF_' + model[0][:6] + '_aligned.npz' for model in models]
    #
    # plot_seed_robustness(best_models, seeds=seeds, c_param=0.5, test_size=0.25, max_iter=100)

    # plot_ablation_r2(models)
    # plot_cumulative_ablation(models)

    # df_linreg = save_neuron_weights(
    #         model_path='models_LNOF_Lasso/J01_1D.pkl',
    #         top_n=262
    #     )
    #
    # print(len(df_linreg))

    # models = ['J24_3D.pkl', 'J54_4D.pkl', 'J18_2D.pkl']

    # Examples of usage:
    # get_model_results(mouse_id='H39') # All sessions and objects for H39
    # get_model_results(mouse_id='J30', session='4D', obj_name='object4')
    # get_model_results(mouse_id='J06', session='4D', obj_name='object4')
    # get_model_results(mouse_id='J30', session='3D', obj_name='object4')
    # get_model_results(csv=True) # Everything saved to CSV
    #
    # print_feature_importance(mouse_id='J01', session='2D', importance_threshold=0.5)
    # print_feature_importance(mouse_id='J54', session='4D', importance_threshold=0.5)
    # print_feature_importance(mouse_id='J18', session='2D', importance_threshold=0.5)
    #
    # plot_importance_distribution(mouse_id='J24', session='3D')
    # plot_importance_distribution(mouse_id='J54', session='4D')
    # plot_importance_distribution(mouse_id='J18', session='2D')

    # plot_metrics_distribution()
    # print_top_models(n=5)

    # best_cfg = perform_lasso_grid_search('J24', '3D')
    # best_cfg = perform_lasso_grid_search('J54', '4D')
    # best_cfg = perform_lasso_grid_search('J18', '2D')
    # print(best_cfg)

    # find_optimal_lasso_params('J06', '4D', 'object4')
    #
    # plot_metrics_boxplots(models_dir='models_LNOF_Lasso', output_dir='plots')
    # plot_metrics_bars_by_day(models_dir='models_LNOF_Lasso', output_dir='plots')
    # plot_mobility_frequency()
    # plot_metrics_frequency_distribution()
    # df_metrics = plot_mobility_frequency()
    # df_mob = get_models_dataframe()
    #
    # df_synth = pd.concat([df_mob.drop(columns=['filename', 'mouse', 'day']), df_metrics['percent_moving']], axis=1)
    # df_synth.sort_values(by='r2', ascending=True)
    #
    # print(df_synth[df_synth['r2'] < 0.4])
    # print(df_synth[df_synth['mae'] > 12])
    # print(f"\nr2: {df_synth['r2'].mean():.2f}\nMED: {df_synth['mae'].mean():.2f}\nmoving%: {df_synth['percent_moving'].mean():.2f}")
    # print(df_mob)

    # plot_chronological_tracking(results_dict)