import os
import re
import joblib
import time
import random
import torch
from scipy import stats
from scipy.stats import kendalltau, spearmanr, wilcoxon
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
from linreg_behavior import load_data, train_lasso, train_test_split_temporal


C_PARAM = 0.5
TEST_SIZE = 0.25
MAX_ITER = 100


def export_tracking_graph_to_csv(mice_list,
                                 models_dir='models_LNOF_Lasso',
                                 csv_dir='MatchData',
                                 output_csv='tracking_metrics_detailed.csv',
                                 threshold=0,
                                 model_is_zero_indexed=True):
    """
    Рассчитывает метрики трекинга для графа сессий и сохраняет их в CSV таблицу.
    Столбцы сгруппированы по метрикам, внутри каждой метрики первые 3 столбца - хронологические.
    """
    # Жестко задаем порядок ребер (сначала хронология, потом прыжки через дни)
    edges = [(1, 2), (2, 3), (3, 4), (1, 3), (1, 4), (2, 4)]
    edge_names = [f"{a}D-{b}D" for a, b in edges]

    # Порядок метрик из твоего запроса
    prefixes = [
        'norm_jac',  # Нормированный Жаккард
        'sig_jac',  # Ненормированный Жаккард (значимые)
        'base_jac',  # Общий Жаккард (все нейроны)
        'sig_match',  # Абс. совпадения значимых
        'base_match'  # Абс. совпадения общие
    ]

    # Динамически собираем идеальный порядок столбцов
    target_columns = ['mouse']
    for prefix in prefixes:
        for edge in edge_names:
            target_columns.append(f"{prefix}_{edge}")

    all_rows = []

    for mouse in mice_list:
        print(f"[{mouse}] Обработка мыши...")

        # Инициализируем словарь NaN'ами, чтобы структура не поплыла, если нет данных
        row_data = {col: np.nan for col in target_columns}
        row_data['mouse'] = mouse

        csv_path = Path(csv_dir) / f"LNOF_{mouse}.csv"
        if not csv_path.exists():
            print(f"  [SKIP] Файл трекинга не найден: {csv_path}")
            all_rows.append(row_data)
            continue

        df_map = pd.read_csv(csv_path)
        day_cols = df_map.columns[:4]

        # 1. Извлекаем значимые нейроны
        sig_neurons_by_day = {}
        for day_num in [1, 2, 3, 4]:
            model_name = f"{mouse}_{day_num}D.pkl"
            model_path = Path(models_dir) / model_name

            if not model_path.exists():
                continue

            data = joblib.load(model_path)
            model = data['model']

            importance = np.linalg.norm(model.coef_, axis=0)
            active_mask = (importance >= threshold) & (importance != 0)

            sig_ids = []
            for name in np.array(data['feature_names'])[active_mask]:
                match = re.search(r'\d+', name)
                if match:
                    n_id = int(match.group())
                    if model_is_zero_indexed:
                        n_id += 1
                    sig_ids.append(n_id)

            sig_neurons_by_day[day_num] = set(sig_ids)

        # 2. Обходим ребра и считаем метрики
        for day_A, day_B in edges:
            edge_name = f"{day_A}D-{day_B}D"

            if (day_A - 1) >= len(day_cols) or (day_B - 1) >= len(day_cols):
                continue

            col_A = day_cols[day_A - 1]
            col_B = day_cols[day_B - 1]

            # Базовые метрики (Фон)
            base_intersection = ((df_map[col_A] != 0) & (df_map[col_B] != 0)).sum()
            base_union = ((df_map[col_A] != 0) | (df_map[col_B] != 0)).sum()
            base_jaccard = base_intersection / base_union if base_union > 0 else 0

            # Метрики значимых нейронов (Топ)
            if day_A in sig_neurons_by_day and day_B in sig_neurons_by_day:
                sig_A = sig_neurons_by_day[day_A]
                sig_B = sig_neurons_by_day[day_B]

                mask_A = df_map[col_A].isin(sig_A)
                mask_B = df_map[col_B].isin(sig_B)

                sig_intersection = (mask_A & mask_B).sum()
                sig_union = (mask_A | mask_B).sum()
                sig_jaccard = sig_intersection / sig_union if sig_union > 0 else 0
            else:
                sig_intersection = np.nan
                sig_jaccard = np.nan

            # Нормировка
            if not np.isnan(sig_jaccard) and base_jaccard > 0:
                normalized_jaccard = sig_jaccard / base_jaccard
            else:
                normalized_jaccard = np.nan

            # Записываем в строку
            row_data[f'norm_jac_{edge_name}'] = normalized_jaccard
            row_data[f'sig_jac_{edge_name}'] = sig_jaccard
            row_data[f'base_jac_{edge_name}'] = base_jaccard
            row_data[f'sig_match_{edge_name}'] = sig_intersection
            row_data[f'base_match_{edge_name}'] = base_intersection

        all_rows.append(row_data)

    # 3. Собираем DataFrame и сохраняем
    df = pd.DataFrame(all_rows)

    # Принудительно упорядочиваем колонки (защита от сдвигов)
    df = df[target_columns]

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\n✓ Таблица метрик графа сохранена: {out_path}")
    print(f"  Форма таблицы: {df.shape[0]} строк, {df.shape[1]} столбцов")

    return df

def analyze_neuron_tracking_graph(mice_list,
                                  models_dir='models_LNOF_Lasso',
                                  csv_dir='LNOF_SynchronizedData',
                                  threshold=0.1,
                                  model_is_zero_indexed=True):
    """
    Строит полносвязный граф по дням сессий, извлекает значимые нейроны
    и рассчитывает нормированную меру Жаккара для их трекинга.
    """
    all_results = {}

    for mouse in mice_list:
        print(f"\n{'=' * 60}")
        print(f"[START] Анализ стабильности трекинга для мыши: {mouse}")
        print(f"{'=' * 60}")

        csv_path = Path(csv_dir) / f"LNOF_{mouse}.csv"
        if not csv_path.exists():
            print(f"[ERROR] CSV файл трекинга не найден: {csv_path}. Пропускаем.")
            continue

        # 1. Загрузка CSV (Предполагаем, что первые 4 столбца — это дни 1, 2, 3, 4)
        df_map = pd.read_csv(csv_path)
        day_cols = df_map.columns[:4]
        print(f"[INFO] Таблица трекинга загружена. Столбцы дней: {list(day_cols)}")

        # 2. Извлечение значимых нейронов из моделей
        sig_neurons_by_day = {}
        for day_num in [1, 2, 3, 4]:
            model_name = f"{mouse}_{day_num}D.pkl"
            model_path = Path(models_dir) / model_name

            if not model_path.exists():
                print(f"[WARNING] Модель {model_name} не найдена. День {day_num} выпадает из графа.")
                continue

            data = joblib.load(model_path)
            model = data['model']
            feature_names = data['feature_names']

            # Считаем важность (L2-норма) и фильтруем по порогу
            importance = np.linalg.norm(model.coef_, axis=0)
            active_mask = importance >= threshold

            sig_ids = []
            for name in np.array(feature_names)[active_mask]:
                match = re.search(r'\d+', name)
                if match:
                    n_id = int(match.group())
                    # Сдвигаем индекс из модели (0-N) к формату CSV (1-N)
                    if model_is_zero_indexed:
                        n_id += 1
                    sig_ids.append(n_id)

            sig_neurons_by_day[day_num] = set(sig_ids)
            print(f"  > День {day_num}D: Найдено {len(sig_ids)} значимых нейронов (порог {threshold}).")

        # 3. Обход полносвязного графа сессий (комбинации по 2)
        mouse_metrics = {}
        normalized_scores = []

        # Функция combinations([1,2,3,4], 2) выдаст: (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
        for day_A, day_B in combinations([1, 2, 3, 4], 2):
            if day_A not in sig_neurons_by_day or day_B not in sig_neurons_by_day:
                continue

            edge_name = f"{day_A}D-{day_B}D"
            print(f"\n  --- Анализ ребра графа: {edge_name} ---")

            col_A = day_cols[day_A - 1]
            col_B = day_cols[day_B - 1]

            # --- Шаг А: Базовая мера Жаккара (популяция в целом) ---
            # Пересечение: нейрон найден (>0) и в дне А, и в дне Б
            base_intersection = ((df_map[col_A] != 0) & (df_map[col_B] != 0)).sum()
            # Объединение: нейрон найден хотя бы в одном из дней
            base_union = ((df_map[col_A] != 0) | (df_map[col_B] != 0)).sum()
            base_jaccard = base_intersection / base_union if base_union > 0 else 0

            # --- Шаг Б: Мера Жаккара для значимых нейронов ---
            sig_A = sig_neurons_by_day[day_A]
            sig_B = sig_neurons_by_day[day_B]

            # Маски: ID ячейки должен принадлежать множеству значимых нейронов этого дня
            mask_A = df_map[col_A].isin(sig_A)
            mask_B = df_map[col_B].isin(sig_B)

            # Совпадение значимого нейрона (он значим и в А, и в Б)
            sig_intersection = (mask_A & mask_B).sum()
            sig_union = (mask_A | mask_B).sum()
            sig_jaccard = sig_intersection / sig_union if sig_union > 0 else 0

            # --- Шаг В: Нормировка ---
            normalized_jaccard = sig_jaccard / base_jaccard if base_jaccard > 0 else 0
            normalized_scores.append(normalized_jaccard)

            mouse_metrics[edge_name] = {
                'base_jaccard': base_jaccard,
                'sig_jaccard': sig_jaccard,
                'normalized_score': normalized_jaccard,
                'matched_sig_count': int(sig_intersection)
            }

            print(f"    Базовый Jaccard (фон):      {base_jaccard:.4f}")
            print(f"    Значимый Jaccard (топ):     {sig_jaccard:.4f} (Абс. совпадений: {sig_intersection})")
            print(f"    Нормированная стабильность: {normalized_jaccard:.4f}")

        # 4. Итоговая агрегация (General)
        general_score = np.mean(normalized_scores) if normalized_scores else 0
        mouse_metrics['general'] = general_score

        print(f"\n[ИТОГ {mouse}] Средняя нормированная стабильность (General): {general_score:.4f}")

        # Интерпретация результата в лог
        if general_score > 1.1:
            print(f"[*] Вывод: У мыши {mouse} 'работающие' нейроны значительно стабильнее фоновых клеток.")
        elif general_score < 0.9:
            print(f"[*] Вывод: У мыши {mouse} наблюдается сильный representational drift (модель меняет якоря).")

        all_results[mouse] = mouse_metrics

    return all_results


def evaluate_unified_metrics(model_filename, real_metrics=True,
                             csv_path='LNOF_SynchronizedData/LNOF_database.csv',
                             models_dir='models_LNOF_Lasso'):
    """
    Унифицированная функция для подсчета PRF1 и Kendall Tau.
    Ранжирование: intense по 'me', linreg по модулю веса.
    Количество предсказаний строго <= количеству уникальных intense нейронов.
    """
    mouse = model_filename[:3]
    day = model_filename[4:6]
    model_path = Path(models_dir) / model_filename

    if not model_path.exists():
        return None

    # --- 1. Подготовка Ground Truth (INTENSE) ---
    # Читаем базу и фильтруем по мыши, сессии и feature='place'
    df_db = pd.read_csv(csv_path)
    df_intense = df_db[(df_db['mouse'] == mouse) & (df_db['session'] == day) & (df_db['feature'] == 'place')].copy()

    if df_intense.empty:
        return None

    # РАНЖИРОВАНИЕ INTENSE: Сортируем по 'me' (по убыванию).
    # drop_duplicates защищает от двойного учета нейрона, оставляя строку с максимальным 'me'
    df_intense = df_intense.sort_values(by='me', ascending=False).drop_duplicates(subset=['neuron_idx'])
    count = len(df_intense)  # Строго уникальное количество place cells

    # Формируем имена вида "neuron_0", "neuron_1" и словарь весов me
    neurons_intense = [f"neuron_{int(idx)}" for idx in df_intense['neuron_idx'].values]
    intense_me_dict = {f"neuron_{int(row['neuron_idx'])}": row['me'] for _, row in df_intense.iterrows()}

    # --- 2. Подготовка предсказаний (LINREG / BASELINE) ---
    data = joblib.load(model_path)
    model = data.get('model')
    all_features = data.get('feature_names')  # Список ['neuron_0', 'neuron_1', ...]
    overall = len(all_features)

    # Создаем бинарный вектор True
    y_true_binary = np.array([1 if f in neurons_intense else 0 for f in all_features])
    # Создаем непрерывный вектор True для Kendall Tau (веса me, 0 если не place cell)
    y_true_weights = np.array([intense_me_dict.get(f, 0.0) for f in all_features])

    if real_metrics:
        calc_type = 'Real Model'
        # РАНЖИРОВАНИЕ LINREG: L2 норма весов (модуль)
        model_weights = np.linalg.norm(model.coef_, axis=0)
        y_pred_weights = model_weights

        # Отбираем активные нейроны (>0)
        active_indices = np.where(model_weights > 0)[0]
        # Сортируем индексы по убыванию веса
        sorted_active = active_indices[np.argsort(model_weights[active_indices])[::-1]]

        # Берем одинаковое количество, но не больше чем в intense (и не больше чем выжило в Lasso)
        linreg_count = min(count, len(sorted_active))
        selected_indices = sorted_active[:linreg_count]

    else:
        calc_type = 'Random Baseline'
        # Бейзлайн: случайный вес для Tau и случайный выбор для бинарной классификации
        y_pred_weights = np.random.rand(overall)
        # Если проверяем бейзлайн, берем случайные нейроны (ровно count штук)
        selected_indices = np.random.choice(overall, size=count, replace=False)
        linreg_count = count

    # Создаем бинарный вектор предсказаний
    y_pred_binary = np.zeros(overall, dtype=int)
    y_pred_binary[selected_indices] = 1

    # --- 3. Расчет метрик ---
    tau, p_val = kendalltau(y_true_weights, y_pred_weights)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, zero_division=np.nan, average='binary'
    )
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel().tolist()

    # --- 4. Упаковка результатов ---
    metrics_dict = {
        'mouse': mouse,
        'session': day,
        'kendall_tau': tau,
        'ktau_p_val': p_val,
        # 'type': calc_type,
        'precision': prec,
        'recall': rec,
        'F1': f1,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
        'linreg_val': linreg_count,
        'intense_val': count,
        'overall': overall
    }

    return metrics_dict


def compare_metric_distributions(file_base='LNOF_SynchronizedData/LNOF_prec_rec_base_tau.csv',
                                 file_real='LNOF_SynchronizedData/LNOF_prec_rec_real_tau.csv',
                                 metric='F1'):
    """
    Рассчитывает критерии Манна-Уитни, Колмогорова-Смирнова и Стьюдента
    для сравнения распределений метрики между бейзлайном и реальной моделью.
    """
    path_base = Path(file_base)
    path_real = Path(file_real)

    if not path_base.exists() or not path_real.exists():
        print("[ERROR] Один или оба файла не найдены. Проверьте пути.")
        return

    # 1. Загрузка данных
    df_base = pd.read_csv(path_base)
    df_real = pd.read_csv(path_real)

    if metric not in df_base.columns or metric not in df_real.columns:
        print(f"[ERROR] Колонка '{metric}' отсутствует в одном из файлов.")
        return

    # Извлекаем значения и удаляем NaN (вдруг где-то метрика не посчиталась)
    sample_base = df_base[metric].dropna().values
    sample_real = df_real[metric].dropna().values

    print(f"\n{'=' * 50}")
    print(f"Статистический анализ метрики: {metric.upper()}")
    print(f"Выборка Baseline: {len(sample_base)} сессий | Выборка Real Model: {len(sample_real)} сессий")
    print(f"Mean Baseline: {np.mean(sample_base):.4f} | Mean Real: {np.mean(sample_real):.4f}")
    print(f"{'=' * 50}")

    results = {}

    # 2. U-критерий Манна-Уитни (Mann-Whitney U test)
    # Непараметрический тест. Проверяет, превосходят ли значения одной выборки значения другой.
    # Идеален, если распределение F1 не является нормальным.
    stat_mw, p_mw = stats.mannwhitneyu(sample_real, sample_base, alternative='two-sided')
    results['Mann-Whitney'] = {'stat': stat_mw, 'p_value': p_mw}

    # 3. Критерий Колмогорова-Смирнова (Kolmogorov-Smirnov 2-sample test)
    # Непараметрический тест. Сравнивает саму ФОРМУ распределений (медианы, дисперсию, асимметрию).
    stat_ks, p_ks = stats.ks_2samp(sample_real, sample_base)
    results['Kolmogorov-Smirnov'] = {'stat': stat_ks, 'p_value': p_ks}

    # 4. T-критерий Стьюдента (Student's t-test)
    # Параметрический тест. Проверяет равенство средних значений.
    # Работает хорошо, если распределения близки к нормальным.
    stat_tt, p_tt = stats.ttest_ind(sample_real, sample_base, equal_var=False)
    # Указываем equal_var=False (T-тест Уэлча), так как дисперсии у реальной модели и бейзлайна могут отличаться
    results['Student T-test'] = {'stat': stat_tt, 'p_value': p_tt}

    # --- Вывод результатов ---
    for test_name, data in results.items():
        p_val = data['p_value']
        stat = data['stat']

        # Определяем уровень значимости
        if p_val < 0.001:
            significance = "*** (p < 0.001)"
        elif p_val < 0.01:
            significance = "** (p < 0.01)"
        elif p_val < 0.05:
            significance = "* (p < 0.05)"
        else:
            significance = "ns (незначимо)"

        print(f"[{test_name}]")
        print(f"  Статистика: {stat:.4f}")
        print(f"  p-value:    {p_val:.2e}  {significance}\n")

    return results


def compare_metric_distributions_wilcoxon(
        file_base='stat_tables/LNOF_prec_rec_base_tau.csv',
        file_real='stat_tables/LNOF_prec_rec_real_tau.csv',
        metric='F1'):
    """
    Рассчитывает парный критерий Уилкоксона для сравнения
    распределений метрики между бейзлайном и реальной моделью.
    """
    path_base = Path(file_base)
    path_real = Path(file_real)

    if not path_base.exists() or not path_real.exists():
        print("[ERROR] Один или оба файла не найдены. Проверьте пути.")
        return

    # 1. Загрузка данных
    df_base = pd.read_csv(path_base)
    df_real = pd.read_csv(path_real)

    if metric not in df_base.columns or metric not in df_real.columns:
        print(f"[ERROR] Колонка '{metric}' отсутствует в одном из файлов.")
        return

    # 2. Строгое попарное выравнивание (Критично для Уилкоксона)
    # Предполагается, что строки в файлах идут в одинаковом порядке (сессия к сессии).
    df_paired = pd.DataFrame({
        'base': df_base[metric],
        'real': df_real[metric]
    }).dropna()

    sample_base = df_paired['base'].values
    sample_real = df_paired['real'].values

    if len(sample_base) < 5:
        print("[ERROR] Недостаточно парных данных для теста Уилкоксона (минимум 5 пар).")
        return

    print(f"\n{'=' * 55}")
    print(f"Парный статистический анализ метрики: {metric.upper()}")
    print(f"Количество валидных пар сессий: {len(sample_base)}")
    print(f"Медиана Baseline: {np.median(sample_base):.4f} | Медиана Real: {np.median(sample_real):.4f}")
    print(f"Среднее Baseline: {np.mean(sample_base):.4f} | Среднее Real: {np.mean(sample_real):.4f}")
    print(f"{'=' * 55}")

    results = {}

    # 3. Парный критерий Уилкоксона (Wilcoxon signed-rank test)
    # Проверяет, есть ли значимый сдвиг в медиане разностей между парными значениями.
    stat_w, p_w = stats.wilcoxon(sample_real, sample_base)
    results['Wilcoxon'] = {'stat': stat_w, 'p_value': p_w}

    # --- Вывод результатов ---
    if p_w < 0.001:
        significance = "*** (p < 0.001)"
    elif p_w < 0.01:
        significance = "** (p < 0.01)"
    elif p_w < 0.05:
        significance = "* (p < 0.05)"
    else:
        significance = "ns (незначимо)"

    print("[Wilcoxon signed-rank test]")
    print(f"  Статистика W: {stat_w:.4f}")
    print(f"  p-value:      {p_w:.2e}  {significance}\n")

    return results


def compare_gnn_vs_lr_wilcoxon(
        gnn_csv='models_LNOF_TGNN_Tanh/gnn_models_summary.csv',
        lr_csv='stat_tables/unified_model_performance_report.csv',
        exclude_mice=['J20', 'J21']):
    """
    Попарно сравнивает метрики R2 и MED между моделями GNN и Линейной Регрессией (LR).
    Использует критерий Уилкоксона и выводит среднее, медиану и 1.96 * SE.
    """
    path_gnn = Path(gnn_csv)
    path_lr = Path(lr_csv)

    if not path_gnn.exists() or not path_lr.exists():
        print(f"[ERROR] Не найден один из файлов. Проверьте пути:\n  - {path_gnn}\n  - {path_lr}")
        return

    # 1. Загрузка данных
    df_gnn = pd.read_csv(path_gnn)
    df_lr = pd.read_csv(path_lr)

    # Приводим названия колонок к нижнему регистру для страховки
    df_gnn.columns = [c.lower() for c in df_gnn.columns]
    df_lr.columns = [c.lower() for c in df_lr.columns]

    # Переименовываем метрики GNN, если необходимо
    if 'test_r2' in df_gnn.columns and 'r2' not in df_gnn.columns:
        df_gnn.rename(columns={'test_r2': 'r2'}, inplace=True)
    if 'test_med' in df_gnn.columns and 'med' not in df_gnn.columns:
        df_gnn.rename(columns={'test_med': 'med'}, inplace=True)

    # --- ИСПРАВЛЕНИЕ: Разделение колонки Session в GNN таблице ---
    # Превращаем 'J54_4D' в mouse='J54' и session='4D'
    if 'session' in df_gnn.columns and 'mouse' not in df_gnn.columns:
        split_data = df_gnn['session'].str.split('_', expand=True)
        if split_data.shape[1] == 2:
            df_gnn['mouse'] = split_data[0]
            df_gnn['session'] = split_data[1]
        else:
            print("[ERROR] Не удалось распарсить колонку session в таблице GNN. Ожидается формат 'J54_4D'.")
            return

    # 2. Исключение дефектных мышей (если задано)
    if exclude_mice:
        df_gnn = df_gnn[~df_gnn['mouse'].isin(exclude_mice)]
        df_lr = df_lr[~df_lr['mouse'].isin(exclude_mice)]

    # 3. Строгое попарное выравнивание (Inner Join)
    # Объединяем по мыши и сессии, чтобы сравнивать строго одни и те же условия
    df_paired = pd.merge(
        df_gnn[['mouse', 'session', 'r2', 'med']],
        df_lr[['mouse', 'session', 'r2', 'med']],
        on=['mouse', 'session'],
        suffixes=('_gnn', '_lr')
    )

    if df_paired.empty:
        print("[ERROR] Не удалось найти пересекающиеся пары (mouse + session) между таблицами.")
        return

    print(f"\n{'=' * 80}")
    print(f"СРАВНЕНИЕ АРХИТЕКТУР: GNN vs Linear Regression (Lasso)")
    print(f"Количество строго парных сессий: {len(df_paired)}")
    print(f"{'=' * 80}\n")

    def get_stars(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    def analyze_metric(metric_name, col_gnn, col_lr):
        # Удаляем NaN в парах, если таковые имеются
        df_clean = df_paired.dropna(subset=[col_gnn, col_lr])

        arr_gnn = df_clean[col_gnn].values
        arr_lr = df_clean[col_lr].values

        if len(arr_gnn) < 5:
            print(f"[ERROR] Недостаточно данных для {metric_name} (менее 5 пар).")
            return

        # Статистики GNN
        mean_gnn = np.mean(arr_gnn)
        med_gnn = np.median(arr_gnn)
        ci_gnn = 1.96 * stats.sem(arr_gnn)

        # Статистики LR
        mean_lr = np.mean(arr_lr)
        med_lr = np.median(arr_lr)
        ci_lr = 1.96 * stats.sem(arr_lr)

        # Парный критерий Уилкоксона
        stat_w, p_w = stats.wilcoxon(arr_gnn, arr_lr)

        # Вывод
        print(f"--- Метрика: {metric_name.upper()} ---")
        print(f"GNN Model:")
        print(f"  Среднее ± 1.96*SE: {mean_gnn:.4f} ± {ci_gnn:.4f}")
        print(f"  Медиана:           {med_gnn:.4f}")
        print(f"Linear Regression:")
        print(f"  Среднее ± 1.96*SE: {mean_lr:.4f} ± {ci_lr:.4f}")
        print(f"  Медиана:           {med_lr:.4f}\n")

        print(f"Wilcoxon Signed-Rank Test:")
        print(f"  W-статистика: {stat_w:.1f}")
        print(f"  p-value:      {p_w:.2e} ({get_stars(p_w)})\n")

    # Анализ обеих метрик
    analyze_metric('R2 (Точность)', 'r2_gnn', 'r2_lr')
    analyze_metric('MED (Ошибка в см)', 'med_gnn', 'med_lr')

    print(f"{'=' * 80}")


def aggregate_pt_metrics(models_dir='models_LNOF_TGNN', save_csv=True, csv_name='gnn_models_summary.csv'):
    """
    Бронебойный парсер метрик. Ищет файлы рекурсивно, игнорирует регистр,
    безопасно обрабатывает битые данные и выводит всю таблицу целиком.
    """
    models_path = Path(models_dir)

    if not models_path.exists():
        print(f"[ОШИБКА] Директория {models_dir} не найдена.")
        return None

    # 1. ИЩЕМ РЕКУРСИВНО И БЕЗ УЧЕТА РЕГИСТРА
    # rglob("*") заглянет во все подпапки, а suffix.lower() поймает .pt, .PT, .pth
    all_files = list(models_path.rglob("*"))
    pt_files = [f for f in all_files if f.is_file() and f.suffix.lower() in ['.pt', '.pth']]

    if not pt_files:
        print(f"[ИНФО] В директории {models_dir} и её подпапках нет .pt файлов.")
        return None

    print(f"  > Найдено файлов для анализа: {len(pt_files)}")

    results = []

    for file_path in pt_files:
        # Очищаем имя от всех возможных суффиксов
        session_name = file_path.name.replace('_st_gnn.pt', '').replace('.pt', '').replace('.pth', '')

        try:
            # Загружаем с разрешением на выполнение (weights_only=False)
            model_data = torch.load(file_path, map_location='cpu', weights_only=False)

            # Защита: проверяем, что это вообще словарь, а не просто веса модели
            if isinstance(model_data, dict) and 'metrics' in model_data:
                metrics = model_data['metrics']

                row = {
                    'Session': session_name,
                    'Test_R2': metrics.get('test_r2', np.nan),
                    'Train_R2': metrics.get('train_r2', np.nan),
                    'Test_MED': metrics.get('test_med_dist', np.nan),
                    'Train_MED': metrics.get('train_med_dist', np.nan),
                    'Active_Neurons': metrics.get('n_active_features', np.nan),
                    'N_Train_Windows': metrics.get('n_train', np.nan),
                    'N_Test_Windows': metrics.get('n_test', np.nan),
                    'File_Name': file_path.name  # Добавил имя файла, чтобы точно знать источник
                }
                results.append(row)
            else:
                print(f"    [ПРОПУСК] Файл {file_path.name} не содержит словаря 'metrics'.")

        except Exception as e:
            print(f"    [ОШИБКА] Не удалось прочитать {file_path.name}: {e}")

    df_results = pd.DataFrame(results)

    if df_results.empty:
        print("Не удалось извлечь метрики ни из одного файла.")
        return None

    # 2. БЕЗОПАСНАЯ КОНВЕРТАЦИЯ ДАННЫХ
    # Если где-то проскочила строка или None, превращаем её в математический NaN
    numeric_cols = ['Test_R2', 'Train_R2', 'Test_MED', 'Train_MED', 'Active_Neurons']
    for col in numeric_cols:
        if col in df_results.columns:
            df_results[col] = pd.to_numeric(df_results[col], errors='coerce')

    # Сортируем (все NaN улетят в самый низ списка, но не удалятся)
    df_results = df_results.sort_values(by='Test_R2', ascending=False, na_position='last').reset_index(drop=True)

    # 3. ОКРУГЛЕНИЕ ДЛЯ КРАСОТЫ
    df_results['Test_R2'] = df_results['Test_R2'].round(4)
    df_results['Train_R2'] = df_results['Train_R2'].round(4)
    df_results['Test_MED'] = df_results['Test_MED'].round(2)
    df_results['Train_MED'] = df_results['Train_MED'].round(2)

    print("\n" + "=" * 90)
    print("📊 СВОДНАЯ ТАБЛИЦА МЕТРИК ST-GNN")
    print("=" * 90)

    # 4. ОТКЛЮЧЕНИЕ ЛИМИТОВ PANDAS ПРИ ВЫВОДЕ В КОНСОЛЬ
    # Это гарантирует, что ты увидишь ВСЕ строки, даже если их 100+ штук
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(df_results.to_string())

    if save_csv:
        csv_path = models_path / csv_name
        df_results.to_csv(csv_path, index=False)
        print(f"\n✓ Таблица успешно сохранена в: {csv_path}")

    return df_results


def get_models_dataframe(models_dir='models_LNOF_Lasso'):
    """
    Парсит папку с моделями и собирает DataFrame с метриками и днем эксперимента.
    Ожидается формат файла типа: MouseID_1D.pkl, H01_2D_object.pkl и т.д.
    """
    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"[ERROR] Directory {models_dir} not found.")
        return pd.DataFrame()

    results = []

    for file in models_path.glob("*.pkl"):
        if "best" in file.name:
            continue

        try:
            data = joblib.load(file)
            metrics = data.get('metrics', {})
            metadata = data.get('metadata', {})

            # Извлекаем R2 и MAE (у нас это test_dist / MED)
            r2 = metrics.get('test_r2', np.nan)
            mae = metrics.get('test_dist', np.nan)  # Mean Euclidean Distance

            # Пытаемся вытащить день эксперимента из названия (ищем 1D, 2D, 3D, 4D)
            day_match = re.search(r'_([1-4])D', file.name)
            day = int(day_match.group(1)) if day_match else None

            # Если в названии нет XD, но есть в session
            if not day and 'session' in metadata:
                day_match = re.search(r'([1-4])D', metadata['session'])
                day = int(day_match.group(1)) if day_match else 0

            mouse = metadata.get('mouse', 'Unknown')

            # Для подписи на оси X удобно использовать Имя_Мыши
            model_name = f"{mouse}_{day}D" if day else file.stem

            results.append({
                'model_name': model_name,
                'mouse': mouse,
                'day': day,
                'r2': r2,
                'mae': mae,
                'filename': file.name
            })
        except Exception as e:
            print(f"[WARNING] Could not process {file.name}: {e}")

    df = pd.DataFrame(results)

    # Сортируем сначала по мыши, затем по дню, чтобы столбики шли логично
    if not df.empty:
        df = df.sort_values(by=['mouse', 'day']).reset_index(drop=True)

    return df


def get_model_results(mouse_id=None, models_dir='models_LNOF_Lasso', csv=True):
    """
    Сканирует папку с моделями и возвращает DataFrame с метриками регрессии (MED/MAE и R2).
    """
    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"[ERROR] Directory {models_dir} not found.")
        return None

    all_results = []
    for file in models_path.glob("*.pkl"):
        try:
            if "best" in file.name:
                continue

            data = joblib.load(file)
            metrics = data.get('metrics', {})
            metadata = data.get('metadata', {})

            entry = {
                'mouse_id': metadata.get('mouse'),
                'session': metadata.get('session'),
                'r2_score': metrics.get('test_r2'),
                'med_dist': metrics.get('test_dist'),  # Mean Euclidean Distance (аналог MAE для 2D)
                'active_neurons': metrics.get('n_active_features'),
                'filename': file.name
            }
            all_results.append(entry)
        except Exception as e:
            print(f"[WARNING] Could not read {file.name}: {e}")

    if not all_results:
        print("No models found in the directory.")
        return None

    df = pd.DataFrame(all_results)

    if mouse_id: df = df[df['mouse_id'] == mouse_id]

    df = df.sort_values(['mouse_id', 'session'])

    if df.empty:
        print("No sessions match the specified criteria.")
    else:
        print(f"\nFound {len(df)} matching model(s):")
        # Выводим R2 и MED (расстояние ошибки)
        print(df[['mouse_id', 'session', 'r2_score', 'med_dist', 'active_neurons']].to_string(index=False))

    if csv and not df.empty:
        csv_filename = os.path.join(models_dir, "model_performance_report.csv")
        df.to_csv(csv_filename, index=False)
        print(f"\n[INFO] Results saved to {csv_filename}")

    return df


def generate_unified_summary_report(models_dir='models_LNOF_Lasso',
                                    data_dir='LNOF_SynchronizedData',
                                    output_csv='unified_model_performance_report.csv',
                                    speed_threshold=1.0):
    """
    Создает сводную таблицу по всем сессиям, объединяя метрики точности модели (R2, MED),
    параметры нейронной сети (кол-во важных нейронов) и поведенческие метрики мыши (скорость).
    """
    models_path = Path(models_dir)
    data_path = Path(data_dir)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Ищем все файлы моделей
    pkl_files = list(models_path.glob("*.pkl"))

    if not pkl_files:
        print(f"[ERROR] Не найдено .pkl файлов в директории {models_dir}")
        return

    print(f"[{'=' * 50}]\nНачат сбор сводной статистики по {len(pkl_files)} сессиям...\n[{'=' * 50}]")

    for model_file in pkl_files:
        model_name = model_file.stem  # Например 'J30_1D'
        mouse = model_name[:3]
        day = model_name[4:6]

        row_data = {
            'mouse': mouse,
            'session': day,
            'model_name': model_name
        }

        # --- 1. ИЗВЛЕЧЕНИЕ ДАННЫХ ИЗ МОДЕЛИ (.pkl) ---
        try:
            data = joblib.load(model_file)
            model = data.get('model')

            # Достаем R2 и MED.
            # Зависит от того, где именно они лежат в твоем словаре (в корне или в 'metrics')
            metrics = data.get('metrics', {})
            row_data['r2'] = metrics.get('test_r2', data.get('test_r2', np.nan))
            row_data['med'] = metrics.get('test_dist', data.get('test_dist', np.nan))  # MED часто сохраняется как MAE

            # Считаем нейроны
            feature_names = data.get('feature_names', [])
            total_neurons = len(feature_names)
            row_data['total_neurons'] = total_neurons

            if hasattr(model, 'coef_'):
                importance = np.linalg.norm(model.coef_, axis=0)
                important_neurons = int(np.sum(importance > 0))
                row_data['important_neurons_count'] = important_neurons
                row_data['important_fraction'] = important_neurons / total_neurons if total_neurons > 0 else np.nan
            else:
                row_data['important_neurons_count'] = np.nan
                row_data['important_fraction'] = np.nan

        except Exception as e:
            print(f"[ERROR] Ошибка при чтении модели {model_name}: {e}")
            continue

        # --- 2. ИЗВЛЕЧЕНИЕ ДАННЫХ ИЗ ТРЕКИНГА (.npz) ---
        npz_file = data_path / f"LNOF_{model_name}_aligned.npz"

        if npz_file.exists():
            try:
                npz_data = np.load(npz_file, allow_pickle=True)

                # Добываем скорость
                if 'speed' in npz_data.files:
                    speed = npz_data['speed']
                elif 'x' in npz_data.files and 'y' in npz_data.files:
                    dx = np.diff(npz_data['x'], prepend=npz_data['x'][0])
                    dy = np.diff(npz_data['y'], prepend=npz_data['y'][0])
                    speed = np.sqrt(dx ** 2 + dy ** 2)
                else:
                    speed = np.array([])

                speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)
                total_frames = len(speed)

                if total_frames > 0:
                    # Поведенческие метрики
                    moving_frames = np.sum(speed >= speed_threshold)
                    row_data['moving_fraction'] = moving_frames / total_frames
                    row_data['avg_speed'] = np.mean(speed)
                else:
                    row_data['moving_fraction'] = np.nan
                    row_data['avg_speed'] = np.nan

            except Exception as e:
                print(f"[ERROR] Ошибка при чтении трекинга для {model_name}: {e}")
                row_data['moving_fraction'] = np.nan
                row_data['avg_speed'] = np.nan
        else:
            print(f"[WARNING] .npz файл для {model_name} не найден. Поведенческие метрики будут пустыми.")
            row_data['moving_fraction'] = np.nan
            row_data['avg_speed'] = np.nan

        # Добавляем собранную строку в общий список
        all_results.append(row_data)

    # --- 3. СБОРКА И СОХРАНЕНИЕ ---
    if not all_results:
        print("[WARNING] Не удалось собрать данные. Таблица пуста.")
        return None

    df_summary = pd.DataFrame(all_results)

    # Сортируем для красоты: сначала мышь, потом день
    df_summary = df_summary.sort_values(by=['mouse', 'session']).reset_index(drop=True)

    # Сохраняем в CSV
    df_summary.to_csv(output_path, index=False)

    print(f"✓ Сводный отчет успешно сохранен: {output_path}")
    print(f"  Форма таблицы: {df_summary.shape[0]} строк, {df_summary.shape[1]} столбцов.")

    return df_summary


def analyze_performance_correlations(csv_path='stat_tables/spearman_kendall_jaccard_stats.csv'):
    """
    Рассчитывает корреляцию Спирмена между целевыми метриками (r2, med)
    и всеми остальными численными параметрами (поведение, топология сети).
    Исключает из анализа мышей J20 и J21.
    Сохраняет результаты в ту же папку.
    """
    path = Path(csv_path)
    if not path.exists():
        print(f"[ERROR] Файл не найден: {path}")
        return None

    # Загружаем сводную таблицу
    df = pd.read_csv(path)

    # --- НОВЫЙ БЛОК: Исключение J20 и J21 ---
    excluded_mice = ['J20', 'J21']
    initial_len = len(df)

    if 'mouse' in df.columns:
        df = df[~df['mouse'].isin(excluded_mice)]
    elif 'model_name' in df.columns:
        # Если колонки mouse нет, ищем по префиксу названия модели
        df = df[~df['model_name'].str.startswith(tuple(excluded_mice))]

    print(f"[INFO] Исключены мыши: {excluded_mice}. Сессий до: {initial_len}, Сессий после: {len(df)}")
    # ----------------------------------------

    # Отбираем только числовые колонки
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Целевые метрики
    targets = ['r2', 'med']

    # Проверяем, есть ли они в таблице
    for t in targets:
        if t not in numeric_cols:
            print(f"[ERROR] Целевая метрика '{t}' отсутствует в таблице.")
            return None

    # Остальные метрики сохраняют свой изначальный порядок слева направо
    other_metrics = [col for col in numeric_cols if col not in targets]

    print(f"\n{'=' * 60}")
    print(f"Анализ корреляций Спирмена (Цели: R2, MED)")
    print(f"{'=' * 60}")

    results_raw = []

    for t in targets:
        row_corr = {'Target': f"{t}_corr"}
        row_pval = {'Target': f"{t}_pval"}

        for m in other_metrics:
            # Берем только те сессии, где есть обе метрики (исключаем NaN)
            valid_data = df[[t, m]].dropna()

            if len(valid_data) > 1:
                corr, pval = spearmanr(valid_data[t], valid_data[m])
            else:
                corr, pval = np.nan, np.nan

            row_corr[m] = corr
            row_pval[m] = pval

        results_raw.append(row_corr)
        results_raw.append(row_pval)

        print(f"  > Обработан таргет: {t.upper()}")

    # Создаем DataFrame
    df_raw = pd.DataFrame(results_raw)

    # Сохраняем в ТУ ЖЕ папку, где лежал исходный файл
    out_dir = path.parent
    path_raw = out_dir / 'performance_correlations_raw.csv'

    df_raw.to_csv(path_raw, index=False)

    print(f"\n✓ Успешно сохранено!")
    print(f"  Таблица с сырыми числами (для скриптов): {path_raw}")


def evaluate_top_vs_rest_models(models_dir='models_LNOF_Lasso',
                                metrics_csv='stat_tables/top_vs_rest_metrics.csv'):
    """
    Проходит по сохраненным моделям Top и Rest, извлекает метрики (R2, MED)
    и количество фичей, после чего собирает их в сводную таблицу.
    Без переобучения.
    """
    dir_top = Path(models_dir) / 'models_top'
    dir_rest = Path(models_dir) / 'models_rest'

    if not dir_top.exists() or not dir_rest.exists():
        print(f"[ERROR] Папки с моделями не найдены. Убедитесь, что модели обучены.")
        return None

    path_metrics_csv = Path(metrics_csv)
    path_metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    # Ищем все файлы топовых моделей (они послужат списком сессий)
    top_files = list(dir_top.glob('*_top.pkl'))

    if not top_files:
        print(f"[WARNING] В папке {dir_top} нет файлов моделей.")
        return None

    results = []

    # Инициализация таймера
    start_time = time.time()
    last_notify_time = start_time
    notify_interval = 1800  # 30 минут в секундах
    models_processed_count = 0

    print(f"\n{'=' * 70}")
    print(f"Запуск сбора метрик: Top vs Rest (Чтение с диска)")
    print(f"Найдено сессий для обработки: {len(top_files)}")
    print(f"{'=' * 70}")

    for top_file in top_files:
        # Извлекаем базовое имя, например 'J30_1D' из 'J30_1D_top.pkl'
        model_name = top_file.name.replace('_top.pkl', '')
        mouse = model_name[:3]
        day = model_name[4:6]

        rest_file = dir_rest / f"{model_name}_rest.pkl"

        row_metrics = {
            'mouse': mouse,
            'session': day,
            'model_name': model_name
        }

        # --- 1. Извлечение из TOP модели ---
        try:
            data_top = joblib.load(top_file)
            metrics_top = data_top.get('metrics', {})

            row_metrics['top_count'] = len(data_top.get('feature_names', []))
            row_metrics['top_r2'] = metrics_top.get('test_r2', np.nan)
            row_metrics['top_med'] = metrics_top.get('test_dist', np.nan)  # Используем правильный ключ test_med
        except Exception as e:
            print(f"[ERROR] Ошибка чтения {top_file.name}: {e}")
            row_metrics['top_count'], row_metrics['top_r2'], row_metrics['top_med'] = 0, np.nan, np.nan

        # --- 2. Извлечение из REST модели ---
        if rest_file.exists():
            try:
                data_rest = joblib.load(rest_file)
                metrics_rest = data_rest.get('metrics', {})

                row_metrics['rest_count'] = len(data_rest.get('feature_names', []))
                row_metrics['rest_r2'] = metrics_rest.get('test_r2', np.nan)
                row_metrics['rest_med'] = metrics_rest.get('test_dist', np.nan)
            except Exception as e:
                print(f"[ERROR] Ошибка чтения {rest_file.name}: {e}")
                row_metrics['rest_count'], row_metrics['rest_r2'], row_metrics['rest_med'] = 0, np.nan, np.nan
        else:
            print(f"[WARNING] Парная модель {rest_file.name} не найдена. Заполняем пропусками.")
            row_metrics['rest_count'], row_metrics['rest_r2'], row_metrics['rest_med'] = 0, np.nan, np.nan

        # Считаем общее количество активных нейронов в оригинальной базовой модели
        row_metrics['base_active_count'] = row_metrics['top_count'] + row_metrics['rest_count']

        results.append(row_metrics)
        models_processed_count += 1

        # --- 3. Таймер ---
        current_time = time.time()
        if current_time - last_notify_time >= notify_interval:
            elapsed_mins = int((current_time - start_time) / 60)
            print(f"\n[⏱ ТАЙМЕР] Прошло {elapsed_mins} минут. Обработано {models_processed_count} пар моделей...\n")
            last_notify_time = current_time

    # --- 4. Сохранение таблицы ---
    df_results = pd.DataFrame(results)

    # Упорядочиваем колонки для читаемости
    cols = ['mouse', 'session', 'model_name',
            'base_active_count', 'top_count', 'rest_count',
            'top_r2', 'rest_r2', 'top_med', 'rest_med']

    # Если вдруг каких-то колонок не оказалось, берем только существующие
    final_cols = [c for c in cols if c in df_results.columns]
    df_results = df_results[final_cols]

    df_results.sort_values(by=['mouse', 'session'], inplace=True)
    df_results.to_csv(path_metrics_csv, index=False)

    total_time = time.time() - start_time
    mins, secs = divmod(total_time, 60)

    print(f"\n{'=' * 70}")
    print(f"✓ Сбор данных завершен за {int(mins)} мин {int(secs)} сек!")
    print(f"✓ Всего обработано пар моделей: {models_processed_count}")
    print(f"✓ Таблица метрик сохранена: {path_metrics_csv}")
    print(f"{'=' * 70}")

    return df_results


def multi_jaccard(*sets):
    """
    Вспомогательная функция для расчета меры Жаккара для произвольного количества множеств.
    Основана на пересечении (intersection) и объединении (union).
    """
    if not sets:
        return 0.0

    # set.intersection и set.union отлично работают с произвольным количеством аргументов
    intersect = set.intersection(*sets)
    union = set.union(*sets)

    return len(intersect) / len(union) if len(union) > 0 else 0.0


def analyze_full_correlations_and_jaccard(ablation_csv='stat_tables/full_ablation_stats.csv',
                                          models_dir='models_LNOF_Lasso',
                                          csv_path='LNOF_SynchronizedData/LNOF_database.csv',
                                          output_csv='stat_tables/spearman_kendall_jaccard_stats.csv'):
    """
    Считает Кендалла (глобально), Спирмена (локально для значений > 0)
    и меру Жаккара для всех сочетаний выделенных множеств, включая MI.
    """
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Загрузка данных
    ablation_df = pd.read_csv(ablation_csv)

    if Path(csv_path).exists():
        df_db = pd.read_csv(csv_path)
    else:
        print(f"[ERROR] База данных {csv_path} не найдена. Анализ невозможен.")
        return

    results = []

    print(f"\n{'=' * 70}")
    print(f"Начат глобальный анализ: Spearman (вкл. MI), Kendall & multi-Jaccard")
    print(f"{'=' * 70}")

    for _, row in ablation_df.iterrows():
        model_name = row['model_session']
        mouse = model_name[:3]
        day = model_name[4:6]

        model_path = Path(models_dir) / f"{model_name}.pkl"
        if not model_path.exists():
            continue

        # --- 2. ИЗВЛЕЧЕНИЕ ДАННЫХ ИЗ МОДЕЛИ ---
        data = joblib.load(model_path)
        model = data.get('model')
        feature_names = np.array(data.get('feature_names'))

        if not hasattr(model, 'coef_'):
            continue

        # Абсолютные веса
        weights = np.linalg.norm(model.coef_, axis=0)

        # --- 3. ИЗВЛЕЧЕНИЕ ABLATION И РАСЧЕТ CRIT ---
        baseline_r2 = row['baseline_r2']
        ablation_r2_vals = np.array([row.get(f'ablation_{feat}', np.nan) for feat in feature_names])

        # crit: разница между полноценной моделью и ablation
        crit = baseline_r2 - ablation_r2_vals

        # Z-score для crit
        std_crit = np.std(crit[~np.isnan(crit)])
        if std_crit > 0:
            zcrit = (crit - np.mean(crit[~np.isnan(crit)])) / std_crit
        else:
            zcrit = np.zeros_like(crit)

        # Замена NaN на 0 для безопасного расчета Кендалла (минусы остаются)
        safe_crit = np.nan_to_num(crit, nan=0.0)
        safe_zcrit = np.nan_to_num(zcrit, nan=0.0)

        # --- 4. ИЗВЛЕЧЕНИЕ ВЗАИМНОЙ ИНФОРМАЦИИ (MI) ---
        df_intense = df_db[(df_db['mouse'] == mouse) & (df_db['session'] == day) & (df_db['feature'] == 'place')].copy()
        df_intense = df_intense.sort_values(by='me', ascending=False).drop_duplicates(subset=['neuron_idx'])
        intense_me_dict = {f"neuron_{int(r['neuron_idx'])}": r['me'] for _, r in df_intense.iterrows()}
        mi_array = np.array([intense_me_dict.get(f, 0.0) for f in feature_names])

        row_data = {
            'mouse': mouse,
            'session': day,
            'model_name': model_name
        }

        # =====================================================================
        # БЛОК 1: КОРРЕЛЯЦИЯ КЕНДАЛЛА (Глобальная, включает 0 и минусы)
        # =====================================================================
        tau_mi_w, p_mi_w = kendalltau(mi_array, weights)
        tau_mi_c, p_mi_c = kendalltau(mi_array, safe_crit)
        tau_mi_zc, p_mi_zc = kendalltau(mi_array, safe_zcrit)

        row_data.update({
            'kendall_mi_vs_weight': tau_mi_w, 'p_kendall_mi_w': p_mi_w,
            'kendall_mi_vs_crit': tau_mi_c, 'p_kendall_mi_c': p_mi_c,
            'kendall_mi_vs_zcrit': tau_mi_zc, 'p_kendall_mi_zc': p_mi_zc
        })

        # =====================================================================
        # БЛОК 2: КОРРЕЛЯЦИЯ СПИРМЕНА (Локальная, только пересечения значений > 0)
        # =====================================================================
        # 2.1 Crit vs Weight
        mask_c_w = (weights > 0) & (crit > 0) & (~np.isnan(crit))
        if np.sum(mask_c_w) > 1:
            sp_c_w, p_sp_c_w = spearmanr(crit[mask_c_w], weights[mask_c_w])
            sp_zc_w, p_sp_zc_w = spearmanr(zcrit[mask_c_w], weights[mask_c_w])
        else:
            sp_c_w, p_sp_c_w, sp_zc_w, p_sp_zc_w = (np.nan,) * 4

        # 2.2 MI vs Weight
        mask_mi_w = (mi_array > 0) & (weights > 0)
        if np.sum(mask_mi_w) > 1:
            sp_mi_w, p_sp_mi_w = spearmanr(mi_array[mask_mi_w], weights[mask_mi_w])
        else:
            sp_mi_w, p_sp_mi_w = np.nan, np.nan

        # 2.3 MI vs Crit (и Z-crit)
        mask_mi_c = (mi_array > 0) & (crit > 0) & (~np.isnan(crit))
        if np.sum(mask_mi_c) > 1:
            sp_mi_c, p_sp_mi_c = spearmanr(mi_array[mask_mi_c], crit[mask_mi_c])
            sp_mi_zc, p_sp_mi_zc = spearmanr(mi_array[mask_mi_c], zcrit[mask_mi_c])
        else:
            sp_mi_c, p_sp_mi_c, sp_mi_zc, p_sp_mi_zc = (np.nan,) * 4

        row_data.update({
            'spearman_crit_vs_weight': sp_c_w, 'p_spearman_c_w': p_sp_c_w,
            'spearman_zcrit_vs_weight': sp_zc_w, 'p_spearman_zc_w': p_sp_zc_w,
            'spearman_mi_vs_weight': sp_mi_w, 'p_spearman_mi_w': p_sp_mi_w,
            'spearman_mi_vs_crit': sp_mi_c, 'p_spearman_mi_c': p_sp_mi_c,
            'spearman_mi_vs_zcrit': sp_mi_zc, 'p_spearman_mi_zc': p_sp_mi_zc
        })

        # =====================================================================
        # БЛОК 3: МНОЖЕСТВЕННЫЙ ЖАККАРД
        # =====================================================================
        set_W = set(feature_names[weights > 0])
        set_C = set(feature_names[safe_crit > 0])
        set_Z = set(feature_names[safe_zcrit > 0])
        set_M = set(feature_names[mi_array > 0])

        row_data.update({
            'jac_weight_mi': multi_jaccard(set_W, set_M),
            'jac_weight_crit': multi_jaccard(set_W, set_C),
            'jac_weight_zcrit': multi_jaccard(set_W, set_Z),
            'jac_mi_crit': multi_jaccard(set_M, set_C),
            'jac_mi_zcrit': multi_jaccard(set_M, set_Z),
            'jac_crit_zcrit': multi_jaccard(set_C, set_Z),
            'jac_overall_with_crit': multi_jaccard(set_W, set_M, set_C),
            'jac_overall_with_zcrit': multi_jaccard(set_W, set_M, set_Z),
            'size_weight': len(set_W),
            'size_mi': len(set_M),
            'size_crit': len(set_C),
            'size_zcrit': len(set_Z)
        })

        results.append(row_data)
        print(f"[{model_name}] Готово. Spearman(MI, W): {sp_mi_w:.3f} | Spearman(MI, C): {sp_mi_c:.3f}")

    # --- 5. СОХРАНЕНИЕ ---
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)

    print(f"\n✓ Успешно! Полная аналитика сохранена в:")
    print(f"  {output_path}")

    return df_results


def find_minimal_sufficient_ensemble(
        data_dir='LNOF_SynchronizedData',
        models_dir='models_LNOF_Lasso',
        output_csv='stat_tables/minimal_sufficient_ensembles.csv',
        c_param=C_PARAM,
        test_size=TEST_SIZE,
        max_iter=MAX_ITER
):
    """
    Итеративно добавляет нейроны (от самых сильных к слабым) с переобучением.
    Останавливается и записывает результат, когда R2 и MED приближаются к
    базовым значениям (R2: дельта <= 0.01, MED: дельта <= 0.1).
    Исключает мышей J20 и J21.
    """
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    excluded_mice = ['J20', 'J21']
    model_files = list(Path(models_dir).glob("*.pkl"))

    # Фильтруем мышей и исключаем модели _top и _rest из предыдущих скриптов
    valid_models = []
    for f in model_files:
        if f.stem.startswith(tuple(excluded_mice)): continue
        if '_top' in f.stem or '_rest' in f.stem: continue
        valid_models.append(f)

    results = []
    start_time = time.time()

    print(f"\n{'=' * 75}")
    print("Поиск минимально достаточного ансамбля (Addition с ранней остановкой)")
    print(f"Цель: \u0394R2 <= 0.01 и \u0394MED <= 0.1 от бейзлайна")
    print(f"{'=' * 75}")

    for model_path in valid_models:
        session_name = model_path.stem

        # 1. Загрузка базовой модели и ее метрик
        base_data = joblib.load(model_path)
        base_model = base_data.get('model')
        base_metrics = base_data.get('metrics', {})

        if not hasattr(base_model, 'coef_'):
            continue

        base_r2 = base_metrics.get('test_r2', np.nan)
        # Учитываем, что MED может лежать по ключу test_dist или test_mae
        base_med = base_metrics.get('test_dist', base_metrics.get('test_mae', np.nan))

        if np.isnan(base_r2) or np.isnan(base_med):
            print(f"[SKIP] {session_name}: отсутствуют базовые метрики R2 или MED.")
            continue

        # 2. Ранжирование нейронов по абсолютному весу
        weights = np.linalg.norm(base_model.coef_, axis=0)
        feat_names = np.array(base_data.get('feature_names'))

        # Сортировка по убыванию веса (от сильных к слабым)
        sorted_idx = np.argsort(weights)[::-1]
        sorted_features = feat_names[sorted_idx]
        total_features = len(sorted_features)

        # 3. Загрузка сырых данных
        data_file = Path(data_dir) / f"LNOF_{session_name}_aligned.npz"
        if not data_file.exists():
            continue

        # Предполагается наличие функций загрузки и сплита
        features_raw, labels = load_data(data_file)
        X_train_full, X_test_full, y_train, y_test = train_test_split_temporal(features_raw, labels,
                                                                               test_size=test_size)
        all_feat_list = list(feat_names)

        print(f"[{session_name}] База -> R2: {base_r2:.3f} | MED: {base_med:.3f}. Ищем минимум...")

        minimal_k = None
        achieved_r2 = None
        achieved_med = None

        # 4. Итеративное добавление (Cumulative Addition)
        for k in range(1, total_features + 1):
            subset = sorted_features[:k]

            # Подготовка данных для текущего сабсета
            if isinstance(X_train_full, pd.DataFrame):
                X_tr = X_train_full[subset]
                X_te = X_test_full[subset]
            else:
                idx = [all_feat_list.index(f) for f in subset]
                X_tr = X_train_full[:, idx]
                X_te = X_test_full[:, idx]

            # Переобучение
            clf, imp, scl, metrics = train_lasso(X_tr, y_train, X_te, y_test, C=c_param, max_iter=max_iter)

            curr_r2 = metrics.get('test_r2', -10.0)
            curr_med = metrics.get('test_dist', metrics.get('test_mae', 1000.0))

            # 5. Проверка условий (Ранняя остановка)
            # R2 может быть даже лучше базового, поэтому проверяем (base_r2 - curr_r2 <= 0.01)
            cond_r2 = (base_r2 - curr_r2) <= 0.01
            # MED тоже может стать лучше (меньше), поэтому проверяем (curr_med - base_med <= 0.1)
            cond_med = (curr_med - base_med) <= 0.1

            if cond_r2 and cond_med:
                minimal_k = k
                achieved_r2 = curr_r2
                achieved_med = curr_med
                break  # Успех! Прерываем цикл для этой сессии

        # Запись результатов
        if minimal_k is not None:
            print(f"  > Успех! Достаточно {minimal_k} нейронов. (R2: {achieved_r2:.3f}, MED: {achieved_med:.3f})")
        else:
            print(f"  > [WARNING] Условия не достигнуты даже при использовании всех нейронов.")
            minimal_k = total_features
            achieved_r2 = curr_r2
            achieved_med = curr_med

        results.append({
            'model_name': session_name,
            'total_neurons': total_features,
            'minimal_neurons_needed': minimal_k,
            'fraction_needed': minimal_k / total_features,
            'base_r2': base_r2,
            'achieved_r2': achieved_r2,
            'base_med': base_med,
            'achieved_med': achieved_med
        })

    # Сохранение таблицы
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res = df_res.sort_values('model_name')
        df_res.to_csv(out_path, index=False)

        elapsed = int((time.time() - start_time) / 60)
        print(f"\n{'=' * 75}")
        print(f"✓ Анализ завершен за {elapsed} минут.")
        print(f"✓ Таблица сохранена: {out_path}")

        mean_fraction = df_res['fraction_needed'].mean() * 100
        print(f"В среднем для достижения порога требуется всего {mean_fraction:.1f}% ансамбля.")
        print(f"{'=' * 75}\n")

    return df_res


def generate_cross_session_wilcoxon_tables(
        match_dir='MatchData',
        models_dir='models_LNOF_Lasso',
        data_dir='LNOF_SynchronizedData',
        test_size=0.25,
        model_is_zero_indexed=True,
        output_csv_dir='stat_tables'):
    """
    Calculates Wilcoxon statistics for cross-session prediction graphs.
    Consolidated into a single English-only ASCII-friendly table.
    """
    output_path = Path(output_csv_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    excluded_mice = ['J20', 'J21']
    match_files = list(Path(match_dir).glob("LNOF_*.csv"))

    results = []

    print(f"\n{'=' * 85}")
    print("Collecting cross-session prediction data for Wilcoxon tests...")
    print(f"{'=' * 85}")

    def build_id_to_idx_map(feature_names, zero_indexed=model_is_zero_indexed):
        mapping = {}
        for idx, name in enumerate(feature_names):
            match = re.search(r'\d+', str(name))
            if match:
                mapping[int(match.group()) + (1 if zero_indexed else 0)] = idx
        return mapping

    # --- 1. DATA EXTRACTION ---
    for file_path in match_files:
        mouse = file_path.stem.replace('LNOF_', '')
        if mouse in excluded_mice: continue

        df_map = pd.read_csv(file_path, header=None)
        if df_map.shape[1] < 4: continue

        models, datasets, feat_maps = {}, {}, {}
        valid_chain = True

        for d in range(4):
            day_str = f"{d + 1}D"
            session_name = f"{mouse}_{day_str}"

            model_path = Path(models_dir) / f"{session_name}.pkl"
            data_file = Path(data_dir) / f"LNOF_{session_name}_aligned.npz"

            if not model_path.exists() or not data_file.exists():
                valid_chain = False
                break

            m_data = joblib.load(model_path)
            models[d] = m_data
            feat_maps[d] = build_id_to_idx_map(m_data.get('feature_names'))

            features_raw, labels_raw = load_data(data_file)
            split_idx = int(len(features_raw) * (1 - test_size))

            X_test_raw = features_raw.iloc[split_idx:].values if isinstance(features_raw,
                                                                            pd.DataFrame) else features_raw[split_idx:]
            y_test = labels_raw.iloc[split_idx:].values if isinstance(labels_raw,
                                                                      (pd.DataFrame, pd.Series)) else labels_raw[
                                                                                                      split_idx:]

            scaler = m_data.get('scaler')
            datasets[d] = (scaler.transform(X_test_raw) if scaler is not None else X_test_raw, y_test)

        if not valid_chain: continue

        for N in range(4):
            X_N, y_N = datasets[N]
            model_N = models[N]['model']

            # 1D Model (Blue)
            model_1 = models[0]['model']
            X_in_1 = np.zeros((len(X_N), len(models[0]['feature_names'])))
            mask_1_N = df_map[0].notna() & (df_map[0] > 0) & df_map[N].notna() & (df_map[N] > 0)
            for _, row in df_map[mask_1_N].iterrows():
                i1, iN = feat_maps[0].get(int(row[0]), -1), feat_maps[N].get(int(row[N]), -1)
                if i1 != -1 and iN != -1: X_in_1[:, i1] = X_N[:, iN]
            r2_1_to_N = r2_score(y_N, model_1.predict(X_in_1))

            # Current day model on 1D neurons (Green)
            X_in_N_sub1 = np.zeros((len(X_N), len(models[N]['feature_names'])))
            for _, row in df_map[mask_1_N].iterrows():
                iN = feat_maps[N].get(int(row[N]), -1)
                if iN != -1: X_in_N_sub1[:, iN] = X_N[:, iN]
            r2_N_sub1 = r2_score(y_N, model_N.predict(X_in_N_sub1))

            # 4D Model (Red)
            model_4 = models[3]['model']
            X_in_4 = np.zeros((len(X_N), len(models[3]['feature_names'])))
            mask_4_N = df_map[3].notna() & (df_map[3] > 0) & df_map[N].notna() & (df_map[N] > 0)
            for _, row in df_map[mask_4_N].iterrows():
                i4, iN = feat_maps[3].get(int(row[3]), -1), feat_maps[N].get(int(row[N]), -1)
                if i4 != -1 and iN != -1: X_in_4[:, i4] = X_N[:, iN]
            r2_4_to_N = r2_score(y_N, model_4.predict(X_in_4))

            # Current day model on 4D neurons (Purple)
            X_in_N_sub4 = np.zeros((len(X_N), len(models[N]['feature_names'])))
            for _, row in df_map[mask_4_N].iterrows():
                iN = feat_maps[N].get(int(row[N]), -1)
                if iN != -1: X_in_N_sub4[:, iN] = X_N[:, iN]
            r2_N_sub4 = r2_score(y_N, model_N.predict(X_in_N_sub4))

            results.append({
                'mouse': mouse, 'day': N + 1,
                'r2_1_to_N': r2_1_to_N, 'r2_N_sub1': r2_N_sub1,
                'r2_4_to_N': r2_4_to_N, 'r2_N_sub4': r2_N_sub4
            })

    df_res = pd.DataFrame(results)

    # --- Formatting Helper ---
    def calc_wilcoxon(arr1, arr2):
        df_pair = pd.DataFrame({'x': arr1, 'y': arr2}).dropna()
        if len(df_pair) < 5: return "N/A (n<5)"

        diff = df_pair['x'] - df_pair['y']
        if np.allclose(diff, 0):
            return "W = 0.0, p = 1.000"

        try:
            w, p = stats.wilcoxon(df_pair['x'], df_pair['y'])
        except Exception:
            return "N/A (Calc Error)"

        if p < 0.001:
            return f"W = {w:.1f}, p < 0.001"
        elif p < 0.01:
            return f"W = {w:.1f}, p < 0.01"
        elif p < 0.05:
            return f"W = {w:.1f}, p < 0.05"
        else:
            return f"W = {w:.1f}, p = {p:.3f}"

    combined_table = []

    # =========================================================================
    # PART 1: Symmetry (Right Graph)
    # Blue (1D -> N) vs Red (4D -> N)
    # =========================================================================
    sym_pairs = [(1, 4), (2, 3), (3, 2), (4, 1)]
    for d_blue, d_red in sym_pairs:
        df_b = df_res[df_res['day'] == d_blue].set_index('mouse')
        df_r = df_res[df_res['day'] == d_red].set_index('mouse')

        aligned = df_b[['r2_1_to_N']].join(df_r[['r2_4_to_N']], how='inner')
        res_str = calc_wilcoxon(aligned['r2_1_to_N'], aligned['r2_4_to_N'])

        combined_table.append({
            'Category': 'Symmetry (Right Graph)',
            'Comparison': f"{d_blue}D Blue vs {d_red}D Red",
            'Wilcoxon_Result': res_str
        })

    # =========================================================================
    # PART 2: Drift Day 1 Neurons (Left vs Right Graph)
    # Green (Current model on 1D pool) vs Blue (1D Model on 1D pool)
    # =========================================================================
    for d in range(1, 5):
        df_d = df_res[df_res['day'] == d]
        res_str = calc_wilcoxon(df_d['r2_N_sub1'], df_d['r2_1_to_N'])
        combined_table.append({
            'Category': 'Day 1 Drift (Green vs Blue)',
            'Comparison': f"{d}D Green vs {d}D Blue",
            'Wilcoxon_Result': res_str
        })

    # =========================================================================
    # PART 3: Drift Day 4 Neurons (Left vs Right Graph)
    # Purple (Current model on 4D pool) vs Red (4D Model on 4D pool)
    # =========================================================================
    for d in range(1, 5):
        df_d = df_res[df_res['day'] == d]
        res_str = calc_wilcoxon(df_d['r2_N_sub4'], df_d['r2_4_to_N'])
        combined_table.append({
            'Category': 'Day 4 Drift (Purple vs Red)',
            'Comparison': f"{d}D Purple vs {d}D Red",
            'Wilcoxon_Result': res_str
        })

    # --- Output ---
    df_combined = pd.DataFrame(combined_table)

    print("\nCOMBINED WILCOXON CROSS-SESSION RESULTS:")
    print(df_combined.to_string(index=False))

    save_path = output_path / 'wilcoxon_cross_session_all.csv'
    df_combined.to_csv(save_path, index=False)

    print(f"\n✓ Success! Single consolidated table saved to: {save_path}")


def generate_demographics_mannwhitney_table(
        demographics_csv='stat_tables/times_genders.csv',
        unified_csv='stat_tables/unified_model_performance_report.csv',
        exclude_mice=['J20', 'J21'],
        output_csv_dir='stat_tables'):
    """
    Анализирует зависимость R2 и MED от пола и времени суток.
    Жестко берет r2 и med из unified_model_performance_report.csv.
    Генерирует сводную таблицу сравнений по U-критерию Манна-Уитни и сохраняет в CSV.
    """
    output_path = Path(output_csv_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    path_demo = Path(demographics_csv)
    path_uni = Path(unified_csv)

    if not path_demo.exists() or not path_uni.exists():
        print(f"[ERROR] Не найден один из файлов:\n  - {path_demo}\n  - {path_uni}")
        return

    # 1. Загрузка данных
    df_demo = pd.read_csv(path_demo)
    df_uni = pd.read_csv(path_uni)

    # Приводим названия колонок к нижнему регистру для страховки
    df_demo.columns = [c.lower() for c in df_demo.columns]
    df_uni.columns = [c.lower() for c in df_uni.columns]

    # 2. Исключение дефектных мышей
    if exclude_mice:
        df_demo = df_demo[~df_demo['mouse'].isin(exclude_mice)]
        df_uni = df_uni[~df_uni['mouse'].isin(exclude_mice)]

    # 3. Объединение таблиц по мыши (берем только нужные колонки)
    df_merged = pd.merge(
        df_uni[['mouse', 'r2', 'med']],
        df_demo,
        on='mouse',
        how='inner'
    )

    if df_merged.empty:
        print("[ERROR] Не удалось объединить таблицы по колонке 'mouse'.")
        return

    print(f"\n{'=' * 85}")
    print("Генерация таблицы: Влияние пола (gender) и времени (time) на R2 и MED")
    print(f"{'=' * 85}")

    table_results = []

    def format_p_value(u, p):
        if p < 0.001:
            return f"U = {u:.1f}, p < 0.001"
        elif p < 0.01:
            return f"U = {u:.1f}, p < 0.01"
        elif p < 0.05:
            return f"U = {u:.1f}, p < 0.05"
        else:
            return f"U = {u:.1f}, p = {p:.3f}"

    def analyze_category(category_col, metric_col, metric_name):
        df_clean = df_merged.dropna(subset=[category_col, metric_col])
        unique_vals = df_clean[category_col].unique()

        if len(unique_vals) != 2:
            return

        group1, group2 = unique_vals[0], unique_vals[1]
        arr1 = df_clean[df_clean[category_col] == group1][metric_col].values
        arr2 = df_clean[df_clean[category_col] == group2][metric_col].values

        if len(arr1) < 3 or len(arr2) < 3:
            return

        # Расчет среднего и 1.96*SE для отображения в таблице
        stat1 = f"{np.mean(arr1):.2f} ± {1.96 * stats.sem(arr1):.2f}"
        stat2 = f"{np.mean(arr2):.2f} ± {1.96 * stats.sem(arr2):.2f}"

        # Тест Манна-Уитни
        u_stat, p_val = stats.mannwhitneyu(arr1, arr2, alternative='two-sided')

        table_results.append({
            'Фактор': category_col.capitalize(),
            'Метрика': metric_name,
            'Группа 1': f"{group1} (n={len(arr1)})",
            'Значение 1': stat1,
            'Группа 2': f"{group2} (n={len(arr2)})",
            'Значение 2': stat2,
            'Манн-Уитни': format_p_value(u_stat, p_val)
        })

    # --- Анализ по категориям ---
    for cat in ['gender', 'time']:
        if cat in df_merged.columns:
            analyze_category(cat, 'r2', "R2")
            analyze_category(cat, 'med', "MED")

    if not table_results:
        print("[!] Не удалось сформировать таблицу (проверьте наличие категорий).")
        return

    # 4. Формирование DataFrame и вывод
    df_results = pd.DataFrame(table_results)

    print("\n" + df_results.to_string(index=False) + "\n")

    # Сохранение в CSV
    save_path = output_path / 'mannwhitney_gender_time.csv'
    df_results.to_csv(save_path, index=False)
    print(f"✓ Таблица сравнений успешно сохранена: {save_path}")


if __name__ == "__main__":

    # find_minimal_sufficient_ensemble()

    # precall_df = get_models_dataframe(models_dir='models_LNOF_RF')
    # precall_df.to_csv('stat_tables/LNOF_prec_rec_RF.csv', index=False)

    # df_summary = aggregate_pt_metrics(models_dir='models_LNOF_TGNN_New', save_csv=True)

    # compare_metric_distributions_wilcoxon()
    # compare_gnn_vs_lr_wilcoxon()

    generate_cross_session_wilcoxon_tables()
    # generate_demographics_mannwhitney_table()

    # compare_gnn_vs_lr_wilcoxon(
    #     gnn_csv='stat_tables/shuffled_report.csv',
    #     lr_csv='stat_tables/unified_model_performance_report.csv',
    #     exclude_mice=['J20', 'J21'])


    # analyze_full_correlations_and_jaccard()
    # evaluate_top_vs_rest_models()
    # df_corr = analyze_performance_correlations('stat_tables/unified_model_performance_report.csv')

    # analyze_ablation_correlations()
    # analyze_ablation_correlations_advanced()

    # models_dir = "models_LNOF_Lasso"
    # models_path = Path(models_dir)
    # filtered_files = [f.name[:3] for f in models_path.glob("*.pkl")]

    # export_tracking_graph_to_csv(filtered_files)
    # generate_unified_summary_report(models_dir='models_LNOF_Lasso_shuffled',
    #                                     data_dir='LNOF_SynchronizedData',
    #                                     output_csv='shuffled_report.csv')

    # metrics = compare_metric_distributions()
    # print(metrics)
    # metrics = pd.DataFrame(metrics)
    # metrics.to_csv('LNOF_SynchronizedData/LNOF_stat.csv', index=False)

    # models_dir = Path("models_LNOF_Lasso")
    # filtered_files = [f.name for f in models_dir.glob("*.pkl")]
    #
    # all_metrics = []
    # np.random.seed(42)
    #
    # for model_name in filtered_files:
    #     # 1. Считаем случайный бейзлайн (Baseline)
    #     baseline_metrics = evaluate_unified_metrics(model_name, real_metrics=None)
    #     if baseline_metrics:
    #         all_metrics.append(baseline_metrics)
    #
    #     # 2. Считаем реальные веса (Model)
    #     # real_metrics = evaluate_unified_metrics(model_name, real_metrics=True)
    #     # if real_metrics:
    #     #     all_metrics.append(real_metrics)
    #
    # # Сохраняем всё в один датафрейм
    # precall_df = pd.DataFrame(all_metrics)
    # precall_df.to_csv('LNOF_SynchronizedData/LNOF_prec_rec_base_tau.csv', index=False)
    # print('\n[OK] Таблица метрик успешно сохранена!')