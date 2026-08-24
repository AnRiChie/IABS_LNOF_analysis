import os
import re
import joblib
import math
import random
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import MultiTaskLasso
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from shapely.geometry import Point
from itertools import product
from matplotlib.ticker import MaxNLocator, FuncFormatter, NullFormatter, MultipleLocator, LogLocator
from matplotlib.patches import Circle
from matplotlib.markers import MarkerStyle
import matplotlib.gridspec as gridspec
import seaborn as sns
import networkx as nx
from adjustText import adjust_text
import shapely.geometry as sg
from descartes import PolygonPatch
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from pathlib import Path
from matplotlib.patches import Patch
from scipy.stats import kendalltau, spearmanr, t
from linreg_behavior import load_data, train_lasso, train_test_split_temporal
from sklearn.metrics import r2_score
from tables_scripts import get_models_dataframe
from tables_scripts import analyze_neuron_tracking_graph
from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles
from scipy.stats import gaussian_kde, skew, zscore
from scipy.optimize import minimize
import itertools


C_PARAM = 0.5
TEST_SIZE = 0.25
MAX_ITER = 100


def plot_importance_distribution(mouse_id, session, models_dir='models_LNOF_Lasso', output_dir='plots'):
    """
    Строит распределение важности нейронов.
    """
    models_path = Path(models_dir)
    output_path = Path(output_dir)

    model_filename = f"{mouse_id}_{session}.pkl"
    model_file = models_path / model_filename

    if not model_file.exists():
        print(f"[ERROR] Model file {model_filename} not found.")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    try:
        data = joblib.load(model_file)
        model = data.get('model')

        # L2 норма по осям X и Y
        importance = np.linalg.norm(model.coef_, axis=0)
        non_zero_weights = importance[importance > 0]

        if len(non_zero_weights) == 0:
            print(f"[WARNING] Model {model_filename} has no non-zero features.")
            return

        plt.figure(figsize=(10, 6))
        plt.hist(non_zero_weights, bins=30, color='purple', alpha=0.7, edgecolor='black')

        plt.title(f"Distribution of Neuron Importance\n"
                  f"Mouse: {mouse_id} | Session: {session}", fontsize=14)
        plt.xlabel("Importance Value", fontsize=12)
        plt.ylabel("Number of Neurons", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        mean_val = np.mean(non_zero_weights)
        plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')

        total_neurons = len(importance)
        active_neurons = len(non_zero_weights)
        plt.annotate(f"Active: {active_neurons}/{total_neurons}\n(Sparsity: {active_neurons / total_neurons:.1%})",
                     xy=(0.95, 0.95), xycoords='axes fraction',
                     ha='right', va='top', bbox=dict(boxstyle="round", fc="white", alpha=0.8))

        plt.legend()
        save_name = f"importance_dist_{mouse_id}_{session}.png"
        plt.savefig(output_path / save_name, dpi=1000, bbox_inches='tight')
        plt.close()

        print(f"✓ Distribution plot saved: {save_name}")

    except Exception as e:
        print(f"[ERROR] Failed to plot importance for {model_filename}: {e}")


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


def plot_metrics_distribution(models_dir='models_LNOF_L2', output_dir='plots', importance_threshold=0.0):
    """
    Collects metrics from all models, filters out zero values,
    creates distribution plots, and saves them to a folder.
    """
    models_path = Path(models_dir)
    output_path = Path(output_dir)

    if not models_path.exists():
        print(f"[ERROR] Directory {models_dir} not found.")
        return

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Collect metrics
    data_list = []
    for file in models_path.glob("*.pkl"):
        if "best" in file.name:
            continue
        try:
            content = joblib.load(file)
            metrics = content.get('metrics', {})

            data_list.append({
                'f1': metrics.get('test_f1', 0),
                'precision': metrics.get('test_precision', 0),
                'recall': metrics.get('test_recall', 0)
            })
        except Exception as e:
            print(f"[WARNING] Could not read {file.name}: {e}")

    if not data_list:
        print("No model metrics found to plot.")
        return

    df = pd.DataFrame(data_list)

    # 2. Filtering non-trivial (non-zero) values
    metrics_to_plot = ['f1', 'precision', 'recall']

    # Set up the figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Distribution of Non-Zero Model Metrics', fontsize=16)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

    for i, metric in enumerate(metrics_to_plot):
        # Filter zero values
        non_zero_values = df[df[metric] > 0][metric]

        if non_zero_values.empty:
            axes[i].text(0.5, 0.5, f'No non-zero {metric} values',
                         ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{metric.upper()} (All Zeros)')
            continue

        # Plotting histogram + KDE (kernel density estimation) approximation
        axes[i].hist(non_zero_values, bins=15, color=colors[i], alpha=0.7, edgecolor='black', density=True)

        # Add labels and title
        axes[i].set_title(f'Distribution of {metric.upper()}')
        axes[i].set_xlabel('Score')
        axes[i].set_ylabel('Density')
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

        # Add mean line
        mean_val = non_zero_values.mean()
        axes[i].axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
        axes[i].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 3. Save the plot
    save_path = output_path / "metrics_distribution.png"
    plt.savefig(save_path)
    plt.close()

    print(f"\n[INFO] Distribution plots saved to: {save_path.absolute()}")


def plot_metrics_bars_by_day(models_dir='models_LNOF_Lasso', output_dir='plots'):
    """
    Строит столбчатые диаграммы для R2 и MAE(MED), раскрашивая столбики в зависимости от дня.
    """
    df = get_models_dataframe(models_dir)
    if df.empty:
        print("[ERROR] No valid data found to plot.")
        return

    # Убираем модели без определенного дня (опционально)
    df = df.dropna(subset=['day'])
    df['day'] = df['day'].astype(int)

    output_df = df.copy()

    # Создаем папку для графиков
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Настраиваем цвета для 4 дней (можно задать свои HEX коды)
    # Day 1: Голубой, Day 2: Оранжевый, Day 3: Зеленый, Day 4: Красный
    day_colors = {
        1: '#1f77b4',
        2: '#ff7f0e',
        3: '#2ca02c',
        4: '#d62728'
    }

    # Сопоставляем каждой строке её цвет
    colors = df['day'].map(day_colors).tolist()

    # --- График 1: R2 Score ---
    plt.figure(figsize=(14, 6))
    bars_r2 = plt.bar(df['model_name'], df['r2'], color=colors, edgecolor='black', alpha=0.8)

    plt.axhline(0, color='black', linewidth=1)  # Линия нуля (R2 бывает отрицательным)
    plt.title(r'Распределение коэффициента детерминации ($R^2$) по моделям и дням', fontsize=16)
    plt.ylabel(r'$R^2$ Score (Чем ближе к 1, тем лучше)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Добавляем легенду для цветов (чтобы не дублировать на каждый бар, делаем кастомную)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=day_colors[d], edgecolor='black', label=f'Day {d}')
                       for d in sorted(df['day'].unique())]
    plt.legend(handles=legend_elements, title="День эксперимента")

    save_path_r2 = output_path / 'R2_distribution_by_day.png'
    plt.savefig(save_path_r2, dpi=1000)
    plt.close()
    print(f"✓ График R2 сохранен: {save_path_r2}")

    # --- График 2: MAE / MED (Mean Euclidean Distance) ---
    plt.figure(figsize=(14, 6))
    bars_mae = plt.bar(df['model_name'], df['mae'], color=colors, edgecolor='black', alpha=0.8)

    plt.title('Распределение средней ошибки предсказания координат (MAE / MED)', fontsize=16)
    plt.ylabel('Ошибка в единицах координат (чем меньше, тем лучше)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.legend(handles=legend_elements, title="День эксперимента")

    save_path_mae = output_path / 'MAE_distribution_by_day.png'
    plt.savefig(save_path_mae, dpi=1000)
    plt.close()
    print(f"✓ График MAE сохранен: {save_path_mae}")

    return output_df


def plot_metrics_boxplots(models_dir='models_LNOF_Lasso', output_dir='plots'):
    """
    Строит Boxplot (ящики с усами) для сравнения распределения метрик между днями.
    Исправлено для совместимости с Seaborn >= 0.13.0
    """
    df = get_models_dataframe(models_dir)
    if df.empty: return
    df = df.dropna(subset=['day'])

    # ПРИНУДИТЕЛЬНО делаем дни строками, чтобы ключи словаря точно совпали
    df['day'] = df['day'].astype(str)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Ключи словаря теперь тоже строки ('1', '2' и т.д.)
    day_palette = {'1': '#1f77b4', '2': '#ff7f0e', '3': '#2ca02c', '4': '#d62728'}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- R2 Boxplot ---
    # Добавлены hue='day' и legend=False для совместимости с новым Seaborn
    sns.boxplot(data=df, x='day', y='r2', ax=axes[0],
                hue='day', palette=day_palette, legend=False)

    # Swarmplot (точки) оставляем серыми (color=".25")
    sns.swarmplot(data=df, x='day', y='r2', ax=axes[0], color=".25", size=6)

    axes[0].set_title(r'$R^2$ Score по дням эксперимента', fontsize=14)
    axes[0].set_xlabel('День (Day)', fontsize=12)
    axes[0].set_ylabel(r'$R^2$', fontsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)

    # --- MAE Boxplot ---
    sns.boxplot(data=df, x='day', y='mae', ax=axes[1],
                hue='day', palette=day_palette, legend=False)
    sns.swarmplot(data=df, x='day', y='mae', ax=axes[1], color=".25", size=6)

    axes[1].set_title('Ошибка декодирования координат (MED) по дням', fontsize=14)
    axes[1].set_xlabel('День (Day)', fontsize=12)
    axes[1].set_ylabel('Ошибка (Distance)', fontsize=12)
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path_box = output_path / 'Metrics_boxplots_by_day.png'
    plt.savefig(save_path_box, dpi=1000)
    plt.close()
    print(f"✓ Boxplot сохранен: {save_path_box}")


def plot_mobility_frequency(data_dir='LNOF_SynchronizedData', output_dir='plots', speed_threshold=5.0):
    """
    Строит частотное распределение сессий по проценту подвижности.
    Ось X: Процент времени в движении (speed >= 5).
    Ось Y: Процент сессий.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"[ERROR] Directory {data_dir} not found.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []

    # 1. Извлекаем данные по всем сессиям
    for file in data_path.glob("*.npz"):
        try:
            data = np.load(file, allow_pickle=True)

            # Достаем день эксперимента
            day_match = re.search(r'_([1-4])D', file.name)
            day = str(day_match.group(1)) if day_match else 'Unknown'

            # Достаем ID мыши
            mouse_match = re.search(r'(?:LNOF_)?([A-Za-z0-9]+)_', file.name)
            mouse = mouse_match.group(1) if mouse_match else 'Unknown'

            # Получаем скорость
            if 'speed' in data.files:
                speed = data['speed']
            elif 'x' in data.files and 'y' in data.files:
                dx = np.diff(data['x'], prepend=data['x'][0])
                dy = np.diff(data['y'], prepend=data['y'][0])
                speed = np.sqrt(dx ** 2 + dy ** 2)
            else:
                continue

            speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)
            total_frames = len(speed)
            if total_frames == 0:
                continue

            # Считаем процент движения (подвижность)
            # Условие из твоего запроса: покой - это speed < 5, значит движение >= 5
            moving_frames = np.sum(speed >= speed_threshold)
            percent_moving = (moving_frames / total_frames) * 100

            results.append({
                'session_name': file.name,
                'day': day,
                'percent_moving': percent_moving
            })

        except Exception as e:
            print(f"[ERROR] Failed to process {file.name}: {e}")

    df = pd.DataFrame(results)
    output_df = df.copy()

    if df.empty:
        print("[ERROR] No valid data extracted.")
        return

    # Оставляем только размеченные дни (от 1 до 4)
    df = df[df['day'].isin(['1', '2', '3', '4'])].copy()

    # 2. Отрисовка распределения
    plt.figure(figsize=(10, 6))

    day_palette = {'1': '#1f77b4', '2': '#ff7f0e', '3': '#2ca02c', '4': '#d62728'}

    # Строим гистограмму
    sns.histplot(
        data=df,
        x='percent_moving',
        hue='day',
        palette=day_palette,
        stat='percent',  # Ось Y будет в процентах от общего числа сессий
        multiple='stack',  # Столбики разных дней ставятся друг на друга
        bins=np.arange(0, 105, 5),  # Шаг корзины (bin) - 5% подвижности
        edgecolor='black',
        alpha=0.8
    )

    # Настраиваем визуал
    plt.title('Частотное распределение подвижности мышей', fontsize=16)
    plt.xlabel('Время в движении за сессию (%)', fontsize=12)
    plt.ylabel('Доля сессий (%)', fontsize=12)

    # Делаем сетку для читаемости
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xlim(0, 100)  # Ограничиваем ось X от 0 до 100%

    # Настраиваем легенду
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=day_palette[str(d)], edgecolor='black', label=f'Day {d}')
                       for d in sorted(df['day'].unique())]
    plt.legend(handles=legend_elements, title="День эксперимента")

    plt.tight_layout()
    save_path = output_path / 'Mobility_Frequency_Distribution.png'
    plt.savefig(save_path, dpi=1000)
    plt.close()

    print(f"✓ График распределения частот сохранен: {save_path}")

    return output_df


def plot_metrics_frequency_distribution(models_dir='models_LNOF_Lasso', output_dir='plots'):
    """
    Строит частотное распределение сессий по качеству моделей (R2 и MED).
    Ось X: Значение метрики (R2 или MED).
    Ось Y: Процент моделей (сессий).
    """
    df = get_models_dataframe(models_dir)

    if df.empty:
        print("[ERROR] No valid data extracted for distributions.")
        return

    # Очищаем данные от NaN и оставляем только размеченные дни (1-4)
    df = df.dropna(subset=['r2', 'mae', 'day'])
    df['day'] = df['day'].astype(str)
    df = df[df['day'].isin(['1', '2', '3', '4'])].copy()

    if df.empty:
        print("[ERROR] No valid data left after dropping NaNs and unknown days.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Цветовая палитра по дням
    day_palette = {'1': '#1f77b4', '2': '#ff7f0e', '3': '#2ca02c', '4': '#d62728'}

    # Создаем полотно с двумя графиками (1 строка, 2 колонки)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- 1. Гистограмма для R2 Score ---
    # Для R2 автоматический подбор корзин (bins='auto') работает хорошо.
    # Если есть жесткие отрицательные выбросы (модель не сошлась), ограничиваем ось X снизу нулем или небольшим минусом.
    sns.histplot(
        data=df,
        x='r2',
        hue='day',
        palette=day_palette,
        stat='percent',
        multiple='stack',
        bins=20,
        edgecolor='black',
        alpha=0.8,
        ax=axes[0]
    )
    axes[0].set_title(r'Частотное распределение качества декодирования ($R^2$)', fontsize=14)
    axes[0].set_xlabel(r'$R^2$ Score (Чем ближе к 1, тем лучше)', fontsize=12)
    axes[0].set_ylabel('Доля моделей (%)', fontsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)

    # Защита от сильных выбросов в R2 (если модель предсказывает хуже среднего)
    min_r2 = max(df['r2'].min() - 0.1, -1.0)  # Не показываем хвост хуже -1.0
    axes[0].set_xlim(left=min_r2, right=1.0)

    # --- 2. Гистограмма для MED (MAE) ---
    sns.histplot(
        data=df,
        x='mae',
        hue='day',
        palette=day_palette,
        stat='percent',
        multiple='stack',
        bins=20,
        edgecolor='black',
        alpha=0.8,
        ax=axes[1]
    )
    axes[1].set_title('Частотное распределение ошибки декодирования (MED)', fontsize=14)
    axes[1].set_xlabel('Ошибка в координатах (Distance)', fontsize=12)
    axes[1].set_ylabel('Доля моделей (%)', fontsize=12)
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)

    # Убираем дефолтные кривые легенды от Seaborn и делаем свою чистую
    axes[0].get_legend().remove()
    if axes[1].get_legend() is not None:
        axes[1].get_legend().remove()

    legend_elements = [Patch(facecolor=day_palette[str(d)], edgecolor='black', label=f'Day {d}')
                       for d in sorted(df['day'].unique())]

    # Размещаем общую легенду для всего Figure (справа или по центру)
    fig.legend(handles=legend_elements, title="День", loc='center right', bbox_to_anchor=(0.98, 0.5))

    # Корректируем отступы, чтобы графики не наезжали на легенду
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    save_path = output_path / 'Metrics_Frequency_Distribution.png'
    plt.savefig(save_path, dpi=1000)
    plt.close()

    print(f"✓ Графики распределения метрик сохранены: {save_path}")


def plot_chronological_tracking(tracking_dict, output_dir='plots'):
    """
    Визуализирует зависимость меры Жаккара от хронологического ребра сессий.

    Параметры:
    - tracking_dict: словарь, возвращаемый функцией analyze_neuron_tracking_graph.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Задаем хронологические ребра (последовательный переход по дням)
    chrono_edges = ['1D-2D', '2D-3D', '3D-4D']

    for mouse, metrics in tracking_dict.items():
        print(f"\n[ОТРИСОВКА] Создание графика трекинга для мыши {mouse}...")

        # 1. Сбор данных для хронологических ребер
        norm_scores = []
        unnorm_scores = []
        valid_chrono_edges = []

        for edge in chrono_edges:
            if edge in metrics:
                valid_chrono_edges.append(edge)
                norm_scores.append(metrics[edge]['normalized_score'])
                # Используем sig_jaccard как ненормированную меру для значимых нейронов
                unnorm_scores.append(metrics[edge]['sig_jaccard'])
            else:
                # Если ребра нет, ставим NaN, чтобы график прервался, но не сломался
                valid_chrono_edges.append(edge)
                norm_scores.append(np.nan)
                unnorm_scores.append(np.nan)

        if all(np.isnan(norm_scores)):
            print(f"[SKIP] Нет хронологических данных для {mouse}")
            continue

        # 2. Сбор данных для расчета глобального бейзлайна (среднее по ВСЕМ ребрам)
        all_norm = []
        all_unnorm = []
        for key, value in metrics.items():
            if '-' in key:  # Отбираем только ключи-ребра (игнорируем 'general')
                all_norm.append(value['normalized_score'])
                all_unnorm.append(value['sig_jaccard'])

        mean_norm = np.nanmean(all_norm)
        mean_unnorm = np.nanmean(all_unnorm)

        # 3. Отрисовка
        plt.figure(figsize=(9, 6))

        # Ось X - это индексы 0, 1, 2...
        x_indices = np.arange(len(valid_chrono_edges))

        # Рисуем ненормированную кривую (прозрачный серый)
        plt.plot(x_indices, unnorm_scores, color='gray', alpha=0.5, linewidth=3,
                 marker='o', markersize=8, label='Ненормированный Jaccard (sig_jaccard)')

        # Рисуем нормированную кривую (красный)
        plt.plot(x_indices, norm_scores, color='red', linewidth=3,
                 marker='o', markersize=8, label='Нормированная стабильность')

        # Рисуем пунктирные линии средних (Бейзлайны)
        plt.axhline(mean_unnorm, color='gray', linestyle='--', alpha=0.5, linewidth=2,
                    label=f'Среднее ненорм. (все ребра): {mean_unnorm:.2f}')

        plt.axhline(mean_norm, color='red', linestyle='--', alpha=0.7, linewidth=2,
                    label=f'Среднее норм. (все ребра): {mean_norm:.2f}')

        # 4. Настройка осей и оформления
        plt.title(f'Динамика стабильности нейронной репрезентации\nМышь: {mouse}', fontsize=14)
        plt.xlabel('Хронологический переход (Дни эксперимента)', fontsize=12)
        plt.ylabel('Мера Жаккара / Коэффициент стабильности', fontsize=12)

        # Настраиваем тики на оси X, чтобы там были надписи ребер
        plt.xticks(x_indices, valid_chrono_edges, fontsize=11)

        # Авто-масштаб уже работает по умолчанию, но можно добавить легкий запас снизу
        current_bottom, current_top = plt.ylim()
        plt.ylim(min(0, current_bottom), current_top * 1.1)

        plt.grid(axis='y', linestyle='--', alpha=0.6)

        # Легенда снаружи или в удобном месте
        plt.legend(loc='upper right', framealpha=0.9)

        plt.tight_layout()
        save_path = output_path / f'tracking_dynamics_{mouse}.png'
        plt.savefig(save_path, dpi=1000)
        plt.close()

        print(f"  ✓ График сохранен: {save_path}")


def plot_retrain_ablation_distribution(filenames, models_dir='models_LNOF_Lasso', data_dir='LNOF_SynchronizedData',
                                       output_dir='plots', test_size=0.25, c_param=.5, max_iter=100):
    """
    Переобучает модель с исключением по одному активному нейрону (Retraining Ablation).
    Строит распределение R2 и подписывает нейроны, чье удаление роняет R2 больше чем на 1 сигму.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ablation_results = []
    baselines = []

    for filename in filenames:
        # Убираем .pkl для красивых подписей и поиска файлов
        clean_name = filename.replace('.pkl', '')
        model_name = clean_name

        # Правильные пути для данных (.npz) и модели (.pkl)
        file_path = Path(data_dir) / f"LNOF_{clean_name}_aligned.npz"
        model_path = Path(models_dir) / filename

        if not file_path.exists():
            print(f"[SKIP] Файл данных не найден: {file_path}")
            continue
        if not model_path.exists():
            print(f"[SKIP] Файл модели не найден: {model_path}")
            continue

        print(f"\n{'=' * 50}\n[{model_name}] Загрузка и обучение бейзлайна...")

        features, labels = load_data(file_path)
        if features is None or labels is None:
            continue

        # 1. Сплит данных
        X_train, X_test, y_train, y_test = train_test_split_temporal(features, labels, test_size=test_size)

        # Загружаем сохраненный бейзлайн
        data = joblib.load(model_path)
        model = data.get('model')

        # Исправлено извлечение метрики из словаря
        base_r2 = data.get('metrics', {}).get('test_r2', np.nan)
        baselines.append({'model': model_name, 'r2': base_r2})

        # 2. Ищем активные нейроны в бейзлайне (чтобы не переобучать на пустышках)
        importance = np.linalg.norm(model.coef_, axis=0)
        active_indices = np.where(importance > 0)[0]
        active_features = features.columns[active_indices].tolist()

        print(f"[{model_name}] Активных нейронов: {len(active_features)}. Начинаем переобучение (Ablation)...")

        session_r2_scores = []

        # 3. Переобучаем модель, исключая по одному активному нейрону
        for feat in active_features:
            # Вырезаем колонку из сырых данных
            X_train_ablated = X_train.drop(columns=[feat])
            X_test_ablated = X_test.drop(columns=[feat])

            # Переобучаем честно с нуля
            _, _, _, metrics_ablated = train_lasso(
                X_train_ablated, y_train, X_test_ablated, y_test,
                C=c_param, max_iter=max_iter
            )

            abl_r2 = metrics_ablated.get('test_r2', np.nan)
            session_r2_scores.append(abl_r2)

            # Достаем номер нейрона для подписи
            neuron_id = re.search(r'\d+', feat).group() if re.search(r'\d+', feat) else feat

            ablation_results.append({
                'model': model_name,
                'neuron_id': neuron_id,
                'r2': abl_r2
            })

        # Считаем порог (сигму) для конкретной сессии, чтобы потом подписать точки
        mean_r2 = np.mean(session_r2_scores)
        std_r2 = np.std(session_r2_scores)
        threshold = mean_r2 - std_r2  # Все, что ниже (хуже) этого порога - критичные нейроны

        print(
            f"[{model_name}] Baseline R2: {base_r2:.4f} | Средний Ablation R2: {mean_r2:.4f} | Порог (1 Сигма): {threshold:.4f}")

    if not ablation_results:
        print("[WARNING] Нет данных для построения графика.")
        return

    # --- Подготовка к визуализации ---
    df_ablation = pd.DataFrame(ablation_results)
    df_base = pd.DataFrame(baselines)

    unique_models = df_ablation['model'].unique()

    plt.figure(figsize=(max(10, len(unique_models) * 2), 7))

    # 1. Рисуем облако ablation-моделей (черные точки)
    sns.stripplot(
        data=df_ablation,
        x='model',
        y='r2',
        color='black',
        alpha=0.5,
        jitter=0.15,  # Разброс по X для читаемости
        size=5,
        label='Ablation (переобучение без 1 нейрона)'
    )

    # 2. Рисуем Baseline (красная точка БЕЗ обводки, как заказывали)
    sns.scatterplot(
        data=df_base,
        x='model',
        y='r2',
        color='red',
        s=40,
        edgecolor='none',
        zorder=10,
        label='Baseline (все нейроны)'
    )

    # 3. Подписываем "супер-якоря" (те, что роняют R2 больше чем на 1 сигму)
    model_to_x = {model: i for i, model in enumerate(unique_models)}

    for model_name in unique_models:
        subset = df_ablation[df_ablation['model'] == model_name]
        mean_r2 = subset['r2'].mean()
        std_r2 = subset['r2'].std()

        # Находим точки-выбросы
        outliers = subset[subset['r2'] < (mean_r2 - std_r2)]
        x_base = model_to_x[model_name]

        # for _, row in outliers.iterrows():
        #     # Добавляем небольшое смещение по X и Y, чтобы текст не сливался с точкой
        #     plt.text(
        #         x_base + 0.15, row['r2'],
        #         f"#{row['neuron_id']}",
        #         fontsize=9,
        #         color='darkred',
        #         ha='left', va='center',
        #         bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1)
        #     )

    # --- Оформление графика ---
    plt.title(
        'Retraining Ablation: Дисперсия $R^2$ при переобучении без одиночных нейронов\n(Подписаны нейроны, роняющие $R^2$ > $1\sigma$)',
        fontsize=14)
    plt.xlabel('Модель (Сессия)', fontsize=12)
    plt.ylabel(r'$R^2$', fontsize=12)

    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xticks(rotation=45, ha='right')

    # Чистим дубликаты легенды
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')

    plt.tight_layout()
    save_path = output_path / 'retrain_ablation_1.png'
    plt.savefig(save_path, dpi=1000)
    plt.close()

    print(f"✓ График Retraining Ablation успешно сохранен: {save_path}")


def format_session_label(session_name):
    """
    Разбирает названия (например, 'J24_3D' или 'JN_TD')
    в формат 'Мышь J[N] день N'.
    """
    try:
        parts = session_name.split('_')
        mouse = parts[0]  # J24, JN и т.д.
        # Убираем букву 'D', если она есть в конце (3D -> 3, TD -> T)
        day = parts[1].replace('D', '') if len(parts) > 1 else "Unknown"
        return f"Мышь {mouse}\nдень {day}"
    except Exception:
        return session_name


def plot_ablation_broken_axis_smart_text(model_sessions, sigma_threshold=3.0,
                                         csv_path='stat_tables/full_ablation_stats.csv',
                                         output_dir='plots'):
    """
    Строит график дисперсии R2 при лезировании нейронов (Broken Axis).
    Стиль: Научный минимализм. Кастомная легенда, увеличенные подписи выбросов.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[ERROR] Файл не найден: {csv_path}")
        return

    df = df[df['model_session'].isin(model_sessions)].copy()
    if df.empty:
        print("[WARNING] Данные для указанных моделей не найдены.")
        return

    x_order = df['model_session'].unique().tolist()
    model_to_x = {model: i for i, model in enumerate(x_order)}

    # Преобразование данных
    id_vars = ['model_session', 'baseline_r2', 'mean_r2', 'std_r2']
    abl_cols = [col for col in df.columns if col.startswith('ablation_neuron_')]

    df_long = pd.melt(df, id_vars=id_vars, value_vars=abl_cols, var_name='neuron_col', value_name='r2')
    df_long = df_long.dropna(subset=['r2']).copy()

    df_long['neuron_id'] = df_long['neuron_col'].apply(
        lambda x: re.search(r'\d+', x).group() if re.search(r'\d+', x) else x
    )

    df_base = df[['model_session', 'baseline_r2']].drop_duplicates()

    base_min, base_max = df_base['baseline_r2'].min(), df_base['baseline_r2'].max()
    abl_min, abl_max = df_long['r2'].min(), df_long['r2'].max()

    gap_exists = (base_min - abl_max) > 0.01

    # --- НАСТРОЙКА СЕТКИ ---
    fig_width = max(8, len(x_order) * 7)
    fig_height = 9

    c_text = '#2C3E50'

    if gap_exists:
        fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1, sharex=True,
            figsize=(fig_width, fig_height),
            gridspec_kw={'height_ratios': [1, 4]}
        )
        fig.subplots_adjust(hspace=0.05)
        axes = [ax_top, ax_bottom]

        ax_top.spines['bottom'].set_visible(False)
        ax_bottom.spines['top'].set_visible(False)
        ax_top.tick_params(labeltop=False, bottom=False)
        ax_bottom.xaxis.tick_bottom()
    else:
        fig, ax_bottom = plt.subplots(figsize=(fig_width, fig_height))
        axes = [ax_bottom]
        ax_top = ax_bottom

    np.random.seed(42)
    texts_to_adjust = []

    for model in x_order:
        subset = df_long[df_long['model_session'] == model].copy()
        x_base = model_to_x[model]

        # Растягиваем точки шире (до 0.45)
        jitters = np.random.uniform(-0.45, 0.45, size=len(subset))
        subset['x_jit'] = x_base + jitters

        # Черные точки БЕЗ обводки
        ax_bottom.scatter(subset['x_jit'], subset['r2'], color='black', alpha=0.5, s=20, edgecolors='none', zorder=3)

        # Красная точка (Baseline)
        base_val = df_base[df_base['model_session'] == model]['baseline_r2'].values[0]
        ax_top.scatter(x_base, base_val, color='#D62728', s=90, edgecolors='none', zorder=10)

        mean_r2 = subset['mean_r2'].iloc[0]
        std_r2 = subset['std_r2'].iloc[0]
        threshold = mean_r2 - (sigma_threshold * std_r2)

        outliers = subset[subset['r2'] < threshold]

        for _, row in outliers.iterrows():
            # УВЕЛИЧЕННЫЙ ШРИФТ ДЛЯ ВЫБРОСОВ (fontsize=14)
            txt = ax_bottom.text(
                row['x_jit'], row['r2'],
                f" {row['neuron_id']} ",
                fontsize=14,
                fontweight='bold',
                color=c_text,
                ha='center', va='center',
                zorder=5
            )
            texts_to_adjust.append(txt)

    if gap_exists:
        margin_base = (base_max - base_min) * 0.5 if base_max != base_min else 0.02
        margin_abl = (abl_max - abl_min) * 0.1

        ax_top.set_ylim(base_min - margin_base, base_max + margin_base)
        ax_bottom.set_ylim(abl_min - margin_abl, abl_max + margin_abl)

        # Засечки разрыва
        d = .012
        kwargs = dict(transform=ax_top.transAxes, color=c_text, clip_on=False, lw=1.5)
        ax_top.plot((-d, +d), (-d, +d), **kwargs)
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs.update(transform=ax_bottom.transAxes)
        ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # Разводка текста (adjust_text)
    if adjust_text and texts_to_adjust:
        adjust_text(
            texts_to_adjust,
            ax=ax_bottom,
            expand_points=(1.5, 1.5),
            expand_text=(1.2, 1.2),
            force_text=(0.1, 0.2),
            arrowprops=dict(arrowstyle="-", color='gray', lw=0.8, alpha=0.6, shrinkA=5, shrinkB=5)
        )

    # --- ОФОРМЛЕНИЕ МИНИМАЛИЗМА ---
    # Функция format_session_label должна быть определена в вашем коде (или можно заменить на x_order)
    try:
        formatted_labels = [format_session_label(m) for m in x_order]
    except NameError:
        formatted_labels = x_order

    ax_bottom.set_xticks(range(len(x_order)))
    ax_bottom.set_xticklabels(formatted_labels, rotation=0, ha='center', fontsize=14, color=c_text)

    for ax in axes:
        ax.grid(axis='y', linestyle=':', alpha=0.4, color='#BDC3C7', zorder=0)
        ax.tick_params(axis='y', labelsize=12, colors=c_text, width=1.2)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['right'].set_visible(False)
        if ax == ax_top:
            ax.spines['top'].set_visible(False)
        if ax == ax_bottom:
            ax.spines['bottom'].set_linewidth(1.2)
        [s.set_color(c_text) for s in ax.spines.values()]

    # --- ПОДПИСЬ Y (ПРИДВИНУТАЯ И ЛАКОНИЧНАЯ) ---
    # Параметр x=0.06 двигает подпись вплотную к шкале (варьируйте от 0.04 до 0.08 при необходимости)
    fig.supylabel(r'$R^2$', fontsize=16, x=0.06, color=c_text)

    # --- КАСТОМНАЯ ЛЕГЕНДА ---
    legend_elements = [
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#D62728', markersize=10, label='Исходная модель'),
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', alpha=0.5, markersize=8,
               label='Удален один нейрон'),
        # Использование TeX строки как маркера:
        mlines.Line2D([0], [0], marker='$\mathbb{N}$', color='w', markerfacecolor=c_text, markeredgecolor=c_text, markersize=12,
               label=f'Отклонение > {sigma_threshold}$\\sigma$')
    ]

    # Легенда внутри верхнего графика справа, без рамки
    ax_top.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=12, labelcolor=c_text)

    plt.subplots_adjust(left=0.1, right=0.98, bottom=0.1)

    safe_names = [re.sub(r'\W+', '', m) for m in model_sessions[:3]]
    prefix = "_".join(safe_names)
    if len(model_sessions) > 3:
        prefix += "_and_others"

    save_path = output_path / f'ablation_{prefix}_sigma{sigma_threshold}.png'

    plt.savefig(save_path, dpi=1000, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Эстетичный график успешно сохранен: {save_path}")


def plot_selective_neuron_graphs(sessions, selectivity='linreg', corr_threshold=0.15,
                                 models_dir='models_LNOF_Lasso', data_dir='LNOF_SynchronizedData',
                                 csv_path='LNOF_SynchronizedData/LNOF_database.csv', output_dir='plots'):
    """
    Строит графы нейронных взаимодействий. Фоновые нейроны серые, селективные выделены цветом.

    Параметры:
    - sessions: список имен сессий, например ['J30_1D', 'J61_2D']
    - selectivity: 'intense' (из базы CSV) или 'linreg' (значимые фичи из Lasso)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Загрузка базы данных intense, если выбрана
    if selectivity == 'intense':
        if not Path(csv_path).exists():
            print(f"[ERROR] База данных не найдена: {csv_path}")
            return
        df_db = pd.read_csv(csv_path)

        # Заранее собираем топ уникальных фичей для палитры
        top_features = df_db['feature'].value_counts().index.tolist()
        base_cmap = plt.get_cmap('tab10')  # Хорошая палитра для категорий
        feature_colors = {feat: base_cmap(i % 10) for i, feat in enumerate(top_features)}

    for session in sessions:
        mouse = session[:3]
        day = session[4:6]

        print(f"\n{'=' * 50}\n[{session}] Обработка графа (Режим: {selectivity})...")

        # 2. Загрузка данных активности (.npz)
        npz_file = Path(data_dir) / f"LNOF_{session}_aligned.npz"
        if not npz_file.exists():
            print(f"[SKIP] Нет файла данных: {npz_file}")
            continue

        data = np.load(npz_file, allow_pickle=True)
        # Получаем транспонированную матрицу: (n_timepoints, n_neurons)
        calcium_raw = data['Calcium'].T
        n_neurons = calcium_raw.shape[1]

        # --- ОПТИМИЗАЦИЯ КОРРЕЛЯЦИЙ ---
        # Вместо вложенных циклов используем векторизованный расчет Спирмена (в 100 раз быстрее)
        print("  > Расчет матрицы корреляций Спирмена...")
        df_calcium = pd.DataFrame(calcium_raw)
        corr_mtx = df_calcium.corr(method='spearman').values
        # Зануляем диагональ (нейрон сам с собой)
        np.fill_diagonal(corr_mtx, 0)

        # 3. Отбор селективных нейронов и их раскраска
        selective_nodes = {}

        if selectivity == 'intense':
            df_sess = df_db[(df_db['mouse'] == mouse) & (df_db['session'] == day)]
            # Группируем фичи для каждого нейрона
            for neuron_idx, group in df_sess.groupby('neuron_idx'):
                features = group['feature'].tolist()

                # Смешивание цветов для мульти-селективных нейронов
                mixed_rgb = np.mean([feature_colors[f][:3] for f in features], axis=0)
                hex_color = mcolors.to_hex(mixed_rgb)

                selective_nodes[int(neuron_idx)] = hex_color

        elif selectivity == 'linreg':
            model_path = Path(models_dir) / f"{session}.pkl"
            if not model_path.exists():
                print(f"[SKIP] Модель Lasso не найдена: {model_path}")
                continue

            model_data = joblib.load(model_path)
            importance = np.linalg.norm(model_data['model'].coef_, axis=0)
            active_indices = np.where(importance > 0)[0]
            feature_names = model_data['feature_names']

            for idx in active_indices:
                name = feature_names[idx]
                match = re.search(r'\d+', name)
                if match:
                    neuron_id = int(match.group())
                    selective_nodes[neuron_id] = 'red'  # В Lasso все селективны на координату (красные)

        print(f"  > Найдено селективных нейронов: {len(selective_nodes)} из {n_neurons}")

        # 4. Построение графа
        G = nx.Graph()
        G.add_nodes_from(range(n_neurons))

        # Добавляем ребра по порогу корреляции (только абсолютные значения)
        edges_to_add = []
        for i in range(n_neurons):
            for j in range(i + 1, n_neurons):
                weight = corr_mtx[i, j]
                if abs(weight) >= corr_threshold:
                    edges_to_add.append((i, j, weight))
        G.add_weighted_edges_from(edges_to_add)

        # 5. Подготовка визуала (Узлы)
        node_colors = []
        node_sizes = []
        labels = {}

        for node in G.nodes():
            if node in selective_nodes:
                node_colors.append(selective_nodes[node])
                node_sizes.append(400)  # Селективные - большие
                labels[node] = str(node)  # Подписываем только их
            else:
                node_colors.append('lightgray')
                node_sizes.append(50)  # Фоновые - мелкие
                # labels[node] останется пустым

        # 6. Подготовка визуала (Ребра)
        edge_colors = []
        edge_widths = []
        edge_alphas = []

        for u, v, d in G.edges(data=True):
            weight = abs(d['weight'])
            # Если оба нейрона фоновые - ребро бледное и тонкое
            if u not in selective_nodes and v not in selective_nodes:
                edge_colors.append('gainsboro')
                edge_widths.append(weight * 2)
                edge_alphas.append(0.3)
            else:
                # Если хотя бы один селективный - выделяем связь
                edge_colors.append('black')  # Можно использовать cmap
                edge_widths.append(weight * 5)
                edge_alphas.append(0.8)

        # 7. Отрисовка
        print("  > Отрисовка графа...")
        plt.figure(figsize=(16, 12))

        # Раскладка (пружинный алгоритм). Для больших графов seed важен для повторяемости
        pos = nx.spring_layout(G, k=0.15, seed=42)

        # Рисуем фоновые ребра
        bg_edges = [e for i, e in enumerate(G.edges()) if edge_alphas[i] == 0.3]
        nx.draw_networkx_edges(G, pos, edgelist=bg_edges,
                               width=[edge_widths[i] for i, e in enumerate(G.edges()) if edge_alphas[i] == 0.3],
                               edge_color='gainsboro', alpha=0.3)

        # Рисуем "активные" ребра поверх
        fg_edges = [e for i, e in enumerate(G.edges()) if edge_alphas[i] == 0.8]
        nx.draw_networkx_edges(G, pos, edgelist=fg_edges,
                               width=[edge_widths[i] for i, e in enumerate(G.edges()) if edge_alphas[i] == 0.8],
                               edge_color='dimgray', alpha=0.8)

        # Рисуем узлы
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors='black', linewidths=0.5)

        # Добавляем метки (только для селективных)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold', font_color='white')

        # 8. Легенда
        legend_elements = []
        if selectivity == 'intense':
            for feat, color in feature_colors.items():
                legend_elements.append(
                    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=12, label=feat))
            legend_elements.append(mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=12,
                                              label='Mixed Selectivity'))
        elif selectivity == 'linreg':
            legend_elements.append(mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=12,
                                              label='Spatial/Coordinate Selectivity (Lasso)'))

        legend_elements.append(mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=8,
                                          label='Non-selective (Background)'))

        plt.legend(handles=legend_elements, loc='upper right', framealpha=0.9, title="Neuron Selectivity")

        plt.title(
            f"Neural Network Topology: {session} (Selectivity: {selectivity.upper()})\nEdges corr $\geq$ {corr_threshold}",
            fontsize=16)
        plt.axis('off')
        plt.tight_layout()

        save_name = output_path / f"graph_{session}_{selectivity}.png"
        plt.savefig(save_name, dpi=1000, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"  ✓ Граф сохранен: {save_name}")


def plot_metrics_sigma_distribution(csv_path='stat_tables/unified_model_performance_report.csv',
                                    output_dir='plots'):
    """
    Строит распределение R2 и MED, выделяет 1, 2, 3 сигмы.
    Подписывает сессии-выбросы (>2 сигм) на графике и выводит в консоль (условие ИЛИ).
    """
    path = Path(csv_path)
    if not path.exists():
        print(f"[ERROR] Сводный файл не найден: {path}")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(path)

    # Очищаем от NaN для корректной статистики
    df_clean = df.dropna(subset=['r2', 'med']).copy()

    if df_clean.empty:
        print("[ERROR] Нет валидных данных R2 и MED для анализа.")
        return

    # --- 1. РАСЧЕТ СТАТИСТИКИ ---
    metrics = {
        'r2': {'data': df_clean['r2'].values, 'name': 'R2 Score', 'color': '#1f77b4'},
        'med': {'data': df_clean['med'].values, 'name': 'MED (Error)', 'color': '#ff7f0e'}
    }

    for m in metrics:
        metrics[m]['mean'] = np.mean(metrics[m]['data'])
        metrics[m]['std'] = np.std(metrics[m]['data'])

        # Считаем Z-score (насколько сигм значение отклоняется от среднего по модулю)
        df_clean[f'{m}_z'] = np.abs((df_clean[m] - metrics[m]['mean']) / metrics[m]['std'])

    # --- 2. ПОИСК АНОМАЛИЙ (УСЛОВИЕ "ИЛИ") ---
    # Сессия аномальна, если хотя бы одна из метрик (R2 или MED) улетела дальше порога
    outliers_2s = df_clean[(df_clean['r2_z'] > 2) | (df_clean['med_z'] > 2)]
    outliers_3s = df_clean[(df_clean['r2_z'] > 3) | (df_clean['med_z'] > 3)]

    # --- ВЫВОД В КОНСОЛЬ ---
    print(f"\n{'=' * 60}")
    print(f"АНАЛИЗ АНОМАЛИЙ РАСПРЕДЕЛЕНИЯ (Контроль Качества)")
    print(f"{'=' * 60}")
    print(f"Всего проанализировано сессий: {len(df_clean)}")

    print(f"\n[!!!] СЕССИИ > 3 СИГМ (Критические выбросы) [!!!]")
    if outliers_3s.empty:
        print("  ✓ Критичных выбросов нет.")
    else:
        for _, row in outliers_3s.iterrows():
            reasons = []
            if row['r2_z'] > 3: reasons.append(f"R2={row['r2']:.3f} ({row['r2_z']:.1f}σ)")
            if row['med_z'] > 3: reasons.append(f"MED={row['med']:.3f} ({row['med_z']:.1f}σ)")
            print(f"  > {row['model_name']}: {', '.join(reasons)}")

    print(f"\n[!] СЕССИИ > 2 СИГМ (Подозрительные) [!]")
    if outliers_2s.empty:
        print("  ✓ Подозрительных сессий нет.")
    else:
        for _, row in outliers_2s.iterrows():
            # Если она уже выведена в 3 сигмах, пропускаем, чтобы не дублировать
            if row['model_name'] in outliers_3s['model_name'].values:
                continue
            reasons = []
            if row['r2_z'] > 2: reasons.append(f"R2={row['r2']:.3f} ({row['r2_z']:.1f}σ)")
            if row['med_z'] > 2: reasons.append(f"MED={row['med']:.3f} ({row['med_z']:.1f}σ)")
            print(f"  > {row['model_name']}: {', '.join(reasons)}")
    print(f"{'=' * 60}\n")

    # --- 3. ОТРИСОВКА ГРАФИКОВ ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for idx, m in enumerate(['r2', 'med']):
        ax = axes[idx]
        mean = metrics[m]['mean']
        std = metrics[m]['std']
        color = metrics[m]['color']

        # Строим гистограмму с кривой плотности KDE
        sns.histplot(df_clean, x=m, kde=True, ax=ax, color=color, alpha=0.4, edgecolor='black')

        # Высота графика (нужна для красивой расстановки текста)
        y_max = ax.get_ylim()[1]

        # Отрисовка линий сигм
        sigmas = {
            1: ('green', 0.5, '--', r'$1\sigma$'),
            2: ('orange', 0.8, '-.', r'$2\sigma$'),
            3: ('red', 1.0, ':', r'$3\sigma$')
        }

        # Центральная линия (Mean)
        ax.axvline(mean, color='black', linestyle='-', linewidth=2, label=f'Mean: {mean:.3f}')

        for mult, (s_color, s_alpha, s_ls, s_label) in sigmas.items():
            # Линия вправо (+N сигм)
            ax.axvline(mean + mult * std, color=s_color, linestyle=s_ls, alpha=s_alpha, linewidth=2,
                       label=f'{s_label} ($\pm${mult * std:.2f})')
            # Линия влево (-N сигм)
            ax.axvline(mean - mult * std, color=s_color, linestyle=s_ls, alpha=s_alpha, linewidth=2)

        # Подпись аномальных сессий (Топ > 2 Сигм для ТЕКУЩЕЙ метрики)
        outliers_current_metric = df_clean[df_clean[f'{m}_z'] > 2]

        # Чтобы текст не накладывался друг на друга, вводим джиттер по высоте
        y_offset_step = y_max * 0.05
        current_y = y_max * 0.1

        for _, row in outliers_current_metric.iterrows():
            # Рисуем красную точку на оси X для выброса
            ax.scatter(row[m], 0, color='red', s=100, zorder=5, edgecolors='black')

            # Подписываем модель
            ax.text(row[m], current_y, f"{row['model_name']}",
                    color='darkred', fontsize=10, fontweight='bold',
                    ha='center', va='bottom', rotation=90,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

            # Смещаем высоту для следующей подписи
            current_y += y_offset_step
            if current_y > y_max * 0.5:
                current_y = y_max * 0.1  # Сброс высоты, если улетели слишком высоко

        ax.set_title(f"Распределение {metrics[m]['name']}\n(Mean={mean:.3f}, Std={std:.3f})", fontsize=14)
        ax.set_xlabel(metrics[m]['name'], fontsize=12)
        ax.set_ylabel('Количество сессий', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        # Легенду выводим только уникальную (убираем дубликаты линий + и -)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best')

    plt.suptitle("Анализ выбросов: Отклонение сессий от среднего на $1\sigma, 2\sigma, 3\sigma$", fontsize=16, y=1.05)
    plt.tight_layout()

    save_path = output_path / 'sigma_distribution_outliers.png'
    plt.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.close()

    print(f"✓ График распределений сохранен: {save_path}")


def plot_global_euler_diagram(sessions=None,
                              sets_to_plot=('intense', 'linreg', 'zcrit'),
                              color_scheme='cmyk',
                              ablation_csv='stat_tables/full_ablation_stats.csv',
                              database_csv='LNOF_SynchronizedData/LNOF_database.csv',
                              models_dir='models_LNOF_Lasso',
                              output_dir='plots'):
    """
    Визуализирует пересечение глобальных пулов нейронов без контуров.
    Имитирует физическое смешивание цветов (CMYK уходит в черный, RGB - в белый).
    Фон строго белый, цвета текста зависят от выбранной цветовой схемы.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    excluded_mice = ['J20', 'J21']

    try:
        df_abl = pd.read_csv(ablation_csv)
        df_db = pd.read_csv(database_csv)
    except FileNotFoundError as e:
        print(f"[ERROR] Не найден файл базы данных: {e}")
        return

    all_sessions = df_abl['model_session'].unique().tolist()
    if sessions is None:
        sessions_to_process = all_sessions
    else:
        sessions_to_process = [s for s in sessions if s in all_sessions]

    sessions_to_process = [s for s in sessions_to_process if s[:3] not in excluded_mice]

    print(f"\n{'=' * 70}")
    print(f"Сборка глобального графа пересечений (Сессий: {len(sessions_to_process)})")
    print(f"Исключены: {excluded_mice}")
    print(f"{'=' * 70}")

    global_sets = {
        'intense': set(),
        'linreg': set(),
        'zcrit': set(),
        'crit': set()
    }

    # 1. Итерация по сессиям
    for sess in sessions_to_process:
        mouse = sess[:3]
        day = sess[4:6]

        # --- INTENSE ---
        if 'intense' in sets_to_plot:
            df_int = df_db[(df_db['mouse'] == mouse) & (df_db['session'] == day) & (df_db['feature'] == 'place')]
            for idx in df_int['neuron_idx'].unique():
                global_sets['intense'].add(f"{sess}_neuron_{int(idx)}")

        # --- LINREG, CRIT & ZCRIT ---
        model_path = Path(models_dir) / f"{sess}.pkl"
        if not model_path.exists():
            continue

        data = joblib.load(model_path)
        model = data.get('model')
        if not hasattr(model, 'coef_'):
            continue

        feats = np.array(data.get('feature_names'))
        weights = np.linalg.norm(model.coef_, axis=0)

        # LINREG
        if 'linreg' in sets_to_plot:
            for f in feats[weights > 0]:
                global_sets['linreg'].add(f"{sess}_{f}")

        # CRIT & ZCRIT
        if 'crit' in sets_to_plot or 'zcrit' in sets_to_plot:
            abl_row = df_abl[df_abl['model_session'] == sess].iloc[0]
            base_r2 = abl_row['baseline_r2']
            abl_r2_vals = np.array([abl_row.get(f'ablation_{f}', np.nan) for f in feats])

            # Расчет падения качества (Δ)
            crit = base_r2 - abl_r2_vals
            valid_crit = crit[~np.isnan(crit)]

            # Базовый CRIT: роняют метрику (Δ > 0)
            if 'crit' in sets_to_plot:
                for f in feats[~np.isnan(crit) & (crit > 0)]:
                    global_sets['crit'].add(f"{sess}_{f}")

            # Z-CRIT: стандартизованное падение
            if 'zcrit' in sets_to_plot:
                if len(valid_crit) > 0:
                    std_c = np.std(valid_crit)
                    mean_c = np.mean(valid_crit)
                    if std_c > 0:
                        zcrit = (crit - mean_c) / std_c
                        for f in feats[~np.isnan(zcrit) & (zcrit > 0)]:
                            global_sets['zcrit'].add(f"{sess}_{f}")

    # 2. Подготовка данных
    subsets = [global_sets[k] for k in sets_to_plot if k in global_sets]

    label_map = {
        'intense': 'Клетки места',
        'linreg': 'Участвующие',
        'zcrit': 'Важные', #r'z-$\delta$',
        'crit': r'$\delta$'
    }
    labels = [label_map.get(k, k.upper()) for k in sets_to_plot if k in global_sets]

    if len(subsets) < 2:
        print("[ERROR] Недостаточно множеств для пересечения (минимум 2).")
        return

    # 3. Визуализация
    fig, ax = plt.subplots(figsize=(14, 14))

    # Жестко белый фон
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Словари идеального смешивания цветов для 7 областей
    color_maps = {
        'cmyk': {
            '100': '#00FFFF', '010': '#FF00FF', '001': '#FFFF00',  # Cyan, Magenta, Yellow
            '110': '#0000FF', '101': '#00FF00', '011': '#FF0000',  # Синий, Зеленый, Красный
            '111': '#000000'  # Черный центр
        },
        'rgb': {
            '100': '#FF0000', '010': '#00FF00', '001': '#0000FF',  # Red, Green, Blue
            '110': '#FFFF00', '101': '#FF00FF', '011': '#00FFFF',  # Желтый, Маджента, Циан
            '111': '#FFFFFF'  # Белый центр
        },
        'bi': {
            '100': '#D60270', '010': '#9B4F96', '001': '#0038A8',  # Pink, Purple, Blue
            '110': '#B82983', '101': '#6B1D8C', '011': '#4D439F',  # Промежуточные градиенты
            '111': '#5A328C'  # Глубокий фиолетовый
        }
    }

    scheme = color_scheme.lower()
    selected_map = color_maps.get(scheme, color_maps['cmyk'])

    # Рисуем базовый граф
    if len(subsets) == 3:
        v = venn3(subsets, set_labels=labels, ax=ax)
    else:
        v = venn2(subsets, set_labels=labels, ax=ax)

    if v:
        # Перезакраска секторов
        for region_id, color in selected_map.items():
            if len(subsets) == 2 and len(region_id) == 3:
                continue

            patch = v.get_patch_by_id(region_id)
            if patch:
                patch.set_color(color)
                patch.set_alpha(1.0)  # Полная плотность цвета, никаких полупрозрачностей
                patch.set_edgecolor('none')  # Строго без обводки

        # Настройка ВНЕШНИХ текстовых подписей (названия сетов)
        for text in v.set_labels:
            if text:
                text.set_fontsize(30)
                text.set_fontweight('bold')
                text.set_color('black')  # Внешние подписи всегда черные, т.к. фон белый

        # Настройка ВНУТРЕННИХ текстовых подписей (числа нейронов)
        for region_id in selected_map.keys():
            if len(subsets) == 2 and len(region_id) == 3:
                continue

            label_txt = v.get_label_by_id(region_id)
            if label_txt:
                label_txt.set_fontsize(25)
                label_txt.set_fontweight('bold')

                # Логика контраста текста внутри кругов:
                if scheme == 'cmyk':
                    # Белый текст везде, кроме желтого круга (001)
                    label_txt.set_color('black' if region_id == '001' else 'white')
                elif scheme == 'rgb':
                    # В RGB белый текст только на синем (001), на остальных - черный (вкл. белый центр)
                    label_txt.set_color('white' if region_id == '001' else 'black')
                else:  # bi
                    # Для глубоких цветов BI-флага везде хорошо смотрится белый
                    label_txt.set_color('white')

    ax.axis('off')

    file_prefix = f"global_euler_{'_'.join(sets_to_plot)}_{scheme}"

    fig.savefig(output_path / f"{file_prefix}.svg", format='svg', bbox_inches='tight', facecolor='white')
    fig.savefig(output_path / f"{file_prefix}.png", format='png', dpi=1000, bbox_inches='tight', facecolor='white')

    plt.close()
    print(f"✓ Диаграмма {scheme.upper()} успешно сохранена в {output_path}")


def plot_r2_dependencies(top_metrics_csv='stat_tables/top_vs_rest_metrics_abl.csv',
                         unified_csv='stat_tables/unified_model_performance_report.csv',
                         skj_csv='stat_tables/spearman_kendall_jaccard_stats.csv',
                         output_dir='plots'):
    """
    Строит графики зависимостей от R^2.
    Порядок изменен: слева - Top нейроны, справа - MI.
    Прозрачность линии тренда (0.6) и доверительного интервала (0.35) синхронизированы.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        df_top = pd.read_csv(top_metrics_csv)
        df_uni = pd.read_csv(unified_csv)
        df_skj = pd.read_csv(skj_csv)
    except FileNotFoundError as e:
        print(f"[ERROR] Не найден один из файлов: {e}")
        return

    df = df_top[['model_name', 'top_count', 'top_r2']].merge(
        df_uni[['model_name', 'r2', 'total_neurons']], on='model_name', how='inner'
    )
    df = df.merge(
        df_skj[['model_name', 'size_mi']], on='model_name', how='inner'
    )

    df['mi_percentage'] = (df['size_mi'] / df['total_neurons']) * 100

    df_mi = df.dropna(subset=['mi_percentage', 'r2']).copy()
    df_top_n = df.dropna(subset=['top_count', 'top_r2']).copy()

    if df_mi.empty or df_top_n.empty:
        print("[ERROR] Недостаточно данных для построения графиков.")
        return

    corr_mi, pval_mi = spearmanr(df_mi['mi_percentage'], df_mi['r2'])
    corr_top, pval_top = spearmanr(df_top_n['top_count'], df_top_n['top_r2'])

    def get_stars(p):
        return r"{\ast}{\ast}{\ast}" if p < 0.001 else r"{\ast}{\ast}" if p < 0.01 else r"{\ast}" if p < 0.05 else "{ns}"

    # ==========================================
    # ЭСТЕТИЧЕСКАЯ ОТРИСОВКА
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax1, ax2 = axes

    c_text = '#2C3E50'
    c_purple = '#8E44AD'
    c_yellow = '#F39C12'

    # ------------------------------------------
    # График 1 (Слева): Top Neurons vs Top R2 (Желтые точки)
    # ------------------------------------------
    sns.regplot(
        data=df_top_n, x='top_count', y='top_r2', ax=ax1,
        color='grey',
        scatter_kws={'alpha': 0.8, 's': 50, 'color': c_yellow, 'edgecolors': 'none', 'zorder': 3},
        line_kws={'linestyle': '--', 'linewidth': 2, 'alpha': 0.6, 'zorder': 2}
    )

    # Принудительная установка alpha=0.35 для заливки доверительного интервала (CI)
    for collection in ax1.collections:
        if type(collection).__name__ == 'PolyCollection':
            collection.set_alpha(0.35)

    stars2 = get_stars(pval_top)
    ax1.text(0.05, 0.95, f"$\\rho = {corr_top:.2f}^{{{stars2}}}$",
             transform=ax1.transAxes, ha='left', va='top',
             fontsize=14, color=c_text, bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=5), zorder=5)

    ax1.set_xlabel(r'Количество сильных нейронов (>3$\sigma$)', fontsize=13, labelpad=10, color=c_text)
    ax1.set_ylabel(r'$R^2$', fontsize=15, labelpad=10, color=c_text)  # R^2 написан только слева

    # ------------------------------------------
    # График 2 (Справа): MI vs R2 (Фиолетовые точки)
    # ------------------------------------------
    sns.regplot(
        data=df_mi, x='mi_percentage', y='r2', ax=ax2,
        color='grey',
        scatter_kws={'alpha': 0.6, 's': 50, 'color': c_purple, 'edgecolors': 'none', 'zorder': 3},
        line_kws={'linestyle': '--', 'linewidth': 2, 'alpha': 0.6, 'zorder': 2}
    )

    # Принудительная установка alpha=0.35 для заливки доверительного интервала (CI)
    for collection in ax2.collections:
        if type(collection).__name__ == 'PolyCollection':
            collection.set_alpha(0.35)

    stars1 = get_stars(pval_mi)
    ax2.text(0.05, 0.95, f"$\\rho = {corr_mi:.2f}^{{{stars1}}}$",
             transform=ax2.transAxes, ha='left', va='top',
             fontsize=14, color=c_text, bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=5), zorder=5)

    ax2.set_xlabel(r'Доля информативных нейронов (%)', fontsize=13, labelpad=10, color=c_text)
    ax2.set_ylabel('')  # Скрываем подпись оси Y справа

    # ------------------------------------------
    # ОБЩЕЕ ФОРМАТИРОВАНИЕ ОПТИКИ
    # ------------------------------------------
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        [s.set_color(c_text) for s in ax.spines.values()]

        ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, colors=c_text)
        ax.grid(axis='both', linestyle=':', alpha=0.4, color='#BDC3C7', zorder=0)

    plt.tight_layout(w_pad=3.0)

    save_path = output_path / 'r2_ensemble_dependencies_aesthetic.png'
    plt.savefig(save_path, dpi=1000, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Эстетичный график зависимостей сохранен: {save_path}")


def plot_neuron_composition_stacked_barplot(sessions=None,
                                            top_metrics_csv='stat_tables/top_vs_rest_metrics_abl.csv',
                                            unified_csv='stat_tables/unified_model_performance_report.csv',
                                            output_dir='plots'):
    """
    Строит stacked barplot состава нейронных ансамблей (Сильные, Значимые, Неважные).
    Стиль: Научный минимализм (Nature/Science). Увеличены шрифты подписей осей.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Загрузка данных
    try:
        df_top = pd.read_csv(top_metrics_csv)
        df_uni = pd.read_csv(unified_csv)
    except FileNotFoundError as e:
        print(f"[ERROR] Не найден файл таблиц: {e}")
        return

    df = df_top[['model_name', 'top_count', 'rest_count']].merge(
        df_uni[['model_name', 'total_neurons']], on='model_name', how='inner'
    )

    if sessions is not None:
        df = df[df['model_name'].isin(sessions)]
        if df.empty:
            print(f"[ERROR] Указанные сессии не найдены.")
            return

    # 2. Вычисление и форматирование данных
    df['zero_weight_count'] = df['total_neurons'] - (df['top_count'] + df['rest_count'])

    # Сортировка по имени
    df = df.sort_values('model_name')

    # Парсер для переименования оси X (J30_1D -> Мышь J30 день 1)
    def format_session_name(name):
        match = re.match(r'(J\d+)_(\d)D', name)
        if match:
            return f"Мышь {match.group(1)}\nдень {match.group(2)}"
        return name

    df['formatted_name'] = df['model_name'].apply(format_session_name)
    plot_data = df.set_index('formatted_name')[['zero_weight_count', 'rest_count', 'top_count']]

    # Переименовываем колонки для легенды
    plot_data.rename(columns={
        'zero_weight_count': 'Неучаствующие',
        'rest_count': 'Значимые',
        'top_count': 'Сильные (>3$\sigma$)'
    }, inplace=True)

    # 3. Отрисовка
    # Ширина рассчитывается динамически, высота фиксирована
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.8), 7))

    # Палитра: Светло-серый, Желто-песочный, Темно-красный
    # Порядок цветов соответствует порядку колонок (Неважные -> Значимые -> Сильные)
    colors = ['#E0E0E0', '#E6B800', '#C00000']

    # Отрисовка столбцов без обводки
    plot_data.plot(
        kind='bar',
        stacked=True,
        color=colors,
        ax=ax,
        edgecolor='none',
        width=0.65,  # Делаем столбики чуть тоньше для элегантности
        zorder=3  # Помещаем столбики ПОВЕРХ сетки
    )

    # 4. Эстетика и минимализм (Nature/Science style)
    # УВЕЛИЧЕН ШРИФТ подписи оси Y (fontsize=18)
    ax.set_ylabel('Нейроны по вкладу в декодирование места', fontsize=18, labelpad=15, color='#2C3E50')
    ax.set_xlabel('')  # Убираем слово "Сессия"

    # УВЕЛИЧЕН ШРИФТ тиков (значений) на осях (labelsize=14)
    ax.tick_params(axis='x', rotation=0, labelsize=14, colors='#2C3E50', length=0)  # length=0 убирает засечки снизу
    ax.tick_params(axis='y', labelsize=14, colors='#2C3E50', width=1.2)

    # Очистка рамок (справа и сверху)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    [s.set_color('#2C3E50') for s in ax.spines.values()]

    # Горизонтальная сетка (под столбцами, zorder=0)
    ax.grid(axis='y', linestyle='--', alpha=0.4, color='#BDC3C7', zorder=0)

    # 5. Легенда (внутри, прозрачная, без рамки)
    # Используем reverse, чтобы порядок в легенде был сверху вниз (Сильные -> Значимые -> Неважные)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1], labels[::-1],
        loc='upper left',
        fontsize=14,
        frameon=False,  # Убираем рамку и фон
        handlelength=1.5,  # Длина квадратиков цвета
        handletextpad=0.8  # Отступ от цвета до текста
    )

    plt.tight_layout()

    # 6. Сохранение
    save_path = output_path / 'neuron_composition_stacked_aesthetic.png'
    plt.savefig(save_path, dpi=1000, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n{'=' * 60}")
    print(f"✓ Эстетичный Stacked Barplot успешно сохранен: {save_path}")
    print(f"  Отображено сессий: {len(df)}")
    print(f"{'=' * 60}")


def plot_delta_vs_mi_dependencies(ablation_csv='stat_tables/full_ablation_stats.csv',
                                  database_csv='LNOF_SynchronizedData/LNOF_database.csv',
                                  output_dir='plots'):
    """
    Строит график зависимости нормированного падения R2 (z-delta) от Mutual Information (MI).
    Логарифмические координаты (log2) с умной нумерацией осей.
    Добавлена серая полупрозрачная аппроксимация с 95% доверительным интервалом.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        df_abl = pd.read_csv(ablation_csv)
        df_db = pd.read_csv(database_csv)
    except FileNotFoundError as e:
        print(f"[ERROR] Не найден файл: {e}")
        return

    print(f"\n{'=' * 70}")
    print(f"Сборка пула нейронов для графика z-\u03B4 от MI (log2 шкала)...")

    mi_all, zcrit_all = [], []
    feat_cols = [col for col in df_abl.columns if col.startswith('ablation_')]
    feature_names = [col.replace('ablation_', '') for col in feat_cols]

    for _, row in df_abl.iterrows():
        model_name = row['model_session']
        mouse, day = model_name[:3], model_name[4:6]

        base_r2 = row['baseline_r2']
        abl_vals = row[feat_cols].values.astype(float)
        crit = base_r2 - abl_vals

        valid_crit = crit[~np.isnan(crit)]
        if len(valid_crit) > 0 and np.std(valid_crit) > 0:
            zcrit = (crit - np.mean(valid_crit)) / np.std(valid_crit)
        else:
            zcrit = np.zeros_like(crit)

        df_intense = df_db[(df_db['mouse'] == mouse) & (df_db['session'] == day) & (df_db['feature'] == 'place')].copy()
        df_intense = df_intense.sort_values(by='me', ascending=False).drop_duplicates(subset=['neuron_idx'])
        me_dict = {f"neuron_{int(r['neuron_idx'])}": r['me'] for _, r in df_intense.iterrows()}

        mi = np.array([me_dict.get(f, 0.0) for f in feature_names])

        mi_all.extend(mi)
        zcrit_all.extend(zcrit)

    df_plot = pd.DataFrame({'mi': mi_all, 'zcrit': zcrit_all}).dropna()
    df_zcrit = df_plot[(df_plot['mi'] > 0) & (df_plot['zcrit'] > 0)].copy()

    print(f"Пул для z-\u03B4: {len(df_zcrit)} нейронов")
    print(f"{'=' * 70}")

    corr_zcrit, p_zcrit = spearmanr(df_zcrit['mi'], df_zcrit['zcrit'])

    def get_stars(p):
        return r"{\ast}{\ast}{\ast}" if p < 0.001 else r"{\ast}{\ast}" if p < 0.01 else r"{\ast}" if p < 0.05 else "{ns}"

    # ==========================================
    # ЭСТЕТИЧЕСКАЯ ОТРИСОВКА
    # ==========================================
    fig, ax = plt.subplots(figsize=(8, 6))

    c_green = '#27AE60'
    c_text = '#2C3E50'

    # --- ЧЕСТНЫЙ РАСЧЕТ ТРЕНДА И 95% ДОВЕРИТЕЛЬНОГО ИНТЕРВАЛА В LOG2 ПРОСТРАНСТВЕ ---
    log2_x = np.log2(df_zcrit['mi']).values
    log2_y = np.log2(df_zcrit['zcrit']).values

    m, b = np.polyfit(log2_x, log2_y, 1)

    x_trend_log = np.linspace(log2_x.min(), log2_x.max(), 100)
    y_trend_log = m * x_trend_log + b

    # Математика доверительного интервала (Standard Error of Regression)
    n = len(log2_x)
    y_hat = m * log2_x + b
    s_err = np.sqrt(np.sum((log2_y - y_hat) ** 2) / (n - 2))

    # Квантиль t-распределения для 95% (двухсторонний)
    t_val = t.ppf(0.975, n - 2)

    mean_x = np.mean(log2_x)
    ssx = np.sum((log2_x - mean_x) ** 2)

    # Ширина доверительного интервала в логарифмическом пространстве
    ci = t_val * s_err * np.sqrt(1 / n + (x_trend_log - mean_x) ** 2 / ssx)

    # Возврат координат в линейное пространство (экспоненцирование по базе 2)
    x_trend = 2 ** x_trend_log
    y_trend = 2 ** y_trend_log
    y_lower = 2 ** (y_trend_log - ci)
    y_upper = 2 ** (y_trend_log + ci)

    # Отрисовка
    ax.scatter(df_zcrit['mi'], df_zcrit['zcrit'], alpha=0.5, s=35, color=c_green, edgecolors='none', zorder=3)

    # Серая линия тренда и заливка с требуемыми уровнями прозрачности
    ax.plot(x_trend, y_trend, color='grey', linestyle='--', linewidth=2, alpha=0.6, zorder=4)
    ax.fill_between(x_trend, y_lower, y_upper, color='grey', alpha=0.35, zorder=2)

    # Настройка осей
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)

    # ИСПРАВЛЕННАЯ НУМЕРАЦИЯ
    def tick_formatter(val, pos):
        if val < 0.125:
            return f"$2^{{{int(np.round(np.log2(val)))}}}$"
        return f"{val:g}"

    custom_formatter = FuncFormatter(tick_formatter)

    ax.xaxis.set_major_formatter(custom_formatter)
    ax.yaxis.set_major_formatter(custom_formatter)

    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    # ТЕКСТ СПИРМЕНА
    stars = get_stars(p_zcrit)
    stat_text = f"$\\rho = {corr_zcrit:.2f}^{{{stars}}}$"

    ax.text(0.05, 0.95, stat_text,
            transform=ax.transAxes, ha='left', va='top',
            fontsize=14, color=c_text,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=5), zorder=5)

    ax.set_xlabel(r'MI место & $Ca^{2+}$-активность', fontsize=26, labelpad=10, color=c_text)
    ax.set_ylabel('Падение точности предсказания (z-значения)', fontsize=26, labelpad=10, color=c_text)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    [s.set_color(c_text) for s in ax.spines.values()]

    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, colors=c_text)
    ax.grid(axis='both', linestyle=':', alpha=0.4, color='#BDC3C7', zorder=0)

    plt.tight_layout()

    save_path = output_path / 'zdelta_vs_mi_log2_with_ci.png'
    plt.savefig(save_path, dpi=1000, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Эстетичный график с доверительным интервалом успешно сохранен: {save_path}")


def plot_heavy_tailed_distributions(sessions=None,
                                    ablation_csv='stat_tables/full_ablation_stats.csv',
                                    database_csv='LNOF_SynchronizedData/LNOF_database.csv',
                                    output_dir='plots'):
    """
    Строит распределения MI и Delta R2 с двойным разрывом осей.
    Эстетика: Увеличенные шрифты, идеальные двойные засечки на разрывах.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        df_abl = pd.read_csv(ablation_csv)
        df_db = pd.read_csv(database_csv)
    except FileNotFoundError as e:
        print(f"[ERROR] Не найден файл: {e}")
        return

    excluded_mice = ['J20', 'J21']
    all_sessions = df_abl['model_session'].unique().tolist()

    if sessions is None:
        sessions_to_process = [s for s in all_sessions if s[:3] not in excluded_mice]
    else:
        sessions_to_process = [s for s in sessions if s in all_sessions and s[:3] not in excluded_mice]

    if not sessions_to_process:
        print("[ERROR] Нет валидных сессий.")
        return

    mi_all, crit_all = [], []
    feat_cols = [col for col in df_abl.columns if col.startswith('ablation_')]
    feature_names = [col.replace('ablation_', '') for col in feat_cols]

    for sess in sessions_to_process:
        row = df_abl[df_abl['model_session'] == sess].iloc[0]
        mouse, day = sess[:3], sess[4:6]

        base_r2 = row['baseline_r2']
        abl_vals = row[feat_cols].values.astype(float)
        crit = base_r2 - abl_vals

        df_intense = df_db[(df_db['mouse'] == mouse) & (df_db['session'] == day) & (df_db['feature'] == 'place')].copy()
        df_intense = df_intense.sort_values(by='me', ascending=False).drop_duplicates(subset=['neuron_idx'])
        me_dict = {f"neuron_{int(r['neuron_idx'])}": r['me'] for _, r in df_intense.iterrows()}
        mi = np.array([me_dict.get(f, 0.0) for f in feature_names])

        mi_all.extend(mi)
        crit_all.extend(crit)

    # 1. Данные без обрезки
    df_mi_raw = pd.Series(mi_all).dropna()
    df_mi = df_mi_raw[df_mi_raw > 0].values

    df_crit = pd.Series(crit_all).dropna().values
    df_crit_z = zscore(df_crit)  # Перевод в z-значения

    # 2. Расчет KDE
    stats = {}
    for name, data in zip(['MI', 'Delta'], [df_mi, df_crit_z]):
        kde = gaussian_kde(data, bw_method=0.2)
        x_eval = np.linspace(data.min(), data.max(), 1000)
        kde_vals = kde(x_eval)

        stats[name] = {
            'data': data,
            'x': x_eval, 'y': kde_vals,
            'mode': x_eval[np.argmax(kde_vals)],
            'skew': skew(data)
        }

    # ==========================================
    # Вспомогательная функция для градиента
    # ==========================================
    def add_linear_gradient(ax, x_vals, y_vals, color):
        x_grid = np.linspace(x_vals.min(), x_vals.max(), 500)
        y_grid = np.linspace(0, y_vals.max() * 1.15, 500)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = Y.copy()

        kde_interp = np.interp(x_grid, x_vals, y_vals)
        Z = np.where(Y <= kde_interp[None, :], Z, np.nan)

        c_rgba = mcolors.to_rgba(color)
        cmap_colors = [(c_rgba[0], c_rgba[1], c_rgba[2], 0.0),
                       (c_rgba[0], c_rgba[1], c_rgba[2], 0.65)]
        custom_cmap = mcolors.LinearSegmentedColormap.from_list("grad", cmap_colors)

        ax.pcolormesh(X, Y, Z, cmap=custom_cmap, shading='auto', zorder=1)

    # ==========================================
    # 3. ПОСТРОЕНИЕ СЕТКИ (GridSpec)
    # ==========================================
    fig = plt.figure(figsize=(16, 6))

    gs_main = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.2)
    ax_mi = fig.add_subplot(gs_main[0])

    # Невидимая ось поверх MI для идеального центрирования подписи X
    ax_mi_common = fig.add_subplot(gs_main[0], frameon=False)
    ax_mi_common.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax_mi_common.grid(False)

    gs_delta = gs_main[1].subgridspec(1, 3, width_ratios=[0.5, 4, 0.5], wspace=0.04)
    ax_d_left = fig.add_subplot(gs_delta[0])
    ax_d_center = fig.add_subplot(gs_delta[1], sharey=ax_d_left)
    ax_d_right = fig.add_subplot(gs_delta[2], sharey=ax_d_left)

    # Невидимая ось поверх Delta
    ax_d_common = fig.add_subplot(gs_main[1], frameon=False)
    ax_d_common.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax_d_common.grid(False)

    colors = {'MI': '#1F77B4', 'Delta': '#D62728'}
    c_text = '#2C3E50'

    # ==========================================
    # ОТРИСОВКА MI
    # ==========================================
    x_mi, y_mi = stats['MI']['x'], stats['MI']['y']
    c_mi = colors['MI']

    add_linear_gradient(ax_mi, x_mi, y_mi, c_mi)
    ax_mi.plot(x_mi, y_mi, color=c_mi, linewidth=2.5, zorder=3)
    ax_mi.axvline(stats['MI']['mode'], color='grey', linestyle='--', alpha=0.6, linewidth=2, zorder=2)

    # Увеличенный коэффициент
    ax_mi.text(0.95, 0.95, f"$\\mu_3 = {stats['MI']['skew']:.2f}$",
               transform=ax_mi.transAxes, ha='right', va='top',
               fontsize=30, color=c_text, bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))

    ax_mi.set_xlim(left=0)
    ax_mi.set_ylim(bottom=0, top=y_mi.max() * 1.15)

    # Увеличенные заголовки и подписи
    ax_mi_common.set_title('Распределение информативности нейронов', fontsize=17, pad=15, fontweight='bold', color=c_text)
    ax_mi_common.set_xlabel(r'MI место & $Ca^{2+}$-активность', fontsize=15, labelpad=25, color=c_text)

    ax_mi.spines['top'].set_visible(False)
    ax_mi.spines['right'].set_visible(False)
    ax_mi.spines['left'].set_linewidth(1.2)
    ax_mi.spines['bottom'].set_linewidth(1.2)
    [s.set_color(c_text) for s in ax_mi.spines.values()]
    ax_mi.tick_params(axis='both', which='major', labelsize=13, width=1.2, colors=c_text)
    ax_mi.grid(axis='y', linestyle='-', alpha=0.15, color=c_text, zorder=0)

    # ==========================================
    # ОТРИСОВКА DELTA (Двойной разрыв)
    # ==========================================
    x_d, y_d = stats['Delta']['x'], stats['Delta']['y']
    c_d = colors['Delta']
    raw_delta = stats['Delta']['data']
    mode_d = stats['Delta']['mode']

    core_min, core_max = -2.5, 2.5

    cut_left_start = raw_delta.min() - 0.5
    cut_left_end = core_min - 1.0
    cut_right_start = core_max + 1.0
    cut_right_end = raw_delta.max() + 0.5

    delta_axes = [ax_d_left, ax_d_center, ax_d_right]

    for ax in delta_axes:
        add_linear_gradient(ax, x_d, y_d, c_d)
        ax.plot(x_d, y_d, color=c_d, linewidth=2.5, zorder=3)
        ax.axvline(mode_d, color='grey', linestyle='--', alpha=0.6, linewidth=2, zorder=2)

        ax.set_ylim(bottom=0, top=y_d.max() * 1.15)
        ax.grid(axis='y', linestyle='-', alpha=0.15, color=c_text, zorder=0)

        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.2)
        [s.set_color(c_text) for s in ax.spines.values()]
        ax.tick_params(axis='x', which='major', labelsize=13, width=1.2, colors=c_text)

    # Применяем лимиты
    ax_d_left.set_xlim(cut_left_start, cut_left_end)
    ax_d_center.set_xlim(core_min, core_max)
    ax_d_right.set_xlim(cut_right_start, cut_right_end)

    ax_d_left.set_xticks([np.round(raw_delta.min(), 1)])
    ax_d_center.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
    ax_d_right.set_xticks([np.round(raw_delta.max(), 1)])

    # Скрытие внутренних границ
    ax_d_left.spines['right'].set_visible(False)
    ax_d_center.spines['left'].set_visible(False)
    ax_d_center.spines['right'].set_visible(False)
    ax_d_right.spines['left'].set_visible(False)
    ax_d_right.spines['right'].set_visible(False)

    ax_d_left.spines['left'].set_linewidth(1.2)
    ax_d_left.tick_params(axis='y', which='major', labelsize=13, width=1.2, colors=c_text)
    ax_d_center.tick_params(axis='y', left=False, labelleft=False)
    ax_d_right.tick_params(axis='y', left=False, labelleft=False)

    # --- ЭСТЕТИЧНЫЕ ДВОЙНЫЕ ЗАСЕЧКИ (Double Slashes) ---
    d = 0.02  # Высота засечек
    tilt_c = 0.015  # Наклон для центрального графика
    tilt_s = tilt_c * 8  # Усиленный наклон для узких боковых графиков

    # Толщина линий засечек строго равна толщине осей (lw=1.2)
    kw_l = dict(transform=ax_d_left.transAxes, color=c_text, clip_on=False, lw=1.2)
    kw_c = dict(transform=ax_d_center.transAxes, color=c_text, clip_on=False, lw=1.2)
    kw_r = dict(transform=ax_d_right.transAxes, color=c_text, clip_on=False, lw=1.2)

    # Разрыв 1 (Левый) -> 1 линия слева + 1 линия справа = 2 засечки
    ax_d_left.plot((1 - tilt_s, 1 + tilt_s), (-d, +d), **kw_l)
    ax_d_center.plot((-tilt_c, +tilt_c), (-d, +d), **kw_c)

    # Разрыв 2 (Правый) -> 1 линия слева + 1 линия справа = 2 засечки
    ax_d_center.plot((1 - tilt_c, 1 + tilt_c), (-d, +d), **kw_c)
    ax_d_right.plot((-tilt_s, +tilt_s), (-d, +d), **kw_r)

    # --- ИДЕАЛЬНОЕ ВЫРАВНИВАНИЕ ПОДПИСЕЙ ---
    ax_d_common.set_title('Распределение влияния удаления нейронов', fontsize=17, pad=15, fontweight='bold', color=c_text)
    ax_d_common.set_xlabel('Падение точности предсказания (z-значения)', fontsize=15, labelpad=25, color=c_text)

    ax_d_common.text(0.95, 0.95, f"$\\mu_3 = {stats['Delta']['skew']:.2f}$",
                     transform=ax_d_common.transAxes, ha='right', va='top',
                     fontsize=30, color=c_text, bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))

    save_path = output_path / 'heavy_tail_smooth_zscore.png'
    plt.savefig(save_path, dpi=1000, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Эстетичный анализ (Z-score + Double Broken Axis) завершен!")
    for k in ['MI', 'Delta']:
        print(f"  [{k}] Skewness: {stats[k]['skew']:.2f} | Mode: {stats[k]['mode']:.4f}")
    print(f"✓ График сохранен: {save_path}\n{'=' * 70}")


def train_and_get_r2(X_subset, y):
    """
    Автономная функция для быстрого переобучения MultiTaskLasso и возврата R2.
    Временное разбиение: первые 75% времени - обучение, последние 25% - тест.
    """
    if X_subset.shape[1] == 0:
        return 0.0

    split_idx = int(len(X_subset) * 0.75)
    X_train, X_test = X_subset[:split_idx], X_subset[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Используем стандартную альфу (можешь поменять на свой параметр C, если используешь обертку)
    model = MultiTaskLasso(alpha=0.1, max_iter=1000)
    try:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return r2_score(y_test, preds)
    except Exception:
        return 0.0


def plot_cumulative_retrain_ablation_addition(
        session_name,
        metric='test_r2',
        data_dir='LNOF_SynchronizedData',
        models_dir='models_LNOF_Lasso',
        output_dir='plots',
        step_size=5
):
    c_param = 0.5
    test_size = 0.25
    max_iter = 100

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    mouse = session_name[:3]
    day = session_name[4:5]

    # 1. Загрузка базовой модели
    model_path = Path(models_dir) / f"{session_name}.pkl"
    if not model_path.exists():
        print(f"[ERROR] Базовая модель {session_name} не найдена.")
        return

    data = joblib.load(model_path)
    base_coef = data.get('model').coef_
    all_features = np.array(data.get('feature_names'))
    all_feat_list = list(all_features)
    num_features = len(all_features)

    # 2. Ранжирование нейронов
    weights = np.linalg.norm(base_coef, axis=0)
    sorted_idx_desc = np.argsort(weights)[::-1]
    sorted_idx_asc = np.argsort(weights)

    np.random.seed(42)
    rand_idx = np.random.permutation(num_features)

    feat_desc = all_features[sorted_idx_desc]
    feat_asc = all_features[sorted_idx_asc]
    feat_rand = all_features[rand_idx]

    # 3. Загрузка данных
    data_file = Path(data_dir) / f"LNOF_{session_name}_aligned.npz"
    if not data_file.exists():
        print(f"[ERROR] Файл данных {data_file} не найден.")
        return

    features_raw, labels = load_data(data_file)
    X_train_full, X_test_full, y_train, y_test = train_test_split_temporal(features_raw, labels, test_size=test_size)

    def evaluate_subset(features_subset, eval_metric):
        if len(features_subset) == 0:
            # Для R2 0 признаков = 0 объясненной дисперсии.
            # Для дистанции (MAE/Med) - предсказывать нечем, ошибка бесконечна (возвращаем NaN для графика)
            return np.nan if eval_metric == 'test_dist' else 0.0

        if isinstance(X_train_full, pd.DataFrame):
            X_tr = X_train_full[features_subset]
            X_te = X_test_full[features_subset]
        else:
            idx = [all_feat_list.index(f) for f in features_subset]
            X_tr = X_train_full[:, idx]
            X_te = X_test_full[:, idx]

        clf, imp, scl, metrics = train_lasso(X_tr, y_train, X_te, y_test, C=c_param, max_iter=max_iter)
        return metrics.get(eval_metric, 0.0)

    # 4. Формируем шаги
    steps = list(range(0, num_features, step_size))
    if steps[-1] != num_features:
        steps.append(num_features)

    print(f"\n{'=' * 70}")
    print(f"[{session_name}] Начат процесс переобучения (Ablation & Addition). Метрика: {metric}")
    print(f"{'=' * 70}")

    r2_abl_red, r2_abl_blue, r2_abl_grey = [], [], []
    r2_add_green, r2_add_yellow, r2_add_grey = [], [], []

    # 5. Цикл переобучения
    for k in steps:
        r2_abl_red.append(evaluate_subset(feat_desc[k:], metric))
        r2_abl_blue.append(evaluate_subset(feat_asc[k:], metric))
        r2_abl_grey.append(evaluate_subset(feat_rand[k:], metric))

        r2_add_green.append(evaluate_subset(feat_desc[:k], metric))
        r2_add_yellow.append(evaluate_subset(feat_asc[:k], metric))
        r2_add_grey.append(evaluate_subset(feat_rand[:k], metric))

        if k % (step_size * 5) == 0 and k > 0:
            print(f"  > Пройдено {k}/{num_features} нейронов...")

    # ==========================================
    # 6. ЭСТЕТИЧЕСКАЯ ОТРИСОВКА
    # ==========================================
    c_red = '#D62728'
    c_blue = '#1F77B4'
    c_green = '#2CA02C'
    c_yellow = '#F39C12'
    c_grey = '#7F8C8D'
    fill_color = '#BDC3C7'

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(top=0.90, bottom=0.25, wspace=0.15)
    ax1, ax2 = axes

    # --- ДИНАМИЧЕСКОЕ ОТКЛЮЧЕНИЕ МАРКЕРОВ ---
    # Ширина одного sub-plot примерно 6 дюймов. В пунктах (points) это 6 * 72 = 432 pt.
    # Если расстояние между точками (432 / кол-во шагов) меньше радиуса маркера (2 pt), убираем маркеры.
    pt_distance = 432 / len(steps)
    marker_style = 'o' if pt_distance > 10 else None
    marker_size = 4

    # --- ПАНЕЛЬ 1: УДАЛЕНИЕ (Ablation) ---
    ax1.fill_between(steps, r2_abl_red, r2_abl_blue, color=fill_color, alpha=0.35, edgecolor='none', zorder=1)

    ax1.plot(steps, r2_abl_grey, color=c_grey, lw=2, linestyle='--', alpha=0.6, zorder=2)
    ax1.plot(steps, r2_abl_red, color=c_red, lw=2.5, marker=marker_style, markersize=marker_size, zorder=3)
    ax1.plot(steps, r2_abl_blue, color=c_blue, lw=2.5, marker=marker_style, markersize=marker_size, zorder=3)

    ax1.set_xlabel("Удаление нейронов", fontsize=13, labelpad=10, color='#2C3E50')

    # Динамическая подпись оси Y
    y_label = r"$R^2$" if metric == 'test_r2' else "MED"
    ax1.set_ylabel(y_label, fontsize=13, labelpad=10, color='#2C3E50')

    # --- ПАНЕЛЬ 2: ДОБАВЛЕНИЕ (Addition) ---
    ax2.fill_between(steps, r2_add_green, r2_add_yellow, color=fill_color, alpha=0.35, edgecolor='none', zorder=1)

    ax2.plot(steps, r2_add_grey, color=c_grey, lw=2, linestyle='--', alpha=0.6, zorder=2)
    ax2.plot(steps, r2_add_green, color=c_green, lw=2.5, marker=marker_style, markersize=marker_size, zorder=3)
    ax2.plot(steps, r2_add_yellow, color=c_yellow, lw=2.5, marker=marker_style, markersize=marker_size, zorder=3)

    ax2.set_xlabel("Добавление нейронов", fontsize=13, labelpad=10, color='#2C3E50')

    # --- ОЧИСТКА И ФОРМАТИРОВАНИЕ ОСЕЙ ---
    for ax in axes:
        ax.grid(axis='y', linestyle='--', alpha=0.5, color='#BDC3C7', zorder=0)
        ax.grid(axis='x', linestyle=':', alpha=0.3, color='#BDC3C7', zorder=0)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['left'].set_color('#2C3E50')
        ax.spines['bottom'].set_linewidth(1.2)
        ax.spines['bottom'].set_color('#2C3E50')

        ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, colors='#2C3E50')

    # --- ЛЕГЕНДА СНИЗУ ---
    # Маркеры в легенде рисуем всегда, независимо от плотности точек на графике, для читаемости
    legend_elements = [
        mlines.mlines.Line2D([0], [0], color=c_red, lw=2.5, marker='o', markersize=5, label='Сильных'),
        mlines.mlines.Line2D([0], [0], color=c_blue, lw=2.5, marker='o', markersize=5, label='Слабых'),
        mlines.mlines.Line2D([0], [0], color=c_grey, lw=2, linestyle='--', alpha=0.6, label='Случайно'),
        mlines.mlines.Line2D([0], [0], color=c_green, lw=2.5, marker='o', markersize=5, label='Сильных'),
        mlines.mlines.Line2D([0], [0], color=c_yellow, lw=2.5, marker='o', markersize=5, label='Слабых')
    ]

    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.12),
               ncol=5, frameon=False, fontsize=12, handlelength=2.5, columnspacing=1.5)

    mouse_formatted = mouse.replace('J', '') if mouse.startswith('J') else mouse
    fig.suptitle(f"Мышь J{mouse_formatted} день {day}", fontsize=18, fontweight='bold', y=0.98, color='#2C3E50')

    # 7. Сохранение
    save_path = output_path / f'cumulative_retrain_{session_name}_{metric}.png'
    plt.savefig(save_path, dpi=1000, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n✓ Готово! График высшего качества сохранен: {save_path}")


def plot_global_euler_diagram_4d_area(mouse_name,
                                      color_scheme='cmyk',
                                      data_dir='MatchData',
                                      output_dir='plots'):
    """
    Визуализирует 4D пересечение с физичным вычислением площадей и цветов.
    Тройные пересечения смешиваются субтрактивно, четверное (ядро) — принудительно окрашивается в белый.
    Текст автоматически инвертируется (черный/белый) в зависимости от яркости фона.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data_file = Path(data_dir) / f"LNOF_{mouse_name}.csv"

    try:
        df = pd.read_csv(data_file, header=None)
    except FileNotFoundError as e:
        print(f"[ERROR] Не найден файл: {e}")
        return

    # 1. Извлечение сетов нейронов
    days_sets = []
    for col_idx in range(4):
        if col_idx < len(df.columns):
            active_idx = df.index[df[col_idx].notna() & (df[col_idx] != 0)].tolist()
            days_sets.append(set(active_idx))
        else:
            days_sets.append(set())

    # 2. Подготовка пропорций
    max_size = max(len(s) for s in days_sets)
    if max_size == 0:
        print("Все сеты пусты.")
        return

    R_arr = [math.sqrt(len(s) / max_size) for s in days_sets]

    target_areas = {}
    for i, j in itertools.combinations(range(4), 2):
        intersection_size = len(days_sets[i].intersection(days_sets[j]))
        target_areas[(i, j)] = (intersection_size / max_size) * math.pi

    def circle_overlap(d, r1, r2):
        if d >= r1 + r2: return 0.0
        if d <= abs(r1 - r2): return math.pi * min(r1, r2) ** 2
        r1_sq, r2_sq, d_sq = r1 ** 2, r2 ** 2, d ** 2
        d1 = (r1_sq - r2_sq + d_sq) / (2 * d)
        d2 = d - d1
        h = math.sqrt(max(0, r1_sq - d1 ** 2))
        a1 = r1_sq * math.acos(max(-1, min(1, d1 / r1))) - d1 * h
        a2 = r2_sq * math.acos(max(-1, min(1, d2 / r2))) - d2 * h
        return a1 + a2

    # 3. Оптимизация координат центров
    def objective(vars):
        y1, x2, y2, y3, x4, y4 = vars
        centers = [(0, y1), (x2, y2), (0, y3), (x4, y4)]
        loss = 0
        for (i, j), target in target_areas.items():
            d = math.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
            loss += (circle_overlap(d, R_arr[i], R_arr[j]) - target) ** 2
        return loss

    init_guess = [R_arr[0], R_arr[1], 0, -R_arr[2], -R_arr[3], 0]
    bnds = ((0, None), (0, None), (None, None), (None, 0), (None, 0), (None, None))

    res = minimize(objective, init_guess, bounds=bnds, method='Nelder-Mead')
    y1_opt, x2_opt, y2_opt, y3_opt, x4_opt, y4_opt = res.x
    centers = [(0, y1_opt), (x2_opt, y2_opt), (0, y3_opt), (x4_opt, y4_opt)]

    # 4. Построение геометрии
    circles = [Point(x, y).buffer(r) if r > 0 else Point(x, y).buffer(0.001)
               for (x, y), r in zip(centers, R_arr)]

    color_maps = {
        'cmyk': ['#00FFFF', '#FF00FF', '#FFFF00', '#80FF80'],
        'rgb': ['#FF0000', '#00FF00', '#0000FF', '#FFB300']
    }
    scheme = color_scheme.lower()
    rgb_bases = [mcolors.to_rgb(c) for c in color_maps.get(scheme, color_maps['cmyk'])]

    fig, ax = plt.subplots(figsize=(14, 14))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # 5. Вычисление и отрисовка полигонов пересечения
    for r in range(1, 5):
        for combo in itertools.combinations(range(4), r):
            poly = circles[combo[0]]
            for idx in combo[1:]: poly = poly.intersection(circles[idx])

            excluded = [i for i in range(4) if i not in combo]
            for idx in excluded: poly = poly.difference(circles[idx])

            if poly.is_empty: continue

            subset = set(days_sets[combo[0]])
            for idx in combo[1:]: subset = subset.intersection(days_sets[idx])
            for idx in excluded: subset = subset.difference(days_sets[idx])

            count = len(subset)
            if count == 0: continue

            # Физичное смешивание
            if scheme == 'cmyk':
                blend_r = math.prod(rgb_bases[i][0] for i in combo)
                blend_g = math.prod(rgb_bases[i][1] for i in combo)
                blend_b = math.prod(rgb_bases[i][2] for i in combo)
            else:
                blend_r = 1.0 - math.prod(1.0 - rgb_bases[i][0] for i in combo)
                blend_g = 1.0 - math.prod(1.0 - rgb_bases[i][1] for i in combo)
                blend_b = 1.0 - math.prod(1.0 - rgb_bases[i][2] for i in combo)

            blend_color = (blend_r, blend_g, blend_b)

            # ИСКЛЮЧЕНИЕ: Белый центр для 4-х дневного ансамбля
            if len(combo) == 4:
                blend_color = (1.0, 1.0, 1.0)

            # Вычисляем визуальную яркость цвета для контраста текста
            lum = 0.299 * blend_color[0] + 0.587 * blend_color[1] + 0.114 * blend_color[2]
            txt_color = 'white' if lum < 0.6 else 'black'

            # Отрисовка
            polys_to_draw = [poly] if poly.geom_type == 'Polygon' else poly.geoms
            for p in polys_to_draw:
                x, y = p.exterior.xy
                ax.fill(x, y, color=blend_color, alpha=1.0, edgecolor='none')

            # Вывод цифры: без рамок, крупный шрифт
            cx, cy = poly.centroid.x, poly.centroid.y
            ax.text(cx, cy, str(count), ha='center', va='center',
                    color=txt_color, fontsize=30, fontweight='bold')

    # 6. Внешние лейблы на русском
    for i, (cx, cy) in enumerate(centers):
        if R_arr[i] == 0: continue
        length = math.hypot(cx, cy)
        if length == 0: length = 1
        offset = R_arr[i] + 0.15
        label_x = cx + (cx / length) * offset
        label_y = cy + (cy / length) * offset

        ax.text(label_x, label_y, f'День {i + 1}', fontsize=32,
                fontweight='bold', color='#1a1a1a', ha='center', va='center')

    ax.axis('equal')
    ax.axis('off')

    file_prefix = f"euler_4d_area_{mouse_name}_{scheme}"
    fig.savefig(output_path / f"{file_prefix}.svg", format='svg', bbox_inches='tight', facecolor='white')
    fig.savefig(output_path / f"{file_prefix}.png", format='png', dpi=1000, bbox_inches='tight', facecolor='white')

    plt.close()
    print(f"✓ Диаграмма 4D с честными площадями и белым центром сохранена в {output_path}")


def plot_r2_and_top_neurons_dependencies(top_metrics_csv='stat_tables/top_vs_rest_metrics_abl.csv',
                                         unified_csv='stat_tables/unified_model_performance_report.csv',
                                         output_dir='plots'):
    """
    Строит два графика:
    1. Зависимость R2 от количества сильных нейронов (две модели).
    2. Зависимость количества сильных нейронов от размера сети (со Спирменом).
    Стиль: Научный минимализм. Мягкая пастельная палитра.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Загрузка данных
    try:
        df_top = pd.read_csv(top_metrics_csv)
        df_uni = pd.read_csv(unified_csv)
    except FileNotFoundError as e:
        print(f"[ERROR] Не найден один из файлов: {e}")
        return

    # 2. Объединение таблиц по имени сессии
    df = df_top[['model_name', 'top_count', 'top_r2']].merge(
        df_uni[['model_name', 'r2', 'total_neurons']], on='model_name', how='inner'
    )

    # 3. Очистка от NaN
    df_clean = df.dropna(subset=['top_count', 'top_r2', 'r2', 'total_neurons'])

    if df_clean.empty:
        print("[ERROR] Недостаточно валидных данных для построения графиков.")
        return

    # 4. Расчет коэффициента Спирмена для правого графика
    corr, pval = spearmanr(df_clean['total_neurons'], df_clean['top_count'])

    def get_stars(p):
        return r"{\ast}{\ast}{\ast}" if p < 0.001 else r"{\ast}{\ast}" if p < 0.01 else r"{\ast}" if p < 0.05 else "{ns}"

    # ==========================================
    # ЭСТЕТИЧЕСКАЯ ОТРИСОВКА
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax1, ax2 = axes

    c_text = '#2C3E50'
    # Мягкая пастельная палитра (из референса)
    c_blue = '#64B5F6'  # Светло-голубой (Полноценная модель)
    c_yellow = '#F39C12'  # Песочно-желтый (Сильные нейроны)
    c_pink = '#EC407A'  # Пыльно-розовый (Правый график)

    # ------------------------------------------
    # График 1 (Слева): R2 vs Количество сильных нейронов
    # ------------------------------------------
    # Регрессия для Полноценной модели (голубые точки)
    sns.regplot(
        data=df_clean, x='top_count', y='r2', ax=ax1,
        color=c_blue,
        scatter_kws={'alpha': 0.7, 's': 60, 'edgecolors': 'none', 'zorder': 3},
        line_kws={'linestyle': '--', 'linewidth': 2, 'alpha': 0.8, 'zorder': 2},
        label='Полноценная модель'
    )

    # Регрессия для Top-модели (желтые точки)
    sns.regplot(
        data=df_clean, x='top_count', y='top_r2', ax=ax1,
        color=c_yellow,
        scatter_kws={'alpha': 0.7, 's': 60, 'edgecolors': 'none', 'zorder': 3},
        line_kws={'linestyle': '--', 'linewidth': 2, 'alpha': 0.8, 'zorder': 2},
        label='Сильные нейроны (>3$\sigma$)'
    )

    # Легенда: без рамки
    ax1.legend(fontsize=12, loc='lower right', frameon=False, labelcolor=c_text)

    ax1.set_xlabel('Количество сильных нейронов (>3$\sigma$)', fontsize=13, labelpad=10, color=c_text)
    ax1.set_ylabel(r'$R^2$', fontsize=15, labelpad=10, color=c_text)

    # ------------------------------------------
    # График 2 (Справа): Сильные нейроны vs Общее количество
    # ------------------------------------------
    sns.regplot(
        data=df_clean, x='total_neurons', y='top_count', ax=ax2,
        color=c_pink,
        scatter_kws={'alpha': 0.7, 's': 60, 'edgecolors': 'none', 'zorder': 3},
        line_kws={'linestyle': '--', 'linewidth': 2, 'alpha': 0.8, 'zorder': 2}
    )

    # Текст Спирмена в левом верхнем углу
    stars = get_stars(pval)
    stat_text = f"$\\rho = {corr:.2f}^{{{stars}}}$"

    ax2.text(0.05, 0.95, stat_text,
             transform=ax2.transAxes, ha='left', va='top',
             fontsize=14, color=c_text,
             bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=5), zorder=5)

    ax2.set_xlabel('Общее количество нейронов в сессии', fontsize=13, labelpad=10, color=c_text)
    ax2.set_ylabel('Количество сильных нейронов (>3$\sigma$)', fontsize=13, labelpad=10, color=c_text)

    # ------------------------------------------
    # ОБЩЕЕ ФОРМАТИРОВАНИЕ ОПТИКИ
    # ------------------------------------------
    for ax in [ax1, ax2]:
        # Устанавливаем легкую прозрачность доверительного интервала (0.25 для мягкости)
        for collection in ax.collections:
            if type(collection).__name__ == 'PolyCollection':
                collection.set_alpha(0.25)

        # Очистка рамок
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        [s.set_color(c_text) for s in ax.spines.values()]

        # Настройка тиков и сетки
        ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, colors=c_text)
        ax.grid(axis='both', linestyle=':', alpha=0.4, color='#BDC3C7', zorder=0)

    plt.tight_layout(w_pad=3.0)

    # Сохранение
    save_path = output_path / 'r2_and_top_topology_dependencies_aesthetic.png'
    plt.savefig(save_path, dpi=1000, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Эстетичные графики (Пастельная палитра) успешно сохранены: {save_path}")


def plot_longitudinal_survival_aesthetic_selector(
        feature_type='linreg',  # Доступно: 'linreg', 'delta', 'z-delta', 'intense'
        match_dir='MatchData',
        models_dir='models_LNOF_Lasso',
        ablation_csv='stat_tables/full_ablation_stats.csv',
        database_csv='LNOF_SynchronizedData/LNOF_database.csv',
        output_dir='plots',
        model_is_zero_indexed=True):
    """
    Отслеживает выживаемость ансамблей с нормировкой на локальный Жаккард.
    Интеллектуальное позиционирование подписей: защита от наезда на ось X.
    Раздельное масштабирование Y-осей (правая строго до 1). Легенда справа.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    excluded_mice = ['J20', 'J21']
    match_files = list(Path(match_dir).glob("LNOF_*.csv"))

    df_abl = pd.read_csv(ablation_csv) if feature_type in ['delta', 'z-delta'] else None
    df_db = pd.read_csv(database_csv) if feature_type == 'intense' else None

    results = []

    print(f"\n{'=' * 75}")
    print(f"Расчет выживаемости (Режим: {feature_type.upper()} | Выживаемость с дня 1)")
    print(f"{'=' * 75}")

    for file_path in match_files:
        mouse = file_path.stem.replace('LNOF_', '')
        if mouse in excluded_mice:
            continue

        df_map = pd.read_csv(file_path, header=None)
        if df_map.shape[1] < 4:
            continue

        tracked_sets = []
        important_sets = []
        valid_chain = True

        for day_idx in range(4):
            day = f"{day_idx + 1}D"
            model_session = f"{mouse}_{day}"

            col_data = df_map[day_idx]
            tracked_mask = col_data.notna() & (col_data > 0)
            tracked_sets.append(set(df_map[tracked_mask].index))

            important_local_ids = []

            if feature_type == 'linreg':
                model_path = Path(models_dir) / f"{model_session}.pkl"
                if not model_path.exists(): valid_chain = False; break
                data = joblib.load(model_path)
                model = data.get('model')
                if not hasattr(model, 'coef_'): valid_chain = False; break

                coefs = np.array(model.coef_)
                if coefs.ndim > 1:
                    weights = np.linalg.norm(coefs, axis=0)
                else:
                    weights = np.abs(coefs)

                feat_names = np.array(data.get('feature_names'))
                valid_features = feat_names[weights > 0]

            elif feature_type in ['delta', 'z-delta']:
                if df_abl is None or df_abl.empty: valid_chain = False; break
                row = df_abl[df_abl['model_session'] == model_session]
                if row.empty: valid_chain = False; break
                row = row.iloc[0]

                base_r2 = row['baseline_r2']
                feat_cols = [c for c in row.index if c.startswith('ablation_')]
                abl_vals = row[feat_cols].values.astype(float)
                crits = base_r2 - abl_vals
                valid_mask = ~np.isnan(crits)

                if feature_type == 'delta':
                    active_idx = np.where(valid_mask & (crits > 0))[0]
                else:
                    valid_crits = crits[valid_mask]
                    if len(valid_crits) > 0 and np.std(valid_crits) > 0:
                        zcrits = np.zeros_like(crits)
                        zcrits[valid_mask] = (valid_crits - np.mean(valid_crits)) / np.std(valid_crits)
                        active_idx = np.where(valid_mask & (zcrits > 0))[0]
                    else:
                        active_idx = []
                valid_features = [feat_cols[i].replace('ablation_', '') for i in active_idx]

            elif feature_type == 'intense':
                if df_db is None or df_db.empty: valid_chain = False; break
                df_int = df_db[(df_db['mouse'] == mouse) & (df_db['session'] == day) & (df_db['feature'] == 'place')]
                df_int = df_int[df_int['me'] > 0]
                valid_features = [f"neuron_{int(idx)}" for idx in df_int['neuron_idx']]

            else:
                print(f"[ERROR] Неизвестный feature_type: {feature_type}")
                return

            for name in valid_features:
                match = re.search(r'\d+', str(name))
                if match:
                    n_id = int(match.group())
                    if model_is_zero_indexed:
                        n_id += 1
                    important_local_ids.append(n_id)

            important_mask = col_data.isin(important_local_ids)
            important_sets.append(set(df_map[important_mask].index))

        if not valid_chain or len(important_sets[0]) == 0:
            continue

        B_1 = len(tracked_sets[0])
        V_1 = len(important_sets[0])

        for step in range(4):
            day_num = step + 1

            B_d = len(set.intersection(*tracked_sets[:step + 1]))
            V_d = len(set.intersection(*important_sets[:step + 1]))

            if step == 0:
                J_adj = 1.0
            else:
                prev_set = tracked_sets[step - 1]
                curr_set = tracked_sets[step]
                intersect_adj = len(prev_set.intersection(curr_set))
                union_adj = len(prev_set.union(curr_set))
                J_adj = intersect_adj / union_adj if union_adj > 0 else 0

            base_frac = B_d / B_1 if B_1 > 0 else 0
            imp_frac = V_d / V_1 if V_1 > 0 else 0

            if J_adj > 0:
                norm_base = base_frac / J_adj
                norm_imp = imp_frac / J_adj
                norm_B_d = B_d / J_adj
                norm_V_d = V_d / J_adj
            else:
                norm_base = 0
                norm_imp = 0
                norm_B_d = 0
                norm_V_d = 0

            results.append({
                'mouse': mouse,
                'day': day_num,
                'base_frac': base_frac,
                'imp_frac': imp_frac,
                'norm_base': norm_base,
                'norm_imp': norm_imp,
                'B_d': B_d,
                'V_d': V_d,
                'norm_B_d': norm_B_d,
                'norm_V_d': norm_V_d
            })

        print(f"[{mouse}] Обработано. Базовый пул 1D: Все={B_1}, Целевые={V_1}")

    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("[ERROR] Нет данных для выбранного критерия.")
        return

    # =========================================================================
    # ЭСТЕТИЧЕСКАЯ ОТРИСОВКА
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')
    ax1, ax2 = axes

    c_text = '#2C3E50'
    c_agg_base = 'black'

    c_line_imp = '#D81B60'  # Magenta
    c_marker_imp = '#D81B60'
    c_ci_imp = '#D81B60'

    mice = df_res['mouse'].unique()
    ind_imp_colors = sns.color_palette("husl", len(mice))

    # --- 1. Индивидуальные сессии ---
    for i, mouse_id in enumerate(mice):
        m_data = df_res[df_res['mouse'] == mouse_id]

        ax1.plot(m_data['day'], m_data['norm_base'], color='grey', alpha=0.20, lw=1.5, zorder=1)
        ax1.plot(m_data['day'], m_data['norm_imp'], color=ind_imp_colors[i], alpha=0.30, lw=1.5, zorder=2)

        ax2.plot(m_data['day'], m_data['base_frac'], color='grey', alpha=0.20, lw=1.5, zorder=1)
        ax2.plot(m_data['day'], m_data['imp_frac'], color=ind_imp_colors[i], alpha=0.30, lw=1.5, zorder=2)

    # --- 2. Глобальные тренды (Агрегация + CI) ---
    sns.lineplot(data=df_res, x='day', y='norm_base', ax=ax1, color=c_agg_base,
                 errorbar=('ci', 95), marker='o', markersize=8, linewidth=2.5,
                 markeredgecolor='none', err_kws={'facecolor': 'grey', 'alpha': 0.20, 'linewidth': 0}, zorder=4)

    sns.lineplot(data=df_res, x='day', y='norm_imp', ax=ax1, color=c_line_imp,
                 errorbar=('ci', 95), marker='^', markersize=10, linewidth=2.5,
                 markerfacecolor=c_marker_imp, markeredgecolor='none',
                 err_kws={'facecolor': c_ci_imp, 'alpha': 0.25, 'linewidth': 0}, zorder=5)

    sns.lineplot(data=df_res, x='day', y='base_frac', ax=ax2, color=c_agg_base,
                 errorbar=('ci', 95), marker='o', markersize=8, linewidth=2.5,
                 markeredgecolor='none', err_kws={'facecolor': 'grey', 'alpha': 0.20, 'linewidth': 0}, zorder=4)

    sns.lineplot(data=df_res, x='day', y='imp_frac', ax=ax2, color=c_line_imp,
                 errorbar=('ci', 95), marker='^', markersize=10, linewidth=2.5,
                 markerfacecolor=c_marker_imp, markeredgecolor='none',
                 err_kws={'facecolor': c_ci_imp, 'alpha': 0.25, 'linewidth': 0}, zorder=5)

    # --- 3. Изолированные масштабы осей ---
    agg_stats = df_res.groupby('day').agg({
        'norm_base': 'mean', 'norm_imp': 'mean',
        'base_frac': 'mean', 'imp_frac': 'mean',
        'B_d': 'mean', 'V_d': 'mean',
        'norm_B_d': 'mean', 'norm_V_d': 'mean'
    }).reset_index()

    # Левый график (масштабируется по черной линии)
    max_black_norm = (agg_stats['norm_base'] + 2.5 * df_res.groupby('day')['norm_base'].sem()).max()
    y_max1 = max(1.05, max_black_norm * 1.15)
    ax1.set_ylim(-0.05, y_max1)

    # Правый график (жестко до 1, с воздухом для маркеров)
    y_max2 = 1.05
    ax2.set_ylim(-0.05, y_max2)

    # --- 4. Интеллектуальные аннотации с защитой от наезда на ось X ---
    def get_smart_offsets(y_b, y_i, y_max):
        """Возвращает смещения так, чтобы подписи не пересекали оси и друг друга."""
        off_b, va_b = 12, 'bottom'
        off_i, va_i = -15, 'top'

        # Разводим текст при пересечении
        if y_i > y_b:
            off_b, va_b = -15, 'top'
            off_i, va_i = 12, 'bottom'

        # Защита нижней оси X (Пропорциональное прижатие)
        if va_i == 'top' and y_i < y_max * 0.05:
            ratio = max(0.1, y_i / (y_max * 0.05))
            off_i = -3 - (12 * ratio)  # Плавно укорачиваем отступ от -15 до -3

        if va_b == 'top' and y_b < y_max * 0.05:
            ratio = max(0.1, y_b / (y_max * 0.05))
            off_b = -3 - (12 * ratio)

        return off_b, va_b, off_i, va_i

    for _, row in agg_stats.iterrows():
        d = row['day']

        # Аннотации левого графика
        y_b1, y_i1 = row['norm_base'], row['norm_imp']
        off_b1, va_b1, off_i1, va_i1 = get_smart_offsets(y_b1, y_i1, y_max1)

        ax1.annotate(f"{int(np.round(row['norm_B_d']))}", (d, y_b1), textcoords="offset points", xytext=(0, off_b1),
                     ha='center', va=va_b1, fontsize=10, color=c_agg_base, weight='bold', zorder=6)
        ax1.annotate(f"{int(np.round(row['norm_V_d']))}", (d, y_i1), textcoords="offset points", xytext=(0, off_i1),
                     ha='center', va=va_i1, fontsize=10, color=c_line_imp, weight='bold', zorder=6)

        # Аннотации правого графика
        y_b2, y_i2 = row['base_frac'], row['imp_frac']
        off_b2, va_b2, off_i2, va_i2 = get_smart_offsets(y_b2, y_i2, y_max2)

        ax2.annotate(f"{int(np.round(row['B_d']))}", (d, y_b2), textcoords="offset points", xytext=(0, off_b2),
                     ha='center', va=va_b2, fontsize=10, color=c_agg_base, weight='bold', zorder=6)
        ax2.annotate(f"{int(np.round(row['V_d']))}", (d, y_i2), textcoords="offset points", xytext=(0, off_i2),
                     ha='center', va=va_i2, fontsize=10, color=c_line_imp, weight='bold', zorder=6)

    # --- 5. Оформление ---
    ax1.set_title("Нормированный дрейф", fontsize=15, pad=15, fontweight='bold', color=c_text)
    ax2.set_title("Абсолютный дрейф", fontsize=15, pad=15, fontweight='bold', color=c_text)

    ax1.set_ylabel("Доля нейронов соответствующих дню 1", fontsize=14, labelpad=10, color=c_text)
    ax2.set_ylabel("")

    for ax in axes:
        ax.set_facecolor('white')
        ax.set_xlabel("День", fontsize=13, labelpad=10, color=c_text)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlim(0.8, 4.2)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        [s.set_color(c_text) for s in ax.spines.values()]

        ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, colors=c_text)
        ax.grid(axis='both', linestyle=':', alpha=0.4, color='#BDC3C7', zorder=0)

    # --- 6. Кастомная легенда (Только на правом графике) ---
    labels_dict = {
        'linreg': 'Участвующие',
        'delta': 'Важные',
        'z-delta': 'Важные',
        'intense': 'Информативные'
    }
    target_label = labels_dict.get(feature_type, 'Целевые')

    legend_elements = [
        mlines.Line2D([], [], color=c_agg_base, marker='o', markersize=8, markeredgecolor='none', lw=2.5,
                      label='Все нейроны'),
        mlines.Line2D([], [], color=c_line_imp, marker='^', markerfacecolor=c_marker_imp, markeredgecolor='none',
                      markersize=10, lw=2.5, label=target_label)
    ]
    ax2.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=12, labelcolor=c_text)

    plt.tight_layout(w_pad=3.0)

    save_path = output_path / f'longitudinal_survival_{feature_type}.png'
    plt.savefig(save_path, dpi=1000, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Эстетичный график (режим {feature_type}) сохранен: {save_path}")


def plot_top_neurons_longitudinal_counts(
        top_csv='stat_tables/top_neurons_ablation_3.0sigma.csv',
        match_dir='MatchData',
        output_dir='plots',
        model_is_zero_indexed=True):
    """
    Отслеживает КОЛИЧЕСТВО сильных нейронов (>3 sigma) по дням.
    Эстетика: Alpine Green (DS4 Mint).
    Индивидуальные траектории только для выживших нейронов (разноцветные).
    Умные аннотации текста (зеленая линия всегда подписана сверху, кроме 1-го дня).
    Легенда на правом графике.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    excluded_mice = ['J20', 'J21']
    match_files = list(Path(match_dir).glob("LNOF_*.csv"))

    try:
        df_top = pd.read_csv(top_csv)
    except FileNotFoundError:
        print(f"[ERROR] Не найден файл: {top_csv}")
        return

    # Отфильтруем только строки с именами нейронов
    df_names = df_top[df_top['data_type'] == 'Neuron Names'].copy()
    top_cols = [c for c in df_names.columns if c.startswith('top_')]

    results = []

    print(f"\n{'=' * 75}")
    print("Расчет выживаемости (Абсолютные количества | Alpine Green Theme)")
    print(f"{'=' * 75}")

    for file_path in match_files:
        mouse = file_path.stem.replace('LNOF_', '')
        if mouse in excluded_mice:
            continue

        df_map = pd.read_csv(file_path, header=None)
        if df_map.shape[1] < 4:
            continue

        tracked_sets = []
        important_sets = []
        valid_chain = True

        for day_idx in range(4):
            day = f"{day_idx + 1}D"

            # 1. Все затреканные клетки дня (для Жаккарда)
            col_data = df_map[day_idx]
            tracked_mask = col_data.notna() & (col_data > 0)
            tracked_sets.append(set(df_map[tracked_mask].index))

            # 2. Извлекаем топовые нейроны из CSV
            row = df_names[(df_names['mouse'] == mouse) & (df_names['session'] == day)]

            important_local_ids = []
            if not row.empty:
                # Берем все заполненные имена нейронов для этой сессии
                neuron_names = row.iloc[0][top_cols].dropna().values

                for name in neuron_names:
                    match = re.search(r'\d+', str(name))
                    if match:
                        n_id = int(match.group())
                        if model_is_zero_indexed:
                            n_id += 1
                        important_local_ids.append(n_id)

            # Находим глобальные индексы важных нейронов
            important_mask = col_data.isin(important_local_ids)
            important_sets.append(set(df_map[important_mask].index))

        # Если в первый день важных нет, пропускаем
        if len(important_sets[0]) == 0:
            continue

        # --- МАТЕМАТИЧЕСКИЙ РАСЧЕТ ---
        for step in range(4):
            day_num = step + 1

            N_top_d = len(important_sets[step])
            V_d = len(set.intersection(*important_sets[:step + 1]))

            if step == 0:
                J_adj = 1.0
            else:
                prev_set = tracked_sets[step - 1]
                curr_set = tracked_sets[step]
                intersect_adj = len(prev_set.intersection(curr_set))
                union_adj = len(prev_set.union(curr_set))
                J_adj = intersect_adj / union_adj if union_adj > 0 else 0

            if J_adj > 0:
                norm_N_top = N_top_d / J_adj
                norm_V_d = V_d / J_adj
            else:
                norm_N_top = 0
                norm_V_d = 0

            results.append({
                'mouse': mouse,
                'day': day_num,
                'N_top_d': N_top_d,
                'V_d': V_d,
                'norm_N_top': norm_N_top,
                'norm_V_d': norm_V_d
            })

        print(f"[{mouse}] Обработано. В 1D найдено топовых: {len(important_sets[0])}")

    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("[ERROR] Нет данных для построения.")
        return

    # =========================================================================
    # ЭСТЕТИЧЕСКАЯ ОТРИСОВКА (Alpine Green Theme)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')
    ax1, ax2 = axes

    c_text = '#2C3E50'
    c_agg_base = 'black'

    # --- Свежий Alpine Green (Мятно-изумрудный, как DS4) ---
    c_line_imp = '#12A57A'
    c_marker_imp = '#12A57A'
    c_ci_imp = '#12A57A'

    mice = df_res['mouse'].unique()
    ind_imp_colors = sns.color_palette("husl", len(mice))

    # --- 1. Индивидуальные сессии (ТОЛЬКО для целевых нейронов) ---
    for i, mouse_id in enumerate(mice):
        m_data = df_res[df_res['mouse'] == mouse_id]

        # Левый: Нормированные целевые
        ax1.plot(m_data['day'], m_data['norm_V_d'], color=ind_imp_colors[i], alpha=0.30, lw=1.5, zorder=2)

        # Правый: Абсолютные целевые
        ax2.plot(m_data['day'], m_data['V_d'], color=ind_imp_colors[i], alpha=0.30, lw=1.5, zorder=2)

    # --- 2. Глобальные тренды (Агрегация + CI) ---
    # Левый
    sns.lineplot(data=df_res, x='day', y='norm_N_top', ax=ax1, color=c_agg_base,
                 errorbar=('ci', 95), marker='o', markersize=8, linewidth=2.5,
                 markeredgecolor='none', err_kws={'facecolor': 'grey', 'alpha': 0.20, 'linewidth': 0}, zorder=4)

    sns.lineplot(data=df_res, x='day', y='norm_V_d', ax=ax1, color=c_line_imp,
                 errorbar=('ci', 95), marker='^', markersize=10, linewidth=2.5,
                 markerfacecolor=c_marker_imp, markeredgecolor='none',
                 err_kws={'facecolor': c_ci_imp, 'alpha': 0.25, 'linewidth': 0}, zorder=5)

    # Правый
    sns.lineplot(data=df_res, x='day', y='N_top_d', ax=ax2, color=c_agg_base,
                 errorbar=('ci', 95), marker='o', markersize=8, linewidth=2.5,
                 markeredgecolor='none', err_kws={'facecolor': 'grey', 'alpha': 0.20, 'linewidth': 0}, zorder=4)

    sns.lineplot(data=df_res, x='day', y='V_d', ax=ax2, color=c_line_imp,
                 errorbar=('ci', 95), marker='^', markersize=10, linewidth=2.5,
                 markerfacecolor=c_marker_imp, markeredgecolor='none',
                 err_kws={'facecolor': c_ci_imp, 'alpha': 0.25, 'linewidth': 0}, zorder=5)

    # --- 3. Интеллектуальные аннотации с учетом номера Дня ---
    def get_smart_offsets(y_b, y_i, y_max, current_day):
        """Возвращает смещения так, чтобы подписи не пересекали оси и друг друга."""
        off_b, va_b = 12, 'bottom'  # Черная по умолчанию сверху

        # Для 1-го дня зеленая линия снизу, для остальных - всегда сверху
        if current_day == 1:
            off_i, va_i = -15, 'top'
        else:
            off_i, va_i = 12, 'bottom'

        # Разводим текст при пересечении
        if current_day == 1 and y_i > y_b:
            off_b, va_b = -15, 'top'
            off_i, va_i = 12, 'bottom'
        elif current_day > 1 and abs(y_b - y_i) < y_max * 0.08:
            # Если зеленая подпись идет наверх, и черная линия близко,
            # мы уступаем место и отправляем черную подпись вниз, чтобы они не слиплись
            off_b, va_b = -15, 'top'

        # Защита нижней оси X (Пропорциональное прижатие)
        if va_i == 'top' and y_i < y_max * 0.05:
            ratio = max(0.1, y_i / (y_max * 0.05))
            off_i = -3 - (12 * ratio)

        if va_b == 'top' and y_b < y_max * 0.05:
            ratio = max(0.1, y_b / (y_max * 0.05))
            off_b = -3 - (12 * ratio)

        return off_b, va_b, off_i, va_i

    agg_stats = df_res.groupby('day').agg({
        'norm_N_top': 'mean', 'norm_V_d': 'mean',
        'N_top_d': 'mean', 'V_d': 'mean'
    }).reset_index()

    # Считаем максимальные значения для правильного отступа алгоритма
    max_left = max(agg_stats['norm_N_top'].max(), agg_stats['norm_V_d'].max())
    max_right = max(agg_stats['N_top_d'].max(), agg_stats['V_d'].max())
    global_max_left = max_left * 1.25
    global_max_right = max_right * 1.25

    ax1.set_ylim(-0.5, global_max_left)
    ax2.set_ylim(-0.5, global_max_right)

    for _, row in agg_stats.iterrows():
        d = row['day']

        # Левый график (Нормированные) - передаем номер дня 'd'
        y_b1, y_i1 = row['norm_N_top'], row['norm_V_d']
        off_b1, va_b1, off_i1, va_i1 = get_smart_offsets(y_b1, y_i1, global_max_left, d)

        ax1.annotate(f"{y_b1:.1f}", (d, y_b1), textcoords="offset points", xytext=(0, off_b1),
                     ha='center', va=va_b1, fontsize=11, color=c_agg_base, weight='bold', zorder=6)
        ax1.annotate(f"{y_i1:.1f}", (d, y_i1), textcoords="offset points", xytext=(0, off_i1),
                     ha='center', va=va_i1, fontsize=11, color=c_line_imp, weight='bold', zorder=6)

        # Правый график (Абсолютные) - передаем номер дня 'd'
        y_b2, y_i2 = row['N_top_d'], row['V_d']
        off_b2, va_b2, off_i2, va_i2 = get_smart_offsets(y_b2, y_i2, global_max_right, d)

        ax2.annotate(f"{y_b2:.1f}", (d, y_b2), textcoords="offset points", xytext=(0, off_b2),
                     ha='center', va=va_b2, fontsize=11, color=c_agg_base, weight='bold', zorder=6)
        ax2.annotate(f"{y_i2:.1f}", (d, y_i2), textcoords="offset points", xytext=(0, off_i2),
                     ha='center', va=va_i2, fontsize=11, color=c_line_imp, weight='bold', zorder=6)

    # --- 4. Оформление ---
    ax1.set_title("Нормированный дрейф", fontsize=15, pad=15, fontweight='bold', color=c_text)
    ax2.set_title("Абсолютный дрейф", fontsize=15, pad=15, fontweight='bold', color=c_text)

    ax1.set_ylabel("Количество сильных нейронов", fontsize=14, labelpad=10, color=c_text)
    ax2.set_ylabel("")

    for ax in axes:
        ax.set_facecolor('white')
        ax.set_xlabel("День", fontsize=13, labelpad=10, color=c_text)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlim(0.8, 4.2)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        [s.set_color(c_text) for s in ax.spines.values()]

        ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, colors=c_text)
        ax.grid(axis='both', linestyle=':', alpha=0.4, color='#BDC3C7', zorder=0)

    # Легенда
    legend_elements = [
        mlines.Line2D([], [], color=c_agg_base, marker='o', markersize=8, markeredgecolor='none', lw=2.5,
                      label='Все нейроны'),
        mlines.Line2D([], [], color=c_line_imp, marker='^', markerfacecolor=c_marker_imp, markeredgecolor='none',
                      markersize=10, lw=2.5, label='Соответствующие дню 1')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=12, labelcolor=c_text)

    plt.tight_layout(w_pad=3.0)

    save_path = output_path / 'longitudinal_top_neurons_counts_alpine.png'
    plt.savefig(save_path, dpi=1000, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Эстетичный график с количествами сохранен: {save_path}")


def plot_top_neurons_survival_absolute_with_baseline(
        top_csv='stat_tables/top_neurons_ablation_3.0sigma.csv',
        match_dir='MatchData',
        models_dir='models_LNOF_Lasso',
        output_dir='plots',
        model_is_zero_indexed=True):
    """
    Отслеживает абсолютное количество топовых нейронов (>3 sigma) по дням.
    Отрисовывает 1 график (сохраняя пропорции).
    Индивидуальные сессии разноцветные.
    Добавлен серый пунктирный бейзлайн случайного физического выживания ВАЖНЫХ нейронов из 1-го дня.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    excluded_mice = ['J20', 'J21']
    match_files = list(Path(match_dir).glob("LNOF_*.csv"))

    try:
        df_top = pd.read_csv(top_csv)
    except FileNotFoundError:
        print(f"[ERROR] Не найден файл: {top_csv}")
        return

    df_names = df_top[df_top['data_type'] == 'Neuron Names'].copy()
    top_cols = [c for c in df_names.columns if c.startswith('top_')]

    results = []
    rand_results = []

    print(f"\n{'=' * 75}")
    print("Расчет выживаемости (Абсолютные количества + Случайный бейзлайн важных x10)")
    print(f"{'=' * 75}")

    for file_path in match_files:
        mouse = file_path.stem.replace('LNOF_', '')
        if mouse in excluded_mice:
            continue

        df_map = pd.read_csv(file_path, header=None)
        if df_map.shape[1] < 4:
            continue

        tracked_sets = []
        important_sets = []

        for day_idx in range(4):
            day = f"{day_idx + 1}D"

            col_data = df_map[day_idx]
            tracked_mask = col_data.notna() & (col_data > 0)
            tracked_sets.append(set(df_map[tracked_mask].index))

            row = df_names[(df_names['mouse'] == mouse) & (df_names['session'] == day)]
            important_local_ids = []

            if not row.empty:
                neuron_names = row.iloc[0][top_cols].dropna().values
                for name in neuron_names:
                    match = re.search(r'\d+', str(name))
                    if match:
                        n_id = int(match.group())
                        if model_is_zero_indexed:
                            n_id += 1
                        important_local_ids.append(n_id)

            important_mask = col_data.isin(important_local_ids)
            important_sets.append(set(df_map[important_mask].index))

        if len(important_sets[0]) == 0:
            continue

        # --- 1. Извлекаем пул ВАЖНЫХ нейронов из модели 1-го дня ---
        model_path = Path(models_dir) / f"{mouse}_1D.pkl"
        day1_important_pool = []

        if model_path.exists():
            m_data = joblib.load(model_path)
            model = m_data.get('model')
            if hasattr(model, 'coef_'):
                coefs = np.array(model.coef_)
                weights = np.linalg.norm(coefs, axis=0) if coefs.ndim > 1 else np.abs(coefs)
                feat_names = np.array(m_data.get('feature_names'))

                for name in feat_names[weights > 0]:
                    match = re.search(r'\d+', str(name))
                    if match:
                        n_id = int(match.group())
                        if model_is_zero_indexed:
                            n_id += 1
                        day1_important_pool.append(n_id)

        # Убеждаемся, что эти важные нейроны реально затрекались в 1-й день
        day1_important_pool = list(set(day1_important_pool).intersection(tracked_sets[0]))
        num_strong_day1 = len(important_sets[0])

        # --- 2. Расчет случайного бейзлайна (10 итераций из ВАЖНЫХ) ---
        for _ in range(10):
            # Защита: берем не больше, чем есть в пуле важных
            sample_size = min(num_strong_day1, len(day1_important_pool))

            if sample_size > 0:
                rand_day1 = set(np.random.choice(day1_important_pool, size=sample_size, replace=False))
            else:
                rand_day1 = set()

            for step in range(4):
                # Смотрим, сколько из этих случайных физически отследились в день D
                rand_survived = len(rand_day1.intersection(*tracked_sets[:step + 1]))
                rand_results.append({
                    'mouse': mouse,
                    'day': step + 1,
                    'rand_survived': rand_survived
                })

        # --- 3. Расчет целевых метрик ---
        for step in range(4):
            day_num = step + 1
            N_top_d = len(important_sets[step])
            V_d = len(set.intersection(*important_sets[:step + 1]))

            results.append({
                'mouse': mouse,
                'day': day_num,
                'N_top_d': N_top_d,
                'V_d': V_d
            })

        print(f"[{mouse}] Обработано. Сильных в 1D: {num_strong_day1}")

    df_res = pd.DataFrame(results)
    df_rand = pd.DataFrame(rand_results)

    if df_res.empty:
        print("[ERROR] Нет данных для построения.")
        return

    # =========================================================================
    # ЭСТЕТИЧЕСКАЯ ОТРИСОВКА (Alpine Green Theme, 1 график)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(7, 6))  # Сохраняем соотношение сторон одной панели
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    c_text = '#2C3E50'
    c_agg_base = 'black'
    c_line_imp = '#12A57A'  # Alpine Green
    c_rand = 'grey'

    mice = df_res['mouse'].unique()
    individual_session_colors = sns.color_palette("husl", len(mice))

    # 1. Индивидуальные сессии (РАЗНОЦВЕТНЫЕ траектории для выживших)
    for i, mouse_id in enumerate(mice):
        m_data = df_res[df_res['mouse'] == mouse_id]
        ax.plot(m_data['day'], m_data['V_d'], color=individual_session_colors[i], alpha=0.40, lw=1.5, zorder=2)

    # 2. Глобальные тренды
    # Черная линия
    sns.lineplot(data=df_res, x='day', y='N_top_d', ax=ax, color=c_agg_base,
                 errorbar=('ci', 95), marker='o', markersize=8, linewidth=2.5,
                 markeredgecolor='none', err_kws={'facecolor': 'grey', 'alpha': 0.20, 'linewidth': 0}, zorder=5)

    # Серая пунктирная линия (Случайные ВАЖНЫЕ)
    sns.lineplot(data=df_rand, x='day', y='rand_survived', ax=ax, color=c_rand,
                 errorbar=('ci', 95), linestyle='--', marker='s', markersize=6, linewidth=1.5,
                 markeredgecolor='none', err_kws={'facecolor': c_rand, 'alpha': 0.15, 'linewidth': 0}, zorder=3)

    # Alpine Green линия
    sns.lineplot(data=df_res, x='day', y='V_d', ax=ax, color=c_line_imp,
                 errorbar=('ci', 95), marker='^', markersize=10, linewidth=2.5,
                 markerfacecolor=c_line_imp, markeredgecolor='none',
                 err_kws={'facecolor': c_line_imp, 'alpha': 0.25, 'linewidth': 0}, zorder=6)

    # 3. Умные аннотации
    agg_res = df_res.groupby('day').mean(numeric_only=True).reset_index()
    agg_rand = df_rand.groupby('day').mean(numeric_only=True).reset_index()

    global_max = max(agg_res['N_top_d'].max(), agg_res['V_d'].max(), agg_rand['rand_survived'].max()) * 1.25
    ax.set_ylim(-0.5, global_max)

    for d in range(1, 5):
        y_b = agg_res.loc[agg_res['day'] == d, 'N_top_d'].values[0]
        y_v = agg_res.loc[agg_res['day'] == d, 'V_d'].values[0]
        y_r = agg_rand.loc[agg_rand['day'] == d, 'rand_survived'].values[0]

        if d == 1:
            ax.annotate(f"{y_v:.1f}", (d, y_v), textcoords="offset points", xytext=(0, 12),
                        ha='center', va='bottom', fontsize=11, color=c_agg_base, weight='bold', zorder=7)
        else:
            points = [(y_b, c_agg_base), (y_r, c_rand), (y_v, c_line_imp)]
            points.sort(key=lambda x: x[0])

            for i, (y_val, color) in enumerate(points):
                if i == 0:
                    off, va = -14, 'top'
                elif i == len(points) - 1:
                    off, va = 12, 'bottom'
                else:
                    dist_up = points[i + 1][0] - y_val
                    dist_down = y_val - points[i - 1][0]
                    if dist_up > dist_down:
                        off, va = 12, 'bottom'
                    else:
                        off, va = -14, 'top'

                if va == 'top' and y_val < global_max * 0.05:
                    off, va = 12, 'bottom'

                ax.annotate(f"{y_val:.1f}", (d, y_val), textcoords="offset points", xytext=(0, off),
                            ha='center', va=va, fontsize=11, color=color, weight='bold', zorder=7)

    # 4. Оформление осей и рамки
    ax.set_title("Абсолютный дрейф", fontsize=14, pad=15, fontweight='bold', color=c_text)
    ax.set_ylabel("Количество сильных нейронов", fontsize=13, labelpad=10, color=c_text)
    ax.set_xlabel("День", fontsize=13, labelpad=10, color=c_text)

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlim(0.8, 4.2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    [s.set_color(c_text) for s in ax.spines.values()]

    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, colors=c_text)
    ax.grid(axis='both', linestyle=':', alpha=0.4, color='#BDC3C7', zorder=0)

    # 5. Легенда (Добавлен бейзлайн)
    legend_elements = [
        mlines.Line2D([], [], color=c_agg_base, marker='o', markersize=8, markeredgecolor='none', lw=2.5,
                      label='Все нейроны'),
        mlines.Line2D([], [], color=c_rand, marker='s', markersize=6, linestyle='--', markeredgecolor='none', lw=1.5,
                      label='Случайные участвующие'),
        mlines.Line2D([], [], color=c_line_imp, marker='^', markerfacecolor=c_line_imp, markeredgecolor='none',
                      markersize=10, lw=2.5, label='Соответствующие 1 дню')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=11, labelcolor=c_text)

    plt.tight_layout()

    save_path = output_path / 'longitudinal_top_neurons_absolute_with_baseline_colorful.png'
    plt.savefig(save_path, dpi=1000, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ График абсолютного дрейфа со строгим бейзлайном сохранен: {save_path}")


def plot_r2_and_top_neurons_dependencies_exact_colors(top_metrics_csv='stat_tables/top_vs_rest_metrics_abl.csv',
                                                      unified_csv='stat_tables/unified_model_performance_report.csv',
                                                      output_dir='plots'):
    """
    Строит два графика:
    1. Зависимость R2 от количества сильных нейронов (Топовые vs Остальные).
    2. Зависимость R2 топовых от R2 остальных важных нейронов (со Спирменом).
    Стиль: Научный минимализм. Яркая палитра.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Загрузка данных (нам нужна только таблица top_vs_rest, так как там есть и top_r2, и rest_r2)
    try:
        df = pd.read_csv(top_metrics_csv)
    except FileNotFoundError as e:
        print(f"[ERROR] Не найден файл: {e}")
        return

    # 3. Очистка от NaN для нужных колонок
    df_clean = df.dropna(subset=['top_count', 'top_r2', 'rest_r2'])

    if df_clean.empty:
        print("[ERROR] Недостаточно валидных данных для построения графиков.")
        return

    # 4. Расчет коэффициента Спирмена для правого графика (Rest R2 vs Top R2)
    corr, pval = spearmanr(df_clean['rest_r2'], df_clean['top_r2'])

    def get_stars(p):
        return r"{\ast}{\ast}{\ast}" if p < 0.001 else r"{\ast}{\ast}" if p < 0.01 else r"{\ast}" if p < 0.05 else "{ns}"

    # ==========================================
    # ЭСТЕТИЧЕСКАЯ ОТРИСОВКА (Точные цвета)
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax1, ax2 = axes

    c_text = '#2C3E50'

    # Левый график:
    c_golden_yellow = '#FBC02D'  # Яркий, золотой (Топовые нейроны)
    c_vibrant_green = '#43A047'  # Насыщенный зеленый (Остальные важные)

    # Правый график:
    c_deep_purple = '#8E24AA'  # Глубокий фиолетовый (Корреляция R2)
    # ----------------------------------------

    # ------------------------------------------
    # График 1 (Слева): R2 vs Количество сильных нейронов
    # ------------------------------------------

    # 1. Точки: "Только сильные нейроны" (желтые, zorder повыше)
    ax1.scatter(df_clean['top_count'], df_clean['top_r2'],
                color=c_golden_yellow, alpha=0.7, s=60, edgecolors='none', zorder=4,
                label='Сильные нейроны (>3$\sigma$)')

    # 2. Точки: "Остальные важные нейроны" (зеленые, zorder пониже)
    ax1.scatter(df_clean['top_count'], df_clean['rest_r2'],
                color=c_vibrant_green, alpha=0.7, s=60, edgecolors='none', zorder=3,
                label='Значимые нейроны')

    # 3. Линии регрессии (прозрачные CI, zorder=2)
    sns.regplot(data=df_clean, x='top_count', y='top_r2', ax=ax1,
                scatter=False, color=c_golden_yellow,
                line_kws={'linestyle': '--', 'linewidth': 2.5, 'alpha': 0.9, 'zorder': 2})
    sns.regplot(data=df_clean, x='top_count', y='rest_r2', ax=ax1,
                scatter=False, color=c_vibrant_green,
                line_kws={'linestyle': '--', 'linewidth': 2.5, 'alpha': 0.9, 'zorder': 2})

    # Легенда
    handles, labels = ax1.get_legend_handles_labels()
    # Оставляем порядок таким, чтобы желтые (топ) были сверху, зеленые (остальные) снизу
    ax1.legend(handles, labels, fontsize=12, loc='lower right', frameon=False, labelcolor=c_text)

    ax1.set_xlabel('Количество сильных нейронов (>3$\sigma$)', fontsize=13, labelpad=10, color=c_text)
    ax1.set_ylabel(r'$R^2$', fontsize=15, labelpad=10, color=c_text)

    # ------------------------------------------
    # График 2 (Справа): R2 сильных vs R2 остальных
    # ------------------------------------------
    sns.regplot(
        data=df_clean, x='rest_r2', y='top_r2', ax=ax2,
        color=c_deep_purple,
        scatter_kws={'alpha': 0.8, 's': 70, 'edgecolors': 'none', 'zorder': 3},
        line_kws={'linestyle': '--', 'linewidth': 2.5, 'alpha': 0.9, 'zorder': 2}
    )

    # Текст Спирмена в левом верхнем углу
    stars = get_stars(pval)
    stat_text = f"$\\rho = {corr:.2f}^{{{stars}}}$"

    ax2.text(0.05, 0.93, stat_text,
             transform=ax2.transAxes, ha='left', va='top',
             fontsize=16, color=c_text, weight='bold',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=5), zorder=5)

    ax2.set_xlabel('$R^2$, значимые нейроны', fontsize=13, labelpad=10, color=c_text)
    ax2.set_ylabel('$R^2$, сильные нейроны (>3$\sigma$)', fontsize=13, labelpad=10, color=c_text)

    # ------------------------------------------
    # ОБЩЕЕ ФОРМАТИРОВАНИЕ ОПТИКИ
    # ------------------------------------------
    for ax in [ax1, ax2]:
        # Делаем доверительный интервал (CI) более насыщенным
        for collection in ax.collections:
            if type(collection).__name__ == 'PolyCollection':
                collection.set_alpha(0.3)

        # Очистка рамок
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        [s.set_color(c_text) for s in ax.spines.values()]

        # Настройка тиков и сетки
        ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, colors=c_text, length=6)
        ax.grid(axis='both', linestyle=':', alpha=0.5, color='#BDC3C7', zorder=0)

    # Подгонка лимитов осей для левого графика, чтобы легенда поместилась красиво
    ax1.set_xlim(left=0)

    plt.tight_layout(w_pad=3.0)

    # Сохранение
    save_path = output_path / 'r2_top_vs_rest_dependencies.png'
    plt.savefig(save_path, dpi=1000, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Графики с точными яркими цветами успешно сохранены: {save_path}")


def plot_r2_and_network_size_dependencies(unified_csv='stat_tables/unified_model_performance_report.csv',
                                          output_dir='plots'):
    """
    Строит 3 графика зависимостей:
    1. R2 от общего количества нейронов в сессии.
    2. R2 от доли активных нейронов (Lasso weights > 0).
    3. Доли активных нейронов от общего количества нейронов.
    Исключает мышей J20 и J21.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Загрузка данных
    path = Path(unified_csv)
    if not path.exists():
        print(f"[ERROR] Не найден файл: {path}")
        return

    df = pd.read_csv(path)

    # Исключаем J20 и J21
    excluded_mice = []
    if 'mouse' in df.columns:
        df = df[~df['mouse'].isin(excluded_mice)].copy()
    elif 'model_name' in df.columns:
        df = df[~df['model_name'].str.startswith(tuple(excluded_mice))].copy()

    # Проверяем/создаем колонку с долей активных нейронов
    if 'important_fraction' not in df.columns:
        if 'important_neurons_count' in df.columns and 'total_neurons' in df.columns:
            df['important_fraction'] = df['important_neurons_count'] / df['total_neurons']
        else:
            print("[ERROR] В таблице нет данных о количестве важных нейронов.")
            return

    # Переводим в проценты для лучшей читаемости на графике
    df['important_percentage'] = df['important_fraction'] * 100

    # 3. Очистка от NaN
    df_clean = df.dropna(subset=['r2', 'total_neurons', 'important_percentage'])

    if df_clean.empty:
        print("[ERROR] Недостаточно валидных данных для построения графиков.")
        return

    def get_stars(p):
        return r"{\ast}{\ast}{\ast}" if p < 0.001 else r"{\ast}{\ast}" if p < 0.01 else r"{\ast}" if p < 0.05 else "^{ns}"

    # Расчет корреляций Спирмена
    corr1, pval1 = spearmanr(df_clean['total_neurons'], df_clean['r2'])
    corr2, pval2 = spearmanr(df_clean['important_percentage'], df_clean['r2'])
    corr3, pval3 = spearmanr(df_clean['total_neurons'], df_clean['important_neurons_count'])

    # ==========================================
    # ЭСТЕТИЧЕСКАЯ ОТРИСОВКА
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    c_text = '#2C3E50'

    # Цветовая палитра
    c_color1 = '#3498DB'  # Синий
    c_color2 = '#E74C3C'  # Красный
    c_color3 = '#9B59B6'  # Фиолетовый

    # ------------------------------------------
    # График 1: R2 vs Total Neurons
    # ------------------------------------------
    ax1 = axes[0]
    sns.regplot(
        data=df_clean, x='total_neurons', y='r2', ax=ax1,
        color=c_color1,
        scatter_kws={'alpha': 0.7, 's': 70, 'edgecolors': 'none', 'zorder': 3},
        line_kws={'linestyle': '--', 'linewidth': 2.5, 'alpha': 0.8, 'zorder': 2}
    )

    stat_text1 = f"$\\rho = {corr1:.2f}^{{{get_stars(pval1)}}}$"
    ax1.text(0.05, 0.95, stat_text1, transform=ax1.transAxes, ha='left', va='top',
             fontsize=15, color=c_text, weight='bold',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=5), zorder=5)

    ax1.set_xlabel('Общее количество нейронов', fontsize=13, labelpad=10, color=c_text)
    ax1.set_ylabel('$R^2$', fontsize=14, labelpad=10, color=c_text)
    # ax1.set_title("Качество vs Размер сети", fontsize=15, pad=15, fontweight='bold', color=c_text)

    # ------------------------------------------
    # График 2: R2 vs Active Fraction
    # ------------------------------------------
    ax2 = axes[1]
    sns.regplot(
        data=df_clean, x='important_percentage', y='r2', ax=ax2,
        color=c_color2,
        scatter_kws={'alpha': 0.7, 's': 70, 'edgecolors': 'none', 'zorder': 3},
        line_kws={'linestyle': '--', 'linewidth': 2.5, 'alpha': 0.8, 'zorder': 2}
    )

    stat_text2 = f"$\\rho = {corr2:.2f}^{{{get_stars(pval2)}}}$"
    ax2.text(0.05, 0.95, stat_text2, transform=ax2.transAxes, ha='left', va='top',
             fontsize=15, color=c_text, weight='bold',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=5), zorder=5)

    ax2.set_xlabel('Доля участвующих нейронов (%)', fontsize=13, labelpad=10, color=c_text)
    ax2.set_ylabel('$R^2$', fontsize=14, labelpad=10, color=c_text)
    # ax2.set_title("Качество vs Разреженность ансамбля", fontsize=15, pad=15, fontweight='bold', color=c_text)

    # ------------------------------------------
    # График 3: Active Fraction vs Total Neurons
    # ------------------------------------------
    ax3 = axes[2]
    sns.regplot(
        data=df_clean, x='total_neurons', y='important_neurons_count', ax=ax3,
        color=c_color3,
        scatter_kws={'alpha': 0.7, 's': 70, 'edgecolors': 'none', 'zorder': 3},
        line_kws={'linestyle': '--', 'linewidth': 2.5, 'alpha': 0.8, 'zorder': 2}
    )

    stat_text3 = f"$\\rho = {corr3:.2f}^{{{get_stars(pval3)}}}$"
    ax3.text(0.95, 0.95, stat_text3, transform=ax3.transAxes, ha='right', va='top',
             fontsize=15, color=c_text, weight='bold',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=5), zorder=5)

    ax3.set_xlabel('Общее количество нейронов', fontsize=13, labelpad=10, color=c_text)
    ax3.set_ylabel('Количество участвующих нейронов', fontsize=14, labelpad=10, color=c_text)
    # ax3.set_title("Масштабирование ансамбля", fontsize=15, pad=15, fontweight='bold', color=c_text)

    # ------------------------------------------
    # ОБЩЕЕ ФОРМАТИРОВАНИЕ ОПТИКИ
    # ------------------------------------------
    for ax in axes:
        # Полупрозрачные доверительные интервалы (CI)
        for collection in ax.collections:
            if type(collection).__name__ == 'PolyCollection':
                collection.set_alpha(0.2)

        # Очистка рамок
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        [s.set_color(c_text) for s in ax.spines.values()]

        ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, colors=c_text, length=6)
        ax.grid(axis='both', linestyle=':', alpha=0.5, color='#BDC3C7', zorder=0)

    plt.tight_layout(w_pad=4.0)

    # Сохранение
    save_path = output_path / 'r2_topology_and_sparsity_dependencies.png'
    plt.savefig(save_path, dpi=1000, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n✓ Успешно! Тройной график зависимостей сохранен: {save_path}")


def plot_cross_session_r2_dynamics_aesthetic(
        match_dir='MatchData',
        models_dir='models_LNOF_Lasso',
        data_dir='LNOF_SynchronizedData',
        output_dir='plots',
        test_size=0.25,
        model_is_zero_indexed=True):
    """
    Строит графики изменения R2 на тестовых выборках.
    Стиль: Научный минимализм. Индивидуальные траектории скрыты.
    Левый график масштабируется максимально плотно. Правый уходит в минус минимально.
    Добавлена отраженная красная линия для визуального сравнения дрейфа.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    excluded_mice = ['J20', 'J21']
    match_files = list(Path(match_dir).glob("LNOF_*.csv"))

    results = []

    print(f"\n{'=' * 75}")
    print("Расчет перекрестного предсказания (Test Set Evaluation)")
    print(f"{'=' * 75}")

    def build_id_to_idx_map(feature_names, zero_indexed=model_is_zero_indexed):
        mapping = {}
        for idx, name in enumerate(feature_names):
            match = re.search(r'\d+', str(name))
            if match:
                n_id = int(match.group())
                if zero_indexed:
                    n_id += 1
                mapping[n_id] = idx
        return mapping

    for file_path in match_files:
        mouse = file_path.stem.replace('LNOF_', '')
        if mouse in excluded_mice:
            continue

        df_map = pd.read_csv(file_path, header=None)
        if df_map.shape[1] < 4:
            continue

        models = {}
        datasets = {}
        feat_maps = {}
        valid_chain = True

        for d in range(4):
            day_str = f"{d + 1}D"
            session_name = f"{mouse}_{day_str}"

            model_path = Path(models_dir) / f"{session_name}.pkl"
            if not model_path.exists():
                valid_chain = False
                break
            m_data = joblib.load(model_path)
            models[d] = m_data
            feat_maps[d] = build_id_to_idx_map(m_data.get('feature_names'))

            # Предполагается, что функция load_data у вас уже определена в скрипте
            data_file = Path(data_dir) / f"LNOF_{session_name}_aligned.npz"
            if not data_file.exists():
                valid_chain = False
                break

            features_raw, labels_raw = load_data(data_file)
            split_idx = int(len(features_raw) * (1 - test_size))

            if isinstance(features_raw, pd.DataFrame):
                X_test_raw = features_raw.iloc[split_idx:].values
            else:
                X_test_raw = features_raw[split_idx:]

            if isinstance(labels_raw, pd.DataFrame) or isinstance(labels_raw, pd.Series):
                y_test = labels_raw.iloc[split_idx:].values
            else:
                y_test = labels_raw[split_idx:]

            scaler = m_data.get('scaler')
            X_scaled = scaler.transform(X_test_raw) if scaler is not None else X_test_raw

            datasets[d] = (X_scaled, y_test)

        if not valid_chain:
            continue

        for N in range(4):
            X_N, y_N = datasets[N]

            # --- 1. Полноценная модель текущего дня ---
            model_N = models[N]['model']
            pred_full = model_N.predict(X_N)
            r2_full = r2_score(y_N, pred_full)

            # --- 2. Модель 1D предсказывает День N ---
            model_1 = models[0]['model']
            feat_count_1 = len(models[0]['feature_names'])
            X_in_1 = np.zeros((len(X_N), feat_count_1))

            mask_1_N = df_map[0].notna() & (df_map[0] > 0) & df_map[N].notna() & (df_map[N] > 0)
            for _, row in df_map[mask_1_N].iterrows():
                id_1, id_N = int(row[0]), int(row[N])
                idx_1 = feat_maps[0].get(id_1, -1)
                idx_N = feat_maps[N].get(id_N, -1)
                if idx_1 != -1 and idx_N != -1:
                    X_in_1[:, idx_1] = X_N[:, idx_N]

            pred_1 = model_1.predict(X_in_1)
            r2_1_to_N = r2_score(y_N, pred_1)

            # --- 3. Модель ND на нейронах 1-го дня ---
            feat_count_N = len(models[N]['feature_names'])
            X_in_N_sub1 = np.zeros((len(X_N), feat_count_N))

            for _, row in df_map[mask_1_N].iterrows():
                id_N = int(row[N])
                idx_N = feat_maps[N].get(id_N, -1)
                if idx_N != -1:
                    X_in_N_sub1[:, idx_N] = X_N[:, idx_N]

            pred_N_sub1 = model_N.predict(X_in_N_sub1)
            r2_N_sub1 = r2_score(y_N, pred_N_sub1)

            # --- 4. Модель 4D предсказывает День N ---
            model_4 = models[3]['model']
            feat_count_4 = len(models[3]['feature_names'])
            X_in_4 = np.zeros((len(X_N), feat_count_4))

            mask_4_N = df_map[3].notna() & (df_map[3] > 0) & df_map[N].notna() & (df_map[N] > 0)
            for _, row in df_map[mask_4_N].iterrows():
                id_4, id_N = int(row[3]), int(row[N])
                idx_4 = feat_maps[3].get(id_4, -1)
                idx_N = feat_maps[N].get(id_N, -1)
                if idx_4 != -1 and idx_N != -1:
                    X_in_4[:, idx_4] = X_N[:, idx_N]

            pred_4 = model_4.predict(X_in_4)
            r2_4_to_N = r2_score(y_N, pred_4)

            # --- 5. Модель ND на нейронах 4-го дня ---
            X_in_N_sub4 = np.zeros((len(X_N), feat_count_N))
            for _, row in df_map[mask_4_N].iterrows():
                id_N = int(row[N])
                idx_N = feat_maps[N].get(id_N, -1)
                if idx_N != -1:
                    X_in_N_sub4[:, idx_N] = X_N[:, idx_N]

            pred_N_sub4 = model_N.predict(X_in_N_sub4)
            r2_N_sub4 = r2_score(y_N, pred_N_sub4)

            results.append({
                'mouse': mouse,
                'day': N + 1,
                'r2_full': r2_full,
                'r2_1_to_N': r2_1_to_N,
                'r2_N_sub1': r2_N_sub1,
                'r2_4_to_N': r2_4_to_N,
                'r2_N_sub4': r2_N_sub4
            })

        print(f"[{mouse}] Обработана матрица перекрестных предсказаний.")

    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("[ERROR] Нет данных для построения.")
        return

    # Добавляем колонку для отраженной красной линии (День 1 <-> 4)
    df_res['day_reflected'] = 5 - df_res['day']

    # =========================================================================
    # ЭСТЕТИЧЕСКАЯ ОТРИСОВКА
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')
    ax1, ax2 = axes

    c_text = '#2C3E50'
    c_full = 'black'  # Полноценные модели
    c_sub1 = '#12A57A'  # Важные из 1 дня (Alpine Green)
    c_sub4 = '#8E44AD'  # Важные из 4 дня (Глубокий пурпурный)
    c_mod1 = '#1F77B4'  # Синяя (Модель 1 дня)
    c_mod4 = '#D62728'  # Красная (Модель 4 дня)

    # --- 1. Левый график (Базовый дрейф) ---
    sns.lineplot(data=df_res, x='day', y='r2_full', ax=ax1, color=c_full, errorbar=('ci', 95),
                 marker='o', markersize=8, linewidth=2.5, markeredgecolor='none',
                 err_kws={'facecolor': c_full, 'alpha': 0.15, 'linewidth': 0}, zorder=4)

    sns.lineplot(data=df_res, x='day', y='r2_N_sub1', ax=ax1, color=c_sub1, errorbar=('ci', 95),
                 marker='^', markersize=9, linewidth=2.5, markeredgecolor='none',
                 err_kws={'facecolor': c_sub1, 'alpha': 0.20, 'linewidth': 0}, zorder=5)

    sns.lineplot(data=df_res, x='day', y='r2_N_sub4', ax=ax1, color=c_sub4, errorbar=('ci', 95),
                 marker='v', markersize=9, linewidth=2.5, markeredgecolor='none',
                 err_kws={'facecolor': c_sub4, 'alpha': 0.20, 'linewidth': 0}, zorder=6)

    # --- 2. Правый график (Изменение нейронной конфигурации) ---
    sns.lineplot(data=df_res, x='day', y='r2_full', ax=ax2, color=c_full, errorbar=('ci', 95),
                 marker='o', markersize=8, linewidth=2.5, markeredgecolor='none',
                 err_kws={'facecolor': c_full, 'alpha': 0.15, 'linewidth': 0}, zorder=4)

    sns.lineplot(data=df_res, x='day', y='r2_1_to_N', ax=ax2, color=c_mod1, errorbar=('ci', 95),
                 marker='s', markersize=8, linewidth=2.5, markeredgecolor='none',
                 err_kws={'facecolor': c_mod1, 'alpha': 0.20, 'linewidth': 0}, zorder=5)

    sns.lineplot(data=df_res, x='day', y='r2_4_to_N', ax=ax2, color=c_mod4, errorbar=('ci', 95),
                 marker='D', markersize=8, linewidth=2.5, markeredgecolor='none',
                 err_kws={'facecolor': c_mod4, 'alpha': 0.20, 'linewidth': 0}, zorder=6)

    # ОТРАЖЕННАЯ красная линия (пунктир, полупрозрачная, без маркеров, x=day_reflected)
    sns.lineplot(data=df_res, x='day_reflected', y='r2_4_to_N', ax=ax2, color=c_mod4, errorbar=None,
                 linewidth=1.5, alpha=0.4, zorder=3)

    # --- 3. Динамическое масштабирование осей (Максимальное использование пространства) ---
    # Считаем агрегаты для точного определения границ с учетом доверительных интервалов (CI ~ 2*sem)
    agg_left = df_res.groupby('day')[['r2_full', 'r2_N_sub1', 'r2_N_sub4']].agg(['mean', 'sem'])
    max_left = (agg_left.xs('mean', level=1, axis=1) + 2.5 * agg_left.xs('sem', level=1, axis=1)).max().max()
    min_left = (agg_left.xs('mean', level=1, axis=1) - 2.5 * agg_left.xs('sem', level=1, axis=1)).min().min()

    # Левый: Максимально плотно облегает линии
    margin_left = (max_left - min_left) * 0.05
    ax1.set_ylim(min_left - margin_left, max_left + margin_left)

    # Правый: Уходит в минус минимально
    agg_right = df_res.groupby('day')[['r2_full', 'r2_1_to_N', 'r2_4_to_N']].agg(['mean', 'sem'])
    max_right = (agg_right.xs('mean', level=1, axis=1) + 2.5 * agg_right.xs('sem', level=1, axis=1)).max().max()
    min_right = (agg_right.xs('mean', level=1, axis=1) - 2.5 * agg_right.xs('sem', level=1, axis=1)).min().min()

    margin_right = (max_right - min_right) * 0.05
    # Заставляем нижнюю границу быть не ниже -0.05, если сами данные не падают сильно ниже
    lower_bound_right = min(-0.05, min_right - margin_right)
    ax2.set_ylim(lower_bound_right, max_right + margin_right)

    # --- 4. Оформление и Типографика ---
    ax1.set_title("Базовый дрейф", fontsize=15, pad=15, fontweight='bold', color=c_text)
    ax2.set_title("Изменение нейронной конфигурации", fontsize=15, pad=15, fontweight='bold', color=c_text)

    ax1.set_ylabel("Средний $R^2$", fontsize=14, labelpad=10, color=c_text)
    ax2.set_ylabel("")

    for ax in axes:
        ax.set_facecolor('white')
        ax.set_xlabel("День", fontsize=13, labelpad=10, color=c_text)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlim(0.8, 4.2)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        [s.set_color(c_text) for s in ax.spines.values()]

        ax.tick_params(axis='both', which='major', labelsize=12, width=1.2, colors=c_text)
        ax.grid(axis='both', linestyle=':', alpha=0.4, color='#BDC3C7', zorder=0)

        # Жесткая фиксация прозрачности заливок (PolyCollection)
        for collection in ax.collections:
            if type(collection).__name__ == 'PolyCollection':
                collection.set_alpha(0.20)

    # --- 5. Лаконичные Легенды ---
    leg1 = [
        mlines.Line2D([], [], color=c_full, marker='o', markersize=8, markeredgecolor='none', lw=2.5,
                      label='Полноценные модели'),
        mlines.Line2D([], [], color=c_sub4, marker='v', markersize=9, markeredgecolor='none', lw=2.5,
                      label='Участвующие нейроны из 4 дня'),
        mlines.Line2D([], [], color=c_sub1, marker='^', markersize=9, markeredgecolor='none', lw=2.5,
                      label='Участвующие нейроны из 1 дня')
    ]
    ax1.legend(handles=leg1, loc='best', frameon=False, fontsize=11, labelcolor=c_text)

    leg2 = [
        mlines.Line2D([], [], color=c_full, marker='o', markersize=8, markeredgecolor='none', lw=2.5,
                      label='Полноценные модели'),
        mlines.Line2D([], [], color=c_mod4, marker='D', markersize=8, markeredgecolor='none', lw=2.5,
                      label='Модели 4 дня'),
        mlines.Line2D([], [], color=c_mod1, marker='s', markersize=8, markeredgecolor='none', lw=2.5,
                      label='Модели 1 дня'),
        # mlines.Line2D([], [], color=c_mod4, linestyle='--', alpha=0.4, lw=2.5, label='Отраженная Модель 4 дня')
    ]
    ax2.legend(handles=leg2, loc='best', frameon=False, fontsize=11, labelcolor=c_text)

    plt.tight_layout(w_pad=4.0)

    save_path = output_path / 'cross_session_r2_dynamics_aesthetic.png'
    plt.savefig(save_path, dpi=1000, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Успех! Графики сохранены: {save_path}")


def plot_mobility_longitudinal_dependencies(data_dir='LNOF_SynchronizedData', output_dir='plots'):
    """
    Строит 3 независимых графика зависимости локомоции от дня эксперимента.
    Эстетика: Выцветающие столбцы (от темного к светлому), тонкая прозрачная серая линия тренда,
    идеальное расстояние между звездочками значимости.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"[ERROR] Директория {data_dir} не найдена.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []
    excluded_mice = ['J20', 'J21']

    print(f"\n{'=' * 75}")
    print("Извлечение метрик локомоции по дням...")
    print(f"{'=' * 75}")

    # 1. Извлекаем данные
    for file in data_path.glob("*.npz"):
        try:
            mouse_match = re.search(r'(?:LNOF_)?([A-Za-z0-9]+)_', file.name)
            mouse = mouse_match.group(1) if mouse_match else 'Unknown'

            if mouse in excluded_mice:
                continue

            day_match = re.search(r'_([1-4])D', file.name)
            if not day_match:
                continue
            day_num = int(day_match.group(1))

            data = np.load(file, allow_pickle=True)

            if 'speed' in data.files:
                speed = data['speed']
            elif 'x' in data.files and 'y' in data.files:
                dx = np.diff(data['x'], prepend=data['x'][0])
                dy = np.diff(data['y'], prepend=data['y'][0])
                speed = np.sqrt(dx ** 2 + dy ** 2)
            else:
                continue

            speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)
            total_frames = len(speed)

            if total_frames == 0:
                continue

            percent_moving_5 = np.sum(speed > 5.0) / total_frames * 100
            percent_moving_1 = np.sum(speed > 1.0) / total_frames * 100
            mean_speed = np.mean(speed)

            results.append({
                'mouse': mouse,
                'day': day_num,
                'percent_moving_5': percent_moving_5,
                'percent_moving_1': percent_moving_1,
                'mean_speed': mean_speed
            })

        except Exception as e:
            print(f"[ERROR] Ошибка при обработке {file.name}: {e}")

    df = pd.DataFrame(results)

    if df.empty:
        print("[ERROR] Нет валидных данных для построения.")
        return

    # Новая функция для математически ровных звездочек
    def get_stars(p):
        return r"{\ast}{\ast}{\ast}" if p < 0.001 else r"{\ast}{\ast}" if p < 0.01 else r"{\ast}" if p < 0.05 else "{ns}"

    metrics_config = [
        {
            'metric': 'percent_moving_5',
            'ylabel': 'Доля времени (%)',
            'base_color': '#E74C3C',  # Красный
            'filename': 'mobility_intensive_speed5.png'
        },
        {
            'metric': 'percent_moving_1',
            'ylabel': 'Доля времени (%)',
            'base_color': '#3498DB',  # Синий
            'filename': 'mobility_general_speed1.png'
        },
        {
            'metric': 'mean_speed',
            'ylabel': 'Средняя скорость (у.е.)',
            'base_color': '#27AE60',  # Зеленый
            'filename': 'mobility_mean_speed.png'
        }
    ]

    c_text = '#2C3E50'

    # ==========================================
    # ЭСТЕТИЧЕСКАЯ ОТРИСОВКА (Раздельно)
    # ==========================================
    for cfg in metrics_config:
        fig, ax = plt.subplots(figsize=(7, 6))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        metric = cfg['metric']
        corr, pval = spearmanr(df['day'], df[metric])

        # Создаем выцветающий градиент (от темного к светлому)
        # reverse=True ставит темный цвет в начало списка
        palette = sns.light_palette(cfg['base_color'], n_colors=5, reverse=True)[:4]

        # 1. Группа тонких слитных столбцов
        days = sorted(df['day'].unique())
        group_width = 0.65

        for d in days:
            vals = df[df['day'] == d][metric].values
            n = len(vals)
            if n == 0: continue

            # Сортируем по убыванию для красивого каскадного силуэта
            vals_sorted = np.sort(vals)[::-1]

            w = group_width / n
            x_pos = d - (group_width / 2) + np.arange(n) * w + w / 2

            ax.bar(x_pos, vals_sorted, width=w, color=palette[d - 1], edgecolor='none', alpha=0.9, zorder=2)

        # 2. Усредняющая линия (тонкая, прозрачная, серая)
        means = df.groupby('day')[metric].mean()
        ax.plot(means.index, means.values, color='grey', linewidth=1.5, alpha=0.5, linestyle='-', zorder=4)

        # 3. Аннотация Спирмена со строгим позиционированием звездочек
        stat_text = f"$\\rho = {corr:.2f}^{{{get_stars(pval)}}}$"
        ax.text(0.95, 0.95, stat_text, transform=ax.transAxes, ha='right', va='top',
                fontsize=14, color=c_text, weight='bold',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=5), zorder=5)

        # 4. Оформление осей
        ax.set_title("Побежки мышей", fontsize=15, pad=15, fontweight='bold', color=c_text)
        ax.set_xlabel("День", fontsize=13, labelpad=10, color=c_text)
        ax.set_ylabel(cfg['ylabel'], fontsize=14, labelpad=10, color=c_text)

        y_max = df[metric].max() * 1.15
        ax.set_ylim(0, y_max)

        ax.set_xticks(days)
        ax.set_xlim(0.5, 4.5)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        [s.set_color(c_text) for s in ax.spines.values()]

        ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, colors=c_text)

        ax.grid(axis='y', linestyle=':', alpha=0.5, color='#BDC3C7', zorder=0)

        plt.tight_layout()

        # Сохранение текущего графика
        save_path = output_path / cfg['filename']
        plt.savefig(save_path, dpi=1000, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    print(f"✓ Все 3 эстетичных графика сохранены отдельно в: {output_path}")


def plot_top_neurons_longitudinal_counts_bidirectional(
        top_csv='stat_tables/top_neurons_ablation_3.0sigma.csv',
        match_dir='MatchData',
        output_dir='plots',
        model_is_zero_indexed=True):
    """
    Двунаправленное отслеживание КОЛИЧЕСТВА сильных нейронов (>3 sigma).
    Отображает только график абсолютного дрейфа.
    Прямой проход (Alpine Green): Выжившие из Дня 1.
    Обратный проход (Magenta): Выжившие из Дня 4.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    excluded_mice = ['J20', 'J21']
    match_files = list(Path(match_dir).glob("LNOF_*.csv"))

    try:
        df_top = pd.read_csv(top_csv)
    except FileNotFoundError:
        print(f"[ERROR] Не найден файл: {top_csv}")
        return

    df_names = df_top[df_top['data_type'] == 'Neuron Names'].copy()
    top_cols = [c for c in df_names.columns if c.startswith('top_')]

    results = []

    print(f"\n{'=' * 75}")
    print("Расчет выживаемости (Bidirectional Pass | Absolute Drift)")
    print(f"{'=' * 75}")

    for file_path in match_files:
        mouse = file_path.stem.replace('LNOF_', '')
        if mouse in excluded_mice:
            continue

        df_map = pd.read_csv(file_path, header=None)
        if df_map.shape[1] < 4:
            continue

        tracked_sets = []
        important_sets = []
        valid_chain = True

        for day_idx in range(4):
            day = f"{day_idx + 1}D"
            col_data = df_map[day_idx]
            tracked_mask = col_data.notna() & (col_data > 0)
            tracked_sets.append(set(df_map[tracked_mask].index))

            row = df_names[(df_names['mouse'] == mouse) & (df_names['session'] == day)]
            important_local_ids = []
            if not row.empty:
                neuron_names = row.iloc[0][top_cols].dropna().values
                for name in neuron_names:
                    match = re.search(r'\d+', str(name))
                    if match:
                        n_id = int(match.group())
                        if model_is_zero_indexed:
                            n_id += 1
                        important_local_ids.append(n_id)

            important_mask = col_data.isin(important_local_ids)
            important_sets.append(set(df_map[important_mask].index))

        if len(important_sets[0]) == 0 or len(important_sets[3]) == 0:
            continue

        # --- ДВУНАПРАВЛЕННЫЙ МАТЕМАТИЧЕСКИЙ РАСЧЕТ ---
        for step in range(4):
            day_num = step + 1
            N_top_d = len(important_sets[step])

            # Прямой проход (Кумулятивное пересечение от 1-го дня до step)
            V_d_fwd = len(set.intersection(*important_sets[:step + 1]))
            # Обратный проход (Кумулятивное пересечение от 4-го дня назад до step)
            V_d_rev = len(set.intersection(*important_sets[step:]))

            results.append({
                'mouse': mouse,
                'day': day_num,
                'N_top_d': N_top_d,
                'V_d_fwd': V_d_fwd,
                'V_d_rev': V_d_rev
            })

        print(f"[{mouse}] Обработано (Bidirectional).")

    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("[ERROR] Нет данных для построения.")
        return

    # =========================================================================
    # ЭСТЕТИЧЕСКАЯ ОТРИСОВКА
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    c_text = '#2C3E50'
    c_base = 'black'
    c_fwd = '#12A57A'  # Alpine Green
    c_rev = '#D81B60'  # Magenta

    mice = df_res['mouse'].unique()
    # Индивидуальные цвета только для визуального отличия мышей (основаны на серых тонах для лаконичности)
    ind_colors = sns.color_palette("husl", len(mice))

    # --- 1. Индивидуальные сессии (Разноцветные траектории) ---
    for i, mouse_id in enumerate(mice):
        m_data = df_res[df_res['mouse'] == mouse_id]
        ax.plot(m_data['day'], m_data['V_d_fwd'], color=ind_colors[i], alpha=0.25, lw=1.5, zorder=1)
        ax.plot(m_data['day'], m_data['V_d_rev'], color=ind_colors[i], alpha=0.25, lw=1.5, zorder=1)

    # --- 2. Глобальные тренды (Агрегация + CI) ---
    sns.lineplot(data=df_res, x='day', y='N_top_d', ax=ax, color=c_base,
                 errorbar=('ci', 95), marker='o', markersize=8, linewidth=2.5, markeredgecolor='none',
                 err_kws={'facecolor': 'grey', 'alpha': 0.15, 'linewidth': 0}, zorder=4)

    sns.lineplot(data=df_res, x='day', y='V_d_fwd', ax=ax, color=c_fwd,
                 errorbar=('ci', 95), marker='^', markersize=10, linewidth=2.5, markerfacecolor=c_fwd,
                 markeredgecolor='none',
                 err_kws={'facecolor': c_fwd, 'alpha': 0.20, 'linewidth': 0}, zorder=5)

    sns.lineplot(data=df_res, x='day', y='V_d_rev', ax=ax, color=c_rev,
                 errorbar=('ci', 95), marker='D', markersize=8, linewidth=2.5, markerfacecolor=c_rev,
                 markeredgecolor='none',
                 err_kws={'facecolor': c_rev, 'alpha': 0.20, 'linewidth': 0}, zorder=6)

    # --- 3. Интеллектуальный Anti-Collision Алгоритм для аннотаций ---
    def get_3_offsets(d, y_b, y_f, y_r, y_max):
        """Возвращает безопасные смещения (offset, va, ha) для трех аннотаций"""
        off_b, va_b, ha_b = (0, 12), 'bottom', 'center'
        off_f, va_f, ha_f = (0, 12), 'bottom', 'center'
        off_r, va_r, ha_r = (0, 12), 'bottom', 'center'

        if d == 1:
            off_b, va_b = (0, 12), 'bottom'
            off_f, va_f = (0, -15), 'top'  # Fwd совпадает с Black (N_top_d) в 1D
            off_r, va_r = (0, 12), 'bottom'
        elif d == 4:
            off_b, va_b = (0, 12), 'bottom'
            off_r, va_r = (0, -15), 'top'  # Rev совпадает с Black (N_top_d) в 4D
            off_f, va_f = (0, 12), 'bottom'
        else:
            off_b, va_b = (0, 12), 'bottom'  # Black всегда на самом верху
            # Если Fwd и Rev близки друг к другу, разводим их в стороны
            if abs(y_f - y_r) < y_max * 0.08:
                if y_f >= y_r:
                    off_f, va_f, ha_f = (-12, 0), 'center', 'right'
                    off_r, va_r, ha_r = (12, 0), 'center', 'left'
                else:
                    off_r, va_r, ha_r = (-12, 0), 'center', 'right'
                    off_f, va_f, ha_f = (12, 0), 'center', 'left'
            else:
                if y_f >= y_r:
                    off_f, va_f = (0, 12), 'bottom'
                    off_r, va_r = (0, -15), 'top'
                else:
                    off_r, va_r = (0, 12), 'bottom'
                    off_f, va_f = (0, -15), 'top'

        return (off_b, va_b, ha_b), (off_f, va_f, ha_f), (off_r, va_r, ha_r)

    agg_stats = df_res.groupby('day').agg({
        'N_top_d': 'mean', 'V_d_fwd': 'mean', 'V_d_rev': 'mean'
    }).reset_index()

    global_max = agg_stats[['N_top_d', 'V_d_fwd', 'V_d_rev']].max().max() * 1.25
    ax.set_ylim(-0.5, global_max)

    for _, row in agg_stats.iterrows():
        d = row['day']
        yb, yf, yr = row['N_top_d'], row['V_d_fwd'], row['V_d_rev']
        (ob, vb, hb), (of, vf, hf), (or_, vr, hr) = get_3_offsets(d, yb, yf, yr, global_max)

        ax.annotate(f"{yb:.1f}", (d, yb), textcoords="offset points", xytext=ob, ha=hb, va=vb, fontsize=11,
                    color=c_base, weight='bold', zorder=6)
        ax.annotate(f"{yf:.1f}", (d, yf), textcoords="offset points", xytext=of, ha=hf, va=vf, fontsize=11, color=c_fwd,
                    weight='bold', zorder=6)
        ax.annotate(f"{yr:.1f}", (d, yr), textcoords="offset points", xytext=or_, ha=hr, va=vr, fontsize=11,
                    color=c_rev, weight='bold', zorder=6)

    # --- 4. Оформление ---
    ax.set_title("Абсолютный дрейф", fontsize=15, pad=15, fontweight='bold', color=c_text)
    ax.set_ylabel("Количество сильных нейронов", fontsize=14, labelpad=10, color=c_text)
    ax.set_xlabel("День", fontsize=13, labelpad=10, color=c_text)

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlim(0.8, 4.2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    [s.set_color(c_text) for s in ax.spines.values()]

    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, colors=c_text)
    ax.grid(axis='both', linestyle=':', alpha=0.4, color='#BDC3C7', zorder=0)

    # Легенда
    legend_elements = [
        mlines.Line2D([], [], color=c_base, marker='o', markersize=8, lw=2.5, label='Все нейроны'),
        mlines.Line2D([], [], color=c_fwd, marker='^', markersize=10, lw=2.5,
                      label='Соответствующие дню 1'),
        mlines.Line2D([], [], color=c_rev, marker='D', markersize=8, lw=2.5,
                      label='Соответствующие дню 4')
    ]
    ax.legend(handles=legend_elements, loc='best', frameon=False, fontsize=11, labelcolor=c_text)

    plt.tight_layout()

    save_path = output_path / 'longitudinal_top_neurons_counts_bidirectional_single.png'
    plt.savefig(save_path, dpi=1000, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Одинарный двунаправленный график успешно сохранен: {save_path}")


def plot_weight_survival_dependency(
        match_dir='MatchData',
        models_dir='models_LNOF_Lasso',
        output_dir='plots',
        model_is_zero_indexed=True):
    """
    Строит график зависимости абсолютного веса нейрона в 1-й день
    от максимального дня, до которого этот нейрон "дожил" (успешно затрекался).
    Рассчитывает и выводит коэффициент Спирмена.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    excluded_mice = ['J20', 'J21']
    match_files = list(Path(match_dir).glob("LNOF_*.csv"))

    results = []

    print(f"\n{'=' * 75}")
    print("Расчет связи: Исходный вес (1D) vs Продолжительность жизни ансамбля")
    print(f"{'=' * 75}")

    for file_path in match_files:
        mouse = file_path.stem.replace('LNOF_', '')
        if mouse in excluded_mice:
            continue

        # Загрузка карты трекинга
        df_map = pd.read_csv(file_path, header=None)
        if df_map.shape[1] < 4:
            continue

        # Загрузка модели 1-го дня
        model_path = Path(models_dir) / f"{mouse}_1D.pkl"
        if not model_path.exists():
            continue

        data = joblib.load(model_path)
        model = data.get('model')
        if not hasattr(model, 'coef_'):
            continue

        # Получаем абсолютные веса (модуль)
        coefs = np.array(model.coef_)
        if coefs.ndim > 1:
            weights = np.linalg.norm(coefs, axis=0)
        else:
            weights = np.abs(coefs)

        feat_names = np.array(data.get('feature_names'))

        # Оставляем только важные нейроны (>0)
        important_mask = weights > 0
        valid_feats = feat_names[important_mask]
        valid_weights = weights[important_mask]

        # Ищем каждый важный нейрон в матрице трекинга
        for name, w in zip(valid_feats, valid_weights):
            match = re.search(r'\d+', str(name))
            if match:
                n_id = int(match.group())
                if model_is_zero_indexed:
                    n_id += 1

                # Находим строку нейрона в матрице (по 1-му дню)
                row_idx = df_map[df_map[0] == n_id].index
                if len(row_idx) > 0:
                    # Извлекаем значения трекинга для 4-х дней
                    row_vals = df_map.iloc[row_idx[0], :4].values

                    # Находим все дни, где нейрон затрекался (>0 и не NaN)
                    valid_cols = [c for c in range(4) if pd.notna(row_vals[c]) and row_vals[c] > 0]

                    if valid_cols:
                        # Максимальный индекс колонки (0=1D, 1=2D...) + 1 дает последний день
                        last_day = max(valid_cols) + 1

                        results.append({
                            'mouse': mouse,
                            'neuron_id': n_id,
                            'weight_1D': w,
                            'last_day': last_day
                        })

        print(f"[{mouse}] Обработано. Нейронов 1D: {len(valid_feats)}")

    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("[ERROR] Нет данных для построения.")
        return

    # --- Расчет коэффициента Спирмена ---
    corr, pval = spearmanr(df_res['last_day'], df_res['weight_1D'])

    def get_stars(p):
        return r"^{\ast\ast\ast}" if p < 0.001 else r"^{\ast\ast}" if p < 0.01 else r"^{\ast}" if p < 0.05 else "^{ns}"

    # =========================================================================
    # ЭСТЕТИЧЕСКАЯ ОТРИСОВКА
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    c_text = '#2C3E50'
    c_dots = '#3498DB'  # Яркий синий (Flat Blue)
    c_line = '#E74C3C'  # Красный для тренда
    c_box = '#BDC3C7'  # Светло-серый для боксплотов

    # 1. Боксплоты на заднем фоне (для показа распределения плотности)
    sns.boxplot(
        data=df_res, x='last_day', y='weight_1D', ax=ax,
        color=c_box, width=0.4, showfliers=False, boxprops=dict(alpha=0.3, edgecolor='none'), zorder=1
    )

    # 2. Стрипплот (Jitter) для показа индивидуальных нейронов
    sns.stripplot(
        data=df_res, x='last_day', y='weight_1D', ax=ax,
        color=c_dots, alpha=0.5, size=5, jitter=0.2, zorder=2
    )

    # 3. Линия регрессии (Тренд)
    sns.regplot(
        x=df_res['last_day'] - 1, y=df_res['weight_1D'], ax=ax,
        scatter=False, color=c_line, line_kws={'linestyle': '--', 'linewidth': 3, 'zorder': 3}
    )

    # 4. Аннотация Спирмена
    stat_text = f"$\\rho = {corr:.2f}{get_stars(pval)}$\n$n = {len(df_res)}$ нейронов"
    ax.text(0.05, 0.95, stat_text, transform=ax.transAxes, ha='left', va='top',
            fontsize=15, color=c_text, weight='bold',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=5), zorder=5)

    # 5. Оформление
    ax.set_title("Продолжительность 'жизни' нейрона в зависимости от его важности в 1D",
                 fontsize=15, pad=15, fontweight='bold', color=c_text)

    ax.set_xlabel("Последний день активности (Выживаемость)", fontsize=13, labelpad=10, color=c_text)
    ax.set_ylabel("Абсолютный вес Lasso (День 1)", fontsize=14, labelpad=10, color=c_text)

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['Только 1D', 'До 2D', 'До 3D', 'До 4D'])

    # Эстетика осей
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    [s.set_color(c_text) for s in ax.spines.values()]

    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, colors=c_text)
    ax.grid(axis='y', linestyle=':', alpha=0.5, color='#BDC3C7', zorder=0)

    plt.tight_layout()

    save_path = output_path / 'weight_survival_dependency.png'
    plt.savefig(save_path, dpi=1000, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n✓ Успешно! График зависимости выживаемости от веса сохранен: {save_path}")
    print(f"  > Spearman \u03C1: {corr:.3f} (p={pval:.2e})")


def plot_combined_longitudinal_survival(
        match_dir='MatchData',
        models_dir='models_LNOF_Lasso',
        ablation_csv='stat_tables/full_ablation_stats.csv',
        top_csv='stat_tables/top_neurons_ablation_3.0sigma.csv',
        output_dir='plots',
        model_is_zero_indexed=True):
    """
    Отслеживает выживаемость сразу 4-х ансамблей (от Дня 1):
    - Все нейроны (Черный)
    - Важные (z-delta > 0) (Синий)
    - Участвующие (веса > 0) (Маджента)
    - Сильные (>3sigma) (Alpine Green)
    Убраны цифры с правого графика (процентов), легенда лаконичная.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    excluded_mice = ['J20', 'J21']
    match_files = list(Path(match_dir).glob("LNOF_*.csv"))

    try:
        df_abl = pd.read_csv(ablation_csv)
        df_top = pd.read_csv(top_csv)
        df_names = df_top[df_top['data_type'] == 'Neuron Names'].copy()
        top_cols = [c for c in df_names.columns if c.startswith('top_')]
    except FileNotFoundError as e:
        print(f"[ERROR] Не найден файл таблиц: {e}")
        return

    results = []

    print(f"\n{'=' * 75}")
    print("Расчет комбинированной выживаемости (Все + Важные + Участвующие + Сильные)")
    print(f"{'=' * 75}")

    for file_path in match_files:
        mouse = file_path.stem.replace('LNOF_', '')
        if mouse in excluded_mice:
            continue

        df_map = pd.read_csv(file_path, header=None)
        if df_map.shape[1] < 4:
            continue

        sets_all, sets_val, sets_lin, sets_top = [], [], [], []
        valid_chain = True

        for day_idx in range(4):
            day = f"{day_idx + 1}D"
            model_session = f"{mouse}_{day}"

            # 1. ВСЕ НЕЙРОНЫ
            col_data = df_map[day_idx]
            tracked_mask = col_data.notna() & (col_data > 0)
            sets_all.append(set(df_map[tracked_mask].index))

            # Загрузка модели для Важных и Участвующих
            model_path = Path(models_dir) / f"{model_session}.pkl"
            if not model_path.exists():
                valid_chain = False
                break

            data = joblib.load(model_path)
            model = data.get('model')
            if not hasattr(model, 'coef_'):
                valid_chain = False
                break

            coefs = np.array(model.coef_)
            weights = np.linalg.norm(coefs, axis=0) if coefs.ndim > 1 else np.abs(coefs)
            feat_names = np.array(data.get('feature_names'))

            # 2. УЧАСТВУЮЩИЕ (Веса Lasso > 0)
            lin_ids = []
            for name in feat_names[weights > 0]:
                match = re.search(r'\d+', str(name))
                if match:
                    lin_ids.append(int(match.group()) + (1 if model_is_zero_indexed else 0))
            sets_lin.append(set(df_map[col_data.isin(lin_ids)].index))

            # 3. ВАЖНЫЕ (z-crit > 0)
            val_ids = []
            abl_rows = df_abl[df_abl['model_session'] == model_session]
            if not abl_rows.empty:
                abl_row = abl_rows.iloc[0]
                base_r2 = abl_row['baseline_r2']
                abl_r2_vals = np.array([abl_row.get(f'ablation_{f}', np.nan) for f in feat_names])

                crit = base_r2 - abl_r2_vals
                valid_crit = crit[~np.isnan(crit)]

                if len(valid_crit) > 0 and np.std(valid_crit) > 0:
                    zcrit = (crit - np.mean(valid_crit)) / np.std(valid_crit)
                    for name in feat_names[~np.isnan(zcrit) & (zcrit > 0)]:
                        match = re.search(r'\d+', str(name))
                        if match:
                            val_ids.append(int(match.group()) + (1 if model_is_zero_indexed else 0))

            sets_val.append(set(df_map[col_data.isin(val_ids)].index))

            # 4. СИЛЬНЫЕ (>3sigma)
            row = df_names[(df_names['mouse'] == mouse) & (df_names['session'] == day)]
            top_ids = []
            if not row.empty:
                for name in row.iloc[0][top_cols].dropna().values:
                    match = re.search(r'\d+', str(name))
                    if match:
                        top_ids.append(int(match.group()) + (1 if model_is_zero_indexed else 0))
            sets_top.append(set(df_map[col_data.isin(top_ids)].index))

        if not valid_chain or len(sets_all[0]) == 0:
            continue

        # Сохраняем пулы Дня 1
        b_all, b_val = len(sets_all[0]), len(sets_val[0])
        b_lin, b_top = len(sets_lin[0]), len(sets_top[0])

        for step in range(4):
            day_num = step + 1

            # Кумулятивное пересечение (Выжившие с 1-го дня)
            s_all = len(set.intersection(*sets_all[:step + 1]))
            s_val = len(set.intersection(*sets_val[:step + 1]))
            s_lin = len(set.intersection(*sets_lin[:step + 1]))
            s_top = len(set.intersection(*sets_top[:step + 1]))

            results.append({
                'mouse': mouse, 'day': day_num,
                'abs_all': s_all, 'abs_val': s_val, 'abs_lin': s_lin, 'abs_top': s_top,
                'frac_all': (s_all / b_all * 100) if b_all > 0 else 0,
                'frac_val': (s_val / b_val * 100) if b_val > 0 else 0,
                'frac_lin': (s_lin / b_lin * 100) if b_lin > 0 else 0,
                'frac_top': (s_top / b_top * 100) if b_top > 0 else 0
            })

        print(f"[{mouse}] Обработано.")

    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("[ERROR] Нет данных для построения.")
        return

    # =========================================================================
    # ЭСТЕТИЧЕСКАЯ ОТРИСОВКА
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('white')
    ax1, ax2 = axes

    c_text = '#2C3E50'
    colors = {
        'all': 'black',
        'val': '#3498DB',  # Синий (Важные)
        'lin': '#D81B60',  # Маджента (Участвующие)
        'top': '#12A57A'  # Alpine Green (Сильные)
    }

    markers = {'all': 'o', 'val': 's', 'lin': 'D', 'top': '^'}

    agg = df_res.groupby('day').mean(numeric_only=True).reset_index()

    # --- Отрисовка линий ---
    for ax, prefix in zip([ax1, ax2], ['abs_', 'frac_']):
        for key in colors.keys():
            sns.lineplot(
                data=df_res, x='day', y=f'{prefix}{key}', ax=ax, color=colors[key],
                errorbar=('ci', 95), marker=markers[key], markersize=9, linewidth=2.5,
                markeredgecolor='none', err_kws={'facecolor': colors[key], 'alpha': 0.15, 'linewidth': 0}, zorder=4
            )

    # --- Умный радиальный алгоритм для подписей (ТОЛЬКО для левого графика) ---
    def annotate_4_points(ax, day, values_dict, max_y):
        sorted_items = sorted(values_dict.items(), key=lambda x: x[1][0], reverse=True)

        placements = [
            (0, 14, 'bottom', 'center'),
            (0, -16, 'top', 'center'),
            (-16, 0, 'center', 'right'),
            (16, 0, 'center', 'left')
        ]

        for i, (name, (y_val, color)) in enumerate(sorted_items):
            ox, oy, va, ha = placements[i]

            if va == 'top' and y_val < max_y * 0.05:
                oy = -5  # Защита от наезда на нижнюю ось

            txt = f"{y_val:.1f}"
            ax.annotate(txt, (day, y_val), textcoords="offset points", xytext=(ox, oy),
                        ha=ha, va=va, fontsize=10, color=color, weight='bold', zorder=6)

    # Применяем аннотации только к левому (абсолютному) графику
    max_abs = agg[['abs_all', 'abs_val', 'abs_lin', 'abs_top']].max().max() * 1.2
    max_frac = 115

    ax1.set_ylim(-0.5, max_abs)
    ax2.set_ylim(-2, max_frac)

    for _, row in agg.iterrows():
        d = row['day']

        # Подписи только для абсолютных количеств
        dict_abs = {
            'all': (row['abs_all'], colors['all']), 'val': (row['abs_val'], colors['val']),
            'lin': (row['abs_lin'], colors['lin']), 'top': (row['abs_top'], colors['top'])
        }
        annotate_4_points(ax1, d, dict_abs, max_abs)
        # Для ax2 (проценты) функцию annotate_4_points больше не вызываем!

    # --- Оформление ---
    ax1.set_title("Выживаемость ансамблей (Сырые количества)", fontsize=15, pad=15, fontweight='bold', color=c_text)
    ax2.set_title("Доля сохраненных нейронов от Дня 1", fontsize=15, pad=15, fontweight='bold', color=c_text)

    ax1.set_ylabel("Количество нейронов (шт.)", fontsize=14, labelpad=10, color=c_text)
    ax2.set_ylabel("Доля нейронов соответствующих дню 1", fontsize=14, labelpad=10, color=c_text)

    for ax in axes:
        ax.set_facecolor('white')
        ax.set_xlabel("День", fontsize=13, labelpad=10, color=c_text)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlim(0.8, 4.2)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        [s.set_color(c_text) for s in ax.spines.values()]

        ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, colors=c_text)
        ax.grid(axis='both', linestyle=':', alpha=0.4, color='#BDC3C7', zorder=0)

    # Кастомная, лаконичная легенда на правом графике
    legend_elements = [
        mlines.Line2D([], [], color=colors['all'], marker=markers['all'], markersize=8, markeredgecolor='none', lw=2.5,
                      label='Все активные нейроны'),
        mlines.Line2D([], [], color=colors['lin'], marker=markers['lin'], markersize=8, markeredgecolor='none', lw=2.5,
                      label='Участвующие'),
        mlines.Line2D([], [], color=colors['val'], marker=markers['val'], markersize=8, markeredgecolor='none', lw=2.5,
                      label='Важные'),
        mlines.Line2D([], [], color=colors['top'], marker=markers['top'], markersize=9, markeredgecolor='none', lw=2.5,
                      label='Сильные (>3$\\sigma$)')
    ]
    ax2.legend(handles=legend_elements, loc='best', frameon=False, fontsize=14, labelcolor=c_text)

    plt.tight_layout(w_pad=4.0)

    save_path = output_path / 'longitudinal_combined_survival_4lines.png'
    plt.savefig(save_path, dpi=1000, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ График 4-х популяций сохранен: {save_path}")


def plot_combined_r2_left_panels_with_baseline(
        top_metrics_csv='stat_tables/top_vs_rest_metrics_abl.csv',
        unified_csv='stat_tables/unified_model_performance_report.csv',
        models_dir='models_LNOF_Lasso',
        data_dir='LNOF_SynchronizedData',
        output_dir='plots',
        test_size=0.25):
    """
    Объединяет графики сравнения R2 (Полноценная vs Сильные, Остальные vs Сильные).
    Случайный бейзлайн отображается точками и тонкой пунктирной регрессией (с CI) на обоих графиках.
    Бейзлайн исключен из легенды.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Загрузка таблиц
    try:
        df_top = pd.read_csv(top_metrics_csv)
        df_uni = pd.read_csv(unified_csv)
    except FileNotFoundError as e:
        print(f"[ERROR] Не найден один из файлов: {e}")
        return

    # Объединение
    df = df_top[['model_name', 'top_count', 'top_r2', 'rest_r2']].merge(
        df_uni[['model_name', 'r2', 'total_neurons']], on='model_name', how='inner'
    )
    df_clean = df.dropna(subset=['top_count', 'top_r2', 'rest_r2', 'r2', 'total_neurons'])

    if df_clean.empty:
        print("[ERROR] Недостаточно валидных данных для построения графиков.")
        return

    print(f"\n{'=' * 75}")
    print("Генерация случайного бейзлайна (10 итераций на сессию)...")
    print(f"{'=' * 75}")

    random_baselines = []

    # 2. Расчет случайного бейзлайна "на лету"
    for idx, row in df_clean.iterrows():
        model_name = row['model_name']
        top_count = int(row['top_count'])

        if top_count <= 0:
            continue

        model_path = Path(models_dir) / f"{model_name}.pkl"
        data_file = Path(data_dir) / f"LNOF_{model_name}_aligned.npz"

        if not model_path.exists() or not data_file.exists():
            continue

        # Загрузка модели
        m_data = joblib.load(model_path)
        model = m_data.get('model')
        scaler = m_data.get('scaler')

        if not hasattr(model, 'coef_'):
            continue

        coefs = np.array(model.coef_)
        weights = np.linalg.norm(coefs, axis=0) if coefs.ndim > 1 else np.abs(coefs)
        important_idx = np.where(weights > 0)[0]

        if len(important_idx) == 0:
            continue

        # Загрузка данных
        features_raw, labels_raw = load_data(data_file)
        split_idx = int(len(features_raw) * (1 - test_size))

        if isinstance(features_raw, pd.DataFrame):
            X_test_raw = features_raw.iloc[split_idx:].values
            y_test = labels_raw.iloc[split_idx:].values
        else:
            X_test_raw = features_raw[split_idx:]
            y_test = labels_raw[split_idx:]

        X_test = scaler.transform(X_test_raw) if scaler is not None else X_test_raw

        # 10 случайных итераций
        for _ in range(10):
            sample_size = min(top_count, len(important_idx))
            rand_idx = np.random.choice(important_idx, size=sample_size, replace=False)

            X_rand = np.zeros_like(X_test)
            X_rand[:, rand_idx] = X_test[:, rand_idx]

            y_pred = model.predict(X_rand)
            r2_rand = r2_score(y_test, y_pred)

            random_baselines.append({
                'top_count': top_count,
                'r2_rand': r2_rand
            })

    df_rand = pd.DataFrame(random_baselines)
    print(f"Сгенерировано {len(df_rand)} точек случайного бейзлайна.")

    # ==========================================
    # ЭСТЕТИЧЕСКАЯ ОТРИСОВКА
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('white')
    ax1, ax2 = axes

    c_text = '#2C3E50'
    c_blue = '#64B5F6'  # Светло-голубой (Полноценная модель)
    c_yellow = '#FBC02D'  # Единый золотой (Сильные нейроны)
    c_green = '#43A047'  # Насыщенный зеленый (Остальные важные)
    c_rand = 'grey'  # Серый (Случайный бейзлайн)

    # ------------------------------------------
    # График 1 (Слева): Полноценная vs Сильные
    # ------------------------------------------
    # 1. Бейзлайн (регрессия тонким пунктиром + точки, на заднем фоне)
    sns.regplot(data=df_rand, x='top_count', y='r2_rand', ax=ax1, color=c_rand,
                scatter_kws={'alpha': 0.25, 's': 6, 'edgecolors': 'none', 'zorder': 1},
                line_kws={'linestyle': ':', 'linewidth': 1.5, 'alpha': 0.6, 'zorder': 1})

    # 2. Полноценная модель
    sns.regplot(data=df_clean, x='top_count', y='r2', ax=ax1, color=c_blue,
                scatter_kws={'alpha': 0.7, 's': 60, 'edgecolors': 'none', 'zorder': 3},
                line_kws={'linestyle': '--', 'linewidth': 2.5, 'alpha': 0.9, 'zorder': 2},
                label='Полноценная модель')

    # 3. Сильные нейроны
    sns.regplot(data=df_clean, x='top_count', y='top_r2', ax=ax1, color=c_yellow,
                scatter_kws={'alpha': 0.8, 's': 60, 'edgecolors': 'none', 'zorder': 4},
                line_kws={'linestyle': '--', 'linewidth': 2.5, 'alpha': 0.9, 'zorder': 2},
                label='Сильные нейроны (>3$\sigma$)')

    # ------------------------------------------
    # График 2 (Справа): Остальные vs Сильные
    # ------------------------------------------
    # 1. Бейзлайн (регрессия тонким пунктиром + точки, на заднем фоне)
    sns.regplot(data=df_rand, x='top_count', y='r2_rand', ax=ax2, color=c_rand,
                scatter_kws={'alpha': 0.25, 's': 6, 'edgecolors': 'none', 'zorder': 1},
                line_kws={'linestyle': ':', 'linewidth': 1.5, 'alpha': 0.6, 'zorder': 1})

    # 2. Остальные важные нейроны
    sns.regplot(data=df_clean, x='top_count', y='rest_r2', ax=ax2, color=c_green,
                scatter_kws={'alpha': 0.7, 's': 60, 'edgecolors': 'none', 'zorder': 3},
                line_kws={'linestyle': '--', 'linewidth': 2.5, 'alpha': 0.9, 'zorder': 2},
                label='Значимые нейроны')

    # 3. Сильные нейроны
    sns.regplot(data=df_clean, x='top_count', y='top_r2', ax=ax2, color=c_yellow,
                scatter_kws={'alpha': 0.8, 's': 60, 'edgecolors': 'none', 'zorder': 4},
                line_kws={'linestyle': '--', 'linewidth': 2.5, 'alpha': 0.9, 'zorder': 2},
                label='Сильные нейроны (>3$\sigma$)')

    # ------------------------------------------
    # ОБЩЕЕ ФОРМАТИРОВАНИЕ ОПТИКИ
    # ------------------------------------------
    for ax in [ax1, ax2]:
        # Полупрозрачный доверительный интервал (затронет и бейзлайн, и основные регрессии)
        for collection in ax.collections:
            if type(collection).__name__ == 'PolyCollection':
                collection.set_alpha(0.20)
                collection.set_zorder(0)

        ax.set_xlabel('Количество сильных нейронов (>3$\sigma$)', fontsize=13, labelpad=10, color=c_text)
        ax.set_ylabel(r'$R^2$', fontsize=15, labelpad=10, color=c_text)

        # Лимиты X
        ax.set_xlim(left=0)

        # Очистка рамок
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        [s.set_color(c_text) for s in ax.spines.values()]

        ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, colors=c_text)
        ax.grid(axis='both', linestyle=':', alpha=0.4, color='#BDC3C7', zorder=0)

        # Обработка легенды (Убираем бейзлайн, инвертируем порядок)
        handles, labels = ax.get_legend_handles_labels()
        # Фильтруем технические метки и бейзлайн (если Seaborn добавил r2_rand)
        valid_indices = [i for i, l in enumerate(labels) if l not in ['r2_rand', 'top_count', '']]
        handles_filtered = [handles[i] for i in valid_indices]
        labels_filtered = [labels[i] for i in valid_indices]

        ax.legend(handles_filtered[::-1], labels_filtered[::-1], fontsize=11, loc='lower right', frameon=False,
                  labelcolor=c_text)

    plt.tight_layout(w_pad=3.0)

    # Сохранение
    save_path = output_path / 'r2_topology_combined_with_baseline.png'
    plt.savefig(save_path, dpi=1000, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n✓ Успех! Комбинированный график с пунктирным бейзлайном сохранен: {save_path}")


def plot_combined_r2_single_panel_with_baseline(
        top_metrics_csv='stat_tables/top_vs_rest_metrics_abl.csv',
        unified_csv='stat_tables/unified_model_performance_report.csv',
        models_dir='models_LNOF_Lasso',
        data_dir='LNOF_SynchronizedData',
        output_dir='plots',
        test_size=0.25):
    """
    Объединяет графики сравнения R2 на одной панели.
    Шрифты увеличены в 3 раза, оси строго черные.
    Случайный бейзлайн подписан в легенде.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Загрузка таблиц
    try:
        df_top = pd.read_csv(top_metrics_csv)
        df_uni = pd.read_csv(unified_csv)
    except FileNotFoundError as e:
        print(f"[ERROR] Не найден один из файлов: {e}")
        return

    # Объединение
    df = df_top[['model_name', 'top_count', 'top_r2', 'rest_r2']].merge(
        df_uni[['model_name', 'r2', 'total_neurons']], on='model_name', how='inner'
    )
    df_clean = df.dropna(subset=['top_count', 'top_r2', 'rest_r2', 'r2', 'total_neurons'])

    if df_clean.empty:
        print("[ERROR] Недостаточно валидных данных для построения графиков.")
        return

    print(f"\n{'=' * 75}")
    print("Генерация случайного бейзлайна (10 итераций на сессию)...")
    print(f"{'=' * 75}")

    random_baselines = []

    # 2. Расчет случайного бейзлайна "на лету"
    for idx, row in df_clean.iterrows():
        model_name = row['model_name']
        top_count = int(row['top_count'])

        if top_count <= 0:
            continue

        model_path = Path(models_dir) / f"{model_name}.pkl"
        data_file = Path(data_dir) / f"LNOF_{model_name}_aligned.npz"

        if not model_path.exists() or not data_file.exists():
            continue

        # Загрузка модели
        m_data = joblib.load(model_path)
        model = m_data.get('model')
        scaler = m_data.get('scaler')

        if not hasattr(model, 'coef_'):
            continue

        coefs = np.array(model.coef_)
        weights = np.linalg.norm(coefs, axis=0) if coefs.ndim > 1 else np.abs(coefs)
        important_idx = np.where(weights > 0)[0]

        if len(important_idx) == 0:
            continue

        # Загрузка данных
        features_raw, labels_raw = load_data(data_file)
        split_idx = int(len(features_raw) * (1 - test_size))

        if isinstance(features_raw, pd.DataFrame):
            X_test_raw = features_raw.iloc[split_idx:].values
            y_test = labels_raw.iloc[split_idx:].values
        else:
            X_test_raw = features_raw[split_idx:]
            y_test = labels_raw[split_idx:]

        X_test = scaler.transform(X_test_raw) if scaler is not None else X_test_raw

        # 10 случайных итераций
        for _ in range(10):
            sample_size = min(top_count, len(important_idx))
            rand_idx = np.random.choice(important_idx, size=sample_size, replace=False)

            X_rand = np.zeros_like(X_test)
            X_rand[:, rand_idx] = X_test[:, rand_idx]

            y_pred = model.predict(X_rand)
            r2_rand = r2_score(y_test, y_pred)

            random_baselines.append({
                'top_count': top_count,
                'r2_rand': r2_rand
            })

    df_rand = pd.DataFrame(random_baselines)
    print(f"Сгенерировано {len(df_rand)} точек случайного бейзлайна.")

    # ==========================================
    # ЭСТЕТИЧЕСКАЯ ОТРИСОВКА (ГИГАНТСКИЙ МАСШТАБ)
    # ==========================================
    # Увеличиваем полотно, чтобы текст 45pt не слипался
    fig, ax = plt.subplots(figsize=(24, 16))
    fig.patch.set_facecolor('white')

    c_text = 'black'  # Строгий черный вместо '#2C3E50'
    c_blue = '#64B5F6'
    c_yellow = '#FBC02D'
    c_green = '#43A047'
    c_rand = 'grey'

    # 1. Бейзлайн (регрессия тонким пунктиром + точки, на заднем фоне)
    sns.regplot(data=df_rand, x='top_count', y='r2_rand', ax=ax, color=c_rand,
                scatter_kws={'alpha': 0.25, 's': 40, 'edgecolors': 'none', 'zorder': 1},
                line_kws={'linestyle': ':', 'linewidth': 3, 'alpha': 0.6, 'zorder': 1},
                label='Случайный бейзлайн')

    # 4. Сильные нейроны
    sns.regplot(data=df_clean, x='top_count', y='top_r2', ax=ax, color=c_yellow,
                scatter_kws={'alpha': 0.8, 's': 200, 'edgecolors': 'none', 'zorder': 4},
                line_kws={'linestyle': '--', 'linewidth': 6, 'alpha': 0.9, 'zorder': 2},
                label='Сильные нейроны (>3$\sigma$)')

    # 3. Остальные важные нейроны
    sns.regplot(data=df_clean, x='top_count', y='rest_r2', ax=ax, color=c_green,
                scatter_kws={'alpha': 0.7, 's': 200, 'edgecolors': 'none', 'zorder': 3},
                line_kws={'linestyle': '--', 'linewidth': 6, 'alpha': 0.9, 'zorder': 2},
                label='Остальные уч. нейроны')

    # 2. Полноценная модель
    sns.regplot(data=df_clean, x='top_count', y='r2', ax=ax, color=c_blue,
                scatter_kws={'alpha': 0.7, 's': 200, 'edgecolors': 'none', 'zorder': 3},
                line_kws={'linestyle': '--', 'linewidth': 6, 'alpha': 0.9, 'zorder': 2},
                label='Все нейроны')

    # ==========================================
    # ОБЩЕЕ ФОРМАТИРОВАНИЕ ОПТИКИ
    # ==========================================
    for collection in ax.collections:
        if type(collection).__name__ == 'PolyCollection':
            collection.set_alpha(0.20)
            collection.set_zorder(0)

    # Увеличенные в 3 раза подписи осей (было 13 и 15 -> стало 39 и 45)
    ax.set_xlabel('Количество сильных нейронов (>3$\sigma$)', fontsize=39, labelpad=20, color=c_text)
    ax.set_ylabel(r'$R^2$', fontsize=45, labelpad=20, color=c_text)

    # Лимиты X
    ax.set_xlim(left=0)

    # Очистка рамок и настройка утолщенных осей (под стать гигантскому тексту)
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(3.5)
    ax.spines['bottom'].set_linewidth(3.5)
    [s.set_color(c_text) for s in ax.spines.values()]

    # Увеличенные в 3 раза тики (было 12 -> стало 36)
    ax.tick_params(axis='both', which='major', labelsize=36, width=3.5, length=12, colors=c_text)
    ax.grid(axis='both', linestyle=':', alpha=0.4, color='black', zorder=0)

    # Обработка легенды (Оставляем бейзлайн)
    handles, labels = ax.get_legend_handles_labels()
    # Фильтруем только технические метки, оставляем наш подписанный бейзлайн
    valid_indices = [i for i, l in enumerate(labels) if l not in ['r2_rand', 'top_count', '']]
    handles_filtered = [handles[i] for i in valid_indices]
    labels_filtered = [labels[i] for i in valid_indices]

    # Увеличенная легенда (было 11 -> стало 33)
    ax.legend(handles_filtered[::-1], labels_filtered[::-1], fontsize=33, loc='lower right', frameon=False,
              labelcolor=c_text)

    plt.tight_layout(pad=3.0)

    # Сохранение
    save_path = output_path / 'r2_topology_single_panel_giant_text.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white') # Снизил DPI до 300, так как холст 24x16 дюймов
    plt.close()

    print(f"\n✓ Успех! Объединенный график (Масштаб х3, Черные оси) сохранен: {save_path}")


def plot_comparability_threshold_dependency(
        metrics_csv='stat_tables/top_vs_rest_metrics_abl.csv',
        r2_threshold=0.15,
        output_dir='plots'):
    """
    Строит два графика:
    1. Локальная вероятность сопоставимости (как распределена по бинам).
    2. Накопительная вероятность (для всех сессий >= X сильных нейронов) + Индекс Джини.
    Размер точек пропорционален количеству сессий.
    """
    path_csv = Path(metrics_csv)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not path_csv.exists():
        print(f"[ERROR] Файл не найден: {path_csv}")
        return

    df = pd.read_csv(path_csv)
    req_cols = ['top_count', 'top_r2', 'rest_r2']
    if not all(c in df.columns for c in req_cols):
        print(f"[ERROR] В таблице отсутствуют нужные колонки: {req_cols}")
        return

    df_clean = df.dropna(subset=req_cols).copy()
    if df_clean.empty:
        print("[ERROR] Нет данных после очистки NaN.")
        return

    # --- 1. Подготовка данных ---
    df_clean['r2_diff'] = (df_clean['top_r2'] - df_clean['rest_r2']).abs()
    df_clean['is_comparable'] = (df_clean['r2_diff'] <= r2_threshold).astype(int)

    # ДАННЫЕ ЛЕВОГО ГРАФИКА: Локальные бины (ровно X нейронов)
    agg_df = df_clean.groupby('top_count').agg(
        success_rate=('is_comparable', lambda x: x.mean() * 100),
        session_count=('is_comparable', 'count')
    ).reset_index()

    # ДАННЫЕ ПРАВОГО ГРАФИКА: Накопительные (>= X нейронов)
    thresholds = sorted(df_clean['top_count'].unique())
    cum_success = []
    cum_counts = []

    for th in thresholds:
        # Берем все сессии, где сильных нейронов >= текущего порога
        subset = df_clean[df_clean['top_count'] >= th]
        rate = subset['is_comparable'].mean() * 100

        cum_success.append(rate)
        cum_counts.append(len(subset))

    cum_df = pd.DataFrame({
        'threshold': thresholds,
        'cum_success_rate': cum_success,
        'remaining_sessions': cum_counts
    })

    # --- 2. Расчет статистик ---
    # Спирмен (Левый график)
    corr, pval = spearmanr(agg_df['top_count'], agg_df['success_rate'])

    # Индекс Джини (Правый график)
    def gini_coefficient(x):
        """Классический расчет коэффициента Джини для массива значений"""
        x = np.array(x, dtype=np.float64)
        if np.amin(x) < 0:
            x -= np.amin(x)  # Сдвиг на случай отрицательных значений
        x += 1e-10  # Защита от деления на ноль
        x = np.sort(x)
        n = len(x)
        return (np.sum((2 * np.arange(1, n + 1) - n - 1) * x)) / (n * np.sum(x))

    gini_val = gini_coefficient(cum_df['cum_success_rate'])

    def get_stars(p):
        return r"^{\ast\ast\ast}" if p < 0.001 else r"^{\ast\ast}" if p < 0.01 else r"^{\ast}" if p < 0.05 else "^{ns}"

    # =========================================================================
    # ЭСТЕТИЧЕСКАЯ ОТРИСОВКА
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('white')
    ax1, ax2 = axes

    c_text = '#2C3E50'
    c_dots = '#3498DB'  # Синий (Левый)
    c_line = '#E74C3C'  # Красный тренд (Левый)
    c_cum = '#8E44AD'  # Глубокий пурпурный (Правый накопительный)

    # --------------------------------------------------
    # ЛЕВЫЙ ГРАФИК: Локальная вероятность
    # --------------------------------------------------
    scatter = ax1.scatter(agg_df['top_count'], agg_df['success_rate'],
                          s=agg_df['session_count'] * 40,
                          color=c_dots, alpha=0.75, edgecolors='white', linewidth=1.5, zorder=3)

    sns.regplot(data=agg_df, x='top_count', y='success_rate', ax=ax1,
                scatter=False, color=c_line,
                line_kws={'linestyle': '--', 'linewidth': 2.5, 'zorder': 2},
                ci=95, scatter_kws={'alpha': 0.2})

    stars = get_stars(pval)
    stat_text_left = f"$\\rho = {corr:.2f}{stars}$"
    ax1.text(0.05, 0.95, stat_text_left, transform=ax1.transAxes, ha='left', va='top',
             fontsize=15, color=c_text, weight='bold',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=5), zorder=5)

    ax1.set_title(rf"Локальная вероятность ($\Delta R^2 \leq {r2_threshold}$)",
                  fontsize=15, pad=15, fontweight='bold', color=c_text)
    ax1.set_xlabel(r"Количество сильных нейронов (>3$\sigma$)", fontsize=13, labelpad=10, color=c_text)
    ax1.set_ylabel("Доля сопоставимых сессий (%)", fontsize=13, labelpad=10, color=c_text)

    # Легенда размеров (Левый)
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, num=4, func=lambda s: s / 40)
    labels = [str(int(round(float(re.findall(r"[-+]?(?:\d*\.*\d+)", l)[0])))) for l in labels]
    legend = ax1.legend(handles, labels, title="Кол-во\nсессий", loc="lower right",
                        frameon=False, fontsize=11, title_fontsize=12, labelcolor=c_text)
    legend.get_title().set_fontweight('bold')

    # --------------------------------------------------
    # ПРАВЫЙ ГРАФИК: Накопительная вероятность (>= X)
    # --------------------------------------------------
    # Размер точек пропорционален оставшимся сессиям (коэффициент меньше, т.к. сессий больше)
    scatter_cum = ax2.scatter(cum_df['threshold'], cum_df['cum_success_rate'],
                              s=cum_df['remaining_sessions'] * 15,
                              color=c_cum, alpha=0.8, edgecolors='white', linewidth=1.5, zorder=3)

    ax2.plot(cum_df['threshold'], cum_df['cum_success_rate'],
             color=c_cum, linestyle='-', linewidth=2.5, alpha=0.5, zorder=2)

    # Аннотация Джини
    stat_text_right = f"Gini = {gini_val:.3f}"
    ax2.text(0.05, 0.95, stat_text_right, transform=ax2.transAxes, ha='left', va='top',
             fontsize=15, color=c_text, weight='bold',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=5), zorder=5)

    ax2.set_title(r"Накопительная вероятность (для сессий $\geq X$ нейронов)",
                  fontsize=15, pad=15, fontweight='bold', color=c_text)
    ax2.set_xlabel(r"Порог количества сильных нейронов ($X$)", fontsize=13, labelpad=10, color=c_text)
    ax2.set_ylabel("Доля сопоставимых сессий (%)", fontsize=13, labelpad=10, color=c_text)

    # Легенда размеров (Правый)
    handles_c, labels_c = scatter_cum.legend_elements(prop="sizes", alpha=0.6, num=4, func=lambda s: s / 15)
    labels_c = [str(int(round(float(re.findall(r"[-+]?(?:\d*\.*\d+)", l)[0])))) for l in labels_c]
    legend_c = ax2.legend(handles_c, labels_c, title="Осталось\nсессий", loc="lower right",
                          frameon=False, fontsize=11, title_fontsize=12, labelcolor=c_text)
    legend_c.get_title().set_fontweight('bold')

    # --------------------------------------------------
    # ОБЩЕЕ ФОРМАТИРОВАНИЕ ОПТИКИ
    # --------------------------------------------------
    for ax in [ax1, ax2]:
        ax.set_ylim(-5, 110)
        ax.set_xlim(left=0)
        ax.set_facecolor('white')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        [s.set_color(c_text) for s in ax.spines.values()]

        ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, colors=c_text)
        ax.grid(axis='both', linestyle=':', alpha=0.4, color='#BDC3C7', zorder=0)

        for collection in ax.collections:
            if type(collection).__name__ == 'PolyCollection':
                collection.set_alpha(0.15)

    plt.tight_layout(w_pad=4.0)

    save_path = output_path / 'comparability_probability_with_cumulative.png'
    plt.savefig(save_path, dpi=1000, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n✓ Успешно! Двойной график сохранен: {save_path}")
    print(f"  > Спирмен (Левый): ρ = {corr:.3f} (p-value: {pval:.2e})")
    print(f"  > Индекс Джини (Правый): {gini_val:.3f}")


def plot_global_euler_diagram_present(sessions=None,
                              sets_to_plot=('intense', 'linreg', 'zcrit'),
                              color_scheme='cmyk',
                              ablation_csv='stat_tables/full_ablation_stats.csv',
                              database_csv='LNOF_SynchronizedData/LNOF_database.csv',
                              models_dir='models_LNOF_Lasso',
                              output_dir='plots'):
    """
    Визуализирует пересечение глобальных пулов нейронов.
    Дизайн: прозрачные круги, только толстые цветные контуры.
    Числа убраны, подписи размещены СНАРУЖИ кругов черным цветом.
    Включена строгая фильтрация для MI > 0.04.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    excluded_mice = ['J20', 'J21']

    try:
        df_abl = pd.read_csv(ablation_csv)
        df_db = pd.read_csv(database_csv)
    except FileNotFoundError as e:
        print(f"[ERROR] Не найден файл базы данных: {e}")
        return

    all_sessions = df_abl['model_session'].unique().tolist()
    if sessions is None:
        sessions_to_process = all_sessions
    else:
        sessions_to_process = [s for s in sessions if s in all_sessions]

    sessions_to_process = [s for s in sessions_to_process if s[:3] not in excluded_mice]

    print(f"\n{'=' * 70}")
    print(f"Сборка глобального графа пересечений (Сессий: {len(sessions_to_process)})")
    print(f"Фильтрация: MI > 0.04")
    print(f"{'=' * 70}")

    global_sets = {
        'intense': set(),
        'linreg': set(),
        'zcrit': set(),
        'crit': set()
    }

    # Ищем правильное название колонки MI
    mi_col = 'me' if 'me' in df_db.columns else None
    if not mi_col:
        print("[WARNING] Колонка 'MI' не найдена в базе данных. Фильтрация по порогу пропущена.")

    # 1. Итерация по сессиям
    for sess in sessions_to_process:
        mouse = sess[:3]
        day = sess[4:6]

        # --- INTENSE (С фильтром MI > 0.04) ---
        if 'intense' in sets_to_plot:
            if mi_col:
                df_int = df_db[(df_db['mouse'] == mouse) &
                               (df_db['session'] == day) &
                               (df_db['feature'] == 'place') &
                               (df_db[mi_col] > 0.04)]
            else:
                df_int = df_db[(df_db['mouse'] == mouse) &
                               (df_db['session'] == day) &
                               (df_db['feature'] == 'place')]

            for idx in df_int['neuron_idx'].unique():
                global_sets['intense'].add(f"{sess}_neuron_{int(idx)}")

        # --- LINREG, CRIT & ZCRIT ---
        model_path = Path(models_dir) / f"{sess}.pkl"
        if not model_path.exists():
            continue

        data = joblib.load(model_path)
        model = data.get('model')
        if not hasattr(model, 'coef_'):
            continue

        feats = np.array(data.get('feature_names'))
        weights = np.linalg.norm(model.coef_, axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)

        # LINREG
        if 'linreg' in sets_to_plot:
            for f in feats[weights > 0]:
                global_sets['linreg'].add(f"{sess}_{f}")

        # CRIT & ZCRIT
        if 'crit' in sets_to_plot or 'zcrit' in sets_to_plot:
            abl_row = df_abl[df_abl['model_session'] == sess].iloc[0]
            base_r2 = abl_row['baseline_r2']
            abl_r2_vals = np.array([abl_row.get(f'ablation_{f}', np.nan) for f in feats])

            crit = base_r2 - abl_r2_vals
            valid_crit = crit[~np.isnan(crit)]

            if 'crit' in sets_to_plot:
                for f in feats[~np.isnan(crit) & (crit > 0)]:
                    global_sets['crit'].add(f"{sess}_{f}")

            if 'zcrit' in sets_to_plot:
                if len(valid_crit) > 0:
                    std_c = np.std(valid_crit)
                    mean_c = np.mean(valid_crit)
                    if std_c > 0:
                        zcrit = (crit - mean_c) / std_c
                        for f in feats[~np.isnan(zcrit) & (zcrit > 0)]:
                            global_sets['zcrit'].add(f"{sess}_{f}")

    # 2. Подготовка данных
    subsets = [global_sets[k] for k in sets_to_plot if k in global_sets]

    label_map = {
        'intense': 'Клетки места',
        'linreg': 'Участвующие',
        'zcrit': 'Важные',
        'crit': r'$\delta$'
    }
    labels = [label_map.get(k, k.upper()) for k in sets_to_plot if k in global_sets]

    if len(subsets) < 2:
        print("[ERROR] Недостаточно множеств для пересечения (минимум 2).")
        return

    # 3. Визуализация
    fig, ax = plt.subplots(figsize=(14, 14))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    c_purple = '#8E44AD' # 0000FF
    c_green = '#27AE60' # 00FF00

    color_maps = {
        'cmyk': {'100': '#00FFFF', '010': '#FF00FF', '001': '#FFFF00'},
        'rgb': {'100': '#1F77B4', '010': '#8E44AD', '001': '#27AE60'},
        'bi': {'100': '#D60270', '010': '#9B4F96', '001': '#0038A8'}
    }

    scheme = color_scheme.lower()
    selected_map = color_maps.get(scheme, color_maps['cmyk'])

    # Инициализация графа и извлечение кругов контура
    if len(subsets) == 3:
        v = venn3(subsets, set_labels=labels, ax=ax)
        c = venn3_circles(subsets, ax=ax)
        colors_for_circles = [selected_map['100'], selected_map['010'], selected_map['001']]
    else:
        v = venn2(subsets, set_labels=labels, ax=ax)
        c = venn2_circles(subsets, ax=ax)
        colors_for_circles = [selected_map['100'], selected_map['010']]

    if v:
        # УБИРАЕМ ЗАЛИВКУ
        for patch in v.patches:
            if patch:
                patch.set_facecolor('none')
                patch.set_edgecolor('none')
                patch.set_alpha(0)

        # СКРЫВАЕМ ВНУТРЕННИЕ ЧИСЛА
        for text in v.subset_labels:
            if text:
                text.set_visible(False)

        # НАСТРАИВАЕМ ВНЕШНИЕ ПОДПИСИ (Черный цвет, крупный шрифт)
        for text in v.set_labels:
            if text:
                text.set_color('black')
                text.set_fontsize(36)
                text.set_fontweight('bold')

        # РИСУЕМ ТОЛСТЫЕ КОНТУРЫ КРУГОВ
        for i in range(len(subsets)):
            if c and c[i]:
                c[i].set_facecolor('none')
                c[i].set_edgecolor(colors_for_circles[i])
                c[i].set_linewidth(8)  # Толстый контур
                c[i].set_alpha(1.0)

    ax.axis('off')

    file_prefix = f"global_euler_outlines_{'_'.join(sets_to_plot)}_{scheme}"

    fig.savefig(output_path / f"{file_prefix}.svg", format='svg', bbox_inches='tight', facecolor='white')
    fig.savefig(output_path / f"{file_prefix}.png", format='png', dpi=1000, bbox_inches='tight', facecolor='white')

    plt.close()
    print(f"✓ Диаграмма {scheme.upper()} (контуры + внешние подписи) сохранена в {output_path}")


def plot_cumulative_retrain_addition_dual(
        session_name,
        data_dir='LNOF_SynchronizedData',
        models_dir='models_LNOF_Lasso',
        output_dir='plots',
        step_size=5,
        r2_threshold_diff=0.03,
        med_threshold_diff=0.32
):
    """
    Строит графики переобучения (добавление нейронов) для R2 и MED.
    Ось X переведена в процент от числа УЧАСТВУЮЩИХ нейронов.
    Оси имеют стандартный лейаут, но границы X расширены (-3 до 103),
    чтобы график не пересекался с осью Y.
    Цвета: Зеленый (Сверху) и Желтый (Снизу).
    """
    c_param = 0.5
    test_size = 0.25
    max_iter = 100

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    mouse = session_name[:3]
    day = session_name[4:5]

    # 1. Загрузка базовой модели
    model_path = Path(models_dir) / f"{session_name}.pkl"
    if not model_path.exists():
        print(f"[ERROR] Базовая модель {session_name} не найдена.")
        return

    data = joblib.load(model_path)
    base_coef = data.get('model').coef_
    all_features = np.array(data.get('feature_names'))
    all_feat_list = list(all_features)
    num_tot = len(all_features)

    # 2. Изоляция только УЧАСТВУЮЩИХ нейронов (вес > 0)
    weights = np.linalg.norm(base_coef, axis=0) if base_coef.ndim > 1 else np.abs(base_coef)
    important_mask = weights >= 0
    num_part = np.sum(important_mask)

    if num_part == 0:
        print(f"[ERROR] В модели {session_name} нет участвующих нейронов.")
        return

    part_features = all_features[important_mask]
    part_weights = weights[important_mask]

    sorted_idx_desc = np.argsort(part_weights)[::-1]
    sorted_idx_asc = np.argsort(part_weights)

    np.random.seed(42)
    rand_idx = np.random.permutation(num_part)

    feat_desc = part_features[sorted_idx_desc]  # Сверху (Сильные)
    feat_asc = part_features[sorted_idx_asc]  # Снизу (Слабые)
    feat_rand = part_features[rand_idx]  # Случайно

    # 3. Загрузка данных
    data_file = Path(data_dir) / f"LNOF_{session_name}_aligned.npz"
    if not data_file.exists():
        print(f"[ERROR] Файл данных {data_file} не найден.")
        return

    features_raw, labels = load_data(data_file)
    X_train_full, X_test_full, y_train, y_test = train_test_split_temporal(features_raw, labels, test_size=test_size)

    def evaluate_subset(features_subset):
        if len(features_subset) == 0:
            return 0.0, np.nan

        if isinstance(X_train_full, pd.DataFrame):
            X_tr = X_train_full[features_subset]
            X_te = X_test_full[features_subset]
        else:
            idx = [all_feat_list.index(f) for f in features_subset]
            X_tr = X_train_full[:, idx]
            X_te = X_test_full[:, idx]

        clf, imp, scl, metrics = train_lasso(X_tr, y_train, X_te, y_test, C=c_param, max_iter=max_iter)

        r2 = metrics.get('test_r2', 0.0)
        med = metrics.get('test_dist', metrics.get('test_med', metrics.get('test_mae', np.nan)))
        return r2, med

    # Получаем абсолютный бейзлайн (полная модель)
    base_r2, base_med = evaluate_subset(all_features)

    # 4. Формируем шаги (только до num_part) и переводим их в проценты
    steps = list(range(0, num_part, step_size))
    if steps[-1] != num_part:
        steps.append(num_part)

    steps_pct = [(k / num_part) * 100 for k in steps]

    print(f"\n{'=' * 70}")
    print(f"[{session_name}] Начат процесс переобучения (Addition). R2 и MED.")
    print(f"{'=' * 70}")

    r2_green, med_green = [], []
    r2_yellow, med_yellow = [], []
    r2_grey, med_grey = [], []

    # 5. Цикл переобучения
    for k in steps:
        r2_g, med_g = evaluate_subset(feat_desc[:k])
        r2_green.append(r2_g)
        med_green.append(med_g)

        r2_y, med_y = evaluate_subset(feat_asc[:k])
        r2_yellow.append(r2_y)
        med_yellow.append(med_y)

        r2_gr, med_gr = evaluate_subset(feat_rand[:k])
        r2_grey.append(r2_gr)
        med_grey.append(med_gr)

        if k % (step_size * 5) == 0 and k > 0:
            print(f"  > Пройдено {k}/{num_part} нейронов ({(k / num_part) * 100:.1f}%)...")

    # --- Поиск пороговых значений в процентах ---
    thresh_r2_idx = next((i for i, v in enumerate(r2_green) if v >= base_r2 - r2_threshold_diff), None)
    thresh_r2_x = steps_pct[thresh_r2_idx] if thresh_r2_idx is not None else None

    thresh_med_idx = next(
        (i for i, v in enumerate(med_green) if not np.isnan(v) and v <= base_med + med_threshold_diff), None)
    thresh_med_x = steps_pct[thresh_med_idx] if thresh_med_idx is not None else None

    # --- ВЫВОД РЕЗУЛЬТАТОВ ДЛЯ ТЕКУЩЕЙ СЕССИИ ---
    print(f"\n{'=' * 70}")
    print(f"СВОДКА ДЛЯ СЕССИИ: {session_name}")
    print(f"Всего нейронов: {num_tot} | Участвующих (>0): {num_part}")

    if thresh_r2_x is not None:
        print(f"\n[R2] Достигнут порог (Δ <= {r2_threshold_diff}): {thresh_r2_x:.1f}% от участвующих")
    else:
        print(f"\n[R2] Порог не достигнут даже при использовании всех участвующих нейронов.")

    if thresh_med_x is not None:
        print(f"[MED] Достигнут порог (Δ <= {med_threshold_diff} см): {thresh_med_x:.1f}% от участвующих")
    print(f"{'=' * 70}\n")

    # ==========================================
    # 6. ЭСТЕТИЧЕСКАЯ ОТРИСОВКА
    # ==========================================
    c_green = '#FBC02D'  # Сверху
    c_yellow = '#43A047'  # Снизу
    c_grey = '#7F8C8D'  # Случайно
    fill_color = '#BDC3C7'
    c_text = '#2C3E50'

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(top=0.90, bottom=0.25, wspace=0.15)
    ax1, ax2 = axes

    pt_distance = 432 / len(steps_pct)
    marker_style = 'o' if pt_distance > 10 else None
    marker_size = 4

    # --- ПАНЕЛЬ 1: R2 ---
    ax1.fill_between(steps_pct, r2_green, r2_yellow, color=fill_color, alpha=0.35, edgecolor='none', zorder=1)

    ax1.plot(steps_pct, r2_grey, color=c_grey, lw=2, linestyle='--', alpha=0.6, zorder=2)
    ax1.plot(steps_pct, r2_yellow, color=c_yellow, lw=2.5, marker=marker_style, markersize=marker_size, zorder=3)
    ax1.plot(steps_pct, r2_green, color=c_green, lw=2.5, marker=marker_style, markersize=marker_size, zorder=4)

    if thresh_r2_x is not None:
        ax1.axvline(x=thresh_r2_x, color='black', linestyle='-.', alpha=0.4, linewidth=1.5, zorder=1)

    ax1.set_ylabel(r"$R^2$", fontsize=15, labelpad=10, color=c_text)
    ax1.yaxis.set_major_locator(MultipleLocator(0.1))

    # --- ПАНЕЛЬ 2: MED ---
    ax2.fill_between(steps_pct, med_green, med_yellow, color=fill_color, alpha=0.35, edgecolor='none', zorder=1)

    ax2.plot(steps_pct, med_grey, color=c_grey, lw=2, linestyle='--', alpha=0.6, zorder=2)
    ax2.plot(steps_pct, med_yellow, color=c_yellow, lw=2.5, marker=marker_style, markersize=marker_size, zorder=3)
    ax2.plot(steps_pct, med_green, color=c_green, lw=2.5, marker=marker_style, markersize=marker_size, zorder=4)

    if thresh_med_x is not None:
        ax2.axvline(x=thresh_med_x, color='black', linestyle='-.', alpha=0.4, linewidth=1.5, zorder=1)

    ax2.set_ylabel("MED (см)", fontsize=13, labelpad=10, color=c_text)
    ax2.yaxis.set_major_locator(MultipleLocator(1.0))

    # --- ОЧИСТКА И ФОРМАТИРОВАНИЕ ОСЕЙ ---
    for ax in axes:
        ax.grid(axis='y', linestyle='--', alpha=0.5, color='#BDC3C7', zorder=0)
        ax.grid(axis='x', linestyle=':', alpha=0.3, color='#BDC3C7', zorder=0)

        # Расширяем границы, чтобы данные не касались осей Y, но сохраняем стандартный лейаут
        ax.set_xlim(-3, 103)
        # Фиксируем тики на ровных значениях
        ax.set_xticks(np.arange(0, 101, 20))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['left'].set_color(c_text)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.spines['bottom'].set_color(c_text)

        ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, colors=c_text)

    # ЕДИНАЯ ПОДПИСЬ ДЛЯ ОСИ X
    fig.supxlabel("Доля добавленных нейронов (%)", fontsize=14, color=c_text, y=0.15)

    # --- ЛЕГЕНДА СНИЗУ ---
    legend_elements = [
        mlines.Line2D([0], [0], color=c_green, lw=2.5, marker='o', markersize=5, label='Сверху'),
        mlines.Line2D([0], [0], color=c_grey, lw=2, linestyle='--', alpha=0.6, label='Случайно'),
        mlines.Line2D([0], [0], color=c_yellow, lw=2.5, marker='o', markersize=5, label='Снизу')
    ]

    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.10),
               ncol=3, frameon=False, fontsize=12, handlelength=2.5, columnspacing=2.0)

    mouse_formatted = mouse.replace('J', '') if mouse.startswith('J') else mouse
    fig.suptitle(f"Мышь J{mouse_formatted} день {day}", fontsize=18, fontweight='bold', y=0.98, color=c_text)

    # 7. Сохранение
    save_path = output_path / f'cumulative_addition_dual_{session_name}.png'
    plt.savefig(save_path, dpi=1000, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n✓ Готово! Двойной график Addition (R2 + MED) сохранен: {save_path}")


def plot_top_neurons_survival_absolute_with_baseline_present(
        top_csv='stat_tables/top_neurons_ablation_3.0sigma.csv',
        match_dir='MatchData',
        models_dir='models_LNOF_Lasso',
        output_dir='plots',
        model_is_zero_indexed=True):
    """
    Отслеживает абсолютное количество топовых нейронов (>3 sigma) по дням.
    Отрисовывает 1 график (сохраняя пропорции).
    Индивидуальные траектории скрыты для минимализма.
    Добавлен серый пунктирный бейзлайн случайного физического выживания ВАЖНЫХ нейронов.
    Легенда размещена в безопасной зоне по оси Y (между 0.5 и 4).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    excluded_mice = ['J20', 'J21']
    match_files = list(Path(match_dir).glob("LNOF_*.csv"))

    try:
        df_top = pd.read_csv(top_csv)
    except FileNotFoundError:
        print(f"[ERROR] Не найден файл: {top_csv}")
        return

    df_names = df_top[df_top['data_type'] == 'Neuron Names'].copy()
    top_cols = [c for c in df_names.columns if c.startswith('top_')]

    results = []
    rand_results = []

    print(f"\n{'=' * 75}")
    print("Расчет выживаемости (Абсолютные количества + Случайный бейзлайн важных x10)")
    print(f"{'=' * 75}")

    for file_path in match_files:
        mouse = file_path.stem.replace('LNOF_', '')
        if mouse in excluded_mice:
            continue

        df_map = pd.read_csv(file_path, header=None)
        if df_map.shape[1] < 4:
            continue

        tracked_sets = []
        important_sets = []

        for day_idx in range(4):
            day = f"{day_idx + 1}D"

            col_data = df_map[day_idx]
            tracked_mask = col_data.notna() & (col_data > 0)
            tracked_sets.append(set(df_map[tracked_mask].index))

            row = df_names[(df_names['mouse'] == mouse) & (df_names['session'] == day)]
            important_local_ids = []

            if not row.empty:
                neuron_names = row.iloc[0][top_cols].dropna().values
                for name in neuron_names:
                    match = re.search(r'\d+', str(name))
                    if match:
                        n_id = int(match.group())
                        if model_is_zero_indexed:
                            n_id += 1
                        important_local_ids.append(n_id)

            important_mask = col_data.isin(important_local_ids)
            important_sets.append(set(df_map[important_mask].index))

        if len(important_sets[0]) == 0:
            continue

        # --- 1. Извлекаем пул ВАЖНЫХ нейронов из модели 1-го дня ---
        model_path = Path(models_dir) / f"{mouse}_1D.pkl"
        day1_important_pool = []

        if model_path.exists():
            m_data = joblib.load(model_path)
            model = m_data.get('model')
            if hasattr(model, 'coef_'):
                coefs = np.array(model.coef_)
                weights = np.linalg.norm(coefs, axis=0) if coefs.ndim > 1 else np.abs(coefs)
                feat_names = np.array(m_data.get('feature_names'))

                for name in feat_names[weights > 0]:
                    match = re.search(r'\d+', str(name))
                    if match:
                        n_id = int(match.group())
                        if model_is_zero_indexed:
                            n_id += 1
                        day1_important_pool.append(n_id)

        # Убеждаемся, что эти важные нейроны реально затрекались в 1-й день
        day1_important_pool = list(set(day1_important_pool).intersection(tracked_sets[0]))
        num_strong_day1 = len(important_sets[0])

        # --- 2. Расчет случайного бейзлайна (10 итераций из ВАЖНЫХ) ---
        for _ in range(10):
            # Защита: берем не больше, чем есть в пуле важных
            sample_size = min(num_strong_day1, len(day1_important_pool))

            if sample_size > 0:
                rand_day1 = set(np.random.choice(day1_important_pool, size=sample_size, replace=False))
            else:
                rand_day1 = set()

            for step in range(4):
                # Смотрим, сколько из этих случайных физически отследились в день D
                rand_survived = len(rand_day1.intersection(*tracked_sets[:step + 1]))
                rand_results.append({
                    'mouse': mouse,
                    'day': step + 1,
                    'rand_survived': rand_survived
                })

        # --- 3. Расчет целевых метрик ---
        for step in range(4):
            day_num = step + 1
            N_top_d = len(important_sets[step])
            V_d = len(set.intersection(*important_sets[:step + 1]))

            results.append({
                'mouse': mouse,
                'day': day_num,
                'N_top_d': N_top_d,
                'V_d': V_d
            })

        print(f"[{mouse}] Обработано. Сильных в 1D: {num_strong_day1}")

    df_res = pd.DataFrame(results)
    df_rand = pd.DataFrame(rand_results)

    if df_res.empty:
        print("[ERROR] Нет данных для построения.")
        return

    # =========================================================================
    # ЭСТЕТИЧЕСКАЯ ОТРИСОВКА (Alpine Green Theme, 1 график)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(7, 6))  # Сохраняем соотношение сторон одной панели
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    c_text = '#2C3E50'
    c_agg_base = 'black'
    c_line_imp = '#12A57A'  # Alpine Green
    c_rand = 'grey'

    # 1. Глобальные тренды
    # Черная линия
    sns.lineplot(data=df_res, x='day', y='N_top_d', ax=ax, color=c_agg_base,
                 errorbar=('ci', 95), marker='o', markersize=8, linewidth=2.5,
                 markeredgecolor='none', err_kws={'facecolor': 'grey', 'alpha': 0.20, 'linewidth': 0}, zorder=5)

    # Серая пунктирная линия (Случайные ВАЖНЫЕ)
    sns.lineplot(data=df_rand, x='day', y='rand_survived', ax=ax, color=c_rand,
                 errorbar=('ci', 95), linestyle='--', marker='s', markersize=6, linewidth=1.5,
                 markeredgecolor='none', err_kws={'facecolor': c_rand, 'alpha': 0.15, 'linewidth': 0}, zorder=3)

    # Alpine Green линия
    sns.lineplot(data=df_res, x='day', y='V_d', ax=ax, color=c_line_imp,
                 errorbar=('ci', 95), marker='^', markersize=10, linewidth=2.5,
                 markerfacecolor=c_line_imp, markeredgecolor='none',
                 err_kws={'facecolor': c_line_imp, 'alpha': 0.25, 'linewidth': 0}, zorder=6)

    # 2. Умные аннотации
    agg_res = df_res.groupby('day').mean(numeric_only=True).reset_index()
    agg_rand = df_rand.groupby('day').mean(numeric_only=True).reset_index()

    global_max = max(agg_res['N_top_d'].max(), agg_res['V_d'].max(), agg_rand['rand_survived'].max()) * 1.25
    ax.set_ylim(-0.5, global_max)

    for d in range(1, 5):
        y_b = agg_res.loc[agg_res['day'] == d, 'N_top_d'].values[0]
        y_v = agg_res.loc[agg_res['day'] == d, 'V_d'].values[0]
        y_r = agg_rand.loc[agg_rand['day'] == d, 'rand_survived'].values[0]

        if d == 1:
            ax.annotate(f"{y_v:.1f}", (d, y_v), textcoords="offset points", xytext=(0, 12),
                        ha='center', va='bottom', fontsize=11, color=c_agg_base, weight='bold', zorder=7)
        else:
            points = [(y_b, c_agg_base), (y_r, c_rand), (y_v, c_line_imp)]
            points.sort(key=lambda x: x[0])

            for i, (y_val, color) in enumerate(points):
                if i == 0:
                    off, va = -14, 'top'
                elif i == len(points) - 1:
                    off, va = 12, 'bottom'
                else:
                    dist_up = points[i + 1][0] - y_val
                    dist_down = y_val - points[i - 1][0]
                    if dist_up > dist_down:
                        off, va = 12, 'bottom'
                    else:
                        off, va = -14, 'top'

                if va == 'top' and y_val < global_max * 0.05:
                    off, va = 12, 'bottom'

                ax.annotate(f"{y_val:.1f}", (d, y_val), textcoords="offset points", xytext=(0, off),
                            ha='center', va=va, fontsize=11, color=color, weight='bold', zorder=7)

    # 3. Оформление осей и рамки
    ax.set_title("Абсолютный дрейф", fontsize=14, pad=15, fontweight='bold', color=c_text)
    ax.set_ylabel("Количество сильных нейронов", fontsize=13, labelpad=10, color=c_text)
    ax.set_xlabel("День", fontsize=13, labelpad=10, color=c_text)

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlim(0.8, 4.2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    [s.set_color(c_text) for s in ax.spines.values()]

    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, colors=c_text)
    ax.grid(axis='both', linestyle=':', alpha=0.4, color='#BDC3C7', zorder=0)

    # 4. Легенда (Размещена на пустом месте между 4 и 0.5 по Y)
    legend_elements = [
        mlines.Line2D([], [], color=c_agg_base, marker='o', markersize=8, markeredgecolor='none', lw=2.5,
                      label='Все нейроны'),
        mlines.Line2D([], [], color=c_rand, marker='s', markersize=6, linestyle='--', markeredgecolor='none', lw=1.5,
                      label='Случайные участвующие'),
        mlines.Line2D([], [], color=c_line_imp, marker='^', markerfacecolor=c_line_imp, markeredgecolor='none',
                      markersize=10, lw=2.5, label='Соответствующие 1 дню')
    ]

    # bbox_to_anchor привязывает легенду к координатам осей (x=0.98, y=0.35)
    # Это позиционирует её справа, но приподнимает над самыми нижними точками 4-го дня
    ax.legend(handles=legend_elements, bbox_to_anchor=(0.96, 0.25), loc='center right',
              frameon=False, fontsize=11, labelcolor=c_text)

    plt.tight_layout()

    save_path = output_path / 'longitudinal_top_neurons_absolute_with_baseline_clean.png'
    plt.savefig(save_path, dpi=1000, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ График абсолютного дрейфа сохранен: {save_path}")


def plot_ablation_broken_axis_smart_text(model_sessions, sigma_threshold=3,
                                         csv_path='stat_tables/full_ablation_stats.csv',
                                         output_dir='plots'):
    """
    Строит график дисперсии R2 при лезировании нейронов (Broken Axis).
    Все элементы (шрифты, точки, линии) увеличены примерно в 3 раза.
    Квадратный лейаут для одной сессии. R2 вынесено за пределы,
    а на верхней оси оставлены только значения исходных моделей.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[ERROR] Файл не найден: {csv_path}")
        return

    df = df[df['model_session'].isin(model_sessions)].copy()
    if df.empty:
        print("[WARNING] Данные для указанных моделей не найдены.")
        return

    x_order = df['model_session'].unique().tolist()
    model_to_x = {model: i for i, model in enumerate(x_order)}

    # Преобразование данных
    id_vars = ['model_session', 'baseline_r2', 'mean_r2', 'std_r2']
    abl_cols = [col for col in df.columns if col.startswith('ablation_neuron_')]

    df_long = pd.melt(df, id_vars=id_vars, value_vars=abl_cols, var_name='neuron_col', value_name='r2')
    df_long = df_long.dropna(subset=['r2']).copy()

    df_long['neuron_id'] = df_long['neuron_col'].apply(
        lambda x: re.search(r'\d+', x).group() if re.search(r'\d+', x) else x
    )

    df_base = df[['model_session', 'baseline_r2']].drop_duplicates()

    base_min, base_max = df_base['baseline_r2'].min(), df_base['baseline_r2'].max()
    abl_min, abl_max = df_long['r2'].min(), df_long['r2'].max()

    gap_exists = (base_min - abl_max) > 0.01

    # --- НАСТРОЙКА КВАДРАТНОГО ЛЕЙАУТА И УВЕЛИЧЕННЫХ РАЗМЕРОВ ---
    # Для 1 сессии будет квадрат 12x12
    fig_width = 12 if len(x_order) == 1 else max(12, len(x_order) * 8)
    fig_height = 12

    c_text = '#2C3E50'

    if gap_exists:
        fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1, sharex=True,
            figsize=(fig_width, fig_height),
            gridspec_kw={'height_ratios': [1, 4]}
        )
        fig.subplots_adjust(hspace=0.05)
        axes = [ax_top, ax_bottom]

        ax_top.spines['bottom'].set_visible(False)
        ax_bottom.spines['top'].set_visible(False)
        ax_top.tick_params(labeltop=False, bottom=False)
        ax_bottom.xaxis.tick_bottom()
    else:
        fig, ax_bottom = plt.subplots(figsize=(fig_width, fig_height))
        axes = [ax_bottom]
        ax_top = ax_bottom

    np.random.seed(42)
    texts_to_adjust = []
    x_outliers = []  # НОВОЕ
    y_outliers = []  # НОВОЕ

    for model in x_order:
        subset = df_long[df_long['model_session'] == model].copy()
        x_base = model_to_x[model]

        # Растягиваем точки шире
        jitters = np.random.uniform(-0.45, 0.45, size=len(subset))
        subset['x_jit'] = x_base + jitters

        # Увеличенные черные точки (s=150 вместо 20)
        ax_bottom.scatter(subset['x_jit'], subset['r2'], color='black', alpha=0.5, s=600, edgecolors='none', zorder=3)

        # Увеличенная красная точка (s=600 вместо 90)
        base_val = df_base[df_base['model_session'] == model]['baseline_r2'].values[0]
        ax_top.scatter(x_base, base_val, color='#D62728', s=1000, edgecolors='none', zorder=10)

        mean_r2 = subset['mean_r2'].iloc[0]
        std_r2 = subset['std_r2'].iloc[0]
        threshold = mean_r2 - (sigma_threshold * std_r2)

        outliers = subset[subset['r2'] < threshold]

        for _, row in outliers.iterrows():
            x_outliers.append(row['x_jit'])  # НОВОЕ
            y_outliers.append(row['r2'])  # НОВОЕ

            # ОГРОМНЫЙ ШРИФТ ДЛЯ ВЫБРОСОВ (fontsize=36)
            txt = ax_bottom.text(
                row['x_jit'], row['r2'],
                f" {row['neuron_id']} ",
                fontsize=36,
                fontweight='bold',
                color=c_text,
                ha='center', va='center',
                zorder=5
            )
            texts_to_adjust.append(txt)

    if gap_exists:
        margin_base = (base_max - base_min) * 0.5 if base_max != base_min else 0.02
        margin_abl = (abl_max - abl_min) * 0.1

        ax_top.set_ylim(base_min - margin_base, base_max + margin_base)
        ax_bottom.set_ylim(abl_min - margin_abl, abl_max + margin_abl)

        # ТОЛЬКО значения исходной модели на верхней оси
        base_vals = df_base['baseline_r2'].unique()
        ax_top.set_yticks(base_vals)

        # Увеличенные засечки разрыва (lw=3.0)
        d = .015
        kwargs = dict(transform=ax_top.transAxes, color=c_text, clip_on=False, lw=3.0)
        ax_top.plot((-d, +d), (-d, +d), **kwargs)
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs.update(transform=ax_bottom.transAxes)
        ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # Разводка текста
    if adjust_text and texts_to_adjust:
        adjust_text(
            texts_to_adjust,
            x=x_outliers,
            y=y_outliers,
            ax=ax_bottom,
            expand_points=(1.8, 1.8),  # Создает невидимый радиус вокруг точки
            expand_text=(1.2, 1.2),  # Буфер между самими подписями
            force_points=(1.5, 1.5),  # Сила отталкивания текста от точек
            force_text=(1.0, 1.0),  # Сила отталкивания текста от текста
            arrowprops=dict(
                arrowstyle="-",
                color='gray',
                lw=2.0,
                alpha=0.6,
                shrinkA=15,  # Отступ стрелки от центра гигантской точки
                shrinkB=5  # Отступ стрелки от текста
            )
        )

    # --- ОФОРМЛЕНИЕ МИНИМАЛИЗМА (Толстые линии и гигантские шрифты) ---
    try:
        formatted_labels = [format_session_label(m) for m in x_order]
    except NameError:
        formatted_labels = x_order

    ax_bottom.set_xticks(range(len(x_order)))
    ax_bottom.set_xticklabels(formatted_labels, rotation=0, ha='center', fontsize=36, color=c_text)

    for ax in axes:
        ax.grid(axis='y', linestyle=':', alpha=0.4, color='#BDC3C7', zorder=0, linewidth=2.0)
        ax.tick_params(axis='y', labelsize=28, colors=c_text, width=3.0, length=8)
        ax.tick_params(axis='x', width=3.0, length=8)
        ax.spines['left'].set_linewidth(3.0)
        ax.spines['right'].set_visible(False)
        if ax == ax_top:
            ax.spines['top'].set_visible(False)
        if ax == ax_bottom:
            ax.spines['bottom'].set_linewidth(3.0)
        [s.set_color(c_text) for s in ax.spines.values()]

    # --- ПОДПИСЬ Y (ВЫНЕСЕНА ДАЛЕКО ВЛЕВО И УВЕЛИЧЕНА) ---
    fig.supylabel(r'$R^2$', fontsize=48, x=-0.05, color=c_text, fontweight='bold')

    # --- КАСТОМНАЯ ЛЕГЕНДА (Увеличенная) ---
    legend_elements = [
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#D62728', markersize=24,
                      label='Исходная модель'),
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', alpha=0.5, markersize=18,
                      label='Удален один нейрон'),
        mlines.Line2D([0], [0], marker='$\mathbb{N}$', color='w', markerfacecolor=c_text, markeredgecolor=c_text,
                      markersize=26,
                      label=f'Отклонение > {sigma_threshold}$\\sigma$')
    ]

    ax_top.legend(handles=legend_elements, loc='lower right', frameon=False, fontsize=24, labelcolor=c_text)

    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.1)

    safe_names = [re.sub(r'\W+', '', m) for m in model_sessions[:3]]
    prefix = "_".join(safe_names)
    if len(model_sessions) > 3:
        prefix += "_and_others"

    save_path = output_path / f'ablation_{prefix}_sigma{sigma_threshold}_HUGE.png'

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Масштабированный квадратный график успешно сохранен: {save_path}")


def plot_delta_vs_mi_dependencies_dual(ablation_csv='stat_tables/full_ablation_stats.csv',
                                       database_csv='LNOF_SynchronizedData/LNOF_database.csv',
                                       models_dir='models_LNOF_Lasso',
                                       output_dir='plots'):
    """
    Строит два графика в логарифмических координатах (log2):
    1. (Слева) Зависимость веса (beta) от MI. Тау Кендалла. Фиолетовый.
    2. (Справа) Зависимость важности (z-delta) от MI. Спирмен. Зеленый.

    Дизайн для презентаций: Огромные шрифты, уменьшенные точки, строгие греческие символы,
    вертикальный порог MI=0.04, полностью унифицированная размерная сетка осей.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        df_abl = pd.read_csv(ablation_csv)
        df_db = pd.read_csv(database_csv)
    except FileNotFoundError as e:
        print(f"[ERROR] Не найден файл: {e}")
        return

    print(f"\n{'=' * 70}")
    print("Сборка пула нейронов для графиков \u03B2 и \u03B4 от MI (log2 шкала)...")

    mi_all, zcrit_all, beta_all = [], [], []
    feat_cols = [col for col in df_abl.columns if col.startswith('ablation_')]
    feature_names = [col.replace('ablation_', '') for col in feat_cols]

    for _, row in df_abl.iterrows():
        model_name = row['model_session']
        mouse, day = model_name[:3], model_name[4:6]

        # 1. Загрузка весов из модели
        model_path = Path(models_dir) / f"{model_name}.pkl"
        if not model_path.exists():
            continue

        m_data = joblib.load(model_path)
        model = m_data.get('model')
        if not hasattr(model, 'coef_'):
            continue

        coefs = np.array(model.coef_)
        weights = np.linalg.norm(coefs, axis=0) if coefs.ndim > 1 else np.abs(coefs)
        feat_names_model = np.array(m_data.get('feature_names'))
        weight_dict = dict(zip(feat_names_model, weights))

        # 2. Вычисление z-delta (z-crit)
        base_r2 = row['baseline_r2']
        abl_vals = row[feat_cols].values.astype(float)
        crit = base_r2 - abl_vals

        valid_crit = crit[~np.isnan(crit)]
        if len(valid_crit) > 0 and np.std(valid_crit) > 0:
            zcrit = (crit - np.mean(valid_crit)) / np.std(valid_crit)
        else:
            zcrit = np.zeros_like(crit)

        # 3. Извлечение MI (me)
        df_intense = df_db[(df_db['mouse'] == mouse) & (df_db['session'] == day) & (df_db['feature'] == 'place')].copy()
        df_intense = df_intense.sort_values(by='me', ascending=False).drop_duplicates(subset=['neuron_idx'])
        me_dict = {f"neuron_{int(r['neuron_idx'])}": r['me'] for _, r in df_intense.iterrows()}

        # Сопоставление
        mi = np.array([me_dict.get(f, 0.0) for f in feature_names])
        betas = np.array([weight_dict.get(f, 0.0) for f in feature_names])

        mi_all.extend(mi)
        zcrit_all.extend(zcrit)
        beta_all.extend(betas)

    df_plot = pd.DataFrame({'mi': mi_all, 'zcrit': zcrit_all, 'beta': beta_all}).dropna()

    # Подготовка данных (теперь правый график не обрезается по 0.04)
    df_left = df_plot[(df_plot['mi'] > 0) & (df_plot['beta'] > 0)].copy()
    df_right = df_plot[(df_plot['mi'] > 0) & (df_plot['zcrit'] > 0)].copy()

    print(f"Пул для \u03B2 vs MI (Слева): {len(df_left)} нейронов")
    print(f"Пул для \u03B4 vs MI (Справа): {len(df_right)} нейронов")
    print(f"{'=' * 70}")

    corr_beta, p_beta = kendalltau(df_left['mi'], df_left['beta'])
    corr_zcrit, p_zcrit = spearmanr(df_right['mi'], df_right['zcrit'])

    # Математически ровные звездочки
    def get_stars(p):
        return r"{\ast}{\ast}{\ast}" if p < 0.001 else r"{\ast}{\ast}" if p < 0.01 else r"{\ast}" if p < 0.05 else "^{ns}"

    # Доверительный интервал в лог-пространстве
    def calc_trend_ci(x, y):
        log2_x = np.log2(x)
        log2_y = np.log2(y)
        m, b = np.polyfit(log2_x, log2_y, 1)

        x_trend_log = np.linspace(log2_x.min(), log2_x.max(), 100)
        y_trend_log = m * x_trend_log + b

        n = len(log2_x)
        y_hat = m * log2_x + b
        s_err = np.sqrt(np.sum((log2_y - y_hat) ** 2) / (n - 2))
        t_val = t.ppf(0.975, n - 2)
        mean_x = np.mean(log2_x)
        ssx = np.sum((log2_x - mean_x) ** 2)

        ci = t_val * s_err * np.sqrt(1 / n + (x_trend_log - mean_x) ** 2 / ssx)

        return 2 ** x_trend_log, 2 ** y_trend_log, 2 ** (y_trend_log - ci), 2 ** (y_trend_log + ci)

    # ==========================================
    # ЭСТЕТИЧЕСКАЯ ОТРИСОВКА (ПРЕЗЕНТАЦИЯ)
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(24, 11))
    fig.patch.set_facecolor('white')
    ax1, ax2 = axes

    c_purple = '#8E44AD'
    c_green = '#27AE60'
    c_text = '#2C3E50'

    font_label = 42
    font_tick = 33
    font_stat = 40
    lw_axes = 3.6
    lw_trend = 6.0

    # Уменьшенный размер точек для снижения визуального шума
    scatter_size = 120

    # --- ЛЕВЫЙ ГРАФИК (ВЕСА vs MI) ---
    x_tr1, y_tr1, y_low1, y_up1 = calc_trend_ci(df_left['mi'], df_left['beta'])

    ax1.scatter(df_left['mi'], df_left['beta'], alpha=0.5, s=scatter_size, color=c_purple, edgecolors='none', zorder=3)
    ax1.plot(x_tr1, y_tr1, color='grey', linestyle='--', linewidth=lw_trend, alpha=0.6, zorder=4)
    ax1.fill_between(x_tr1, y_low1, y_up1, color='grey', alpha=0.35, zorder=2)

    stat_text1 = f"$\\tau = {corr_beta:.2f}^{{{get_stars(p_beta)}}}$"
    ax1.text(0.05, 0.95, stat_text1, transform=ax1.transAxes, ha='left', va='top',
             fontsize=font_stat, color=c_text, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=10), zorder=5)

    # --- ПРАВЫЙ ГРАФИК (ВАЖНОСТЬ vs MI) ---
    x_tr2, y_tr2, y_low2, y_up2 = calc_trend_ci(df_right['mi'], df_right['zcrit'])

    ax2.scatter(df_right['mi'], df_right['zcrit'], alpha=0.5, s=scatter_size, color=c_green, edgecolors='none',
                zorder=3)
    ax2.plot(x_tr2, y_tr2, color='grey', linestyle='--', linewidth=lw_trend, alpha=0.6, zorder=4)
    ax2.fill_between(x_tr2, y_low2, y_up2, color='grey', alpha=0.35, zorder=2)

    stat_text2 = f"$\\rho = {corr_zcrit:.2f}^{{{get_stars(p_zcrit)}}}$"
    ax2.text(0.05, 0.95, stat_text2, transform=ax2.transAxes, ha='left', va='top',
             fontsize=font_stat, color=c_text, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=10), zorder=5)

    # --- ОБЩЕЕ ВЫРАВНИВАНИЕ ОСЕЙ И СЕТОК ---
    # Синхронизируем лимиты осей (чтобы сетки 1 в 1 совпадали по пропорциям)
    global_x_min = min(df_left['mi'].min(), df_right['mi'].min()) * 0.8
    global_x_max = max(df_left['mi'].max(), df_right['mi'].max()) * 1.2

    # Y оси у нас имеют разную физическую природу (веса vs z-score),
    # но мы приведем их к одинаковым лимитам, чтобы шаг сетки был визуально идентичен
    global_y_min = min(df_left['beta'].min(), df_right['zcrit'].min()) * 0.8
    global_y_max = max(df_left['beta'].max(), df_right['zcrit'].max()) * 1.2

    for ax, ylabel in zip([ax1, ax2], [r'Вес ($\beta$)', r'Важность ($\delta$)']):
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=2)

        # Жесткая синхронизация лимитов
        ax.set_xlim(global_x_min, global_x_max)
        ax.set_ylim(global_y_min, global_y_max)

        # Жестко задаем 6 делений: 2^-7, 2^-6, 2^-5, 2^-4, 2^-3, 2^-2 (0.25)
        ax.set_xticks([2 ** i for i in range(-7, -1)])

        # Обновляем лимиты оси X, чтобы график не обрезал крайние точки
        # ax.set_xlim(2 ** -7, 0.25)
        ax.yaxis.set_major_locator(LogLocator(base=2))

        def tick_formatter(val, pos):
            if val < 0.125 or val > 8:
                return f"$2^{{{int(np.round(np.log2(val)))}}}$"
            return f"{val:g}"

        custom_formatter = FuncFormatter(tick_formatter)
        ax.xaxis.set_major_formatter(custom_formatter)
        ax.yaxis.set_major_formatter(custom_formatter)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())

        ax.set_xlabel('MI', fontsize=font_label, labelpad=20, color=c_text)
        ax.set_ylabel(ylabel, fontsize=font_label, labelpad=20, color=c_text)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(lw_axes)
        ax.spines['bottom'].set_linewidth(lw_axes)
        [s.set_color(c_text) for s in ax.spines.values()]

        ax.tick_params(axis='both', which='major', labelsize=font_tick, width=lw_axes, length=12, colors=c_text)
        ax.grid(axis='both', linestyle=':', alpha=0.4, color='#BDC3C7', zorder=0, linewidth=2)
        ax.set_facecolor('white')

        # Добавляем вертикальный пунктирный порог MI = 0.04
        ax.axvline(0.04, color='grey', linestyle='--', linewidth=5, alpha=0.6, zorder=1)

    plt.tight_layout(w_pad=6.0)

    save_path = output_path / 'weights_and_zdelta_vs_mi_huge.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Гигантские графики успешно сохранены: {save_path}")


if __name__ == "__main__":
    # plot_metrics_sigma_distribution()
    models_dir = "models_LNOF_Lasso"
    models_path = Path(models_dir)
    models = [f.name[:6] for f in models_path.glob("*.pkl")]
    models = ['J24_3D.pkl', 'J54_4D.pkl', 'J18_2D.pkl', 'J59_4D.pkl', 'J52_2D.pkl']
    models = [model[:-4] for model in models]
    # plot_retrain_ablation_distribution(models)
    # plot_selective_neuron_graphs([models[1]], selectivity='linreg', corr_threshold=0.25)

    # plot_retrain_ablation_distribution([models[0]])
    # plot_ablation_broken_axis_smart_text(models[:3], sigma_threshold=3)

    # plot_global_euler_diagram(color_scheme='rgb')
    # plot_r2_dependencies()
    # plot_neuron_composition_stacked_barplot(models)
    # plot_delta_vs_mi_dependencies()
    # plot_heavy_tailed_distributions(models)
    # plot_cumulative_retrain_ablation_addition(models[0], metric='test_r2', step_size=1)
    # plot_cumulative_retrain_ablation_addition(models[0], metric='test_dist', step_size=1)
    # plot_global_euler_diagram_4d_area(models[0][:3], color_scheme='cmyk')
    # plot_r2_and_top_neurons_dependencies()
    # plot_longitudinal_survival_aesthetic_selector(feature_type='z-delta')
    # plot_top_neurons_longitudinal_counts()
    # plot_r2_and_top_neurons_dependencies_exact_colors()
    # plot_r2_and_network_size_dependencies()
    # plot_cross_session_r2_dynamics_aesthetic()
    # plot_mobility_longitudinal_dependencies()
    # plot_top_neurons_longitudinal_counts_bidirectional()
    # plot_weight_survival_dependency()
    # plot_combined_longitudinal_survival()
    #
    # plot_combined_r2_left_panels_with_baseline()
    # plot_top_neurons_survival_absolute_with_baseline()
    # plot_comparability_threshold_dependency()

    plot_combined_r2_single_panel_with_baseline()
    # plot_global_euler_diagram_present(color_scheme='rgb')
    # plot_cumulative_retrain_addition_dual(models[0], step_size=1)
    # plot_top_neurons_survival_absolute_with_baseline_present()
    # plot_ablation_broken_axis_smart_text([models[0]])
    # plot_delta_vs_mi_dependencies_dual()
    # plot_combined_longitudinal_survival()