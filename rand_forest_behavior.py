import os
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from linreg_behavior import load_data, calculate_metrics
import sys
from sklearn.model_selection import ParameterGrid


# ==========================================
# 2. ФУНКЦИЯ ОБУЧЕНИЯ RANDOM FOREST
# ==========================================
def train_multioutput_rf(X_train, y_train, X_test, y_test,
                         n_estimators=100, max_depth=20, min_samples_split=5, random_state=42):
    """
    Обучает Multi-output Random Forest Regressor.
    """
    start_time = time.time()
    print("  > Импьютинг и масштабирование данных (StandardScaler)...")

    # 1. Preprocessing
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    # 2. Инициализация и обучение RF
    print(f"  > Обучение Random Forest ({n_estimators} деревьев, max_depth={max_depth})...")
    print("  > Это может занять несколько минут. Задействованы все ядра CPU (n_jobs=-1)...")

    clf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1  # Использовать все доступные потоки процессора
    )

    clf.fit(X_train_scaled, y_train)

    # 3. Инференс
    print("  > Оценка качества (Инференс)...")
    y_train_pred = clf.predict(X_train_scaled)
    y_test_pred = clf.predict(X_test_scaled)

    # 4. Расчет метрик
    train_dist, train_med_dist, train_r2 = calculate_metrics(y_train, y_train_pred)
    test_dist, test_med_dist, test_r2 = calculate_metrics(y_test, y_test_pred)

    elapsed_time = time.time() - start_time
    mins, secs = divmod(int(elapsed_time), 60)

    print(f"\n📊 РЕЗУЛЬТАТЫ RANDOM FOREST (Затрачено: {mins}м {secs}с):")
    print(f"  [Train] MED: {train_med_dist:.2f} | R²: {train_r2:.4f}")
    print(f"  [Test]  MED: {test_med_dist:.2f}  | R²: {test_r2:.4f}")

    # 5. Извлечение важности признаков (Feature Importances)
    # RF выдает единый массив важности признаков (сумма по обеим осям)
    importances = clf.feature_importances_
    n_nonzero = np.sum(importances > 0.001)  # Условный порог значимого влияния
    print(f"  [Info] Активных нейронов (importance > 0.001): {n_nonzero} из {X_train.shape[1]}")

    metrics = {
        'train_dist': train_dist,
        'train_med_dist': train_med_dist,
        'train_r2': train_r2,
        'test_dist': test_dist,
        'test_med_dist': test_med_dist,
        'test_r2': test_r2,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_active_features': int(n_nonzero)
    }

    return clf, imputer, scaler, metrics, importances


def main():
    print(f"Старт пайплайна Multi-output Random Forest.\nМодели будут сохранены в: {MODELS_DIR}")
    os.makedirs(MODELS_DIR, exist_ok=True)

    for session in TARGET_SESSIONS:
        print(f"\n{'=' * 60}")
        print(f"ОБРАБОТКА СЕССИИ: {session}")
        print(f"{'=' * 60}")

        # 1. Загрузка данных
        file_path = Path(DATA_DIR) / f"LNOF_{session}_aligned.npz"
        X, y = load_data(file_path)

        if X is None or y is None:
            continue

        # 2. Хронологический сплит (75% Train, 25% Test)
        split_idx = int(len(X) * 0.75)

        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]

        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]

        print(f"  > Сплит завершен. Train: {len(X_train)} | Test: {len(X_test)}")

        # 3. Запуск обучения RF
        # Настройки: 200 деревьев - хороший баланс между точностью и скоростью.
        # min_samples_split=10 защищает деревья от излишнего переобучения на шуме.
        model, imp, scl, metrics, feat_importances = train_multioutput_rf(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_estimators=350,
            max_depth=40,
            min_samples_split=10,
            random_state=42
        )

        # 4. Сохранение модели (по аналогии с Lasso)
        save_filename = f"{session}_rf.pkl"
        save_path = Path(MODELS_DIR) / save_filename

        model_data = {
            'model': model,
            'imputer': imp,
            'scaler': scl,
            'feature_names': X.columns.tolist(),
            'feature_importances': feat_importances,  # Сохраняем важность для графиков!
            'metrics': metrics
        }

        joblib.dump(model_data, save_path)
        print(f"✓ Модель RF сохранена в: {save_path}")


def prepare_sessions_data(sessions, data_dir):
    """
    Загружает, делает хронологический сплит и скалирует данные ЗАРАНЕЕ,
    чтобы не делать это тысячи раз в цикле Grid Search.
    """
    prepared_data = {}
    print("\n" + "=" * 60)
    print("ЭТАП 1: ПОДГОТОВКА И ПРЕПРОЦЕССИНГ СЕССИЙ")
    print("=" * 60)

    for session in sessions:
        file_path = Path(data_dir) / f"LNOF_{session}_aligned.npz"
        X, y = load_data(file_path)

        if X is None or y is None:
            print(f"  [ПРОПУСК] Сессия {session} не загружена.")
            continue

        # Хронологический сплит 75/25
        split_idx = int(len(X) * 0.75)
        X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
        X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]

        # Импьютинг и скейлинг
        imputer = SimpleImputer(strategy='median')
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)

        # Сохраняем в память готовые numpy массивы
        prepared_data[session] = {
            'X_train': X_train_scaled, 'y_train': y_train.values,
            'X_test': X_test_scaled, 'y_test': y_test.values
        }
        print(f"  ✓ {session}: Подготовлено (Train: {len(X_train)} | Test: {len(X_test)})")

    return prepared_data


# ==========================================
# 3. ОСНОВНОЙ ЦИКЛ GRID SEARCH
# ==========================================
def run_custom_grid_search(sessions_data, output_csv='rf_grid_search_results.csv'):
    # Задаем сетку параметров согласно требованиям
    param_grid = {
        'n_estimators': list(range(50, 501, 50)),  # 50, 100, ..., 500
        'max_depth': list(range(5, 51, 5)),  # 5, 10, ..., 50
        'min_samples_split': list(range(5, 26, 5))  # 2, 3, ..., 25
    }

    grid = list(ParameterGrid(param_grid))
    total_combinations = len(grid)

    print("\n" + "=" * 80)
    print(f"ЭТАП 2: GRID SEARCH (Всего комбинаций: {total_combinations})")
    print(f"Сессий в оценке: {len(sessions_data)}")
    print("=" * 80)

    results_list = []
    global_start = time.time()

    # Символы для анимации
    spinner = ['|', '/', '-', '\\']

    for i, params in enumerate(grid):
        comb_start = time.time()

        # Метрики для данной комбинации параметров по всем сессиям
        comb_test_r2 = []
        comb_train_r2 = []

        # Прогоняем комбинацию через все подготовленные сессии
        for session_name, data in sessions_data.items():
            clf = RandomForestRegressor(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                random_state=42,
                n_jobs=-1  # ОЧЕНЬ ВАЖНО: используем все ядра процессора
            )

            # Обучение
            clf.fit(data['X_train'], data['y_train'])

            # Предсказание
            y_train_pred = clf.predict(data['X_train'])
            y_test_pred = clf.predict(data['X_test'])

            # Считаем R2 (с защитой от сбоев)
            try:
                r2_train = r2_score(data['y_train'], y_train_pred)
                r2_test = r2_score(data['y_test'], y_test_pred)
            except:
                r2_train, r2_test = 0.0, 0.0

            comb_train_r2.append(r2_train)
            comb_test_r2.append(r2_test)

        # Вычисляем агрегированные метрики для этой комбинации
        mean_test_r2 = np.mean(comb_test_r2)
        mean_train_r2 = np.mean(comb_train_r2)
        std_test_r2 = np.std(comb_test_r2)  # Показывает стабильность между сессиями

        # Сохраняем результат
        res_dict = {
            'n_estimators': params['n_estimators'],
            'max_depth': params['max_depth'],
            'min_samples_split': params['min_samples_split'],
            'mean_test_r2': mean_test_r2,
            'std_test_r2': std_test_r2,
            'mean_train_r2': mean_train_r2
        }
        results_list.append(res_dict)

        # Автосохранение в CSV после каждого шага (чтобы не потерять данные при сбое)
        df_results = pd.DataFrame(results_list)
        df_results.to_csv(output_csv, index=False)

        # Вербализация и прогресс
        elapsed = time.time() - comb_start
        total_elapsed = time.time() - global_start
        mins, secs = divmod(int(total_elapsed), 60)

        spin_char = spinner[i % 4]
        progress = (i + 1) / total_combinations
        bar_len = 20
        filled = int(bar_len * progress)
        bar = '=' * filled + '-' * (bar_len - filled)

        sys.stdout.write(
            f"\r{spin_char} Прогресс: [{bar}] {i + 1}/{total_combinations} | "
            f"Текущий R²: {mean_test_r2:.4f} | Лучший R²: {df_results['mean_test_r2'].max():.4f} | "
            f"Время: {mins}м {secs}с"
        )
        sys.stdout.flush()

    print("\n\n" + "=" * 80)
    print("🏁 GRID SEARCH ПОЛНОСТЬЮ ЗАВЕРШЕН!")

    # Вывод Топ-5 комбинаций
    print("\nТоп-5 лучших конфигураций (сортировка по Mean Test R²):")
    top_5 = df_results.sort_values(by='mean_test_r2', ascending=False).head(5)
    print(top_5.to_string(index=False))

    print(f"\nПолный отчет сохранен в файл: {output_csv}")
    return top_5.iloc[0]  # Возвращаем самую лучшую строку параметров


DATA_DIR = 'LNOF_SynchronizedData'
MODELS_DIR = 'models_LNOF_RF'

# models = ['J24_3D.pkl', 'J54_4D.pkl', 'J18_2D.pkl', 'J59_4D.pkl', 'J52_2D.pkl']
# models = [model[:-4] for model in models]

models_dir = "models_LNOF_Lasso"
models_path = Path(models_dir)
models = [f.name[:6] for f in models_path.glob("*.pkl")]

# Список сессий для обработки
TARGET_SESSIONS = models #['J14_3D', 'J19_4D', 'J30_3D', 'J61_1D', 'J14_1D']


if __name__ == "__main__":
    main()
    # DATA_DIR = 'LNOF_SynchronizedData'
    #
    # # 1. Заранее готовим данные (чтобы сэкономить время в цикле)
    # prepared_data = prepare_sessions_data(TARGET_SESSIONS, DATA_DIR)
    #
    # if prepared_data:
    #     # 2. Запускаем поиск
    #     best_params = run_custom_grid_search(prepared_data, output_csv='rf_grid_search_results.csv')
    #
    #     print("\n🏆 ИДЕАЛЬНЫЕ ПАРАМЕТРЫ ДЛЯ ОБУЧЕНИЯ:")
    #     print(f"n_estimators:      {int(best_params['n_estimators'])}")
    #     print(f"max_depth:         {int(best_params['max_depth'])}")
    #     print(f"min_samples_split: {int(best_params['min_samples_split'])}")
    # else:
    #     print("[ОШИБКА] Не удалось загрузить ни одну сессию. Проверь пути.")