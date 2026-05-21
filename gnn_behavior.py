import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.utils import dense_to_sparse

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score


# ==========================================
# 1. АРХИТЕКТУРА ST-GNN (Spatial-Temporal GNN)
# ==========================================
class SpatioTemporalGNN(nn.Module):
    def __init__(self, n_neurons, seq_len, corr_matrix_init=None, hidden_dim=64, output_dim=2, dropout_rate=0.3):
        super(SpatioTemporalGNN, self).__init__()

        self.n_neurons = n_neurons

        # 1. Temporal Component (LSTM)
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)

        # 2. Spatial Component (GCN) - ОБУЧАЕМАЯ МАТРИЦА СМЕЖНОСТИ
        if corr_matrix_init is not None:
            self.spatial_adj = nn.Parameter(torch.tensor(corr_matrix_init, dtype=torch.float32))
        else:
            self.spatial_adj = nn.Parameter(torch.randn(n_neurons, n_neurons) * 0.01)

        self.gcn_weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.gcn_weight)

        # 3. Readout Component
        self.fc1 = nn.Linear(n_neurons * hidden_dim, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)
        self.act = nn.GELU()  # Индустриальный стандарт вместо Tanh/ReLU
        self.dropout = nn.Dropout(p=dropout_rate)

        # Финальный предиктор БЕЗ функции активации, т.к. предсказываем координаты (Z-score)
        self.fc2 = nn.Linear(n_neurons, output_dim)

    def forward(self, x):
        # x shape: (Batch, seq_len, n_neurons)
        B, T, N = x.shape

        # Перестраиваем для LSTM
        x = x.permute(0, 2, 1).contiguous().view(B * N, T, 1)

        # LSTM шаг
        lstm_out, (hn, cn) = self.lstm(x)
        node_embeddings = hn[-1].view(B, N, -1)  # Shape: [Batch, N, hidden_dim]

        # GCN шаг
        adj = torch.tanh(self.spatial_adj)  # Зажимаем силу связей графа в [-1, 1]

        # Идеальное умножение: i - target узел, j - source узел, b - батч, h - фича
        gcn_out = torch.einsum('ij, bjh -> bih', adj, node_embeddings)

        gcn_out = torch.matmul(gcn_out, self.gcn_weight)
        gcn_out = torch.tanh(gcn_out)

        # Readout
        graph_embedding = gcn_out.view(B, -1)

        out = self.fc1(graph_embedding)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


# ==========================================
# 2. ФУНКЦИИ ПОДГОТОВКИ ГРАФА И ДАННЫХ
# ==========================================
def build_adaptive_correlation_graph(calcium_data, target_degree=3, min_threshold=0.15):
    """
    Строит адаптивный граф функциональной коннективности.
    """
    print(f"  > Расчет адаптивной матрицы коннективности (Target Degree: {target_degree})...")

    df_calcium = pd.DataFrame(calcium_data)
    corr_mtx = df_calcium.corr(method='spearman').values
    np.fill_diagonal(corr_mtx, 0)
    abs_corr_mtx = np.abs(corr_mtx)

    N = abs_corr_mtx.shape[0]
    total_possible_edges = (N * (N - 1)) // 2
    desired_edges = (N * target_degree) // 2
    desired_edges = min(desired_edges, total_possible_edges)

    if desired_edges > 0:
        upper_triangle_vals = abs_corr_mtx[np.triu_indices(N, k=1)]
        sorted_corrs = np.sort(upper_triangle_vals)
        threshold_idx = len(sorted_corrs) - desired_edges
        adaptive_threshold = sorted_corrs[threshold_idx]
    else:
        adaptive_threshold = min_threshold

    final_threshold = max(adaptive_threshold, min_threshold)
    adj_matrix = np.where(abs_corr_mtx >= final_threshold, corr_mtx, 0)
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32)

    edge_index, edge_weight = dense_to_sparse(adj_tensor)
    actual_edges = edge_index.shape[1] // 2

    print(f"    - Нейронов: {N}")
    print(f"    - Адаптивный порог: {final_threshold:.3f}")
    print(f"    - Итоговых ребер: {actual_edges} (в ср. {actual_edges * 2 / N:.1f} на нейрон)")

    # ВОЗВРАЩАЕМ САМУ МАТРИЦУ И ПОРОГ ДЛЯ СОХРАНЕНИЯ В .PT
    return edge_index, edge_weight, final_threshold, corr_mtx


def create_temporal_windows(X, y, seq_len=64, overlap=32):
    stride = seq_len - overlap
    if stride <= 0: raise ValueError("[ОШИБКА] Overlap не может быть >= seq_len")

    X_windows, y_windows = [], []
    for i in range(0, len(X) - seq_len, stride):
        X_windows.append(X[i:i + seq_len])
        y_windows.append(y[i + seq_len])
    return np.array(X_windows), np.array(y_windows)


def calculate_metrics(y_true, y_pred):
    dist = np.linalg.norm(y_true - y_pred, axis=1)
    return np.mean(dist), np.median(dist), r2_score(y_true, y_pred)


# ==========================================
# 3. ОСНОВНАЯ ФУНКЦИЯ ОБУЧЕНИЯ ПАЙПЛАЙНА
# ==========================================
def train_st_gnn(X_train_raw, y_train_raw, X_test_raw, y_test_raw,
                 seq_len=64, overlap=32, corr_threshold=0.15, epochs=50, batch_size=128, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[ИНФО] Используемое устройство: {device}")

    # 1. Preprocessing X
    print("  > Импьютинг и масштабирование данных (X и Y)...")
    imputer_x = SimpleImputer(strategy='median')
    X_train_imp = imputer_x.fit_transform(X_train_raw)
    X_test_imp = imputer_x.transform(X_test_raw)

    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train_imp)
    X_test_scaled = scaler_x.transform(X_test_imp)
    n_neurons = X_train_scaled.shape[1]

    # 2. Preprocessing Y
    imputer_y = SimpleImputer(strategy='median')
    y_train_imp = imputer_y.fit_transform(y_train_raw.values)
    y_test_imp = imputer_y.transform(y_test_raw.values)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_imp)
    y_test_scaled = scaler_y.transform(y_test_imp)

    # 3. Создание Графа (забираем матрицу и порог)
    edge_index, edge_weight, actual_threshold, corr_mtx = build_adaptive_correlation_graph(X_train_scaled,
                                                                                           min_threshold=corr_threshold)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    print(f"  > Граф построен. Активных связей: {edge_index.shape[1]}")

    # 4. Нарезка на окна
    print(f"  > Нарезка окон (seq_len={seq_len}, overlap={overlap})...")
    X_train_w, y_train_w = create_temporal_windows(X_train_scaled, y_train_scaled, seq_len=seq_len, overlap=overlap)
    X_test_w, y_test_w = create_temporal_windows(X_test_scaled, y_test_scaled, seq_len=seq_len, overlap=overlap)

    train_data = TensorDataset(torch.FloatTensor(X_train_w), torch.FloatTensor(y_train_w))
    test_data = TensorDataset(torch.FloatTensor(X_test_w), torch.FloatTensor(y_test_w))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # 5. Инициализация Модели
    model = SpatioTemporalGNN(n_neurons=n_neurons, seq_len=seq_len, corr_matrix_init=corr_mtx).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, threshold=1e-3,
                                                           patience=10)

    def get_batch_r2(y_t, y_p):
        try:
            return max(r2_score(y_t, y_p), -1.0)
        except:
            return 0.0

    def get_batch_med(y_t, y_p):
        try:
            return np.mean(np.linalg.norm(y_t - y_p, axis=1))
        except:
            return 0.0

    spinner = ['|', '/', '-', '\\']
    print("\n" + "=" * 80)
    print("🚀 СТАРТ ОБУЧЕНИЯ ST-GNN")
    print("=" * 80)

    global_start_time = time.time()
    best_val_loss = float('inf')
    early_stop_patience = 15
    epochs_no_improve = 0
    min_delta = 0.01

    # 6. Training Loop
    for epoch in range(epochs):
        # --- TRAIN ---
        model.train()
        epoch_loss, epoch_r2_sum, epoch_med_sum = 0, 0, 0
        epoch_start = time.time()

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            out = model(batch_x)  # Модели больше не нужны edge_index!
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

            y_t = batch_y.cpu().detach().numpy()
            y_p = out.cpu().detach().numpy()
            epoch_loss += loss.item()
            epoch_r2_sum += get_batch_r2(y_t, y_p)
            epoch_med_sum += get_batch_med(y_t, y_p)

            elapsed = time.time() - epoch_start
            mins, secs = divmod(int(elapsed), 60)
            progress = (batch_idx + 1) / len(train_loader)
            bar = '=' * int(15 * progress) + '-' * (15 - int(15 * progress))

            sys.stdout.write(
                f"\r{spinner[batch_idx % 4]} Эпоха [{epoch + 1:02d}/{epochs}] [{bar}] Train Loss: {loss.item():.2f} | {mins:02d}:{secs:02d}")
            sys.stdout.flush()

        n_batches = len(train_loader)
        avg_train_loss = epoch_loss / n_batches if n_batches > 0 else 0
        avg_train_r2 = epoch_r2_sum / n_batches if n_batches > 0 else 0
        avg_train_med = epoch_med_sum / n_batches if n_batches > 0 else 0

        # --- VALIDATION ---
        model.eval()
        val_loss, val_r2_sum, val_med_sum = 0, 0, 0
        with torch.no_grad():
            # СТРОГО val_x, чтобы исключить scope leak
            for val_x, val_y in test_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                out = model(val_x)

                v_loss = criterion(out, val_y)
                val_loss += v_loss.item()
                y_t = val_y.cpu().numpy()
                y_p = out.cpu().numpy()
                val_r2_sum += get_batch_r2(y_t, y_p)
                val_med_sum += get_batch_med(y_t, y_p)

        n_val_batches = len(test_loader)
        avg_val_loss = val_loss / n_val_batches if n_val_batches > 0 else 0
        avg_val_r2 = val_r2_sum / n_val_batches if n_val_batches > 0 else 0
        avg_val_med = val_med_sum / n_val_batches if n_val_batches > 0 else 0

        sys.stdout.write(
            f"\r✓ Эпоха [{epoch + 1:02d}/{epochs}] ЗАВЕРШЕНА\n"
            f"   [Train] Loss: {avg_train_loss:.2f} | R²: {avg_train_r2:+.2f} | MED: {avg_train_med:.2f}\n"
            f"   [Val]   Loss: {avg_val_loss:.2f} | R²: {avg_val_r2:+.2f} | MED: {avg_val_med:.2f}\n"
        )

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        if optimizer.param_groups[0]['lr'] < old_lr:
            print(f"   [INFO] Плато! Learning Rate снижен с {old_lr:.2e} до {optimizer.param_groups[0]['lr']:.2e}")

        if avg_val_loss < (best_val_loss - min_delta):
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            print(f"\n[INFO] Early Stopping! Validation Loss не улучшается {early_stop_patience} эпох подряд.")
            break

    total_time = time.time() - global_start_time
    total_mins, total_secs = divmod(int(total_time), 60)
    print("=" * 80)
    print(f"🏁 Обучение завершено за {total_mins} мин {total_secs} сек")

    # 7. Финальный пересчет метрик
    print("  > Финальная оценка качества (с обратным преобразованием координат)...")
    model.eval()

    # Для честной финальной метрики на трейне отключаем drop_last
    train_eval_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=False)
    with torch.no_grad():
        y_train_pred = []
        for tr_x, _ in train_eval_loader:
            out = model(tr_x.to(device))
            y_train_pred.append(out.cpu().numpy())
        y_train_pred = np.vstack(y_train_pred) if y_train_pred else np.array([])

        y_test_pred = []
        for te_x, _ in test_loader:
            out = model(te_x.to(device))
            y_test_pred.append(out.cpu().numpy())
        y_test_pred = np.vstack(y_test_pred) if y_test_pred else np.array([])

    y_train_w_orig = scaler_y.inverse_transform(y_train_w)
    y_train_pred_orig = scaler_y.inverse_transform(y_train_pred)
    y_test_w_orig = scaler_y.inverse_transform(y_test_w)
    y_test_pred_orig = scaler_y.inverse_transform(y_test_pred)

    train_dist, train_med_dist, train_r2 = calculate_metrics(y_train_w_orig, y_train_pred_orig)
    test_dist, test_med_dist, test_r2 = calculate_metrics(y_test_w_orig, y_test_pred_orig)

    print(f"\n📊 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ СЕССИИ:")
    print(f"  [Train] MED: {train_med_dist:.2f} px | R²: {train_r2:.4f}")
    print(f"  [Test]  MED: {test_med_dist:.2f} px  | R²: {test_r2:.4f}")

    metrics = {
        'train_dist': train_dist, 'train_med_dist': train_med_dist, 'train_r2': train_r2,
        'test_dist': test_dist, 'test_med_dist': test_med_dist, 'test_r2': test_r2,
        'n_train': len(X_train_w), 'n_test': len(X_test_w),
        'n_active_features': len(torch.unique(edge_index.flatten())),
        'adaptive_threshold': float(actual_threshold),
        'total_edges': edge_index.shape[1] // 2
    }

    model = model.cpu()
    edge_index = edge_index.cpu()
    edge_weight = edge_weight.cpu()

    return model, edge_index, edge_weight, imputer_x, scaler_x, scaler_y, metrics, corr_mtx


# ==========================================
# 4. ФУНКЦИЯ СОХРАНЕНИЯ (С МАТРИЦЕЙ КОРРЕЛЯЦИЙ)
# ==========================================
def save_gnn_model(model, edge_index, edge_weight, imputer_x, scaler_x, scaler_y, metrics, corr_mtx, filepath):
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        'model_state': model.state_dict(),
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'imputer_x': imputer_x,
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'metrics': metrics,
        'corr_matrix': corr_mtx  # <--- МАТРИЦА СОХРАНЯЕТСЯ ТУТ
    }
    torch.save(model_data, filepath)
    print(f"✓ Модель ST-GNN и метаданные графа сохранены в: {filepath}")


# ==========================================
# 5. УТИЛИТЫ АНАЛИЗА И ПАРСИНГА
# ==========================================
def aggregate_pt_metrics(models_dir='models_LNOF_TGNN', save_csv=True, csv_name='gnn_models_summary.csv'):
    models_path = Path(models_dir)
    if not models_path.exists(): return None

    pt_files = [f for f in models_path.rglob("*") if f.is_file() and f.suffix.lower() in ['.pt', '.pth']]
    if not pt_files: return None

    results = []
    for file_path in pt_files:
        session_name = file_path.name.replace('_st_gnn.pt', '').replace('.pt', '').replace('.pth', '')
        try:
            model_data = torch.load(file_path, map_location='cpu', weights_only=False)
            if isinstance(model_data, dict) and 'metrics' in model_data:
                m = model_data['metrics']
                results.append({
                    'Session': session_name,
                    'Test_R2': m.get('test_r2', np.nan),
                    'Train_R2': m.get('train_r2', np.nan),
                    'Test_MED': m.get('test_med_dist', np.nan),
                    'Train_MED': m.get('train_med_dist', np.nan),
                    'Threshold': m.get('adaptive_threshold', np.nan),
                    'File_Name': file_path.name
                })
        except Exception:
            pass

    df_results = pd.DataFrame(results)
    if df_results.empty: return None

    num_cols = ['Test_R2', 'Train_R2', 'Test_MED', 'Train_MED']
    for c in num_cols: df_results[c] = pd.to_numeric(df_results[c], errors='coerce').round(4)
    df_results = df_results.sort_values(by='Test_R2', ascending=False, na_position='last').reset_index(drop=True)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print("\n" + "=" * 90 + "\n📊 СВОДНАЯ ТАБЛИЦА МЕТРИК ST-GNN\n" + "=" * 90)
        print(df_results.to_string())

    if save_csv: df_results.to_csv(models_path / csv_name, index=False)
    return df_results


def print_model_metrics(session_list, models_dir='models_LNOF_TGNN'):
    models_path = Path(models_dir)
    print("\n" + "=" * 80 + f"\n📊 МЕТРИКИ ST-GNN ({len(session_list)} сессий)\n" + "=" * 80)
    for sess in session_list:
        file_path = models_path / f"{sess}_st_gnn.pt"
        if not file_path.exists(): file_path = models_path / f"{sess}.pt"
        if not file_path.exists(): continue
        try:
            model_data = torch.load(file_path, map_location='cpu', weights_only=False)
            if 'metrics' in model_data:
                m = model_data['metrics']
                print(
                    f"  ✓ {sess:<10} | Test R²: {m.get('test_r2', 'N/A'):.4f} | Train R²: {m.get('train_r2', 'N/A'):.4f}")
        except:
            pass


# ==========================================
# 6. ГЛАВНЫЙ ЦИКЛ УПРАВЛЕНИЯ
# ==========================================
# DATA_DIR = 'LNOF_SynchronizedData'
# MODELS_DIR = 'models_LNOF_TGNN_Final'
#
# # Подгрузка списка сессий
# models_dir = "models_LNOF_Lasso"
# models_path = Path(models_dir)
# if models_path.exists():
#     models_all = [f.name[:6] for f in models_path.glob("*.pkl")]
#     models_for = ['J20_1D', 'J20_2D', 'J20_3D', 'J20_4D', 'J21_1D', 'J21_2D', 'J21_3D', 'J21_4D']
#     TARGET_SESSIONS = list(set(models_all) - set(models_for))
# else:
#     TARGET_SESSIONS = ['J01_1D']  # Пример, если папка пуста


def main():
    pipeline_start = time.time()
    os.makedirs(MODELS_DIR, exist_ok=True)

    for i, session in enumerate(TARGET_SESSIONS):
        sess_start = time.time()
        print(f"\n{'=' * 60}\nОБУЧЕНИЕ СЕССИИ: {session} ({i + 1} из {len(TARGET_SESSIONS)})\n{'=' * 60}")

        file_path = Path(DATA_DIR) / f"LNOF_{session}_aligned.npz"
        if not file_path.exists(): continue

        try:
            data = np.load(file_path, allow_pickle=True)
            X_raw = data['Calcium'].T
            y_raw = np.column_stack((data['x'], data['y']))
        except Exception as e:
            continue

        df_X = pd.DataFrame(X_raw)
        df_y = pd.DataFrame(y_raw)
        split_idx = int(len(df_X) * 0.75)
        X_train, y_train = df_X.iloc[:split_idx], df_y.iloc[:split_idx]
        X_test, y_test = df_X.iloc[split_idx:], df_y.iloc[split_idx:]

        try:
            # РАСПАКОВКА ИЗ 8 ЭЛЕМЕНТОВ
            model, edge_index, edge_weight, imp_x, scl_x, scl_y, metrics, corr_mtx = train_st_gnn(
                X_train_raw=X_train, y_train_raw=y_train,
                X_test_raw=X_test, y_test_raw=y_test,
                seq_len=64, overlap=32, epochs=100, batch_size=64, lr=0.01
            )
        except Exception as e:
            print(f"[ОШИБКА] Сбой во время обучения: {e}")
            continue

        save_path = Path(MODELS_DIR) / f"{session}_st_gnn.pt"
        save_gnn_model(
            model=model, edge_index=edge_index, edge_weight=edge_weight,
            imputer_x=imp_x, scaler_x=scl_x, scaler_y=scl_y,
            metrics=metrics, corr_mtx=corr_mtx, filepath=save_path
        )

        sess_time = time.time() - sess_start
        print(f"⏱ Время: {sess_time:.1f} сек.")

    pipe_time = time.time() - pipeline_start
    print(f"\n{'=' * 60}\n✅ ПАЙПЛАЙН ЗАВЕРШЕН! Общее время: {pipe_time / 60:.2f} мин.\n{'=' * 60}")

# --- НАСТРОЙКИ ---
DATA_DIR = 'LNOF_SynchronizedData'
MODELS_DIR = 'models_LNOF_TGNN_Tanh'

# Здесь ты указываешь одну сессию или список сессий
# Например: TARGET_SESSIONS = ['J30_1D'] или ['H39_1D', 'H39_2D', 'H01_4D']

models_dir = "models_LNOF_Lasso"
models_path = Path(models_dir)
models_all = [f.name[:6] for f in models_path.glob("*.pkl")]
# models_min = Path('models_LNOF_TGNN_New')
# models_for = [f.name[:6] for f in models_min.glob("*.pt")]
# models = ['J24_3D.pkl', 'J54_4D.pkl', 'J18_2D.pkl', 'J59_4D.pkl', 'J52_2D.pkl']
# models = [model[:-4] for model in models]
models_for = ['J20_1D', 'J20_2D', 'J20_3D', 'J20_4D', 'J21_1D', 'J21_2D', 'J21_3D', 'J21_4D']
#
models_all = list(set(models_all) - set(models_for))

TARGET_SESSIONS = models_all

if __name__ == "__main__":
    # main()
    # my_sessions = ['J01_2D']
    # print_model_metrics(my_sessions, models_dir='models_LNOF_TGNN_New')
    aggregate_pt_metrics(models_dir='models_LNOF_TGNN_Tanh')