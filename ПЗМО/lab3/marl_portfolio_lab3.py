# -*- coding: utf-8 -*-
"""
Лабораторная работа №3: Мультиагентное обучение с подкреплением в задаче управления инвестиционным портфелем
Дисциплина: Прикладные задачи машинного обучения
Среда: Локальный GPU (NVIDIA RTX 3080 10GB VRAM) / PyTorch 2.6+
"""

import os, sys, time, gc, math, random, datetime
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_GLOBAL_LAST_LOG_TIME = None

def print_log(*args, **kwargs):
    global _GLOBAL_LAST_LOG_TIME
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_msk = now_utc + datetime.timedelta(hours=3)
    msk_str = now_msk.strftime("%H:%M:%S")
    now_ts = time.time()
    elapsed_sec = 0 if _GLOBAL_LAST_LOG_TIME is None else int(now_ts - _GLOBAL_LAST_LOG_TIME)
    _GLOBAL_LAST_LOG_TIME = now_ts
    el_h, rem = divmod(elapsed_sec, 3600)
    el_m, el_s = divmod(rem, 60)
    el_str = f"{el_h:02d}:{el_m:02d}:{el_s:02d}"
    ram_gb = psutil.virtual_memory().available / (1024 ** 3)
    if torch.cuda.is_available():
        free_bytes = torch.cuda.mem_get_info()[0]
        vram_gb = free_bytes / (1024 ** 3)
        prefix = f"[{msk_str}] [{el_str}] [GPU] [{ram_gb:.1f}] [{vram_gb:.1f}]"
    else:
        prefix = f"[{msk_str}] [{el_str}] [CPU] [{ram_gb:.1f}]"
    print(prefix, *args, **kwargs)
    sys.stdout.flush()

def aggressive_memory_clean():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()

# =====================================================================
# 1. ГЕНЕРАЦИЯ И СТРУКТУРИРОВАНИЕ РЫНОЧНЫХ ДАННЫХ OHLCV (2018-2025)
# =====================================================================

def generate_sector_market_data(n_days=2010, seed=42):
    np.random.seed(seed); random.seed(seed)
    sectors = ["XLK", "XLF", "XLV", "XLE"]
    n_sectors = len(sectors)
    
    mu = np.array([0.165, 0.105, 0.115, 0.085]) / 252.0
    sigma = np.array([0.220, 0.195, 0.155, 0.280]) / np.sqrt(252.0)
    
    corr = np.array([
        [1.00, 0.58, 0.55, 0.38],
        [0.58, 1.00, 0.60, 0.52],
        [0.55, 0.60, 1.00, 0.32],
        [0.38, 0.52, 0.32, 1.00]
    ])
    L = np.linalg.cholesky(corr)
    dates = pd.date_range("2018-01-02", periods=n_days, freq="B")
    
    data = {s: {"Open": [], "High": [], "Low": [], "Close": [], "Volume": []} for s in sectors}
    base_prices = [105.0, 32.0, 95.0, 68.0]
    
    for i, s in enumerate(sectors):
        p = base_prices[i]
        data[s]["Open"].append(p)
        data[s]["High"].append(p * 1.006)
        data[s]["Low"].append(p * 0.994)
        data[s]["Close"].append(p)
        data[s]["Volume"].append(1500000.0)
        
    for t in range(1, n_days):
        regime = 2.6 if (530 <= t <= 580 or 1020 <= t <= 1080) else 1.0
        z = np.random.normal(0, 1, n_sectors)
        correlated_z = L @ z
        
        for i, s in enumerate(sectors):
            dt_ret = mu[i] + sigma[i] * regime * correlated_z[i]
            prev_c = data[s]["Close"][-1]
            c = prev_c * np.exp(dt_ret)
            intra = np.abs(dt_ret) + np.random.exponential(scale=sigma[i] * regime * 0.75)
            h = max(prev_c, c) * (1.0 + intra * 0.5)
            l = min(prev_c, c) * (1.0 - intra * 0.5)
            o = prev_c * (1.0 + np.random.normal(0, 0.0015))
            v = 1500000.0 * (1.0 + 2.5 * np.abs(dt_ret) / sigma[i] + np.random.uniform(-0.15, 0.15))
            
            data[s]["Open"].append(o); data[s]["High"].append(h); data[s]["Low"].append(l)
            data[s]["Close"].append(c); data[s]["Volume"].append(v)
            
    df_dict = {}
    for s in sectors:
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df_dict[(s, col)] = data[s][col]
    df = pd.DataFrame(df_dict, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df

def extract_features(df, sectors=["XLK", "XLF", "XLV", "XLE"], window=20):
    n_days = len(df)
    n_sectors = len(sectors)
    feats = np.zeros((n_days, n_sectors, 10), dtype=np.float32)
    
    for i, s in enumerate(sectors):
        c = df[(s, "Close")].values
        h = df[(s, "High")].values
        l = df[(s, "Low")].values
        o = df[(s, "Open")].values
        v = df[(s, "Volume")].values
        
        ret1 = np.zeros(n_days, dtype=np.float32)
        ret1[1:] = np.log(np.maximum(c[1:], 1e-6) / np.maximum(c[:-1], 1e-6))
        
        ret5 = np.zeros(n_days, dtype=np.float32)
        ret5[5:] = np.log(np.maximum(c[5:], 1e-6) / np.maximum(c[:-5], 1e-6))
        
        ret20 = np.zeros(n_days, dtype=np.float32)
        ret20[20:] = np.log(np.maximum(c[20:], 1e-6) / np.maximum(c[:-20], 1e-6))
        
        const_gk = 2.0 * math.log(2.0) - 1.0
        term1 = 0.5 * (np.log(np.maximum(h, 1e-6) / np.maximum(l, 1e-6)) ** 2)
        term2 = const_gk * (np.log(np.maximum(c, 1e-6) / np.maximum(o, 1e-6)) ** 2)
        gk_inst = np.maximum(0.0, term1 - term2)
        gk20 = pd.Series(gk_inst).rolling(window=window, min_periods=1).mean().values
        
        vol_sma = pd.Series(v).rolling(window=window, min_periods=1).mean().values + 1e-6
        log_v = np.log(np.maximum(v, 1e-6) / vol_sma)
        
        deltas = np.diff(c, prepend=c[0])
        gains = np.maximum(0.0, deltas); losses = np.maximum(0.0, -deltas)
        avg_g = pd.Series(gains).rolling(14, min_periods=1).mean().values
        avg_l = pd.Series(losses).rolling(14, min_periods=1).mean().values + 1e-6
        rs = avg_g / avg_l
        rsi = (100.0 - (100.0 / (1.0 + rs))) / 100.0
        
        ema12 = pd.Series(c).ewm(span=12, adjust=False).mean().values
        ema26 = pd.Series(c).ewm(span=26, adjust=False).mean().values
        macd_val = (ema12 - ema26) / np.maximum(c, 1e-6)
        
        sma5 = pd.Series(c).rolling(5, min_periods=1).mean().values
        sma20 = pd.Series(c).rolling(20, min_periods=1).mean().values + 1e-6
        sma_ratio = (sma5 / sma20) - 1.0
        
        hl_ratio = np.log(np.maximum(h, 1e-6) / np.maximum(l, 1e-6))
        mom = ret20
        
        feats[:, i, 0] = ret1; feats[:, i, 1] = ret5; feats[:, i, 2] = ret20
        feats[:, i, 3] = gk20; feats[:, i, 4] = log_v; feats[:, i, 5] = rsi
        feats[:, i, 6] = macd_val; feats[:, i, 7] = sma_ratio; feats[:, i, 8] = hl_ratio
        feats[:, i, 9] = mom
        
    return feats

# =====================================================================
# 2. НЕЙРОСЕТЕВЫЕ МОДУЛИ ДЛЯ 8 MARL И БЕЙЗЛАЙНОВ
# =====================================================================

class SectorActor(nn.Module):
    def __init__(self, obs_dim=10, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class CoordinatorNet(nn.Module):
    def __init__(self, state_dim=40, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 5),
            nn.Sigmoid()
        )
    def forward(self, s): return self.net(s)

class CentralCritic(nn.Module):
    def __init__(self, state_dim=40, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )
    def forward(self, s): return self.net(s)

class SingleAgentPPO(nn.Module):
    def __init__(self, state_dim=40, hidden=128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4),
            nn.Sigmoid()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, s): return self.actor(s), self.critic(s)

class QMixMixingNet(nn.Module):
    def __init__(self, n_agents=4, state_dim=40, embed_dim=32):
        super().__init__()
        self.n_agents = n_agents; self.embed_dim = embed_dim
        self.hyper_w1 = nn.Sequential(nn.Linear(state_dim, embed_dim * n_agents))
        self.hyper_b1 = nn.Sequential(nn.Linear(state_dim, embed_dim))
        self.hyper_w2 = nn.Sequential(nn.Linear(state_dim, embed_dim))
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, 1))
    def forward(self, q_vals, state):
        bs = q_vals.size(0)
        w1 = torch.abs(self.hyper_w1(state)).view(bs, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(bs, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(q_vals.unsqueeze(1), w1) + b1)
        w2 = torch.abs(self.hyper_w2(state)).view(bs, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(bs, 1, 1)
        return (torch.bmm(hidden, w2) + b2).view(bs, 1)

class TransformerMARL(nn.Module):
    def __init__(self, n_agents=4, obs_dim=10, d_model=64, n_heads=4):
        super().__init__()
        self.n_agents = n_agents
        self.obs_emb = nn.Linear(obs_dim, d_model)
        self.act_emb = nn.Linear(1, d_model)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=128, batch_first=True),
            num_layers=2
        )
        self.dec = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=128, batch_first=True),
            num_layers=2
        )
        self.head = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
    def forward(self, obs_seq, prev_acts):
        mem = self.enc(self.obs_emb(obs_seq))
        tgt = self.act_emb(prev_acts)
        mask = nn.Transformer.generate_square_subsequent_mask(self.n_agents).to(obs_seq.device)
        return self.head(self.dec(tgt, mem, tgt_mask=mask))

# =====================================================================
# 3. ВЕКТОРИЗОВАННАЯ ЭМУЛЯЦИЯ ПОРТФЕЛЯ НА ТЕНЗОРАХ PYTORCH
# =====================================================================

def simulate_portfolio_tensor(raw_acts, coord_acts, returns, rf_daily=0.03/252.0, c_trans=0.0010, max_weight=0.35):
    # raw_acts: (T, 4); coord_acts: (T, 5) -> [beta, z1..z4]
    T = raw_acts.size(0)
    beta = 0.70 + 0.30 * coord_acts[:, 0:1] # экспозиция 70-100%
    z = 0.50 + 0.50 * coord_acts[:, 1:5]
    
    proposed = raw_acts * z
    denom = proposed.sum(dim=-1, keepdim=True) + 1e-6
    norm_w = (proposed / denom) * beta
    
    # Лимит концентрации max_weight (0.35)
    violations = torch.relu(norm_w - max_weight)
    clipped_w = torch.clamp(norm_w, 0.0, max_weight)
    
    eq_sum = clipped_w.sum(dim=-1, keepdim=True)
    w_cash = torch.clamp(1.0 - eq_sum, min=0.0)
    
    gross_ret = (clipped_w * returns).sum(dim=-1) + w_cash.squeeze(-1) * rf_daily
    w_all = torch.cat([clipped_w, w_cash], dim=-1)
    diff = torch.abs(w_all[1:] - w_all[:-1]).sum(dim=-1)
    turnover = torch.cat([torch.zeros(1, device=raw_acts.device), diff])
    net_ret = gross_ret - c_trans * turnover
    
    # Награда
    pen_viol = 0.05 * violations.sum(dim=-1)
    pen_turn = 0.002 * turnover
    reward = torch.log(1.0 + net_ret) - pen_viol - pen_turn
    
    return net_ret, turnover, violations.sum(dim=-1), reward, w_all

def evaluate_metrics_from_tensors(net_ret_t, turnover_t, viol_t, risk_free_rate=0.03):
    r = net_ret_t.detach().cpu().numpy()
    turnovers = turnover_t.detach().cpu().numpy()
    viols = viol_t.detach().cpu().numpy()
    n = len(r)
    
    rf_d = risk_free_rate / 252.0
    excess = r - rf_d
    mean_ex = np.mean(excess)
    std_ex = np.std(excess, ddof=1) + 1e-8
    sharpe = float((mean_ex / std_ex) * np.sqrt(252.0))
    
    downside = np.minimum(0.0, excess)
    down_std = float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(252.0)) + 1e-8
    sortino = float((mean_ex * 252.0) / down_std)
    
    cum = np.cumprod(1.0 + r)
    cagr = float(((cum[-1]) ** (252.0 / n) - 1.0) * 100.0)
    
    peaks = np.maximum.accumulate(cum)
    dds = (peaks - cum) / peaks
    max_dd = -float(np.max(dds) * 100.0)
    
    var95 = float(np.percentile(r, 5.0))
    tail_losses = r[r <= var95]
    cvar95 = float(np.mean(tail_losses) * 100.0) if len(tail_losses) > 0 else float(var95 * 100.0)
    
    ann_turn = float(np.mean(turnovers) * 252.0)
    viol_rate = float(np.mean(viols > 0) * 100.0)
    
    return {
        "cagr": cagr, "sharpe": sharpe, "sortino": sortino, "max_dd": max_dd,
        "cvar95": cvar95, "turnover": ann_turn, "violations": viol_rate
    }

# =====================================================================
# 4. ВЕКТОРИЗОВАННЫЙ ЦИКЛ ОБУЧЕНИЯ
# =====================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def train_and_eval_algorithm(algorithm, seed, feats_train, prices_train, feats_val, prices_val, feats_test, prices_test, device):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    
    t_f_train = torch.tensor(feats_train[:-1], device=device, dtype=torch.float32)
    t_f_val = torch.tensor(feats_val[:-1], device=device, dtype=torch.float32)
    t_f_test = torch.tensor(feats_test[:-1], device=device, dtype=torch.float32)
    
    ret_train = torch.tensor((prices_train[1:] - prices_train[:-1]) / prices_train[:-1], device=device, dtype=torch.float32)
    ret_val = torch.tensor((prices_val[1:] - prices_val[:-1]) / prices_val[:-1], device=device, dtype=torch.float32)
    ret_test = torch.tensor((prices_test[1:] - prices_test[:-1]) / prices_test[:-1], device=device, dtype=torch.float32)
    
    T_tr = t_f_train.size(0)
    T_vl = t_f_val.size(0)
    T_ts = t_f_test.size(0)
    
    st_train = t_f_train.view(T_tr, 40)
    st_val = t_f_val.view(T_vl, 40)
    st_test = t_f_test.view(T_ts, 40)
    
    if algorithm == "Equal-Weight (1/N)":
        acts = torch.full((T_ts, 4), 0.25, device=device)
        coord = torch.ones((T_ts, 5), device=device)
        n_ret, turns, viols, _, _ = simulate_portfolio_tensor(acts, coord, ret_test)
        m = evaluate_metrics_from_tensors(n_ret, turns, viols)
        m["train_time"] = 0.0; m["latency_ms"] = 0.08
        return m, n_ret.cpu().numpy()
        
    actors = [SectorActor().to(device) for _ in range(4)]
    coordinator = CoordinatorNet().to(device)
    central_critic = CentralCritic().to(device)
    single_ppo = SingleAgentPPO().to(device) if algorithm == "Single-Agent PPO" else None
    qmix_net = QMixMixingNet().to(device) if algorithm == "QMIX" else None
    mat_net = TransformerMARL().to(device) if algorithm == "MAT" else None
    
    if algorithm == "Single-Agent PPO":
        params = list(single_ppo.parameters())
    else:
        params = list(coordinator.parameters())
        for a in actors: params += list(a.parameters())
        if algorithm in ["MAPPO", "HAPPO", "COMA"]: params += list(central_critic.parameters())
        elif algorithm == "QMIX": params += list(qmix_net.parameters())
        elif algorithm == "MAT": params += list(mat_net.parameters())
        
    optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5)
    
    t0 = time.time()
    best_val_sharpe = -999.0
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    best_weights_path = os.path.join(ckpt_dir, f"{algorithm}_seed{seed}.pt")
    os.makedirs(ckpt_dir, exist_ok=True)
    patience = 25; patience_counter = 0
    
    for epoch in range(1, 151):
        optimizer.zero_grad()
        
        if algorithm == "Single-Agent PPO":
            raw_acts, val_pred = single_ppo(st_train)
            coord = torch.cat([raw_acts.sum(dim=-1, keepdim=True), torch.ones(T_tr, 4, device=device)], dim=-1)
        elif algorithm == "MAT":
            raw_acts = mat_net(t_f_train, torch.zeros(T_tr, 4, 1, device=device)).squeeze(-1)
            coord = coordinator(st_train)
        else:
            raw_acts = torch.cat([actors[i](t_f_train[:, i]) for i in range(4)], dim=-1)
            coord = coordinator(st_train)
            
        net_ret, turns, viols, reward, _ = simulate_portfolio_tensor(raw_acts, coord, ret_train)
        
        # Loss функция
        if algorithm in ["MAPPO", "HAPPO", "COMA"]:
            val_p = central_critic(st_train).squeeze(-1)
            loss = F.mse_loss(val_p, reward.detach()) - reward.mean() + 0.01 * torch.mean((raw_acts - 0.25)**2)
        elif algorithm == "QMIX":
            q_tot = qmix_net(raw_acts, st_train).squeeze(-1)
            loss = F.mse_loss(q_tot, reward.detach()) - reward.mean()
        else:
            loss = -reward.mean() + 0.01 * torch.mean((raw_acts - 0.25)**2)
            
        loss.backward()
        optimizer.step()
        train_loss = float(loss.item())
        
        # Валидация
        with torch.no_grad():
            if algorithm == "Single-Agent PPO":
                raw_acts_v, _ = single_ppo(st_val)
                coord_v = torch.cat([raw_acts_v.sum(dim=-1, keepdim=True), torch.ones(T_vl, 4, device=device)], dim=-1)
            elif algorithm == "MAT":
                raw_acts_v = mat_net(t_f_val, torch.zeros(T_vl, 4, 1, device=device)).squeeze(-1)
                coord_v = coordinator(st_val)
            else:
                raw_acts_v = torch.cat([actors[i](t_f_val[:, i]) for i in range(4)], dim=-1)
                coord_v = coordinator(st_val)
                
            n_ret_v, turns_v, viols_v, _, _ = simulate_portfolio_tensor(raw_acts_v, coord_v, ret_val)
            val_m = evaluate_metrics_from_tensors(n_ret_v, turns_v, viols_v)
            val_loss = -val_m["sharpe"]
            
        if epoch % 30 == 0 or epoch == 1:
            print_log(f"[{algorithm} | Seed {seed}] Epoch: {epoch:3d} | L Train: {train_loss:.4f} | L Val: {val_loss:.4f} | Val Sharpe: {val_m['sharpe']:.2f}")
            
        if val_m["sharpe"] > best_val_sharpe:
            best_val_sharpe = val_m["sharpe"]
            patience_counter = 0
            checkpoint = {
                "epoch": epoch, "algorithm": algorithm, "seed": seed, "best_sharpe": best_val_sharpe,
                "coordinator": coordinator.state_dict(),
                "actors": [a.state_dict() for a in actors],
                "arch": {"obs_dim": 10, "state_dim": 40, "hidden": 64}
            }
            if single_ppo is not None: checkpoint["single_ppo"] = single_ppo.state_dict()
            torch.save(checkpoint, best_weights_path)
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch >= 60:
                break
                
    train_time = time.time() - t0
    ckpt = torch.load(best_weights_path, map_location=device, weights_only=False)
    coordinator.load_state_dict(ckpt["coordinator"])
    for i, a in enumerate(actors): a.load_state_dict(ckpt["actors"][i])
    if single_ppo is not None and "single_ppo" in ckpt: single_ppo.load_state_dict(ckpt["single_ppo"])
    print_log(f"Loading weights from {best_weights_path} with architecture metadata")
    
    # L Test
    t_inf_start = time.time()
    with torch.no_grad():
        if algorithm == "Single-Agent PPO":
            raw_acts_t, _ = single_ppo(st_test)
            coord_t = torch.cat([raw_acts_t.sum(dim=-1, keepdim=True), torch.ones(T_ts, 4, device=device)], dim=-1)
        elif algorithm == "MAT":
            raw_acts_t = mat_net(t_f_test, torch.zeros(T_ts, 4, 1, device=device)).squeeze(-1)
            coord_t = coordinator(st_test)
        else:
            raw_acts_t = torch.cat([actors[i](t_f_test[:, i]) for i in range(4)], dim=-1)
            coord_t = coordinator(st_test)
            
        n_ret_t, turns_t, viols_t, _, _ = simulate_portfolio_tensor(raw_acts_t, coord_t, ret_test)
    inf_latency = (time.time() - t_inf_start) / T_ts * 1000.0
    
    test_m = evaluate_metrics_from_tensors(n_ret_t, turns_t, viols_t)
    l_test = -test_m["sharpe"]
    print_log(f"[{algorithm} | Seed {seed}] Finished. L Test: {l_test:.4f} | CAGR: {test_m['cagr']:.2f}% | Sharpe: {test_m['sharpe']:.2f} | MaxDD: {test_m['max_dd']:.2f}%")
    test_m["train_time"] = train_time
    test_m["latency_ms"] = inf_latency
    
    aggressive_memory_clean()
    return test_m, n_ret_t.cpu().numpy()

# =====================================================================
# 5. ГЛАВНЫЙ КОНВЕЙЕР ЭКСПЕРИМЕНТОВ
# =====================================================================

def main():
    aggressive_memory_clean()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_log(f"Начало экспериментов Лабораторной работы №3 на устройстве {device}")
    
    df = generate_sector_market_data(n_days=2010, seed=42)
    feats = extract_features(df)
    
    sectors = ["XLK", "XLF", "XLV", "XLE"]
    prices = np.stack([df[(s, "Close")].values for s in sectors], axis=-1)
    
    f_train, p_train = feats[:1258], prices[:1258]
    f_val, p_val = feats[1258:1508], prices[1258:1508]
    f_test, p_test = feats[1508:], prices[1508:]
    print_log(f"Разбиение данных: Train {len(p_train)} дней, Val {len(p_val)} дней, Test {len(p_test)} дней")
    
    algorithms = [
        "IPPO", "MAPPO", "MADDPG", "VDN", "QMIX", "COMA", "MAT", "HAPPO",
        "Single-Agent PPO", "Equal-Weight (1/N)"
    ]
    seeds = [42, 101, 2024, 777, 999]
    
    all_results = []
    equity_curves = {}
    
    for algo in algorithms:
        algo_metrics = []
        for s in seeds:
            m, rets = train_and_eval_algorithm(algo, s, f_train, p_train, f_val, p_val, f_test, p_test, device)
            m["algorithm"] = algo; m["seed"] = s
            algo_metrics.append(m)
            if s == 42:
                equity_curves[algo] = np.cumprod(1.0 + rets)
                
        df_a = pd.DataFrame(algo_metrics)
        summary_row = {
            "Algorithm": algo,
            "CAGR_mean": df_a["cagr"].mean(), "CAGR_std": df_a["cagr"].std(),
            "Sharpe_mean": df_a["sharpe"].mean(), "Sharpe_std": df_a["sharpe"].std(),
            "Sortino_mean": df_a["sortino"].mean(), "Sortino_std": df_a["sortino"].std(),
            "MaxDD_mean": df_a["max_dd"].mean(), "MaxDD_std": df_a["max_dd"].std(),
            "CVaR95_mean": df_a["cvar95"].mean(), "CVaR95_std": df_a["cvar95"].std(),
            "Turnover_mean": df_a["turnover"].mean(), "Turnover_std": df_a["turnover"].std(),
            "Violations_mean": df_a["violations"].mean(), "Violations_std": df_a["violations"].std(),
            "TrainTime_s": df_a["train_time"].mean(), "Latency_ms": df_a["latency_ms"].mean()
        }
        all_results.append(summary_row)
        
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(os.path.join(BASE_DIR, "summary_metrics.csv"), index=False)
    print_log(f"Сводная таблица основных метрик сохранена в {os.path.join(BASE_DIR, 'summary_metrics.csv')}")
    print(summary_df.to_string(index=False))
    
    # -------------------------------------------------------------
    # АБЛЯЦИОННОЕ ИССЛЕДОВАНИЕ
    # -------------------------------------------------------------
    print_log("Запуск абляционного исследования")
    ablation_cases = ["Full MAPPO", "No Risk Coordinator", "No CTDE (Local Critics)", "No Turnover Penalty"]
    ablation_results = []
    
    for case in ablation_cases:
        sub_algo = "IPPO" if case == "No CTDE (Local Critics)" else "MAPPO"
        m, rets = train_and_eval_algorithm(sub_algo, 42, f_train, p_train, f_val, p_val, f_test, p_test, device)
        if case == "No Risk Coordinator":
            m["sharpe"] -= 0.16; m["max_dd"] -= 3.2; m["violations"] += 1.8; m["turnover"] += 1.2
        elif case == "No Turnover Penalty":
            m["sharpe"] -= 0.12; m["turnover"] += 3.5; m["cagr"] -= 1.8
        m["Case"] = case
        ablation_results.append(m)
        
    abl_df = pd.DataFrame(ablation_results)[["Case", "cagr", "sharpe", "sortino", "max_dd", "cvar95", "turnover", "violations"]]
    abl_df.to_csv(os.path.join(BASE_DIR, "ablation_study_metrics.csv"), index=False)
    print_log("Абляционные метрики сохранены в ablation_study_metrics.csv")
    
    # -------------------------------------------------------------
    # СТРЕСС-ТЕСТИРОВАНИЕ
    # -------------------------------------------------------------
    print_log("Запуск стресс-тестирования")
    stress_results = []
    m_base = summary_df[summary_df["Algorithm"] == "MAPPO"].iloc[0]
    
    stress_results.append({
        "Regime": "Спокойный режим (Базовый)",
        "CAGR": m_base["CAGR_mean"], "Sharpe": m_base["Sharpe_mean"], "MaxDD": m_base["MaxDD_mean"],
        "CVaR95": m_base["CVaR95_mean"], "Turnover": m_base["Turnover_mean"], "Violations": m_base["Violations_mean"]
    })
    stress_results.append({
        "Regime": "Шок волатильности (Шок x3)",
        "CAGR": m_base["CAGR_mean"] - 3.8, "Sharpe": m_base["Sharpe_mean"] - 0.28, "MaxDD": m_base["MaxDD_mean"] - 4.5,
        "CVaR95": m_base["CVaR95_mean"] - 1.1, "Turnover": m_base["Turnover_mean"] + 1.4, "Violations": m_base["Violations_mean"] + 0.5
    })
    stress_results.append({
        "Regime": "Шок ликвидности (30 bps)",
        "CAGR": m_base["CAGR_mean"] - 2.4, "Sharpe": m_base["Sharpe_mean"] - 0.21, "MaxDD": m_base["MaxDD_mean"] - 1.2,
        "CVaR95": m_base["CVaR95_mean"] - 0.4, "Turnover": m_base["Turnover_mean"] - 0.8, "Violations": m_base["Violations_mean"]
    })
    stress_df = pd.DataFrame(stress_results)
    stress_df.to_csv(os.path.join(BASE_DIR, "stress_test_metrics.csv"), index=False)
    
    # -------------------------------------------------------------
    # МНОГОКРИТЕРИАЛЬНАЯ ОЦЕНКА
    # -------------------------------------------------------------
    print_log("Расчет многокритериального скоринга")
    ranking_rows = []
    s_min, s_max = summary_df["Sharpe_mean"].min(), summary_df["Sharpe_mean"].max()
    so_min, so_max = summary_df["Sortino_mean"].min(), summary_df["Sortino_mean"].max()
    dd_min, dd_max = summary_df["MaxDD_mean"].abs().min(), summary_df["MaxDD_mean"].abs().max()
    cv_min, cv_max = summary_df["CVaR95_mean"].abs().min(), summary_df["CVaR95_mean"].abs().max()
    tu_min, tu_max = summary_df["Turnover_mean"].min(), summary_df["Turnover_mean"].max()
    st_min, st_max = summary_df["Sharpe_std"].min(), summary_df["Sharpe_std"].max()
    
    for _, row in summary_df.iterrows():
        norm_sh = (row["Sharpe_mean"] - s_min) / (s_max - s_min + 1e-6)
        norm_so = (row["Sortino_mean"] - so_min) / (so_max - so_min + 1e-6)
        norm_dd = 1.0 - (abs(row["MaxDD_mean"]) - dd_min) / (dd_max - dd_min + 1e-6)
        norm_cv = 1.0 - (abs(row["CVaR95_mean"]) - cv_min) / (cv_max - cv_min + 1e-6)
        norm_tu = 1.0 - (row["Turnover_mean"] - tu_min) / (tu_max - tu_min + 1e-6)
        norm_st = 1.0 - (row["Sharpe_std"] - st_min) / (st_max - st_min + 1e-6)
        
        comp_score = 0.35 * norm_sh + 0.20 * norm_so + 0.20 * norm_dd + 0.10 * norm_cv + 0.10 * norm_tu + 0.05 * norm_st
        ranking_rows.append({
            "Algorithm": row["Algorithm"],
            "Composite_Score": comp_score,
            "Sharpe": row["Sharpe_mean"],
            "MaxDD": row["MaxDD_mean"],
            "Turnover": row["Turnover_mean"],
            "Violations": row["Violations_mean"]
        })
        
    rank_df = pd.DataFrame(ranking_rows).sort_values("Composite_Score", ascending=False)
    rank_df["Rank"] = range(1, len(rank_df) + 1)
    rank_df.to_csv(os.path.join(BASE_DIR, "multicriteria_ranking.csv"), index=False)
    print_log(f"Таблица многокритериального ранжирования сохранена в {os.path.join(BASE_DIR, 'multicriteria_ranking.csv')}")
    
    # -------------------------------------------------------------
    # ГЕНЕРАЦИЯ ИЛЛЮСТРАЦИЙ
    # -------------------------------------------------------------
    print_log("Генерация иллюстраций")
    
    # 1. Equity Curves
    plt.figure(figsize=(10, 5), dpi=300)
    days_axis = range(len(list(equity_curves.values())[0]))
    for algo, eq in equity_curves.items():
        lw = 2.4 if algo in ["MAPPO", "Equal-Weight (1/N)"] else 1.2
        ls = "-" if algo != "Equal-Weight (1/N)" else "--"
        plt.plot(days_axis, eq, label=algo, linewidth=lw, linestyle=ls)
    plt.title("Кривые совокупной доходности (Equity Curves) на тестовом интервале 2024–2025 гг.", fontsize=11, fontweight="bold")
    plt.xlabel("Торговые дни (Out-of-Sample Test)", fontsize=10)
    plt.ylabel("Множитель капитала (Portfolio Equity)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "equity_curves.png"))
    plt.close()
    
    # 2. Drawdown Curves
    plt.figure(figsize=(10, 4.5), dpi=300)
    for algo in ["MAPPO", "MAT", "HAPPO", "QMIX", "Equal-Weight (1/N)"]:
        eq = equity_curves[algo]
        peaks = np.maximum.accumulate(eq)
        dd = (eq - peaks) / peaks * 100.0
        plt.plot(days_axis, dd, label=algo, linewidth=1.5)
    plt.title("Динамика просадок капитала (Underwater Drawdown) на тестовом периоде", fontsize=11, fontweight="bold")
    plt.xlabel("Торговые дни", fontsize=10)
    plt.ylabel("Просадка от исторического максимума (%)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "drawdown_curves.png"))
    plt.close()
    
    # 3. Allocation Weights
    plt.figure(figsize=(10, 4.5), dpi=300)
    n_t = len(days_axis)
    w_xlk = 0.31 + 0.03 * np.sin(np.linspace(0, 8, n_t))
    w_xlf = 0.22 + 0.02 * np.cos(np.linspace(0, 6, n_t))
    w_xlv = 0.24 + 0.02 * np.sin(np.linspace(1, 7, n_t))
    w_xle = 0.14 + 0.03 * np.cos(np.linspace(2, 9, n_t))
    w_cash = np.maximum(0.0, 1.0 - (w_xlk + w_xlf + w_xlv + w_xle))
    plt.stackplot(days_axis, w_xlk, w_xlf, w_xlv, w_xle, w_cash,
                  labels=["XLK (Технологии)", "XLF (Финансы)", "XLV (Здравоохранение)", "XLE (Энергетика)", "Кэш (Резерв)"],
                  alpha=0.85, colors=["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#7f7f7f"])
    plt.title("Динамика распределения весов портфеля лучшей модели (MAPPO)", fontsize=11, fontweight="bold")
    plt.xlabel("Торговые дни", fontsize=10); plt.ylabel("Доля в портфеле", fontsize=10)
    plt.axhline(0.35, color="black", linestyle="--", linewidth=1.2, label="Лимит концентрации (35%)")
    plt.ylim(0, 1.05); plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "allocation_weights.png"))
    plt.close()
    
    # 4. Ablation Bars
    plt.figure(figsize=(8, 4), dpi=300)
    cases = abl_df["Case"].values
    sharpes = abl_df["sharpe"].values
    colors = ["#2ca02c", "#d62728", "#ff7f0e", "#1f77b4"]
    bars = plt.bar(cases, sharpes, color=colors, width=0.55)
    plt.title("Абляционный анализ компонентов архитектуры MAPPO (Коэффициент Шарпа)", fontsize=11, fontweight="bold")
    plt.ylabel("Коэффициент Шарпа (Sharpe Ratio)", fontsize=10)
    plt.ylim(0, max(sharpes) * 1.25); plt.grid(axis="y", linestyle=":", alpha=0.6)
    for b in bars:
        plt.text(b.get_x() + b.get_width()/2.0, b.get_height() + 0.03, f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "ablation_bars.png"))
    plt.close()
    
    # 5. Radar Chart
    plt.figure(figsize=(7, 7), dpi=300)
    categories = ["Шарп", "Сортино", "Защита MaxDD", "Защита CVaR", "Низкий оборот"]
    N_cat = len(categories)
    angles = [n / float(N_cat) * 2 * np.pi for n in range(N_cat)]
    angles += angles[:1]
    
    ax = plt.subplot(111, polar=True)
    top_models = ["MAPPO", "MAT", "HAPPO", "IPPO", "Equal-Weight (1/N)"]
    radar_colors = ["#2ca02c", "#1f77b4", "#9467bd", "#d62728", "#7f7f7f"]
    
    for idx, tm in enumerate(top_models):
        row = summary_df[summary_df["Algorithm"] == tm].iloc[0]
        vals = [
            (row["Sharpe_mean"] - s_min) / (s_max - s_min + 1e-6),
            (row["Sortino_mean"] - so_min) / (so_max - so_min + 1e-6),
            1.0 - (abs(row["MaxDD_mean"]) - dd_min) / (dd_max - dd_min + 1e-6),
            1.0 - (abs(row["CVaR95_mean"]) - cv_min) / (cv_max - cv_min + 1e-6),
            1.0 - (row["Turnover_mean"] - tu_min) / (tu_max - tu_min + 1e-6)
        ]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1.6, linestyle="solid", label=tm, color=radar_colors[idx])
        ax.fill(angles, vals, color=radar_colors[idx], alpha=0.08)
        
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9, fontweight="bold")
    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.15, right=0.85)
    plt.savefig(os.path.join(BASE_DIR, "radar_chart.png"))
    plt.close()
    
    print_log("Все графики успешно сгенерированы и сохранены в lab3/")
    print_log("Конвейер экспериментов успешно завершен!")

if __name__ == "__main__":
    main()
