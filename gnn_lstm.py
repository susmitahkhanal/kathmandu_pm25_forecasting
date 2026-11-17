"""GNN-LSTM regression head vs per-station LSTM, same locked split / horizons / metrics / tuning protocol.

Stages (python gnn_lstm.py <stage> ...):
  smoke   tiny timing run
  search  Optuna search (12 trials, TPE seed 42, MedianPruner, 20-epoch cap, floor 3, patience 3, val-MAE objective)
  final   retrain best config (floor 5, patience 5, cap 300), evaluate on test, write comparison tables
  lstm300 retrain tuned per-station LSTM under the same cap-300 regime (parity with GNN final)
  clf     classification head-to-head: GNN-LSTM vs per-station LSTM, identical regime (python gnn_lstm.py clf [seeds])

Model: shared LSTM encoder per station node -> one graph-mixing step over the train-only correlation
adjacency (row-normalised over stations present at that anchor) -> per-station, per-horizon linear heads.
Search space is identical to the LSTM search: hidden, lr, batch_size, dropout. Lookback fixed at 168h.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error

optuna.logging.set_verbosity(optuna.logging.WARNING)

STATIONS = ["Bhaktapur", "Khumaltar", "Ratnapark", "Shankhapark"]
FORECAST_HORIZONS = [24, 48, 72]
OUT_DIR = Path("processed")
device = torch.device("cpu")

N_TRIALS = 12
SEARCH_MAX_EPOCHS, SEARCH_FLOOR_EPOCHS, SEARCH_PATIENCE = 20, 3, 3
FINAL_MAX_EPOCHS, FINAL_FLOOR_EPOCHS, FINAL_PATIENCE = 300, 5, 5
NS, NH = len(STATIONS), len(FORECAST_HORIZONS)


def load_union(target="y_pm25"):
    """Align the four per-station sequence files on anchor timestamp.

    Returns X [N, NS, 168, F], y [N, NS, NH], mask [N, NS] (station has a sequence at that anchor), split [N].
    Missing station-anchors are zero-filled and masked out of the graph and the loss.
    """
    D = {s: np.load(OUT_DIR / f"sequences_{s}.npz", allow_pickle=True) for s in STATIONS}
    anchors = np.array(sorted(set().union(*[set(D[s]["anchor_timestamp"]) for s in STATIONS])))
    pos = {a: i for i, a in enumerate(anchors)}
    N = len(anchors)
    L, F = D[STATIONS[0]]["X"].shape[1:]
    X = np.zeros((N, NS, L, F), dtype=np.float32)
    y = np.zeros((N, NS, NH), dtype=np.int64 if target == "y_class" else np.float32)
    mask = np.zeros((N, NS), dtype=bool)
    split = np.empty(N, dtype="<U5")
    for k, s in enumerate(STATIONS):
        idx = np.array([pos[a] for a in D[s]["anchor_timestamp"]])
        X[idx, k] = D[s]["X"]
        y[idx, k] = D[s][target]
        mask[idx, k] = True
        for i, sp in zip(idx, D[s]["split"]):
            assert split[i] in ("", sp), "split disagrees across stations at same anchor"
            split[i] = sp
    return torch.tensor(X), torch.tensor(y), torch.tensor(mask), split, F


def load_adjacency():
    A = pd.read_csv(OUT_DIR / "adjacency_train_corr.csv", index_col=0).loc[STATIONS, STATIONS].values
    return torch.tensor(A, dtype=torch.float32)


class GNNLSTM(nn.Module):
    def __init__(self, n_feat, hidden, dropout, A, n_out=1):
        super().__init__()
        self.n_out = n_out
        self.register_buffer("A", A)
        self.lstm = nn.LSTM(input_size=n_feat, hidden_size=hidden, num_layers=1, batch_first=True)
        self.gc = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.ModuleList([nn.Linear(2 * hidden, n_out) for _ in range(NH)]) for _ in range(NS)])

    def forward(self, x, m):
        B = x.shape[0]
        _, (h_n, _) = self.lstm(x.reshape(B * NS, *x.shape[2:]))
        h = h_n[-1].reshape(B, NS, -1)
        w = self.A.unsqueeze(0) * m.unsqueeze(1).float()  # only present neighbours contribute
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-8)
        z = torch.relu(self.gc(torch.bmm(w, h)))
        h = self.drop(torch.cat([h, z], dim=-1))
        if self.n_out > 1:  # classification logits [B,NS,NH,C]
            return torch.stack([torch.stack([head(h[:, k]) for head in self.heads[k]], dim=1) for k in range(NS)], dim=1)
        return torch.stack([torch.cat([head(h[:, k]) for head in self.heads[k]], dim=-1) for k in range(NS)], dim=1)  # [B,NS,NH]


def macro_val_mae(model, X, y, m):
    model.eval()
    with torch.no_grad():
        preds = torch.cat([model(X[i:i + 512], m[i:i + 512]) for i in range(0, len(X), 512)])
    model.train()
    maes = []
    for k in range(NS):
        mk = m[:, k]
        err = (preds[mk, k] - y[mk, k]).abs().mean(0)  # per horizon
        maes.append(err.mean().item())
    return float(np.mean(maes))


def run_training(data, hidden, lr, batch_size, dropout, max_epochs, floor_epochs, patience, trial=None, seed=42, verbose=False):
    (X_tr, y_tr, m_tr), (X_va, y_va, m_va), n_feat, A = data
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GNNLSTM(n_feat, hidden, dropout, A).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(X_tr)
    best_mae, best_state, no_improve = float("inf"), None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb, mb = X_tr[idx], y_tr[idx], m_tr[idx]
            opt.zero_grad()
            pred = model(xb, mb)
            se = (pred - yb) ** 2 * mb.unsqueeze(-1)
            loss = (se.sum(dim=(0, 1)) / mb.sum().clamp_min(1)).sum()  # MSE per horizon, summed over horizons
            loss.backward()
            opt.step()
        vmae = macro_val_mae(model, X_va, y_va, m_va)
        if verbose:
            print(f"  epoch {epoch:3d}  val_MAE {vmae:.4f}", flush=True)
        if vmae < best_mae - 1e-4:
            best_mae, best_state, no_improve = vmae, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
        if trial is not None:
            trial.report(vmae, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if epoch >= floor_epochs and no_improve >= patience:
            if verbose:
                print(f"  early stop at epoch {epoch} (best val_MAE {best_mae:.4f})", flush=True)
            break
    return best_mae, best_state, epoch


def get_data(target="y_pm25"):
    X, y, mask, split, F = load_union(target)
    A = load_adjacency()
    sel = lambda sp: torch.tensor(split == sp)
    out = {sp: (X[sel(sp)], y[sel(sp)], mask[sel(sp)]) for sp in ("train", "val", "test")}
    print({sp: len(v[0]) for sp, v in out.items()}, "mask per station (train):", out["train"][2].sum(0).tolist(), flush=True)
    return out, F, A


def stage_search():
    d, F, A = get_data()
    data = (d["train"], d["val"], F, A)

    def objective(trial):
        hidden = trial.suggest_categorical("hidden", [32, 64, 128])
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        t0 = time.time()
        best, _, ep = run_training(data, hidden, lr, batch_size, dropout,
                                   SEARCH_MAX_EPOCHS, SEARCH_FLOOR_EPOCHS, SEARCH_PATIENCE, trial=trial)
        print(f"  trial {trial.number}: val_MAE {best:.4f} epochs {ep} {time.time() - t0:.0f}s", flush=True)
        return best

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=4))
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS)
    print(f"GNN-LSTM: best val_MAE {study.best_value:.4f} params {study.best_params} ({time.time() - t0:.0f}s)", flush=True)
    study.trials_dataframe().to_csv(OUT_DIR / "optuna_gnn_trials.csv", index=False)
    with open(OUT_DIR / "optuna_gnn_best_params.json", "w") as f:
        json.dump({"best_val_mae": study.best_value, **study.best_params}, f, indent=2)


def test_metrics(model, X, y, m):
    model.eval()
    with torch.no_grad():
        preds = torch.cat([model(X[i:i + 512], m[i:i + 512]) for i in range(0, len(X), 512)])
    res = {}
    for k, s in enumerate(STATIONS):
        mk = m[:, k]
        res[s] = {}
        for h, hz in enumerate(FORECAST_HORIZONS):
            p, t = preds[mk, k, h].numpy(), y[mk, k, h].numpy()
            res[s][str(hz)] = {"mae": float(mean_absolute_error(t, p)),
                               "rmse": float(np.sqrt(mean_squared_error(t, p))), "support": int(mk.sum())}
    return res


def stage_final():
    d, F, A = get_data()
    with open(OUT_DIR / "optuna_gnn_best_params.json") as f:
        p = json.load(f)
    print("GNN-LSTM final params", p, flush=True)
    best, state, ep = run_training((d["train"], d["val"], F, A), p["hidden"], p["lr"], p["batch_size"], p["dropout"],
                                   FINAL_MAX_EPOCHS, FINAL_FLOOR_EPOCHS, FINAL_PATIENCE, verbose=True)
    model = GNNLSTM(F, p["hidden"], p["dropout"], A)
    model.load_state_dict(state)
    torch.save(model.state_dict(), OUT_DIR / "gnn_lstm.pt")
    res = test_metrics(model, *d["test"])
    with open(OUT_DIR / "gnn_lstm_eval.json", "w") as f:
        json.dump({"stopped_epoch": ep, "best_val_mae": best, "results": res}, f, indent=2)

    rows = []
    for s in STATIONS:
        base = json.load(open(OUT_DIR / f"reg_{s}_eval.json"))
        tuned_path = OUT_DIR / f"reg_{s}_tuned300_eval.json"
        if not tuned_path.exists():
            tuned_path = OUT_DIR / f"reg_{s}_tuned_eval.json"
        tuned = json.load(open(tuned_path))
        for hz in FORECAST_HORIZONS:
            g = res[s][str(hz)]
            rows.append({"station": s, "horizon": f"+{hz}h",
                         "mae_lstm_base": base[str(hz)]["mae"], "mae_lstm_tuned": tuned[str(hz)]["mae"], "mae_gnn": g["mae"],
                         "rmse_lstm_base": base[str(hz)]["rmse"], "rmse_lstm_tuned": tuned[str(hz)]["rmse"], "rmse_gnn": g["rmse"],
                         "support": g["support"], "support_lstm": base[str(hz)]["support"]})
    df = pd.DataFrame(rows)
    df["mae_gnn_vs_tuned"] = df.mae_gnn - df.mae_lstm_tuned
    df["rmse_gnn_vs_tuned"] = df.rmse_gnn - df.rmse_lstm_tuned
    df.round(3).to_json(OUT_DIR / "gnn_vs_lstm_summary.json", orient="records", indent=2)
    print(df.round(3).to_string(), flush=True)


def stage_lstm300():
    """Retrain tuned per-station LSTM under the GNN final regime (cap 300, floor 5, patience 5)."""
    with open(OUT_DIR / "optuna_reg_best_params.json") as f:
        bp = json.load(f)

    class RegHeadTuned(nn.Module):
        def __init__(self, n_feat, hidden, n_h, dropout):
            super().__init__()
            self.lstm = nn.LSTM(input_size=n_feat, hidden_size=hidden, num_layers=1, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(n_h)])

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            h = self.drop(h_n[-1])
            return [head(h).squeeze(-1) for head in self.heads]

    seeds = [int(x) for x in sys.argv[2:]] or [42]
    for seed, s in [(sd, st) for sd in seeds for st in STATIONS]:
        dd = np.load(OUT_DIR / f"sequences_{s}.npz", allow_pickle=True)
        sp = dd["split"]
        T = {k: (torch.tensor(dd["X"][sp == k], dtype=torch.float32), torch.tensor(dd["y_pm25"][sp == k], dtype=torch.float32))
             for k in ("train", "val", "test")}
        p = bp[s]
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = RegHeadTuned(dd["X"].shape[2], p["hidden"], NH, p["dropout"])
        opt = torch.optim.Adam(model.parameters(), lr=p["lr"])
        X_tr, y_tr = T["train"]
        best, state, nimp = float("inf"), None, 0
        for epoch in range(1, FINAL_MAX_EPOCHS + 1):
            model.train()
            perm = torch.randperm(len(X_tr))
            for i in range(0, len(X_tr), p["batch_size"]):
                idx = perm[i:i + p["batch_size"]]
                opt.zero_grad()
                pr = model(X_tr[idx])
                loss = sum(nn.functional.mse_loss(pr[h], y_tr[idx][:, h]) for h in range(NH))
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                pv = model(T["val"][0])
                v = float(np.mean([mean_absolute_error(T["val"][1][:, h].numpy(), pv[h].numpy()) for h in range(NH)]))
            if v < best - 1e-4:
                best, state, nimp = v, {k: x.clone() for k, x in model.state_dict().items()}, 0
            else:
                nimp += 1
            if epoch >= FINAL_FLOOR_EPOCHS and nimp >= FINAL_PATIENCE:
                break
        model.load_state_dict(state)
        model.eval()
        res = {}
        with torch.no_grad():
            pt = model(T["test"][0])
            for h, hz in enumerate(FORECAST_HORIZONS):
                a, b = T["test"][1][:, h].numpy(), pt[h].numpy()
                res[str(hz)] = {"mae": float(mean_absolute_error(a, b)), "rmse": float(np.sqrt(mean_squared_error(a, b))), "support": int(len(a))}
        with open(OUT_DIR / (f"reg_{s}_tuned300_eval.json" if seed == 42 else f"reg_{s}_tuned300_seed{seed}_eval.json"), "w") as f:
            json.dump(res, f, indent=2)
        print(s, "stopped epoch", epoch, "val", round(best, 4), {k: round(v["mae"], 3) for k, v in res.items()}, flush=True)


CLF_HIDDEN, CLF_LR, CLF_BATCH, CLF_DROPOUT = 64, 1e-3, 128, 0.0  # same untuned config as the notebook LSTM classifier
CLF_MAX_EPOCHS, CLF_FLOOR, CLF_PATIENCE = 40, 5, 5
NC = 4


def clf_report(true, pred):
    """true/pred [n, NH] in 0..3 -> per horizon macro-F1, per-class recall, per-class support."""
    out = {}
    for h, hz in enumerate(FORECAST_HORIZONS):
        t, p = true[:, h], pred[:, h]
        f1 = f1_score(t, p, average="macro", labels=list(range(NC)), zero_division=0)
        recall = [float(((p == c) & (t == c)).sum() / max((t == c).sum(), 1)) for c in range(NC)]
        out[str(hz)] = {"macro_f1": float(f1), "recall": recall, "support": [int((t == c).sum()) for c in range(NC)]}
    return out


class LSTMClf(nn.Module):
    def __init__(self, n_feat, hidden):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_feat, hidden_size=hidden, num_layers=1, batch_first=True)
        self.heads = nn.ModuleList([nn.Linear(hidden, NC) for _ in range(NH)])

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return torch.stack([head(h_n[-1]) for head in self.heads], dim=1)  # [B,NH,C]


def clf_loss(logits, y, w):
    """Class-weighted CE (torch weighted-mean convention) per horizon, summed over horizons. logits [n,NH,C], y [n,NH]."""
    ce = nn.functional.cross_entropy(logits.reshape(-1, NC), y.reshape(-1), weight=w, reduction="none").reshape(y.shape)
    wt = w[y]
    return (ce.sum(0) / wt.sum(0)).sum()


def predict_clf(fwd, model, X, m):
    model.eval()
    with torch.no_grad():
        lg = torch.cat([fwd(model, X[i:i + 512], None if m is None else m[i:i + 512]) for i in range(0, len(X), 512)])
    model.train()
    return lg.argmax(-1).numpy()


def clf_val_f1(fwd, model, va):
    X, y, m = va
    pred, y = predict_clf(fwd, model, X, m), y.numpy()
    if m is None:
        f1s = [f1_score(y[:, h], pred[:, h], average="macro", labels=list(range(NC)), zero_division=0) for h in range(NH)]
    else:
        m = m.numpy()
        f1s = [f1_score(y[m[:, k], k, h], pred[m[:, k], k, h], average="macro", labels=list(range(NC)), zero_division=0)
               for k in range(NS) for h in range(NH)]
    return float(np.mean(f1s))


def train_clf(fwd, model, tr, va, w):
    """tr/va = (X, y, mask-or-None). Weighted CE, best checkpoint by val macro-F1, floor/patience early stop."""
    opt = torch.optim.Adam(model.parameters(), lr=CLF_LR)
    X, y, m = tr
    best, best_state, nimp = -1.0, None, 0
    for epoch in range(1, CLF_MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), CLF_BATCH):
            idx = perm[i:i + CLF_BATCH]
            lg, yb = fwd(model, X[idx], None if m is None else m[idx]), y[idx]
            if m is not None:
                keep = m[idx]  # drop absent station-anchors
                lg, yb = lg[keep], yb[keep]
            opt.zero_grad()
            clf_loss(lg, yb, w).backward()
            opt.step()
        vf1 = clf_val_f1(fwd, model, va)
        if vf1 > best + 1e-4:
            best, best_state, nimp = vf1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            nimp += 1
        if epoch >= CLF_FLOOR and nimp >= CLF_PATIENCE:
            break
    model.load_state_dict(best_state)
    return best, epoch


def stage_clf():
    seeds = [int(x) for x in sys.argv[2:]] or [42, 43, 44]
    cw = json.load(open(OUT_DIR / "class_weights.json"))
    w = torch.tensor([cw[str(c)] for c in range(1, NC + 1)], dtype=torch.float32)
    d, F, A = get_data("y_class")
    d = {sp: (X, y - 1, m) for sp, (X, y, m) in d.items()}  # classes 1..4 -> 0..3
    gnn_fwd = lambda mod, X, m: mod(X, m)
    lstm_fwd = lambda mod, X, m: mod(X)
    seq = {s: np.load(OUT_DIR / f"sequences_{s}.npz", allow_pickle=True) for s in STATIONS}
    for seed in seeds:
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        g = GNNLSTM(F, CLF_HIDDEN, CLF_DROPOUT, A, n_out=NC)
        best, ep = train_clf(gnn_fwd, g, d["train"], d["val"], w)
        pred = predict_clf(gnn_fwd, g, d["test"][0], d["test"][2])
        yt, mt = d["test"][1].numpy(), d["test"][2].numpy()
        res = {s: clf_report(yt[mt[:, k], k], pred[mt[:, k], k]) for k, s in enumerate(STATIONS)}
        json.dump({"seed": seed, "stopped_epoch": ep, "best_val_macro_f1": best, "results": res},
                  open(OUT_DIR / f"clf_cmp_gnn_seed{seed}.json", "w"), indent=2)
        print(f"GNN seed {seed}: epoch {ep} val_F1 {best:.4f} ({time.time() - t0:.0f}s)", flush=True)

        res, info = {}, {}
        for s in STATIONS:
            sp = seq[s]["split"]
            T = {k: (torch.tensor(seq[s]["X"][sp == k], dtype=torch.float32), torch.tensor(seq[s]["y_class"][sp == k] - 1), None)
                 for k in ("train", "val", "test")}
            torch.manual_seed(seed)
            np.random.seed(seed)
            lm = LSTMClf(seq[s]["X"].shape[2], CLF_HIDDEN)
            b, e = train_clf(lstm_fwd, lm, T["train"], T["val"], w)
            p = predict_clf(lstm_fwd, lm, T["test"][0], None)
            res[s], info[s] = clf_report(T["test"][1].numpy(), p), {"stopped_epoch": e, "best_val_macro_f1": b}
        json.dump({"seed": seed, "info": info, "results": res}, open(OUT_DIR / f"clf_cmp_lstm_seed{seed}.json", "w"), indent=2)
        print(f"LSTM seed {seed}: {info} ({time.time() - t0:.0f}s total)", flush=True)

    rows = []
    for s in STATIONS:
        for hz in FORECAST_HORIZONS:
            row = {"station": s, "horizon": f"+{hz}h"}
            for name in ("lstm", "gnn"):
                rs = [json.load(open(OUT_DIR / f"clf_cmp_{name}_seed{sd}.json"))["results"][s][str(hz)] for sd in seeds]
                f1, r4 = [r["macro_f1"] for r in rs], [r["recall"][3] for r in rs]
                row.update({f"f1_{name}": np.mean(f1), f"f1_{name}_sd": np.std(f1),
                            f"c4rec_{name}": np.mean(r4), f"c4rec_{name}_sd": np.std(r4)})
            row["c4_support"] = rs[0]["support"][3]
            rows.append(row)
    df = pd.DataFrame(rows)
    df["f1_gnn_minus_lstm"] = df.f1_gnn - df.f1_lstm
    df.round(3).to_json(OUT_DIR / "clf_gnn_vs_lstm_summary.json", orient="records", indent=2)
    print(df.round(3).to_string(), flush=True)
    print(df.groupby("horizon")[["f1_lstm", "f1_gnn", "c4rec_lstm", "c4rec_gnn"]].mean().round(3).to_string())


def stage_ablate_reg():
    """Graph-removed control: same GNNLSTM, A=identity (each station mixes only with itself). Paired seeds, tuned params."""
    seeds = [int(x) for x in sys.argv[2:]] or [42, 43, 44]
    d, F, A = get_data()
    p = json.load(open(OUT_DIR / "optuna_gnn_best_params.json"))
    arms = {"graph": A, "nograph": torch.eye(NS)}
    rows = []
    for seed in seeds:
        for arm, Aa in arms.items():
            t0 = time.time()
            best, state, ep = run_training((d["train"], d["val"], F, Aa), p["hidden"], p["lr"], p["batch_size"], p["dropout"],
                                           FINAL_MAX_EPOCHS, FINAL_FLOOR_EPOCHS, FINAL_PATIENCE, seed=seed)
            model = GNNLSTM(F, p["hidden"], p["dropout"], Aa)
            model.load_state_dict(state)
            res = test_metrics(model, *d["test"])
            json.dump({"seed": seed, "arm": arm, "stopped_epoch": ep, "best_val_mae": best, "results": res},
                      open(OUT_DIR / f"ablate_reg_{arm}_seed{seed}.json", "w"), indent=2)
            print(f"seed {seed} {arm}: epoch {ep} val_MAE {best:.3f} ({time.time() - t0:.0f}s)", flush=True)
            for s in STATIONS:
                for hz in FORECAST_HORIZONS:
                    rows.append({"seed": seed, "arm": arm, "station": s, "horizon": f"+{hz}h",
                                 "mae": res[s][str(hz)]["mae"], "rmse": res[s][str(hz)]["rmse"]})
    df = pd.DataFrame(rows)
    g = df.groupby(["station", "horizon", "arm"])[["mae", "rmse"]].agg(["mean", "std"]).round(3)
    print(g.to_string(), flush=True)
    piv = df.pivot_table(index=["station", "horizon", "seed"], columns="arm", values="mae")
    piv["nograph_minus_graph"] = piv.nograph - piv.graph
    out = piv.groupby(["station", "horizon"])["nograph_minus_graph"].agg(["mean", "std"]).round(3)
    print(out.to_string(), flush=True)
    print("overall mean MAE per arm:", df.groupby("arm").mae.mean().round(3).to_dict(), flush=True)
    df.round(4).to_json(OUT_DIR / "ablate_reg_summary.json", orient="records", indent=2)


def stage_ablate_clf():
    """Graph-removed control for classification, same regime as stage_clf."""
    seeds = [int(x) for x in sys.argv[2:]] or [42, 43, 44]
    cw = json.load(open(OUT_DIR / "class_weights.json"))
    w = torch.tensor([cw[str(c)] for c in range(1, NC + 1)], dtype=torch.float32)
    d, F, A = get_data("y_class")
    d = {sp: (X, y - 1, m) for sp, (X, y, m) in d.items()}
    fwd = lambda mod, X, m: mod(X, m)
    rows = []
    for seed in seeds:
        for arm, Aa in {"graph": A, "nograph": torch.eye(NS)}.items():
            torch.manual_seed(seed)
            np.random.seed(seed)
            g = GNNLSTM(F, CLF_HIDDEN, CLF_DROPOUT, Aa, n_out=NC)
            best, ep = train_clf(fwd, g, d["train"], d["val"], w)
            pred = predict_clf(fwd, g, d["test"][0], d["test"][2])
            yt, mt = d["test"][1].numpy(), d["test"][2].numpy()
            for k, s in enumerate(STATIONS):
                r = clf_report(yt[mt[:, k], k], pred[mt[:, k], k])
                for hz in FORECAST_HORIZONS:
                    rows.append({"seed": seed, "arm": arm, "station": s, "horizon": f"+{hz}h",
                                 "macro_f1": r[str(hz)]["macro_f1"], "c4_recall": r[str(hz)]["recall"][3]})
            print(f"seed {seed} {arm}: epoch {ep} val_F1 {best:.4f}", flush=True)
    df = pd.DataFrame(rows)
    print(df.groupby(["horizon", "arm"])[["macro_f1", "c4_recall"]].mean().round(3).to_string(), flush=True)
    df.round(4).to_json(OUT_DIR / "ablate_clf_summary.json", orient="records", indent=2)


def stage_smoke():
    d, F, A = get_data()
    sub = lambda t, n: tuple(x[:n] for x in t)
    t0 = time.time()
    best, _, ep = run_training((sub(d["train"], 1024), sub(d["val"], 256), F, A), 64, 1e-3, 64, 0.1, 1, 1, 1, verbose=True)
    print(f"smoke: 1024 train samples, 1 epoch {time.time() - t0:.1f}s  val_MAE {best:.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or f"stage_{sys.argv[1]}" not in globals():
        sys.exit("usage: python gnn_lstm.py <smoke|search|final|lstm300|clf|ablate_reg|ablate_clf> [seeds]")
    globals()[f"stage_{sys.argv[1]}"]()
