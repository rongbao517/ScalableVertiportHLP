import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import Metrics
from uam_airspace_context_od_v2 import ASTT

PROJECT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUT_DIR = PROJECT_DIR / "outputs"

CONTEXT_CHANNEL_NAMES = [
    "is_holiday_or_weekend",
    "temp_max_c",
    "temp_min_c",
    "precip_mm",
    "windspeed_max_kmh",
    "humidity_pct",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the OD-preserving ASTT model (ported from UAM_demand/src_repo, "
        "best NYC architecture: test log-MSE 0.2921) on the 30 kmeans Shanghai sites."
    )
    parser.add_argument("--dataname", type=str, default="shanghai_kmeans30")
    parser.add_argument("--timestep_in", type=int, default=12)
    parser.add_argument("--timestep_out", type=int, default=1)
    parser.add_argument("--n_node", type=int, default=30)
    parser.add_argument("--channel", type=int, default=7)
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "RMSprop", "Adadelta"])
    parser.add_argument("--loss", type=str, default="MSE", choices=["MSE", "MAE"])
    parser.add_argument("--trainratio", type=float, default=0.9)
    parser.add_argument("--trainvalsplit", type=float, default=0.111)
    parser.add_argument("--flowpath", type=str, default=str(DATA_DIR / "shanghai_od_kmeans30_context7ch.npz"))
    parser.add_argument("--adjpath", type=str, default=str(OUT_DIR / "shanghai_adj_matrix_kmeans30.csv"))
    parser.add_argument("--corridor_mask", type=str, default="")
    parser.add_argument("--restricted_mask", type=str, default="")
    parser.add_argument("--gpu", type=int, default=0)
    # Smaller capacity than the NYC defaults (num_heads=8, head_dim=8 -> hidden=64):
    # Shanghai has ~1 month of data (1392 bins) vs NYC's much longer history, and
    # edge_embedding/edge_context_embedding alone cost N*N*hidden params each
    # (900*hidden) -- keep hidden_dim modest to avoid repeating the earlier
    # Shanghai OD attempt's failure (uam_demand_model/, test R2=0.19, underfit
    # from too little data for an NYC-sized architecture).
    parser.add_argument("--num_blocks", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--head_dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--allow_self_loop", action="store_true")
    parser.add_argument("--allow_negative_output", action="store_true")
    parser.add_argument("--no_context_standardize", action="store_true")
    parser.add_argument("--disable_context_graph", action="store_true")
    parser.add_argument("--disable_context_message", action="store_true")
    parser.add_argument("--disable_context_decoder", action="store_true")
    parser.add_argument("--disable_context_film", action="store_true")
    return parser.parse_args()


def set_threads():
    cpu_num = 1
    os.environ["OMP_NUM_THREADS"] = str(cpu_num)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
    os.environ["MKL_NUM_THREADS"] = str(cpu_num)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
    torch.set_num_threads(cpu_num)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def read_matrix(path):
    matrix = np.genfromtxt(path, delimiter=",", dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square adjacency/mask matrix, got {matrix.shape} from {path}")
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_directed_prior(adj_matrix, allow_self=False):
    prior = np.asarray(adj_matrix, dtype=np.float32).copy()
    prior = np.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0)
    prior[prior < 0] = 0.0
    if not allow_self:
        np.fill_diagonal(prior, 0.0)

    row_sum = prior.sum(axis=1, keepdims=True)
    empty = row_sum[:, 0] <= 1e-8
    if np.any(empty):
        fallback = np.ones_like(prior, dtype=np.float32)
        if not allow_self:
            np.fill_diagonal(fallback, 0.0)
        prior[empty] = fallback[empty]
        row_sum = prior.sum(axis=1, keepdims=True)

    return prior / np.maximum(row_sum, 1e-8)


def load_optional_mask(path, n_node):
    if not path:
        return None
    mask = read_matrix(path)
    if mask.shape != (n_node, n_node):
        raise ValueError(f"Mask shape {mask.shape} does not match n_node={n_node}: {path}")
    return torch.tensor((mask > 0).astype(np.float32))


def standardize_context_channels(data, channel, trainratio):
    stats = []
    train_num = int(data.shape[1] * trainratio)
    upper = min(channel, data.shape[0])
    for channel_index in range(1, upper):
        values = data[channel_index, :train_num, :, :]
        mean = float(values.mean())
        std = float(values.std())
        if std < 1e-6:
            std = 1.0
        data[channel_index] = (data[channel_index] - mean) / std
        stats.append(
            {
                "channel": channel_index,
                "name": CONTEXT_CHANNEL_NAMES[channel_index - 1]
                if channel_index - 1 < len(CONTEXT_CHANNEL_NAMES)
                else f"context_{channel_index}",
                "mean": mean,
                "std": std,
            }
        )
    return stats


def load_data(flowpath, channel, trainratio, standardize_context=True):
    data = np.load(flowpath)["arr_0"].astype(np.float32)
    if channel > data.shape[0]:
        raise ValueError(f"Requested channel={channel}, but data only has {data.shape[0]} channels")
    data[0] = np.log1p(data[0])
    context_stats = []
    if standardize_context and channel > 1:
        context_stats = standardize_context_channels(data, channel, trainratio)
    return data, context_stats


def get_xsys(data, mode, trainratio, timestep_in, timestep_out, channel):
    train_num = int(data.shape[1] * trainratio)
    xs, ys = [], []
    if mode == "TRAIN":
        index_range = range(train_num - timestep_out - timestep_in + 1)
    elif mode == "TEST":
        index_range = range(train_num - timestep_in, data.shape[1] - timestep_out - timestep_in + 1)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    for i in index_range:
        x = data[0:channel, i : i + timestep_in, :, :]
        y = data[0, i + timestep_in + timestep_out - 1 : i + timestep_in + timestep_out, :, :]
        xs.append(x)
        ys.append(y)

    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    if xs.ndim == 4:
        xs = xs[:, np.newaxis, :, :, :]
        ys = ys[:, np.newaxis, :, :]

    xs = xs.transpose(0, 3, 4, 1, 2)
    ys = ys.transpose(0, 2, 3, 1)
    return xs, ys


def make_model(args, device):
    adj = read_matrix(args.adjpath)
    if adj.shape != (args.n_node, args.n_node):
        raise ValueError(f"Adjacency shape {adj.shape} does not match n_node={args.n_node}")

    graph_prior = torch.tensor(build_directed_prior(adj, args.allow_self_loop), dtype=torch.float32, device=device)
    corridor_mask = load_optional_mask(args.corridor_mask, args.n_node)
    restricted_mask = load_optional_mask(args.restricted_mask, args.n_node)
    if corridor_mask is not None:
        corridor_mask = corridor_mask.to(device)
    if restricted_mask is not None:
        restricted_mask = restricted_mask.to(device)

    return ASTT(
        graph_prior,
        [args.num_blocks, args.num_heads, args.head_dim],
        [args.timestep_in, args.n_node, args.channel],
        args.dropout,
        out_steps=1,
        corridor_mask=corridor_mask,
        restricted_mask=restricted_mask,
        allow_self=args.allow_self_loop,
        nonnegative_output=not args.allow_negative_output,
        use_context_graph=not args.disable_context_graph,
        use_context_message=not args.disable_context_message,
        use_context_decoder=not args.disable_context_decoder,
        use_context_film=not args.disable_context_film,
    ).to(device)


def make_criterion(loss_name):
    if loss_name == "MSE":
        return nn.MSELoss()
    if loss_name == "MAE":
        return nn.L1Loss()
    raise ValueError(f"Unsupported loss: {loss_name}")


def make_optimizer(args, model):
    if args.optimizer == "RMSprop":
        return torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.001)
    if args.optimizer == "Adadelta":
        return torch.optim.Adadelta(model.parameters(), lr=args.lr)
    return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)


def make_loader(xs, ys, batchsize, shuffle, seed):
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(xs), torch.from_numpy(ys))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(dataset, batchsize, shuffle=shuffle, generator=generator)


def evaluate_model(model, criterion, data_iter, device):
    model.eval()
    loss_sum, n_obs = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss_sum += loss.item() * y.shape[0]
            n_obs += y.shape[0]
    return loss_sum / max(n_obs, 1)


def predict_model(model, data_iter, device):
    model.eval()
    predictions = []
    dynamic_adj_sum = None
    dynamic_adj_count = 0
    with torch.no_grad():
        for x, _y in data_iter:
            x = x.to(device, non_blocking=True)
            pred = model(x)
            predictions.append(pred.cpu().numpy())
            if model.latest_dynamic_adj is not None:
                batch_adj = model.latest_dynamic_adj.cpu()
                batch_sum = batch_adj.sum(dim=0)
                dynamic_adj_sum = batch_sum if dynamic_adj_sum is None else dynamic_adj_sum + batch_sum
                dynamic_adj_count += batch_adj.shape[0]

    pred_np = np.vstack(predictions)
    if dynamic_adj_sum is None:
        return pred_np, None
    return pred_np, (dynamic_adj_sum / max(dynamic_adj_count, 1)).numpy()


def metric_bundle(y_true_log, y_pred_log):
    y_true_log = np.squeeze(y_true_log).copy()
    y_pred_log = np.squeeze(y_pred_log).copy()
    log_metrics = Metrics.evaluate(y_true_log.copy(), y_pred_log.copy())

    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    y_pred[y_pred < 0] = 0
    original_metrics = Metrics.evaluate(y_true.copy(), y_pred.copy())
    return log_metrics, original_metrics


def write_metrics(path, name, mode, torch_score, log_metrics, original_metrics):
    with open(path / f"{name}_prediction_scores.txt", "a", encoding="utf-8") as f:
        f.write(f"{name}, {mode}, Torch {mode} loss, {torch_score:.10e}, {torch_score:.10f}\n")
        f.write(
            "%s, %s, log-space MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n"
            % ((name, mode) + tuple(log_metrics))
        )
        f.write(
            "%s, %s, original-space MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n"
            % ((name, mode) + tuple(original_metrics))
        )


def copy_sources(path):
    current = Path(sys.argv[0]).resolve()
    shutil.copy2(current, path / current.name)
    for filename in ("uam_airspace_context_od_v2.py", "uam_airspace_od.py"):
        model_src = current.parent / filename
        if model_src.exists():
            shutil.copy2(model_src, path / model_src.name)


def main():
    args = parse_args()
    set_threads()
    set_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    model_name = f"ShanghaiASTT_{args.timestep_out}"
    keyword = (
        f"pred_{args.dataname}_{model_name}_{args.channel}_{args.n_node}_"
        f"{datetime.now().strftime('%y%m%d%H%M%S')}"
    )
    save_path = OUT_DIR / "save_shanghai_od_astt" / keyword
    save_path.mkdir(parents=True, exist_ok=False)
    copy_sources(save_path)

    with open(save_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("Using device:", device)
    print("Save path:", save_path.resolve())
    print("Arguments:", args)

    data, context_stats = load_data(
        args.flowpath,
        args.channel,
        args.trainratio,
        standardize_context=not args.no_context_standardize,
    )
    with open(save_path / "context_stats.json", "w", encoding="utf-8") as f:
        json.dump(context_stats, f, indent=2)

    print("data.shape", data.shape)
    print("context_standardized", not args.no_context_standardize)
    train_xs, train_ys = get_xsys(data, "TRAIN", args.trainratio, args.timestep_in, args.timestep_out, args.channel)
    test_xs, test_ys = get_xsys(data, "TEST", args.trainratio, args.timestep_in, args.timestep_out, args.channel)
    print("TRAIN XS.shape YS.shape", train_xs.shape, train_ys.shape)
    print("TEST XS.shape YS.shape", test_xs.shape, test_ys.shape)

    trainval_size = train_xs.shape[0]
    train_size = int(trainval_size * (1 - args.trainvalsplit))
    tr_xs, val_xs = train_xs[:train_size], train_xs[train_size:]
    tr_ys, val_ys = train_ys[:train_size], train_ys[train_size:]

    train_loader = make_loader(tr_xs, tr_ys, args.batchsize, True, args.seed)
    val_loader = make_loader(val_xs, val_ys, args.batchsize, False, args.seed)
    trainval_loader = make_loader(train_xs, train_ys, args.batchsize, False, args.seed)
    test_loader = make_loader(test_xs, test_ys, args.batchsize, False, args.seed)

    model = make_model(args, device)
    n_params = sum(p.numel() for p in model.parameters())
    print("model parameters:", n_params)
    criterion = make_criterion(args.loss)
    optimizer = make_optimizer(args, model)
    best_path = save_path / f"{model_name}.pt"

    print(keyword, "training started", time.ctime())
    best_val_loss = np.inf
    wait = 0
    for epoch in range(args.epoch):
        start = datetime.now()
        model.train()
        loss_sum, n_obs = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            loss_sum += loss.item() * y.shape[0]
            n_obs += y.shape[0]

        train_loss = loss_sum / max(n_obs, 1)
        val_loss = evaluate_model(model, criterion, val_loader, device)
        if val_loss < best_val_loss:
            wait = 0
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
        else:
            wait += 1

        elapsed = (datetime.now() - start).seconds
        line = (
            f"epoch {epoch} time used: {elapsed} seconds train loss: "
            f"{train_loss:.10f} validation loss: {val_loss:.10f}"
        )
        print(line)
        with open(save_path / f"{model_name}_log.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")

        if wait >= args.patience:
            print(f"Early stopping at epoch: {epoch}")
            break

    model.load_state_dict(torch.load(best_path, map_location=device))
    train_loss = evaluate_model(model, criterion, train_loader, device)
    train_pred, train_dynamic_adj = predict_model(model, trainval_loader, device)
    train_log_metrics, train_original_metrics = metric_bundle(train_ys, train_pred)
    write_metrics(save_path, model_name, "train", train_loss, train_log_metrics, train_original_metrics)
    print("*" * 40)
    print(
        "%s, train, log-space MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f"
        % ((model_name,) + tuple(train_log_metrics))
    )
    print(
        "%s, train, original-space MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f"
        % ((model_name,) + tuple(train_original_metrics))
    )

    print(keyword, "testing started", time.ctime())
    test_loss = evaluate_model(model, criterion, test_loader, device)
    test_pred, test_dynamic_adj = predict_model(model, test_loader, device)
    test_log_metrics, test_original_metrics = metric_bundle(test_ys, test_pred)
    np.save(save_path / f"{model_name}_prediction.npy", np.squeeze(test_pred))
    np.save(save_path / f"{model_name}_groundtruth.npy", np.squeeze(test_ys))
    if train_dynamic_adj is not None:
        np.save(save_path / f"{model_name}_train_dynamic_adj_mean.npy", train_dynamic_adj)
    if test_dynamic_adj is not None:
        np.save(save_path / f"{model_name}_test_dynamic_adj_mean.npy", test_dynamic_adj)
    write_metrics(save_path, model_name, "test", test_loss, test_log_metrics, test_original_metrics)
    print("*" * 40)
    print(
        "%s, test, log-space MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f"
        % ((model_name,) + tuple(test_log_metrics))
    )
    print(
        "%s, test, original-space MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f"
        % ((model_name,) + tuple(test_original_metrics))
    )
    print("Model Testing Ended", time.ctime())
    print("Saved model weight:", best_path.resolve())


if __name__ == "__main__":
    main()
