import os, argparse, yaml, torch
from torch import nn, optim
from torch.amp import GradScaler, autocast
from src.utils import set_seed, get_device, save_checkpoint, accuracy
from src.data import get_dataloaders
from src.model import build_model

def parse_cfg():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    return yaml.safe_load(open(p.parse_args().config, "r"))

def make_objects(cfg, device):
    model = build_model(cfg).to(device)
    tr = cfg["train"]
    opt = optim.Adam(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    crit = nn.CrossEntropyLoss(label_smoothing=tr.get("label_smoothing", 0.0))
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=tr["scheduler"]["factor"],
                                               patience=tr["scheduler"]["patience"])
    return model, opt, crit, sch

def run_epoch(loader, model, crit, opt, device, scaler=None, train=True):
    model.train(train)
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            opt.zero_grad(set_to_none=True)
            if scaler:
                with autocast():
                    out = model(x)
                    loss = crit(out, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                out = model(x)
                loss = crit(out, y)
                loss.backward()
                opt.step()
        else:
            with torch.no_grad():
                out = model(x)
                loss = crit(out, y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(out, y) * bs
        total_n += bs
    return total_loss / total_n, total_acc / total_n

def main():
    cfg = parse_cfg()
    set_seed(cfg["seed"])
    device = get_device()
    train_loader, val_loader, _ = get_dataloaders(cfg)
    model, opt, crit, sch = make_objects(cfg, device)
    scaler = GradScaler(enabled=cfg["train"]["mixed_precision"])
    save_dir = cfg["eval"]["save_dir"]; os.makedirs(save_dir, exist_ok=True)
    best_val, bad_epochs = float("inf"), 0
    max_epochs = cfg["train"]["max_epochs"]
    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_acc = run_epoch(train_loader, model, crit, opt, device, scaler, train=True)
        val_loss, val_acc = run_epoch(val_loader, model, crit, opt, device, scaler=None, train=False)
        sch.step(val_loss)
        if val_loss < best_val:
            best_val, bad_epochs = val_loss, 0
            save_checkpoint(model, opt, epoch, os.path.join(save_dir, "best.ckpt"))
        else:
            bad_epochs += 1
        es = cfg["train"]["early_stopping"]
        print(f"Epoch {epoch:02d}: train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")
        if bad_epochs > es["patience"]:
            break

if __name__ == "__main__":
    main()
