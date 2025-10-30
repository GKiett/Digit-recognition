import argparse, yaml, torch
from torch import nn
from src.utils import set_seed, get_device, accuracy
from src.data import get_dataloaders
from src.model import build_model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    return p.parse_args()

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    set_seed(cfg["seed"])
    device = get_device()

    _, _, test_loader = get_dataloaders(cfg)
    model = build_model(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model_state"])

    crit = nn.CrossEntropyLoss()
    model.eval()
    total_loss, total_acc, total_n = 0.0, 0.0, 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = crit(out, y)
            bs = x.size(0)
            total_loss += loss.item() * bs
            total_acc += accuracy(out, y) * bs
            total_n += bs

    print(f"test_loss={total_loss/total_n:.4f}  test_acc={total_acc/total_n:.4f}")

if __name__ == "__main__":
    main()
