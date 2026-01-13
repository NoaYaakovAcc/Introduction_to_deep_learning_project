import torch

def run_epoch(model, loader, optimizer, loss_fn, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total = 0
    correct = 0
    running_loss = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = loss_fn(logits, y)
            preds = torch.argmax(logits, dim=1)

            if train:
                loss.backward()
                optimizer.step()

        bs = y.size(0)
        total += bs
        correct += (preds == y).sum().item()
        running_loss += loss.item() * bs

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc
