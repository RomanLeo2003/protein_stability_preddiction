import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import (
    train_epoch,
    val_epoch,
    compute_metrics,
    add_tensorboard_metrics,
    build_abyssal_model,
)
from tqdm import tqdm
from data_processing import load_data


@hydra.main(version_base=None, config_path="./config", config_name="abyssal")
def train(cfg: DictConfig):
    train_loader, test_loader = load_data(
        train_csv_filename=cfg.train_dataset,
        val_csv_filename=cfg.valid_dataset,
        tokenizer_name=cfg.model.esm_model_name,
        batch_size=cfg.training.batch_size,
    )
    writer = SummaryWriter(log_dir=cfg.training.log_dir)
    EPOCHS = cfg.training.epochs
    DEVICE = cfg.model.device

    print(OmegaConf.to_yaml(cfg))

    model = build_abyssal_model(cfg.model)
    optimizer = getattr(torch.optim, cfg.optimizer.type)(
        model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay
    )
    criterion = getattr(torch.nn, cfg.criterion.type)()

    model.to(DEVICE)
    for epoch in tqdm(range(EPOCHS)):
        print(f"EPOCH {epoch + 1}:")
        model.train(True)
        avg_loss, train_targets, train_preds = train_epoch(
            model, optimizer, criterion, train_loader, DEVICE
        )
        model.eval()

        val_avg_loss, val_targets, val_preds = val_epoch(
            model, criterion, test_loader, DEVICE
        )

        val_metrics = compute_metrics(
            val_targets.cpu().squeeze().numpy(), val_preds.cpu().squeeze().numpy()
        )

        train_metrics = compute_metrics(
            train_targets.detach().cpu().squeeze().numpy(),
            train_preds.detach().cpu().squeeze().numpy(),
        )
        add_tensorboard_metrics(
            writer, epoch, avg_loss, val_avg_loss, train_metrics, val_metrics
        )

        if epoch % 10 == 9:
            print(val_metrics)

    writer.close()


if __name__ == "__main__":
    train()
