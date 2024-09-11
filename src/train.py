import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import gc

from data_processing import load_data
from utils import (
    train_epoch,
    val_epoch,
    compute_metrics,
    add_tensorboard_metrics,
    build_abyssal_model,
)
from losses import WeightedMSELoss
import torch.optim.lr_scheduler as lr_scheduler  # Добавляем для работы с шедулером


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
    checkpoint_dir = cfg.training.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(OmegaConf.to_yaml(cfg))

    model = build_abyssal_model(cfg.model)
    optimizer = getattr(torch.optim, cfg.optimizer.type)(
        model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay
    )
    criterion = getattr(torch.nn, cfg.criterion.type)()

    # Добавляем шедулер (например, StepLR)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=cfg.scheduler.eta_min)

    model.to(DEVICE)

    # Load from checkpoint if available
    start_epoch = 0
    checkpoint_path = os.path.join(checkpoint_dir, "last_checkpoint.pth")
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch + 1}")

    for epoch in tqdm(range(start_epoch, EPOCHS)):
        print(f"TRAIN EPOCH {epoch + 1}:")
        model.train(True)
        avg_loss, train_targets, train_preds = train_epoch(
            model, optimizer, criterion, train_loader, DEVICE
        )
        torch.cuda.empty_cache()
        gc.collect()
        model.eval()
        print(f"EVAL EPOCH {epoch + 1}:")
        val_avg_loss, val_targets, val_preds = val_epoch(
            model, criterion, test_loader, DEVICE
        )

        val_metrics = compute_metrics(
            val_targets.detach().cpu().squeeze().numpy(), val_preds.detach().cpu().squeeze().numpy()
        )

        train_metrics = compute_metrics(
            train_targets.detach().cpu().squeeze().numpy(),
            train_preds.detach().cpu().squeeze().numpy(),
        )
        add_tensorboard_metrics(
            writer, epoch, avg_loss, val_avg_loss, train_metrics, val_metrics
        )
        torch.cuda.empty_cache()
        gc.collect()

        print('train: ', train_metrics)
        print('val: ', val_metrics)

        # Шедулер обновляет learning rate после каждой эпохи
        scheduler.step()

        # Save checkpoint every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            checkpoint_filename = f"checkpoint_epoch_{epoch + 1}.pth"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
            torch.save(checkpoint, checkpoint_path)
            torch.save(checkpoint, os.path.join(checkpoint_dir, "last_checkpoint.pth"))
            print(f"Checkpoint saved at {checkpoint_path}")

    writer.close()


if __name__ == "__main__":
    train()
