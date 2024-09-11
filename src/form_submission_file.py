import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import os

from data_processing import load_data
from utils import (
    val_epoch,
    compute_metrics,
    build_abyssal_model,
)



@hydra.main(version_base=None, config_path="./config", config_name="abyssal")
def create_submission(cfg: DictConfig):
    test_loader, df = load_data(
        train_csv_filename=cfg.train_dataset,
        tokenizer_name=cfg.model.esm_model_name,
        batch_size=cfg.training.batch_size,
    )

    DEVICE = cfg.model.device
    checkpoint_dir = cfg.training.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(OmegaConf.to_yaml(cfg))

    model = build_abyssal_model(cfg.model)
    criterion = getattr(torch.nn, cfg.criterion.type)()

    model.to(DEVICE)
    checkpoint_path = os.path.join(checkpoint_dir, "last_checkpoint.pth")
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    _, val_targets, val_preds = val_epoch(
        model, criterion, test_loader, DEVICE
    )

    df['preds_ddG'] = [int(t[0]) for t in val_preds]

    val_metrics = compute_metrics(
        val_targets.detach().cpu().squeeze().numpy(), val_preds.detach().cpu().squeeze().numpy()
    )
    print(val_metrics)





if __name__ == "__main__":
    train()