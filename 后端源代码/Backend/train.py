import importlib
import json
import os
import warnings
warnings.filterwarnings("ignore")
from typing import Any, Dict, List, Optional, Tuple
import argparse
import pytorch_lightning as pl
import torch
import hydra

torch.set_float32_matmul_precision("high")

from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.strategies.ddp import DDPStrategy
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning.utilities import rank_zero_only

@rank_zero_only
def print_only(message: str):
    """Prints a message only on rank 0."""
    print(message)
    
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    
    # instantiate datamodule 即加载相关路径、参数等
    print_only(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()
    
    # instantiate decouple model 即实例化解耦模型
    print_only(f"Instantiating decouple model <{cfg.decouple_model._target_}>")
    decouple_model: torch.nn.Module = hydra.utils.instantiate(cfg.decouple_model)
    decouple_model.load_state_dict(torch.load(cfg.speechtokenizer_path))
    
    # instantiate detect model 初始化实际的分类（真伪判别）网络
    print(f"Instantiating detect model <{cfg.detect_model._target_}>")
    detect_model: torch.nn.Module = hydra.utils.instantiate(cfg.detect_model)
    
    # instantiate system 
    print_only(f"Instantiating system <{cfg.system._target_}>")
    system: LightningModule = hydra.utils.instantiate(
        cfg.system,
        decouple_model=decouple_model,
        detect_model=detect_model,
    )
    # instantiate callbacks
    callbacks: List[Callback] = []
    if cfg.get("early_stopping"):
        print_only(f"Instantiating early_stopping <{cfg.early_stopping._target_}>")
        callbacks.append(hydra.utils.instantiate(cfg.early_stopping))
    if cfg.get("checkpoint"):
        print_only(f"Instantiating checkpoint <{cfg.checkpoint._target_}>")
        checkpoint: pl.callbacks.ModelCheckpoint = hydra.utils.instantiate(cfg.checkpoint)
        callbacks.append(checkpoint)
        
    # instantiate logger
    print_only(f"Instantiating logger <{cfg.logger._target_}>")
    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name, "logs"), exist_ok=True)
    logger = hydra.utils.instantiate(cfg.logger)
    
    # instantiate trainer
    print_only(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=True),
    )
    
    trainer.fit(system, datamodule=datamodule)
    print_only("Training finished!")
    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(cfg.exp.dir, cfg.exp.name, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    import wandb
    if wandb.run:
        print_only("Closing wandb!")
        wandb.finish()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf_dir",
        default="config/train19.yaml"
    )
    
    args = parser.parse_args()
    cfg = OmegaConf.load(args.conf_dir)
    
    # 保存配置到新的文件，在Exps/ASVspoof19下创建目录，并把当前配置存为config.yaml，便于复现
    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name), exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.exp.dir, cfg.exp.name, "config.yaml"))
    
    train(cfg)
    
