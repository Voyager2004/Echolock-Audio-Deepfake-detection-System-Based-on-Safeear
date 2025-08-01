import importlib
import json
from typing import Any, Dict, List, Optional, Tuple
import argparse
import pytorch_lightning as pl
import torch
import hydra
import csv
import os
import sys
from pathlib import Path
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
    
def test(cfg: DictConfig, args) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    
    # instantiate datamodule
    print_only(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    
    # instantiate decouple model
    print_only(f"Instantiating decouple model <{cfg.decouple_model._target_}>")
    decouple_model: torch.nn.Module = hydra.utils.instantiate(cfg.decouple_model)
    decouple_model.load_state_dict(torch.load(cfg.speechtokenizer_path))
    
    # instantiate detect model
    print(f"Instantiating detect model <{cfg.detect_model._target_}>")
    detect_model: torch.nn.Module = hydra.utils.instantiate(cfg.detect_model)
    
    # instantiate system
    print_only(f"Instantiating system <{cfg.system._target_}>")
    system: LightningModule = hydra.utils.instantiate(
        cfg.system,
        decouple_model=decouple_model,
        detect_model=detect_model,
    )
    
    # instantiate trainer
    print_only(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        strategy=DDPStrategy(find_unused_parameters=True),
    )
    
    trainer.test(system, datamodule=datamodule, ckpt_path=args.ckpt_path)
    

def skip_test():
    """缓存检查函数，返回True表示需要继续测试"""
    cache_file = Path("cached_file.csv")
    score_file = Path("score.csv")
    
    # 如果没有缓存文件直接返回True
    if not cache_file.exists():
        return True

    # 读取缓存文件
    with open(cache_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or reader.fieldnames != ['file_hash', 'prob', 'pred_label', 'hit']:
            print("缓存文件格式错误，缺少必要字段")
            return True
        rows = list(reader)
    
    # 查找需要处理的记录
    processed = False
    for i, row in enumerate(rows):
        # 处理hit=1的记录
        if row['hit'] == '1':
            # 写入score.csv
            with open(score_file, 'w', newline='') as sf:
                writer = csv.writer(sf)
                writer.writerow([
                    "datas/audio/flac/target.flac",  # 固定路径格式
                    row['prob'], 
                    row['pred_label']
                ])
            # 更新hit值
            rows[i]['hit'] = '0'
            processed = True
            break
        
        # 发现-1记录直接返回True
        if row['hit'] == '-1':
            return True
    
    # 如果有更新操作
    if processed:
        # 写回缓存文件
        with open(cache_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return False
    
    return True


def result():
    # 读取score.csv数据
    csv_file = 'score.csv' 
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        try:
            row = next(reader)
            score = float(row[1])  
            label = int(row[2])
        except (StopIteration, IndexError) as e:
            print("score.csv格式错误:", e)
            return
    
    print("Score:", score)
    print("Predict Label:", label)

    # 更新cached_file.csv逻辑
    cache_file = Path("cached_file.csv")
    if not cache_file.exists():
        print("警告：缓存文件不存在")
        return

    # 读取并处理缓存文件
    updated = False
    with open(cache_file, 'r+', newline='') as f:
        # 读取数据
        reader = csv.DictReader(f)
        if reader.fieldnames != ['file_hash', 'prob', 'pred_label', 'hit']:
            print("缓存文件格式异常")
            return
        rows = list(reader)
        
        # 倒序查找第一个hit=-1的行
        for i in reversed(range(len(rows))):
            if rows[i]['hit'] == '-1':
                # 更新记录
                rows[i]['prob'] = str(score)
                rows[i]['pred_label'] = str(label)
                rows[i]['hit'] = '0'
                updated = True
                break
        
        if not updated:
            print("未找到需要更新的记录")
            return
        
        # 回写文件
        f.seek(0)
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        f.truncate()
    
    print("缓存文件更新成功")


if __name__ == "__main__":

    need_to_test=skip_test()
    if not need_to_test:
        # End
        sys.exit(0)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_dir",default="Exps/ASVspoof19/config.yaml")
    parser.add_argument("--ckpt_path",default="best.ckpt") # 模型路径，若不指定则使用最好的模型（best.ckpt）
    
    args = parser.parse_args()
    cfg = OmegaConf.load(args.conf_dir)

    # 保存配置到新的文件，在Exps/ASVspoof19下创建目录，并把当前配置存为config.yaml，便于复现
    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name), exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.exp.dir, cfg.exp.name, "config.yaml"))
    
    test(cfg, args)
    result()
    
    

