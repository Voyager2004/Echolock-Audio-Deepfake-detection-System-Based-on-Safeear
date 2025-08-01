# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import tqdm
import fairseq
import librosa
import soundfile as sf
import torch
import torch.nn.functional as F
from npy_append_array import NpyAppendArray
from pathlib import Path
from feature_utils import get_path_iterator, dump_feature
import shutil
import subprocess
import uuid
import hashlib
import csv
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.disable(logging.INFO)  # 禁止 INFO 及以下级别的所有日志
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_hubert_feature")


class HubertFeatureReader(object):
    """
    使用 Fairseq HuBERT 模型提取音频特征的工具类。
    - 加载指定的 checkpoint。
    - 按块读取音频并提取特征。
    """
    def __init__(self, ckpt_path, layer, max_chunk=1600000):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().to(DEVICE)
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        logger.info(f"TASK CONFIG:\n{self.task.cfg}")
        logger.info(f" max_chunk = {self.max_chunk}")

    def read_audio(self, path, ref_len=None):
        target_sr = self.task.cfg.sample_rate
        wav, sr = librosa.load(path, sr=target_sr, mono=True)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, path, ref_len=None):
        x = self.read_audio(path, ref_len)
        with torch.no_grad():
            x = torch.from_numpy(x).float().to(DEVICE)
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            avg_feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                feat_chunk, _, avg_feat_chunk = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                avg_feat.append(avg_feat_chunk)
        return torch.cat(avg_feat, 1).squeeze(0)

from pathlib import Path
import subprocess

def convert_to_flac(input_path):
    """
    将任意格式的音频文件转成单声道 FLAC，输出到 datas/audio 目录。
    已经是 FLAC 的也会被重新编码为单声道 FLAC。
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} 不存在")

    output_dir = Path("datas/audio/orig")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 输出文件名：原名（不含后缀）+ .flac
    output_path = output_dir / input_path.with_suffix(".flac").name

    # ffmpeg 统一强制转单声道并编码为 flac
    subprocess.run([
        "ffmpeg",
        "-y",                   # 覆盖已存在的输出
        "-i", str(input_path),  
        "-ac", "1",             # 强制单声道
        "-c:a", "flac",         # FLAC 编码
        str(output_path)
    ], check=True)
    
    input_path.unlink()
    return output_path

def write_log(audio_file):
    # 确保 audio_file 是字符串
    audio_str = str(audio_file.name)
    # 1) 修改 /ASVSpoof2019/eval.tsv 的第二行
    eval_path = Path("datas/ASVSpoof2019/eval.tsv")
    if eval_path.exists():
        # 读取所有行
        with eval_path.open("r+", encoding="utf-8") as f:
            lines = f.readlines()
            # 如果不足两行，补齐空行
            if len(lines) < 2:
                lines += ["\n"] * (2 - len(lines))
            # 覆盖第二行
            lines[1] = audio_str + "\n"
            # 回到文件开头写回，并截断多余内容
            f.seek(0)
            f.writelines(lines)
            f.truncate()
    else:
        raise FileNotFoundError(f"{eval_path} 不存在")

    # 2) 修改 ASVspoof2019.LA.cm.eval.trl.txt 的第一行
    trl_path = Path("datas/ASVSpoof2019/ASVspoof2019.LA.cm.eval.trl.txt")
    if trl_path.exists():
        with trl_path.open("r+", encoding="utf-8") as f:
            lines = f.readlines()
            # 如果为空，先加一行占位
            if len(lines) < 1:
                lines = ["\n"]
            # 覆盖第一行
            lines[0] ="LA_0000 " + Path(audio_str).stem + " - A00 spoof" + "\n"  #LA_0000 LA_E_1611480 - A00 spoof
            f.seek(0)
            f.writelines(lines)
            f.truncate()
    else:
        raise FileNotFoundError(f"{trl_path} 不存在")

def dump_feature(reader,audio_dir,save_dir):
    save_dir = Path(save_dir)
    audio_dir = Path(audio_dir)
    audio_files = list(audio_dir.glob("*.flac"))
    for audio_file in audio_files:
        npy_name = audio_file.with_suffix(".npy").name  
        save_path = save_dir / npy_name
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)
        
        feat_f = NpyAppendArray(save_path)
        feat = reader.get_feats(audio_file)
        feat_f.append(feat.cpu().numpy())

        # 处理完成后，把原始 .flac 文件移动到 datas/audio/flac
        archive_dir = Path("datas/audio/flac")
        dest_path = archive_dir / audio_file.name
        if dest_path.exists():
            stem, suffix = dest_path.stem, dest_path.suffix
            i = 1
            # 循环直到找到一个不存在的文件名
            while True:
                new_name = f"{stem}_{i}{suffix}"
                new_dest = archive_dir / new_name
                if not new_dest.exists():
                    dest_path = new_dest
                    break
                i += 1
        shutil.move(str(audio_file), str(dest_path))

        new_path = save_path.parent / dest_path.with_suffix(".npy").name  
        save_path.rename(new_path)

        # 修改log
        write_log(dest_path)
        
    logger.info("finished successfully")


# 在main函数前添加哈希计算函数
def calculate_file_hash(file_path):
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def main(audio_path, audio_dir, save_dir, ckpt_path, layer, max_chunk):

    # ================== 新增哈希检测逻辑 ==================
    # 计算文件哈希值
    current_hash = calculate_file_hash(audio_path)
    # print(f"[Debug]: hash: {current_hash} ")

    csv_path = Path("cached_file.csv")
    fieldnames = ['file_hash', 'prob', 'pred_label', 'hit']
    rows = []
    exists = False

    # print(f"[Debug]: row[file_hash] = {row['file_hash']}")

    # start_time=time.time()
    # 读取现有缓存文件
    if csv_path.exists():
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != fieldnames:
                logger.error("缓存文件格式错误，请检查列名")
                return
            for row in reader:
                print(f"row: {row}")
                if row['file_hash'] == current_hash:
                    row['hit'] = '1'  # 更新匹配记录的hit为1
                    exists = True
                rows.append(row)


    # print(f"[Debug]: CF reached here :before if exists")
    # 如果找到匹配的哈希
    if exists:
        # 写回更新后的缓存文件
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"文件 {audio_path} 已在缓存中，跳过处理")
        os.remove(audio_path)
        # 直接跳过
        return
    

    # print(f"[Debug]: CF reached here :before new_row")
    # 没有找到则添加新记录
    new_row = {
        'file_hash': current_hash,
        'prob': '0',
        'pred_label': '0',
        'hit': '-1'
    }
    rows.append(new_row)
    
    # 写回缓存文件（包含新记录）
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"新增文件记录: {current_hash}")
    # ================== 哈希检测结束 ==================

    # 原有处理流程
    audio_path = convert_to_flac(audio_path)
    reader = HubertFeatureReader(ckpt_path, layer, max_chunk)
    dump_feature(reader, audio_dir, save_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", required=True,help="path to original audio files")
    parser.add_argument("audio_dir", nargs="?", default="datas/audio/orig", help="Directory containing audio files")
    parser.add_argument("save_dir", nargs="?", default="datas/audio/processed", help="Directory to save extracted features")
    parser.add_argument("ckpt_path", nargs="?", default="model_zoos/hubert_base_ls960.pt", help="Path to the checkpoint file")
    parser.add_argument("layer", nargs="?", type=int, default=9, help="Layer number to extract features from")
    parser.add_argument("--max_chunk", type=int, default=1600000, help="Maximum chunk size for processing")
    args = parser.parse_args()

    logger.info(args)

    main(**vars(args))

