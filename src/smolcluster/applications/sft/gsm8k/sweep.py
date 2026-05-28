import copy
import os
import re
import subprocess
import tempfile
from pathlib import Path

import wandb
import yaml

_HERE = Path(__file__).parent          # gsm8k/
_BASE_CONFIG = _HERE / "configs" / "lora_config.yaml"
_LAUNCH_SFT  = _HERE / "scripts" / "launch_sft.sh"

SWEEP_ITERS  = 500
WARMUP_RATIO = 0.05

_TRAIN_RE = re.compile(r"Iter\s+(\d+):\s+Train loss\s+([\d.]+),\s+Learning Rate\s+([\d.e+-]+)")
_VAL_RE   = re.compile(r"Iter\s+(\d+):\s+Val loss\s+([\d.]+)")

sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {"values": [1e-4, 2e-4, 2e-5, 5e-4]},
        "rank":          {"values": [8, 16, 32, 64]},
        "scale":         {"values": [8.0, 16.0, 32.0, 64.0, 128.0]},
    },
}


def train():
    run = wandb.init(project="smolcluster-sft-gsm8k")
    lr    = run.config.learning_rate
    rank  = run.config.rank
    scale = run.config.scale

    cfg = copy.deepcopy(yaml.safe_load(_BASE_CONFIG.read_text()))
    iters = SWEEP_ITERS

    cfg["iters"]                    = iters
    cfg["learning_rate"]            = lr
    cfg["lora_parameters"]["rank"]  = rank
    cfg["lora_parameters"]["scale"] = scale
    cfg["data"]                     = str(_HERE / "data")
    cfg.pop("report_to", None)       # disable mlx_lm's own wandb — we log via stdout

    cfg["lr_schedule"] = {
        "name":        "cosine_decay",
        "arguments":   [lr, iters, lr * 0.01],
        "warmup":      max(1, int(iters * WARMUP_RATIO)),
        "warmup_init": 0.0,
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", dir=_HERE / "configs", delete=False
    ) as f:
        yaml.dump(cfg, f)
        tmp_config = f.name

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["WANDB_DISABLED"]   = "true"   # prevent mlx_lm's WandBCallback from conflicting

        proc = subprocess.Popen(
            ["bash", str(_LAUNCH_SFT),
             "--skip-data", "--foreground", "--config", tmp_config],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        for line in proc.stdout:
            print(line, end="", flush=True)
            m = _TRAIN_RE.search(line)
            if m:
                run.log(
                    {"train_loss": float(m.group(2)), "lr": float(m.group(3))},
                    step=int(m.group(1)),
                )
            m = _VAL_RE.search(line)
            if m:
                run.log({"val_loss": float(m.group(2))}, step=int(m.group(1)))

        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, proc.args)
    finally:
        Path(tmp_config).unlink(missing_ok=True)
        wandb.finish()


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="smolcluster-sft-gsm8k")
    wandb.agent(sweep_id, train, count=20)
