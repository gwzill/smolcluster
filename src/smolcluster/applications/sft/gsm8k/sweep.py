import copy
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
import argparse
import optuna
import yaml

_HERE = Path(__file__).parent          # gsm8k/
_BASE_CONFIG = _HERE / "configs" / "lora_config.yaml"
_LAUNCH_SFT  = _HERE / "scripts" / "launch_sft.sh"

SWEEP_ITERS  = 1869  # 1 full epoch: ceil(7473 train examples / batch_size 4)
WARMUP_RATIO = 0.05
N_TRIALS = 20
_OPTUNA_DIR = _HERE / "data" / "optuna"
STUDY_NAME = "gsm8k_sft_lora_bayes"

_TRAIN_RE = re.compile(r"Iter\s+(\d+):\s+Train loss\s+([\d.]+),\s+Learning Rate\s+([\d.e+-]+)")
_VAL_RE   = re.compile(r"Iter\s+(\d+):\s+Val loss\s+([\d.]+)")


def _completed_count(study: optuna.Study) -> int:
    return sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)


def _stop_after_20(study: optuna.Study, _trial: optuna.trial.FrozenTrial) -> None:
    if _completed_count(study) >= N_TRIALS:
        study.stop()


def train_trial(trial: optuna.trial.Trial, lr: float, rank: int, scale: float, variant: str = "think") -> float:
    last_val_loss = None

    cfg = copy.deepcopy(yaml.safe_load(_BASE_CONFIG.read_text()))
    iters = SWEEP_ITERS

    cfg["iters"]                    = iters
    cfg["learning_rate"]            = lr
    cfg["lora_parameters"]["rank"]  = rank
    cfg["lora_parameters"]["scale"] = scale
    cfg["data"]                     = str(_HERE / "data")
    adapter_path = _HERE / "checkpoints" / f"sft-{variant}" / "adapters"
    adapter_path.mkdir(parents=True, exist_ok=True)
    (_HERE / "checkpoints" / f"sft-{variant}" / "final_model").mkdir(parents=True, exist_ok=True)
    cfg["adapter_path"]             = str(adapter_path)
    cfg["report_to"] = "wandb"

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
        env["WANDB_PROJECT"] = "smolcluster-gsm8k-sweep"

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
            _TRAIN_RE.search(line)
            m = _VAL_RE.search(line)
            if m:
                last_val_loss = float(m.group(2))
                trial.report(last_val_loss, step=int(m.group(1)))

        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, proc.args)
    finally:
        Path(tmp_config).unlink(missing_ok=True)
    if last_val_loss is None:
        raise RuntimeError("No validation loss found in training logs.")
    return last_val_loss


def make_objective(variant: str):
    def objective(trial: optuna.trial.Trial) -> float:
        lr = trial.suggest_categorical("learning_rate", [2e-4, 2e-5, 5e-4])
        rank = trial.suggest_categorical("rank", [8, 16, 32, 64])
        scale = trial.suggest_categorical("scale", [8.0, 16.0, 32.0, 64.0])
        val_loss = train_trial(trial=trial, lr=lr, rank=rank, scale=scale, variant=variant)
        trial.set_user_attr("final_val_loss", val_loss)
        return val_loss
    return objective


if __name__ == "__main__":
    
    _parser = argparse.ArgumentParser(description="GSM8K Optuna sweep")
    _parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=True,
                         help="Prepare data with think/answer tags (default: --think)")
    _args = _parser.parse_args()

    _variant = "think" if _args.think else "nothink"
    _study_db = _OPTUNA_DIR / f"optuna_gsm8k_sweep_{_variant}.db"
    _OPTUNA_DIR.mkdir(parents=True, exist_ok=True)

    _data_script = _HERE / "data" / "prepare_data.py"
    _data_flag = ["--no-think"] if not _args.think else ["--think"]
    subprocess.run(
        [sys.executable, str(_data_script), *_data_flag],
        check=True,
    )

    study = optuna.create_study(
        study_name=f"{STUDY_NAME}_{_variant}",
        storage=f"sqlite:///{_study_db}",
        sampler=optuna.samplers.TPESampler(),
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(
        make_objective(_variant),
        callbacks=[_stop_after_20],
        catch=(subprocess.CalledProcessError, RuntimeError),
    )

    print("Best trial:")
    print(f"  value: {study.best_value}")
    print(f"  params: {study.best_params}")
