from pathlib import Path
import torch
import torch.nn as nn
def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name:str):
  # Create Target Directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents = True, exist_ok = True)

  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model name should end with either .pth or .pt"
  model_save_path = target_dir_path / model_name

  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)
