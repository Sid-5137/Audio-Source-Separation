import torch
import torch.nn as nn

class StereoSpectrogramLoss(nn.Module):
    def __init__(self, w_log_mag=1.0, w_lin_mag=1.0):
        super(StereoSpectrogramLoss, self).__init__()
        self.w_log_mag = w_log_mag
        self.w_lin_mag = w_lin_mag
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        pred_mid = (pred[:, :, 0] + pred[:, :, 1]) / 2
        pred_side = (pred[:, :, 0] - pred[:, :, 1]) / 2
        target_mid = (target[:, :, 0] + target[:, :, 1]) / 2
        target_side = (target[:, :, 0] - target[:, :, 1]) / 2

        # Clamp to ensure non-negative for log1p
        pred_mid = torch.clamp(pred_mid, min=0)
        pred_side = torch.clamp(pred_side, min=0)
        target_mid = torch.clamp(target_mid, min=0)
        target_side = torch.clamp(target_side, min=0)

        # Debugging
        if not (torch.all(torch.isfinite(pred_mid)) and torch.all(torch.isfinite(pred_side))):
            print("NaN/Inf in pred_mid or pred_side")
            print(f"pred_mid: {pred_mid.min()}, {pred_mid.max()}")
            print(f"pred_side: {pred_side.min()}, {pred_side.max()}")
        if not (torch.all(torch.isfinite(target_mid)) and torch.all(torch.isfinite(target_side))):
            print("NaN/Inf in target_mid or target_side")
            print(f"target_mid: {target_mid.min()}, {target_mid.max()}")
            print(f"target_side: {target_side.min()}, {target_side.max()}")

        log_loss_mid = self.mse(torch.log1p(pred_mid), torch.log1p(target_mid))
        log_loss_side = self.mse(torch.log1p(pred_side), torch.log1p(target_side))
        if not (torch.isfinite(log_loss_mid) and torch.isfinite(log_loss_side)):
            print("NaN in log losses")
            print(f"log_loss_mid: {log_loss_mid}, log_loss_side: {log_loss_side}")
        log_loss = (log_loss_mid + log_loss_side) / 2

        lin_loss_mid = self.mse(pred_mid, target_mid)
        lin_loss_side = self.mse(pred_side, target_side)
        if not (torch.isfinite(lin_loss_mid) and torch.isfinite(lin_loss_side)):
            print("NaN in lin losses")
            print(f"lin_loss_mid: {lin_loss_mid}, lin_loss_side: {lin_loss_side}")
        lin_loss = (lin_loss_mid + lin_loss_side) / 2

        total_loss = self.w_log_mag * log_loss + self.w_lin_mag * lin_loss
        if not torch.isfinite(total_loss):
            print("NaN in total_loss")
            print(f"log_loss: {log_loss}, lin_loss: {lin_loss}")
        return total_loss
    
def chunk_spectrogram(spec, chunk_size=512):
    num_frames = spec.shape[-1]
    chunks = []
    for start in range(0, num_frames, chunk_size):
        end = min(start + chunk_size, num_frames)
        chunk = spec[..., start:end]
        if chunk.shape[-1] < chunk_size:
            padding = (0, chunk_size - chunk.shape[-1])
            chunk = torch.nn.functional.pad(chunk, padding)
        chunks.append(chunk)
    return chunks