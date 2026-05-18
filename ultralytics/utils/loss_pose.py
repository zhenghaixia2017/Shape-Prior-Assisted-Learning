# ultralytics/utils/loss_pose.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShapeRegressionLoss(nn.Module):
    """
    形状回归损失：监督形状特征预测高宽比的对数（log(宽/高)）
    """
    def __init__(self, in_channels=256):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 1)
        )

    def forward(self, shape_feat, target_ar):
        """
        shape_feat: 形状编码器输出特征 [B, C, H, W]
        target_ar: 真实高宽比（宽/高），形状 [B]
        """
        pred = self.regressor(shape_feat).squeeze(1)          # [B]
        target_log = torch.log(target_ar + 1e-6)              # 对数变换
        loss = F.mse_loss(pred, target_log)
        return loss
    

class PerBoxShapeRegressionLoss(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 1)
        )

    def forward(self, shape_feat, per_box_info, img_shape):
        losses = []
        B, C, H, W = shape_feat.shape
        h_img, w_img = img_shape
        for item in per_box_info:
            img_idx, box, target_ar, _, _ = item
            x1 = int(box[0] * W / w_img)
            y1 = int(box[1] * H / h_img)
            x2 = int(box[2] * W / w_img)
            y2 = int(box[3] * H / h_img)
            x1 = max(0, min(x1, W-1))
            x2 = max(x1+1, min(x2, W))
            y1 = max(0, min(y1, H-1))
            y2 = max(y1+1, min(y2, H))
            if x1 >= x2 or y1 >= y2:
                continue
            roi = shape_feat[img_idx, :, y1:y2, x1:x2]
            roi_vec = roi.mean(dim=[1,2]).unsqueeze(0)
            pred = self.regressor(roi_vec).squeeze()
            target = torch.log(target_ar + 1e-6)
            loss = F.mse_loss(pred, target)
            losses.append(loss)
        if not losses:
            return torch.tensor(0.0, device=shape_feat.device)
        return torch.stack(losses).mean()