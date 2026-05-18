import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.nn.tasks import DetectionModel, load_checkpoint, intersect_dicts
import argparse


class PerBoxShapeRegressionLoss(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        self.regressor = nn.Linear(in_channels, 1)

    def forward(self, feat, per_box_info, img_shape):
        losses = []
        B, C, H, W = feat.shape
        h_img, w_img = img_shape
        for item in per_box_info:
            img_idx, box, ar_gt, _, _ = item
            x1 = int(box[0] * W / w_img)
            y1 = int(box[1] * H / h_img)
            x2 = int(box[2] * W / w_img)
            y2 = int(box[3] * H / h_img)
            x1 = max(0, min(x1, W - 1))
            x2 = max(x1 + 1, min(x2, W))
            y1 = max(0, min(y1, H - 1))
            y2 = max(y1 + 1, min(y2, H))
            if x1 >= x2 or y1 >= y2:
                continue
            roi = feat[img_idx, :, y1:y2, x1:x2]
            roi_vec = roi.mean(dim=[1, 2])              # [C]
            pred = self.regressor(roi_vec.unsqueeze(0)).squeeze(0)
            target = torch.log(torch.tensor([ar_gt], device=feat.device))
            losses.append(F.mse_loss(pred, target))
        if not losses:
            return torch.tensor(0.0, device=feat.device)
        return torch.stack(losses).mean()


class BoxCenterRegressionLoss(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        self.regressor = nn.Linear(in_channels, 2)          #


    def forward(self, feat, per_box_info, img_shape):
        losses = []
        B, C, H, W = feat.shape
        h_img, w_img = img_shape
        for item in per_box_info:
            img_idx, box, _, cx_gt, cy_gt = item
            x1 = int(box[0] * W / w_img)
            y1 = int(box[1] * H / h_img)
            x2 = int(box[2] * W / w_img)
            y2 = int(box[3] * H / h_img)
            x1 = max(0, min(x1, W - 1))
            x2 = max(x1 + 1, min(x2, W))
            y1 = max(0, min(y1, H - 1))
            y2 = max(y1 + 1, min(y2, H))
            if x1 >= x2 or y1 >= y2:
                continue
            roi = feat[img_idx, :, y1:y2, x1:x2]
            roi_vec = roi.mean(dim=[1, 2])
            pred = self.regressor(roi_vec.unsqueeze(0)).squeeze(0)  # [2]
            target = torch.tensor([cx_gt, cy_gt], device=feat.device)
            losses.append(F.mse_loss(pred, target))
        if not losses:
            return torch.tensor(0.0, device=feat.device)
        return torch.stack(losses).mean()


class PosePriorModel(DetectionModel):
    def __init__(self,
                 cfg='yolov10s_fall_poseprior.yaml',
                 ch=3,
                 nc=None,
                 verbose=True,
                 shape_reg_weight=0.5,
                 global_reg_weight=0.1,
                 center_reg_weight=1.0):
        super().__init__(cfg, ch, nc, verbose)

        self.shape_reg_weight = shape_reg_weight
        self.global_reg_weight = global_reg_weight
        self.center_reg_weight = center_reg_weight

        self.conv_p3 = nn.Conv2d(128, 64, 1)
        self.conv_p4 = nn.Conv2d(256, 64, 1)
        self.conv_p5 = nn.Conv2d(512, 64, 1)
        self.fusion_conv = nn.Conv2d(64 * 3, 64, 1)
        self.shape_reg_loss = PerBoxShapeRegressionLoss(in_channels=64)
        self.center_reg_loss = BoxCenterRegressionLoss(in_channels=64)


        

    def loss(self, batch, preds=None):
        # 1. 基础 YOLO 损失
        base_loss, base_loss_items = super().loss(batch, preds)

        # 2. 收集 ShapeAwareImplicitPose 输出的中间特征
        shape_feats = []
        for m in self.modules():
            if hasattr(m, 'shape_feat'):
                shape_feats.append(m.shape_feat)

        if len(shape_feats) < 3 or not batch.get('per_box_info'):
            total_loss = base_loss
            loss_items = torch.cat([
                base_loss_items,
                torch.zeros(2, device=base_loss.device)   # shape_loss, center_loss
            ])
            return total_loss, loss_items

        feat_p3, feat_p4, feat_p5 = shape_feats[:3]

        # 统一到 P3 尺度
        target_size = feat_p3.shape[-2:]
        feat_p4_up = F.interpolate(feat_p4, size=target_size, mode='bilinear', align_corners=False)
        feat_p5_up = F.interpolate(feat_p5, size=target_size, mode='bilinear', align_corners=False)

        feat_p3_proj = self.conv_p3(feat_p3)
        feat_p4_proj = self.conv_p4(feat_p4_up)
        feat_p5_proj = self.conv_p5(feat_p5_up)
        fused = torch.cat([feat_p3_proj, feat_p4_proj, feat_p5_proj], dim=1)
        fused = self.fusion_conv(fused)   # (B, 64, H, W)

        per_box_info = batch['per_box_info']
        img_shape = batch['img_shape']

        shape_loss = self.shape_reg_loss(fused, per_box_info, img_shape)

        center_loss = self.center_reg_loss(fused, per_box_info, img_shape)

        total_loss = (base_loss +
                      self.shape_reg_weight * shape_loss +
                      self.center_reg_weight * center_loss)

        loss_items = torch.cat([
            base_loss_items,
            shape_loss.detach().unsqueeze(0),
            center_loss.detach().unsqueeze(0)
        ])

        return total_loss, loss_items


# ----------------------------- 自定义训练器 -----------------------------
class PosePriorTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        overrides = dict(overrides) if overrides else {}
        self._weights_path = overrides.pop('weights', None)
        self.shape_reg_weight = overrides.pop('shape_reg_weight', 0.5)
        self.global_reg_weight = overrides.pop('global_reg_weight', 0.1)
        self.center_reg_weight = overrides.pop('center_reg_weight', 1.0)
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = PosePriorModel(
            cfg=cfg if cfg else self.args.model,
            nc=self.data['nc'],
            verbose=verbose,
            shape_reg_weight=self.shape_reg_weight,
            global_reg_weight=self.global_reg_weight,
            center_reg_weight=self.center_reg_weight
        )

        weight_path = self._weights_path or weights
        if weight_path:
            pretrained_model, _ = load_checkpoint(weight_path, device='cpu', fuse=False)
            pretrained_state = pretrained_model.state_dict()
            model_state = model.state_dict()
            matched_state = intersect_dicts(pretrained_state, model_state)
            model.load_state_dict(matched_state, strict=False)
            LOGGER.info(f"Loaded {len(matched_state)}/{len(model_state)} parameters from {weight_path}")

        return model

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)

        bboxes = batch['bboxes']
        batch_idx = batch['batch_idx']
        img_shape = batch['img'].shape[2:]
        h_img, w_img = img_shape

        per_box_info = []
        num_images = batch['img'].shape[0]
        sum_ar = [0.0] * num_images
        count_ar = [0] * num_images

        for i, idx in enumerate(batch_idx):
            box = bboxes[i][:4]
            x1, y1, x2, y2 = box.tolist()
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            ar = w / h
            cx = ((x1 + x2) / 2.0) / w_img
            cy = ((y1 + y2) / 2.0) / h_img
            img_idx = int(idx.item()) if isinstance(idx, torch.Tensor) else int(idx)
            per_box_info.append((img_idx, box, ar, cx, cy))
            sum_ar[img_idx] += ar
            count_ar[img_idx] += 1

        avg_ar = [sum_ar[i] / count_ar[i] if count_ar[i] > 0 else 0.0 for i in range(num_images)]
        batch['per_box_info'] = per_box_info
        batch['avg_ar'] = torch.tensor(avg_ar, device=batch['img'].device)
        batch['img_shape'] = img_shape
        return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='falling_datasets/falling_datasets/data.yaml')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--shape_reg_weight', type=float, default=0.5)
    parser.add_argument('--global_reg_weight', type=float, default=0.1)
    parser.add_argument('--center_reg_weight', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--model', type=str, default='yolov10s_fall_poseprior.yaml')
    args = parser.parse_args()

    overrides = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': args.device,
        'model': args.model,
        'shape_reg_weight': args.shape_reg_weight,
        'global_reg_weight': args.global_reg_weight,
        'center_reg_weight': args.center_reg_weight,
    }

    trainer = PosePriorTrainer(cfg=DEFAULT_CFG, overrides=overrides)
    trainer.train()


if __name__ == '__main__':
    main()