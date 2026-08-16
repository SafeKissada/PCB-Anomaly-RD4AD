"""
RD4AD / Reverse Distillation (Deng & Li, "Anomaly Detection via Reverse
Distillation from One-Class Embedding", CVPR 2022) —
https://arxiv.org/abs/2201.10703

สรุป pipeline (แนวคิดต่างจากอีก 3 baseline ในชุดนี้มากที่สุด):
  1. **Teacher** = pretrained CNN แบบ frozen (เหมือน PatchCore/PaDiM/
     SimpleNet) ดึง multi-scale feature จาก layer1+layer2+layer3
  2. **One-Class Bottleneck Embedding (OCBE)**: หลอมรวม feature ทั้ง 3
     scale เข้าด้วยกันแล้วบีบผ่าน bottleneck แคบๆ — บังคับให้ข้อมูลที่ผ่าน
     ไปยัง student ต้องเป็น "แก่น" ของ normal pattern เท่านั้น (one-class
     compactness) ป้องกันไม่ให้ student ลอกเลียน teacher ได้ตรงๆ แบบ trivial
     (ถ้าไม่มี bottleneck นี้ student อาจแค่เรียน identity mapping ซึ่งจะ
     ทำให้ reconstruct ได้ดีทั้ง normal และ anomaly — เสีย anomaly detection
     ability ไปเลย)
  3. **Student decoder**: โครงสร้างย้อนกลับของ teacher (deep -> shallow)
     พยายาม reconstruct feature ของ teacher จาก bottleneck embedding — ต่าง
     จาก autoencoder ทั่วไปที่ reconstruct ภาพ ตรงนี้ reconstruct **feature
     ของอีกโมเดลหนึ่ง** (knowledge distillation แบบย้อนทิศทาง จึงชื่อ
     "reverse distillation")
  4. Loss: **cosine similarity** ระหว่าง teacher feature กับ student
     feature ที่ scale เดียวกัน ทุก layer — เทรนให้ similarity สูงที่สุด
     สำหรับภาพ normal เท่านั้น
  5. Inference: ภาพ anomaly จะทำให้ teacher feature กับ student feature
     "เพี้ยน" ออกจากกันมากกว่าปกติ (เพราะ student ไม่เคยเห็น anomaly pattern
     เลยระหว่างเทรน) — ใช้ 1-cosine_similarity เป็น anomaly score ต่อ pixel

ข้อควรระวังเฉพาะของ implementation นี้:
  - ใช้ **bilinear interpolation แทน exact-stride transposed convolution**
    ทั้งฝั่ง OCBE (downsample) และ student decoder (upsample) เพื่อความ
    ทนทานต่อขนาดภาพที่หารลงตัวไม่พอดี (เช่นตอน smoke test ใช้ภาพ 64x64)
    — ต่างจาก paper ต้นฉบับที่ออกแบบ exact layer shape ตาม ResNet
    มาตรฐานพอดี ผลลัพธ์ทางคณิตศาสตร์ใกล้เคียงกันมากในทางปฏิบัติ แต่ไม่ใช่
    สถาปัตยกรรมที่ตรงกับ paper 100% ถ้าต้องการ reproduce ตัวเลขในเปเปอร์
    เป๊ะๆ ต้องปรับเป็น exact-stride ตาม input resolution ที่ตายตัว (224x224)
  - ไม่มีการ regularize bottleneck เพิ่มเติมนอกจากขนาด channel/spatial ที่
    บีบแคบลง (ตาม paper) — ถ้า anomaly score ดูไม่ sensitive พอ ลองลด
    bottleneck ให้แคบลงอีกเป็นอันดับแรก
"""
import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

logger = logging.getLogger("RD4AD")

_BACKBONE_FACTORY = {
    "wide_resnet50_2": (torchvision.models.wide_resnet50_2,
                         torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V2),
    "resnet18": (torchvision.models.resnet18,
                 torchvision.models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (torchvision.models.resnet50,
                 torchvision.models.ResNet50_Weights.IMAGENET1K_V2),
}


class _Teacher(nn.Module):
    def __init__(self, backbone_name, layers, device, pretrained=True):
        super().__init__()
        ctor, weights = _BACKBONE_FACTORY[backbone_name]
        if not pretrained:
            logger.warning("pretrained=False: random-init weights — smoke "
                            "test/offline dev เท่านั้น ห้ามใช้รันผลจริง")
        self.backbone = ctor(weights=weights if pretrained else None)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.to(device)
        self.layers = layers
        self._features = {}
        self._hooks = [
            dict(self.backbone.named_modules())[name].register_forward_hook(self._make_hook(name))
            for name in layers
        ]

    def _make_hook(self, name):
        def hook(_m, _i, output):
            self._features[name] = output
        return hook

    @torch.no_grad()
    def forward(self, x):
        self._features = {}
        self.backbone(x)
        return [self._features[name] for name in self.layers]  # [f1, f2, f3] ตื้น->ลึก


def _conv_bn_relu(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))


class _OCBE(nn.Module):
    """One-Class Bottleneck Embedding — หลอม f1,f2,f3 (จาก teacher) เข้า
    bottleneck เดียว โดย resize ทุก scale ให้เท่ากับ f3 (ลึกสุด) ก่อนด้วย
    bilinear interpolation (ดู docstring ของไฟล์ตอนบนสุด — เหตุผลที่ไม่ใช้
    exact-stride conv)
    """
    def __init__(self, c1, c2, c3, bottleneck_ch):
        super().__init__()
        self.proj1 = _conv_bn_relu(c1, c3)
        self.proj2 = _conv_bn_relu(c2, c3)
        self.fuse = nn.Sequential(
            _conv_bn_relu(c3 * 3, c3 * 2),
            nn.Conv2d(c3 * 2, bottleneck_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(bottleneck_ch), nn.ReLU(inplace=True),
        )

    def forward(self, f1, f2, f3):
        target_size = f3.shape[-2:]
        p1 = self.proj1(F.interpolate(f1, size=target_size, mode="bilinear", align_corners=False))
        p2 = self.proj2(F.interpolate(f2, size=target_size, mode="bilinear", align_corners=False))
        fused = torch.cat([p1, p2, f3], dim=1)
        return self.fuse(fused)  # [B, bottleneck_ch, H', W'] (H',W' เล็กกว่า f3 อีกครึ่ง)


class _StudentDecoder(nn.Module):
    """ย้อนโครงสร้าง teacher: bottleneck -> reconstruct f3-shape -> f2-shape -> f1-shape
    (deep-to-shallow ตรงข้ามกับ teacher ที่เป็น shallow-to-deep — ที่มาของชื่อ
    "reverse" distillation)
    """
    def __init__(self, bottleneck_ch, c3, c2, c1):
        super().__init__()
        self.dec3 = _conv_bn_relu(bottleneck_ch, c3)
        self.dec2 = _conv_bn_relu(c3, c2)
        self.dec1 = _conv_bn_relu(c2, c1)

    def forward(self, z, shapes):
        """shapes = [(h1,w1), (h2,w2), (h3,w3)] ขนาดเป้าหมายของ f1,f2,f3 จริง"""
        (h1, w1), (h2, w2), (h3, w3) = shapes
        x3 = self.dec3(F.interpolate(z, size=(h3, w3), mode="bilinear", align_corners=False))
        x2 = self.dec2(F.interpolate(x3, size=(h2, w2), mode="bilinear", align_corners=False))
        x1 = self.dec1(F.interpolate(x2, size=(h1, w1), mode="bilinear", align_corners=False))
        return x1, x2, x3  # เรียงตื้น->ลึก ให้ตรงกับ output ของ teacher


def _cosine_dissimilarity_map(t_feat: torch.Tensor, s_feat: torch.Tensor) -> torch.Tensor:
    """คืน spatial map ของ (1 - cosine similarity) ต่อ pixel ระหว่าง teacher
    กับ student feature ที่ scale เดียวกัน — ค่ายิ่งสูงยิ่งผิดปกติ
    """
    t_norm = F.normalize(t_feat, dim=1)
    s_norm = F.normalize(s_feat, dim=1)
    cos_sim = (t_norm * s_norm).sum(dim=1, keepdim=True)  # [B, 1, H, W]
    return 1 - cos_sim


class RD4AD:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.DEVICE
        self.teacher = _Teacher(cfg.BACKBONE, cfg.FEATURE_LAYERS, self.device,
                                 pretrained=getattr(cfg, "PRETRAINED", True))
        self.ocbe: _OCBE = None
        self.student: _StudentDecoder = None

    def fit(self, normal_loader) -> None:
        first_batch = next(iter(normal_loader))
        images = first_batch[0].to(self.device)
        with torch.no_grad():
            f1, f2, f3 = self.teacher(images)
        c1, c2, c3 = f1.shape[1], f2.shape[1], f3.shape[1]
        bottleneck_ch = c3 * 2  # ตาม paper: ขยาย channel แต่บีบ spatial แทน (compact ในเชิงพื้นที่)

        self.ocbe = _OCBE(c1, c2, c3, bottleneck_ch).to(self.device)
        self.student = _StudentDecoder(bottleneck_ch, c3, c2, c1).to(self.device)

        params = list(self.ocbe.parameters()) + list(self.student.parameters())
        optim = torch.optim.Adam(params, lr=self.cfg.LR)

        self.ocbe.train()
        self.student.train()
        best_loss, patience_ctr = float("inf"), 0
        w1, w2, w3 = self.cfg.LAYER_LOSS_WEIGHTS

        logger.info(f"RD4AD.fit(): เทรน OCBE+student {self.cfg.EPOCHS} epoch "
                    f"(c1={c1}, c2={c2}, c3={c3}, bottleneck={bottleneck_ch})")
        for epoch in range(self.cfg.EPOCHS):
            epoch_loss, n_batches = 0.0, 0
            for batch in normal_loader:
                images = batch[0].to(self.device)
                with torch.no_grad():
                    f1, f2, f3 = self.teacher(images)

                z = self.ocbe(f1, f2, f3)
                shapes = [f1.shape[-2:], f2.shape[-2:], f3.shape[-2:]]
                s1, s2, s3 = self.student(z, shapes)

                loss1 = _cosine_dissimilarity_map(f1, s1).mean()
                loss2 = _cosine_dissimilarity_map(f2, s2).mean()
                loss3 = _cosine_dissimilarity_map(f3, s3).mean()
                loss = w1 * loss1 + w2 * loss2 + w3 * loss3

                optim.zero_grad()
                loss.backward()
                optim.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if epoch % max(self.cfg.EPOCHS // 20, 1) == 0 or epoch == self.cfg.EPOCHS - 1:
                logger.info(f"[epoch {epoch+1}/{self.cfg.EPOCHS}] loss={avg_loss:.4f}")

            if avg_loss < best_loss - 1e-5:
                best_loss, patience_ctr = avg_loss, 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.cfg.PATIENCE:
                    logger.info(f"Early stop ที่ epoch {epoch+1}")
                    break

        self.ocbe.eval()
        self.student.eval()
        logger.info("RD4AD fit เสร็จแล้ว")

    @torch.no_grad()
    def score(self, loader):
        from src.models.base import ScoreResult

        if self.ocbe is None:
            raise RuntimeError("RD4AD.score() ถูกเรียกก่อน fit()")

        image_scores, labels, paths, pixel_maps = [], [], [], []

        for batch in loader:
            images, _orig, _preproc, batch_paths, batch_labels, _size = batch
            images = images.to(self.device)
            B = images.shape[0]

            f1, f2, f3 = self.teacher(images)
            z = self.ocbe(f1, f2, f3)
            shapes = [f1.shape[-2:], f2.shape[-2:], f3.shape[-2:]]
            s1, s2, s3 = self.student(z, shapes)

            maps = []
            for t, s in [(f1, s1), (f2, s2), (f3, s3)]:
                m = _cosine_dissimilarity_map(t, s)  # [B,1,h,w]
                m = F.interpolate(m, size=self.cfg.IMAGE_SIZE, mode="bilinear", align_corners=False)
                maps.append(m)
            combined = torch.stack(maps, dim=0).sum(dim=0)  # [B,1,H,W] รวมทุก scale

            for i in range(B):
                pmap_np = _gaussian_smooth(combined[i, 0].cpu().numpy(), self.cfg.HEATMAP_SIGMA)
                pixel_maps.append(pmap_np)
                image_scores.append(float(pmap_np.max()))

            labels.extend([0 if lb == "normal" else 1 for lb in batch_labels])
            paths.extend(batch_paths)

        return ScoreResult(
            image_scores=np.array(image_scores, dtype=np.float64),
            labels=np.array(labels, dtype=np.int64),
            paths=paths,
            pixel_maps=np.stack(pixel_maps, axis=0),
        )


def _gaussian_smooth(arr, sigma):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(arr, sigma=sigma)
