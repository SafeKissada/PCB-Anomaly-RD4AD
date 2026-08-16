# PCB-Anomaly-RD4AD

Implement [RD4AD / Reverse Distillation](https://arxiv.org/abs/2201.10703)
(Deng & Li, CVPR 2022) สำหรับรันบน dataset PCB defect/false-call เดียวกับ
[`Anomaly-Detection-THESIS`](https://github.com/SafeKissada/Anomaly-Detection-THESIS)

**สถานะ**: implement เสร็จ, smoke test ผ่าน (offline, random-init
backbone, 2 epoch, dummy data), **ยังไม่เคยรันกับข้อมูลจริง หรือเทรนเต็ม
epoch เลย**

## จุดต้องระวังก่อนใช้งานจริง (สำคัญที่สุดในไฟล์นี้)

### 1. ใช้ bilinear interpolation แทน exact-stride convolution — ไม่ตรงกับ paper 100%

Implementation นี้ resize ทุก scale ด้วย `F.interpolate` (bilinear) ทั้ง
ฝั่ง OCBE (downsample รวม f1,f2,f3) และ student decoder (upsample กลับ)
แทนที่จะออกแบบ exact-stride conv/transposed-conv ที่คำนวณ spatial size
ให้พอดีกับ ResNet มาตรฐานที่ input 224×224 เป๊ะๆ ตาม paper ต้นฉบับ

**เหตุผลที่เลือกทำแบบนี้**: ทนทานต่อขนาดภาพที่หารลงตัวไม่พอดี (จำเป็น
สำหรับ smoke test ที่ใช้ภาพ 64×64) — ในทางปฏิบัติผลลัพธ์ทางคณิตศาสตร์
ใกล้เคียงกันมากเมื่อรันที่ 224×224 จริง แต่**ไม่ใช่สถาปัตยกรรมที่ตรงกับ
paper 100%** ถ้าต้องการ reproduce ตัวเลขในเปเปอร์อย้างเคร่งครัดสำหรับ
citation/comparison ที่ต้อง exact — ต้องปรับ `_OCBE`/`_StudentDecoder`
ใน `src/models/rd4ad.py` เป็น exact-stride conv ที่ fix input resolution
เป็น 224×224 ตายตัว

### 2. Bottleneck channel (`bottleneck_ch = c3 * 2`) เป็นค่าที่ตั้งขึ้นเอง ไม่ได้ sweep

Paper ไม่ได้ระบุสูตรตายตัวสำหรับขนาด bottleneck — implementation นี้ใช้
`c3 * 2` (ขยาย channel แต่บีบ spatial ผ่าน stride) เป็นค่าเริ่มต้นที่
สมเหตุสมผล **แต่ยังไม่เคย validate ว่าเหมาะกับ dataset นี้จริง**

ถ้า anomaly score ดูไม่ sensitive พอ (แยก normal/defect ไม่ชัด) จุดแรก
ที่ควรลอง: **ลด bottleneck channel ให้แคบลง** (บังคับ one-class
compactness ให้เข้มขึ้น) — เป็นทิศทางตรงข้ามกับ SimpleNet ที่ปัญหามักมา
จาก `NOISE_STD` ไม่ใช่ bottleneck

### 3. Loss ต่อ layer ใช้ weight เท่ากันหมด (1.0, 1.0, 1.0) — อาจไม่เหมาะกับทุก defect type

`LAYER_LOSS_WEIGHTS` default ให้ layer1 (ตื้น, จับ texture/edge ละเอียด)
กับ layer3 (ลึก, จับ high-level semantic) น้ำหนักเท่ากัน — ถ้า defect ใน
dataset นี้เป็นแบบ **subtle/texture-level** เป็นหลัก (ดูข้อสังเกตใน
`PCB-Anomaly-SimpleNet/README.md` จุดต้องระวังข้อ 1) อาจต้องเพิ่ม
weight ของ layer1/layer2 ให้มากกว่า layer3 — เป็นจุดที่ควร sweep คู่กับ
การเปรียบเทียบผลจาก SimpleNet/DRAEM ว่า defect ส่วนใหญ่ "ตื้น" หรือ
"ลึก" ในเชิง feature hierarchy

### 4. เทรน 2 network พร้อมกัน (OCBE + student) — เวลาเทรนใกล้เคียง DRAEM มากกว่า PatchCore/PaDiM

แม้จะใช้ pretrained teacher (frozen, ไม่ต้องเทรน) แต่ OCBE และ student
decoder เป็น network ที่ต้องเทรนจาก scratch เหมือน DRAEM — วางแผนเวลา
เทรนให้เหมาะสม อย่าคาดหวังว่าจะเร็วเท่า PatchCore/PaDiM ที่ไม่เทรนอะไร
เลย (แค่ fit สถิติ/coreset)

## วิธีรัน

```bash
pip install -r requirements.txt
python tests/smoke_test.py   # เช็ค pipeline ก่อนเสมอ (รันแค่ 2 epoch)
```

แก้ `RUN.py`:
```python
OVERRIDES = dict(
    DATA_ROOT="/path/to/your/dataset",
    SPLIT_CACHE_PATH="/path/to/Anomaly-Detection-THESIS/splits/split_assignment.csv",
    EPOCHS=100,
    LAYER_LOSS_WEIGHTS=(1.0, 1.0, 1.0),  # ปรับตามที่คุยในจุดต้องระวังข้อ 3
)
```
```bash
python RUN.py
```

## Reference

Deng, H., & Li, X. (2022). *Anomaly Detection via Reverse Distillation from
One-Class Embedding.* CVPR 2022. https://arxiv.org/abs/2201.10703
