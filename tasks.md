# Tasks

## [x] Dataset Preparation

Create Clean Data Set (find duration and number of videos), and separate into 3 classes
- Weapon
- Fight
- Background

dataset_id: 10b538da314e4a2d880c60b8a9f64935

## [] Training

### Try ViT-G/16 first (if fail, try to use the smaller model)
```text
[x] vitb: e1c3e0025a5c401c84a263c8dc30d1d6
[x] vitl: 968dae44ab5c46ae8526cfe133640d6b
[x] vitg: 5a333575272144abbde546282fd30f3b
```

### Train Attentive Probe (classify) just 1 epoch - DO NOT use batch size 1, 
and plot acc/loss graphs

## [] Evaluation

## [] Experiments
- Improve YOLOv26 with P2-Head
- Try to use Model Something to Something (SSv2) to follow this [link](https://medium.com/@soumyajit.swain/v-jepa2-and-yolo-combining-vision-and-action-for-real-time-video-understanding-762baf50610b)
