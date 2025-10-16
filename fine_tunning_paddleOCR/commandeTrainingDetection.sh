PYTHONPATH="$PWD"  \
    python ./tools/train.py -c ../OCRSub/OCRSub/B2_Segmentation/PP-OCRv5_server_det_anime.yml \
    -o Global.pretrained_model=../Data/Modèles/PP-OCRv5_server_det_pretrained.pdparams \
    Global.epoch_num=5

PYTHONPATH="$PWD" \
    python ./tools/train.py -c "../OCRSub/OCRSub/B2_Segmentation/PP-OCRv5_server_det_anime.yml" \
    -o Global.pretrained_model="../Data/Modeles/PP-OCRv5_server_det_pretrained.pdparams" \
    Global.epoch_num=50  \
    Global.checkpoints="/home/maxim/code/SubProject/PaddleOCR/output/PP-OCRv5_server_det_anime/latest/latest" \
    wandb.allow_val_change=true \
    wandb.id=k4oy2zem565 \
    wandb.resume=must

# rec
PYTHONPATH="$PWD"  \
    python ./tools/train.py -c ../OCRSub/OCRSub/B2_Segmentation/french_PP-OCRv5_server_rec.yml \
    -o Global.pretrained_model=../Data/Modèles/PP-OCRv5_server_rec_pretrained.pdparams \
    Global.epoch_num=30