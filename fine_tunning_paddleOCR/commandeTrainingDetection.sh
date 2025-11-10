PYTHONPATH="$PWD"  \
    python ./tools/train.py -c ../PaddleOCRAnime/fine_tunning_paddleOCR/PP-OCRv5_server_det_anime.yml \
    -o Global.pretrained_model=../Data/Modèles/PP-OCRv5_server_det_pretrained.pdparams \
    Global.epoch_num=7

PYTHONPATH="$PWD" \
    python ./tools/train.py -c "../PaddleOCRAnime/fine_tunning_paddleOCR/PP-OCRv5_server_det_anime.yml" \
    -o Global.pretrained_model="../Data/Modèles/PP-OCRv5_server_det_pretrained.pdparams" \
    Global.epoch_num=11  \
    Global.checkpoints="/home/maxim/code/SubProject/PaddleOCR_output/PP-OCRv5_server_det_anime/latest/latest"

# rec
PYTHONPATH="$PWD"  \
    python ./tools/train.py -c ../PaddleOCRAnime/fine_tunning_paddleOCR/french_PP-OCRv5_server_rec.yml \
    -o Global.pretrained_model=../Data/Modèles/PP-OCRv5_server_rec_pretrained.pdparams \
    Global.epoch_num=30


PYTHONPATH="$PWD"  \
    python ./tools/train.py -c ../PaddleOCRAnime/fine_tunning_paddleOCR/french_PP-OCRv5_mobile_rec.yml \
    -o Global.pretrained_model=../Data/Modèles/latin_PP-OCRv5_mobile_rec_pretrained.pdparams \
    Global.epoch_num=80 \
    Global.checkpoints="/home/maxim/code/SubProject/PaddleOCR_output/french_PP-OCRv5_mobile_rec/latest/latest"