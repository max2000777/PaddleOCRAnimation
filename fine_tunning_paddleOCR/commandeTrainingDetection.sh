PYTHONPATH="$PWD"  \
    python ./tools/train.py -c "../PaddleOCRAnime/fine_tunning_paddleOCR/PP-OCRv5_server_det_anime.yml" \
    -o Global.pretrained_model="../Data/Modèles/PP-OCRv5_server_det_pretrained.pdparams" \
    Global.epoch_num=7

PYTHONPATH="$PWD"  \
    python ./tools/train.py -c "../PaddleOCRAnime/fine_tunning_paddleOCR/PP-OCRv5_server_det_anime.yml" \
    -o Global.pretrained_model="../Data/Modèles/PP-OCRv5_server_det_pretrained.pdparams" \
    Global.epoch_num=3  \
    Global.checkpoints="/home/maxim/code/SubProject/PaddleOCR_output/PP-OCRv5_server_det_anime/latest/latest"

# det light
PYTHONPATH="$PWD"  \
    python ./tools/train.py -c ../PaddleOCRAnime/fine_tunning_paddleOCR/PP-OCRv5_mobile_det.yml \
    -o Global.pretrained_model=../Data/Modèles/PP-OCRv5_mobile_det_pretrained.pdparams \
    Global.epoch_num=4


# rec
PYTHONPATH="$PWD"  \
    python ./tools/train.py -c ../PaddleOCRAnime/fine_tunning_paddleOCR/french_PP-OCRv5_server_rec.yml \
    -o Global.pretrained_model=../Data/Modèles/PP-OCRv5_server_rec_pretrained.pdparams \
    Global.epoch_num=30


PYTHONPATH="$PWD"  \
    python ./tools/train.py -c ../PaddleOCRAnime/fine_tunning_paddleOCR/french_PP-OCRv5_mobile_rec.yml \
    -o Global.pretrained_model=../Data/Modèles/latin_PP-OCRv5_mobile_rec_pretrained.pdparams \
    Global.epoch_num=20 \
    Global.checkpoints="/home/maxim/code/SubProject/PaddleOCR_output/french_PP-OCRv5_mobile_rec/latest/latest"



python "C:\code\VideOCR\CLI\videocr_cli.py" --video_path "D:\Téléchargement\Banner of the stars\Banner Of The Stars II\01.avi" --use_server_model true --lang fr --output "C:\code\VideOCR\CLI\01.srt" --use_dual_zone true --use_gpu true --text_recognition_batch_size 200 --crop_x 6 --crop_y 275 --crop_width 705 --crop_height 117 --crop_x2 3 --crop_y2 0 --crop_width2 705 --crop_height2 85