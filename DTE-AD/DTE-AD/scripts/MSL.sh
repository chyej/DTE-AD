export CUDA_VISIBLE_DEVICES=0

#python main.py --anormly_ratio 0.8 --num_epochs 50   --batch_size 256  --mode train --dataset MSL  --data_path dataset/MSL --input_c 55    --output_c 55
python main.py --anormly_ratio 0.8  --num_epochs 50      --batch_size 256     --mode test    --dataset MSL   --data_path dataset/MSL  --input_c 55    --output_c 55  --pretrained_model 20