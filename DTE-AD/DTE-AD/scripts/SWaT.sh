export CUDA_VISIBLE_DEVICES=0

#python main.py --anormly_ratio 0.09 --num_epochs 50   --batch_size 256  --mode train --dataset SWaT  --data_path dataset/SWaT   --input_c 51   --output_c 51
python main.py --anormly_ratio 0.09 --num_epochs 50   --batch_size 256     --mode test    --dataset SWaT   --data_path dataset/SWaT   --input_c 51   --output_c 51   --pretrained_model 20

