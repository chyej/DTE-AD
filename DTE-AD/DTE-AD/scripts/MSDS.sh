export CUDA_VISIBLE_DEVICES=0

#python main.py --anormly_ratio 0.9 --num_epochs 50   --batch_size 256  --mode train   --dataset MSDS   --data_path dataset/MSDS   --input_c 10  --output_c 10
python main.py --anormly_ratio 0.9 --num_epochs 50   --batch_size 256     --mode test   --dataset MSDS   --data_path dataset/MSDS     --input_c 10   --output_c 10  --pretrained_model 20