export CUDA_VISIBLE_DEVICES=0

#MSL
python main.py --anormly_ratio 0.8  --num_epochs 50      --batch_size 256     --mode test    --dataset MSL   --data_path dataset/MSL  --input_c 55    --output_c 55  --pretrained_model 20

#SMAP
python main.py --anormly_ratio 0.4  --num_epochs 50        --batch_size 256     --mode test    --dataset SMAP   --data_path dataset/SMAP  --input_c 25    --output_c 25  --pretrained_model 20


#MSDS
python main.py --anormly_ratio 0.9 --num_epochs 50   --batch_size 256     --mode test   --dataset MSDS   --data_path dataset/MSDS     --input_c 10   --output_c 10  --pretrained_model 20

#PSM
python main.py --anormly_ratio 0.9  --num_epochs 50       --batch_size 256     --mode test   --dataset PSM    --data_path dataset/PSM  --input_c 25    --output_c 25  --pretrained_model 20

#SWaT
python main.py --anormly_ratio 0.09 --num_epochs 50   --batch_size 256     --mode test    --dataset SWaT   --data_path dataset/SWaT   --input_c 51   --output_c 51   --pretrained_model 20

#SMD
python main.py --anormly_ratio 0.05 --num_epochs 50   --batch_size 128     --mode test    --dataset machine-1-1   --data_path dataset/SMD     --input_c 38     --pretrained_model 20
python main.py --anormly_ratio 0.05 --num_epochs 50   --batch_size 128     --mode test    --dataset machine-1-2   --data_path dataset/SMD     --input_c 38     --pretrained_model 20
python main.py --anormly_ratio 0.05 --num_epochs 50   --batch_size 128     --mode test    --dataset machine-1-3   --data_path dataset/SMD     --input_c 38     --pretrained_model 20
python main.py --anormly_ratio 0.05 --num_epochs 50   --batch_size 128     --mode test    --dataset machine-1-4   --data_path dataset/SMD     --input_c 38     --pretrained_model 20
python main.py --anormly_ratio 0.05 --num_epochs 50   --batch_size 128     --mode test    --dataset machine-1-5   --data_path dataset/SMD     --input_c 38     --pretrained_model 20
python main.py --anormly_ratio 0.05 --num_epochs 50   --batch_size 128     --mode test    --dataset machine-1-6   --data_path dataset/SMD     --input_c 38     --pretrained_model 20
python main.py --anormly_ratio 0.05 --num_epochs 50   --batch_size 128     --mode test    --dataset machine-1-7   --data_path dataset/SMD     --input_c 38     --pretrained_model 20
python main.py --anormly_ratio 0.05 --num_epochs 50   --batch_size 128     --mode test    --dataset machine-1-8   --data_path dataset/SMD     --input_c 38     --pretrained_model 20

python main.py --anormly_ratio 0.05 --num_epochs 50   --batch_size 128     --mode test    --dataset machine-2-1   --data_path dataset/SMD     --input_c 38     --pretrained_model 20
python main.py --anormly_ratio 0.05 --num_epochs 50   --batch_size 128     --mode test    --dataset machine-2-2   --data_path dataset/SMD     --input_c 38     --pretrained_model 20
python main.py --anormly_ratio 0.05 --num_epochs 50   --batch_size 128     --mode test    --dataset machine-2-3   --data_path dataset/SMD     --input_c 38     --pretrained_model 20
python main.py --anormly_ratio 0.05 --num_epochs 50   --batch_size 128     --mode test    --dataset machine-2-4   --data_path dataset/SMD     --input_c 38     --pretrained_model 20
python main.py --anormly_ratio 0.05 --num_epochs 50   --batch_size 128     --mode test    --dataset machine-2-5   --data_path dataset/SMD     --input_c 38     --pretrained_model 20
python main.py --anormly_ratio 0.05 --num_epochs 50   --batch_size 128     --mode test    --dataset machine-2-6   --data_path dataset/SMD     --input_c 38     --pretrained_model 20
python main.py --anormly_ratio 0.05 --num_epochs 50   --batch_size 128     --mode test    --dataset machine-2-7   --data_path dataset/SMD     --input_c 38     --pretrained_model 20
python main.py --anormly_ratio 0.05 --num_epochs 50   --batch_size 128     --mode test    --dataset machine-2-8   --data_path dataset/SMD     --input_c 38     --pretrained_model 20
python main.py --anormly_ratio 0.05 --num_epochs 50   --batch_size 128     --mode test    --dataset machine-2-9   --data_path dataset/SMD     --input_c 38     --pretrained_model 20

python main.py --anormly_ratio 0.05 --num_epochs 50   --batch_size 128     --mode test    --dataset machine-3-1   --data_path dataset/SMD     --input_c 38     --pretrained_model 20
python main.py --anormly_ratio 0.05 --num_epochs 50   --batch_size 128     --mode test    --dataset machine-3-2   --data_path dataset/SMD     --input_c 38     --pretrained_model 20
python main.py --anormly_ratio 0.05 --num_epochs 50   --batch_size 128     --mode test    --dataset machine-3-3   --data_path dataset/SMD     --input_c 38     --pretrained_model 20
python main.py --anormly_ratio 0.05 --num_epochs 50   --batch_size 128     --mode test    --dataset machine-3-4   --data_path dataset/SMD     --input_c 38     --pretrained_model 20
python main.py --anormly_ratio 0.05 --num_epochs 50   --batch_size 128     --mode test    --dataset machine-3-5   --data_path dataset/SMD     --input_c 38     --pretrained_model 20
python main.py --anormly_ratio 0.05 --num_epochs 50   --batch_size 128     --mode test    --dataset machine-3-6   --data_path dataset/SMD     --input_c 38     --pretrained_model 20
python main.py --anormly_ratio 0.05 --num_epochs 50   --batch_size 128     --mode test    --dataset machine-3-7   --data_path dataset/SMD     --input_c 38     --pretrained_model 20
python main.py --anormly_ratio 0.05 --num_epochs 50   --batch_size 128     --mode test    --dataset machine-3-8   --data_path dataset/SMD     --input_c 38     --pretrained_model 20
python main.py --anormly_ratio 0.05 --num_epochs 50   --batch_size 128     --mode test    --dataset machine-3-9   --data_path dataset/SMD     --input_c 38     --pretrained_model 20
python main.py --anormly_ratio 0.05 --num_epochs 50   --batch_size 128     --mode test    --dataset machine-3-10   --data_path dataset/SMD     --input_c 38     --pretrained_model 20
python main.py --anormly_ratio 0.05 --num_epochs 50   --batch_size 128     --mode test    --dataset machine-3-11   --data_path dataset/SMD     --input_c 38     --pretrained_model 20






