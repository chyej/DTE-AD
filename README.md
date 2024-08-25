Each data set creates a file named "Dataset.sh" (e.g. SMD.sh) in the Scripts folder.

example) 
"SMD.sh" 

  export CUDA_VISIBLE_DEVICES=0

  python main.py --anormly_ratio 0.5 --num_epochs 50   --batch_size 256  --mode train --dataset machine-1-1  --data_path dataset/SMD   --input_c 38
  python main.py --anormly_ratio 0.5 --num_epochs 50   --batch_size 256     --mode test    --dataset machine-1-1   --data_path dataset/SMD     --input_c 38     --pretrained_model 20
  
  (Same goes for other datasets)
