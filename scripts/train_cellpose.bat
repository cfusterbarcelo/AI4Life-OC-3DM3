@echo off
echo Starting Cellpose training...
call conda activate C:\Users\Caterina\anaconda3\envs\cellpose

cellpose --train ^
  --dir D:/Data/Spheroids-Data-OCProject/Individual_Images/zprojection-denoised-cellpose-training/ ^
  --pretrained_model nuclei ^
  --chan 0 --chan2 0 ^
  --train_size ^
  --n_epochs 50 ^
  --batch_size 8 ^
  --learning_rate 0.2 ^
  --weight_decay 1e-5 ^
  --save_every 10 ^
  --model_name_out cellpose-zprojection-denoised ^
  --mask_filter _seg.npy ^
  --verbose

echo Training finished!
pause
