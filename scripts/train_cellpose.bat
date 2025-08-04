@echo off
echo Starting Cellpose training...

call conda activate C:\ProgramData\miniconda3\envs\cellpose

cellpose --train ^
  --dir C:/Users/ac132194/Documents/Fiona_Data/Tiles/CAREamics-output/train/ ^
  --pretrained_model nuclei ^
  --chan 0 --chan2 0 ^
  --train_size ^
  --n_epochs 50 ^
  --batch_size 8 ^
  --learning_rate 0.2 ^
  --weight_decay 1e-5 ^
  --save_every 5 ^
  --model_name_out nuclei_custom ^
  --mask_filter _seg.npy ^
  --verbose

echo Training finished!
pause
