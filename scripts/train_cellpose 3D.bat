@echo off
echo Starting Cellpose training...

call conda activate C:\ProgramData\miniconda3\envs\cellpose

cellpose --train ^
  --dir C:/Users/ac132194/Documents/Fiona_Data/Tiles/3D/CAREamics-output/train/ ^
  --pretrained_model None ^
  --chan 0 --chan2 0 ^
  --train_size ^
  --n_epochs 50 ^
  --batch_size 8 ^
  --learning_rate 0.2 ^
  --weight_decay 1e-5 ^
  --save_every 5 ^
  --model_name_out nuclei_3D_custom ^
  --mask_filter _seg.npy ^
  --do_3D ^
  --verbose

echo Training finished!
pause

python C:/ProgramData/miniconda3/envs/cellpose/lib/site-packages/cellpose/gui/make_train.py ^
  --dir C:/Users/ac132194/Documents/Fiona_Data/Tiles/3D/CAREamics-output/test/ ^
  --anisotropy 8.81 ^
  --crop_size 256 ^
  --z_axis 1