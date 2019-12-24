nvidia-settings -a '[gpu:0]/GPUFanControlState=1'
nvidia-settings -a '[fan:0]/GPUTargetFanSpeed=100'
nvidia-smi -l

nvidia-settings -a '[fan:0]/GPUTargetFanSpeed=60'


source activate pytorch
jupyter notebook



source activate pytorch
spyder

