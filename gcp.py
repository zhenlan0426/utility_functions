conda activate torch-xla-0.5
export XRT_TPU_CONFIG="tpu_worker;0;192.168.0.2:8470"

jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser &
