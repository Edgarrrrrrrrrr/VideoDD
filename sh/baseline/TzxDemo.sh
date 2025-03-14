GPU=$1
DATA=$2
LR=$3
IPC=$4

cd ../..;

CUDA_VISIBLE_DEVICES=${GPU} python TzxDemo.py \
--method CF \
--dataset ${DATA} \
--ipc ${IPC} \
--num_eval 5 \
--epoch_eval_train 500 \
--init real \
--lr_video ${LR} \
--lr_net 0.01 \
--Iteration 5000 \
--model ConvNet3D \
--eval_mode SS \
--eval_it 500 \
--batch_real 64 \
--num_workers 4 \
--sampling_net 0 \
--iter_calib 0 \
--preload \
--data_path  "/root/autodl-tmp/Data"
