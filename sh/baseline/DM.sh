GPU=$1
DATA=$2
LR=$3
IPC=$4

cd ../..;

CUDA_VISIBLE_DEVICES=${GPU} python distill_baseline.py \
--method DM \
--dataset ${DATA} \
--ipc ${IPC} \
--num_eval 5 \
--epoch_eval_train 200 \
--init real \
--lr_img ${LR} \
--lr_net 0.01 \
--Iteration 1000 \
--model ConvNet3D \
--eval_mode SS \
--eval_it 500 \
--batch_real 64 \
--num_workers 4 \
--data_path  /opt/data/private/video_distillation-old/distill_utils/data/
#--Iteration 5000 \
#--preload  \
#--epoch_eval_train 500 \
