score_type=videomme_32

frame_num=32
use_topk=True

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model qwen2_vl \
    --model_args pretrained=./Qwen2-VL-7B-Instruct,use_topk=True,nframes=32 \
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --output_path ./results/${score_type} \
    --verbosity=DEBUG

