$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = 'D:\lmms-eval-main'
$env:LMMS_EVAL_ROOT = 'D:\lmms-eval-main'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'
$python = 'H:\anaconda\envs\remem\python.exe'
$output = 'D:\ReMem\VideoMME\paper_lmms_full_qwen3vl8b'
& $python -m lmms_eval eval `
  --include_path 'D:\ReMem\lmms_tasks' `
  --tasks remem_videomme `
  --model qwen3_vl `
  --model_args 'pretrained=D:\ReMem\models\Qwen3-VL-8B-Instruct,device_map=auto,enable_thinking=False,max_num_frames=32,min_pixels=200704,max_pixels=200704,attn_implementation=sdpa' `
  --batch_size 1 `
  --verbosity INFO `
  --output_path $output `
  --log_samples
exit $LASTEXITCODE
