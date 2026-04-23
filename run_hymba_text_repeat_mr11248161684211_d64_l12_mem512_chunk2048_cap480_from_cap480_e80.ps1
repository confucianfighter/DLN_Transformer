Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location "C:\Users\daylan\DLN_Transformer"

$outDir = ".\runs_hymba_text_repeat_mr11248161684211_d64_l12_mem256to512_chunk2048_cap480_from_cap480_e80"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
Start-Transcript -Path (Join-Path $outDir "terminal_transcript.log") -Append

python external_baselines\hymba_text_repeat_runner.py `
  --output_dir $outDir `
  --data_path .\data\tiny_shakespeare.txt `
  --device cuda `
  --epochs 80 `
  --batch_size 1 `
  --eval_batch_size 4 `
  --n_train 4000 `
  --n_test 512 `
  --split 0.8 `
  --span_mode chunk `
  --chunk_span_len 2048 `
  --blank_len 128 `
  --mem_len 512 `
  --n_layers 12 `
  --embed_dim 64 `
  --d_state 16 `
  --n_heads 4 `
  --n_kv_heads 2 `
  --n_meta_tokens 16 `
  --window_size 128 `
  --attention_context_cap 480 `
  --full_attention_layers 1,4,9,12 `
  --multirate_schedule 1,1,2,4,8,16,16,8,4,2,1,1 `
  --dropout 0.03 `
  --clip 1.0 `
  --lr 0.0001 `
  --weight_decay 0.00001 `
  --optim RMSprop `
  --seed 1111 `
  --log_interval 100000 `
  --repeat_noise_prob 0.02 `
  --repeat_ablate_suffix 0 `
  --source_ablate_suffix 0 `
  --early_stop_exact_below 0.0 `
  --sample_non_exact 10 `
  --print_non_exact 3 `
  --resume .\runs_hymba_text_repeat_mr11248161684211_d64_l12_mem256_chunk2048_cap480_from192_e80\best_recall_checkpoint.pt

Stop-Transcript
