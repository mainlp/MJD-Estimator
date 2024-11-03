export TASK_NAME=mnli
CUDA_VISIBLE_DEVICES=7
python run.py \
  --model_name_or_path bert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /mounts/data/proj/xinpeng/llm-translate/mnli_trained_model_xw/bert/ \
  --load_best_model_at_end \
  --evaluation_strategy steps --save_strategy steps --eval_steps 100 --logging_steps 100 --metric_for_best_model eval_accuracy --overwrite_output_dir