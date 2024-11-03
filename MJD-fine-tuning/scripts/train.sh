CUDA_VISIBLE_DEVICES=7
BNB_CUDA_VERSION=122
TRANSFORMERS_CACHE=/nfs/gdata/bdchen/huggingface_cache
TRAIN_FILE=data/llm-translate/train/chaosnli/train.json
VALIDATION_FILE=data/llm-translate/chaos_dev_test/dev.json
OUTPUT=outputs/chaos_train
python run.py --model_name_or_path /nfs/gdata/bdchen/experiments/project0/mnli_saved_models/mnli_trained_model/bert --train_file $TRAIN_FILE --validation_file $VALIDATION_FILE --do_train --do_eval \
        --per_device_train_batch_size 4 --learning_rate 2e-5 --num_train_epochs 5 --output_dir $OUTPUT --local_data_name chaosnli \
        --problem_type distribution_matching --evaluation_strategy steps --save_strategy steps --eval_steps 20 --logging_steps 20 --load_best_model_at_end --metric_for_best_model eval_macro_F1 --overwrite_output_dir --do_predict --test_file data/llm-translate/chaos_dev_test/test.json
