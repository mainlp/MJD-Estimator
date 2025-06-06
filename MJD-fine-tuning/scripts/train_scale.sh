
CUDA_VISIBLE_DEVICES=6
BNB_CUDA_VERSION=122
TRANSFORMERS_CACHE=/mounts/Users/student/xinpeng/data/runs_models/huggingface
find data/llm-translate/train_softmax/T_10/ -type d -mindepth 1 | while read folder; do
    if [ -d "$folder" ]; then
        echo "Training for $folder"
        TRAIN_FILE=$folder/train.json
        VALIDATION_FILE=data/llm-translate/chaos_dev_test/dev.json
        OUTPUT_SUFFIX=${folder#"data/llm-translate/"}
        OUTPUT=/mounts/data/proj/xinpeng/llm-translate/scale_bert_softmax_mnli/$OUTPUT_SUFFIX
        python run.py --model_name_or_path /mounts/data/proj/xinpeng/llm-translate/mnli_trained_model_xw/bert --train_file $TRAIN_FILE --validation_file $VALIDATION_FILE --do_train --do_eval \
        --per_device_train_batch_size 4 --learning_rate 2e-5 --num_train_epochs 5 --output_dir $OUTPUT --local_data_name chaosnli \
        --problem_type distribution_matching --evaluation_strategy steps --save_strategy steps --eval_steps 20 --logging_steps 20 --load_best_model_at_end --metric_for_best_model eval_macro_F1 --overwrite_output_dir --do_predict --test_file data/llm-translate/chaos_dev_test/test.json --seed 42
    fi
done