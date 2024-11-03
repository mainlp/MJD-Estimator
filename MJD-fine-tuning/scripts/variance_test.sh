CUDA_VISIBLE_DEVICES=6
BNB_CUDA_VERSION=122
TRANSFORMERS_CACHE=/mounts/Users/student/xinpeng/data/runs_models/huggingface

find /mounts/data/proj/xinpeng/llm-translate/scale_roberta_softmax_mnli -type d -mindepth 1 -not -path '*/runs' -not -path '*/runs/*' | while read folder; do
    if [ -d "$folder" ]; then
        echo "Testing with model trained on $folder"
        TEST_FILE=data/llm-translate/variance_test/test_variance.json  # Specify the new test file
        OUTPUT_SUFFIX=$folder
        MODEL_DIR=$folder
        VALIDATION_FILE=data/llm-translate/chaos_dev_test/dev.json
        TRAIN_FILE=data/llm-translate/chaos_dev_test/dev.json
        python run.py --model_name_or_path $MODEL_DIR --test_file $TEST_FILE --do_predict \
        --per_device_eval_batch_size 4 --output_dir $MODEL_DIR --local_data_name chaosnli \
        --problem_type distribution_matching --metric_for_best_model eval_macro_F1 --seed 42 --local_data_name chaosnli \
        --train_file $TRAIN_FILE --validation_file $VALIDATION_FILE --num_train_epochs 0
    fi
done
