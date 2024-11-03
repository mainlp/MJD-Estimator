
CUDA_VISIBLE_DEVICES=0
BNB_CUDA_VERSION=122

TRANSFORMERS_CACHE=/
# transformer models cache dir

directory=/
# where you put the pre-processed "train.json" from MJD-evaluate.json
VALIDATION_FILE=data/chaos_dev_test/dev.json
TEST_FILE=data/chaos_dev_test/test.json
MODEL_NAME=bert
# bert or roberta. First trained on MNLI, and save the checkpoint for distribution fine-tuning

find "$directory" -type f | while read file; do
    folder_path=$(dirname "$file")
    echo "Processing file: $file"
    TRAIN_FILE=$file
    OUTPUT=output/$MODEL_NAME/$folder_path
    python run.py --model_name_or_path mnli_saved_models/mnli_trained_model/$MODEL_NAME --train_file $TRAIN_FILE --validation_file $VALIDATION_FILE --do_train --do_eval \
    --per_device_train_batch_size 4 --learning_rate 2e-5 --num_train_epochs 5 --output_dir $OUTPUT --local_data_name chaosnli \
    --problem_type distribution_matching --evaluation_strategy steps --save_strategy steps --eval_steps 20 --logging_steps 20 --load_best_model_at_end --metric_for_best_model eval_macro_F1 --overwrite_output_dir --do_predict --test_file $TEST_FILE
done

