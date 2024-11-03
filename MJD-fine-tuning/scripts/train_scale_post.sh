# iterate over the different folders and run the training script    
CUDA_VISIBLE_DEVICES=MIG-8ccd2710-9d59-518a-ae7b-c84c597f213e
BNB_CUDA_VERSION=122
TRANSFORMERS_CACHE=/mounts/Users/student/xinpeng/data/runs_models/huggingface
for folder in data/chaos_snli_posterior_as_label/*; do
    if [ -d "$folder" ]; then
        echo "Training for $folder"
        TRAIN_FILE=$folder/train.json
        VALIDATION_FILE=$folder/val.json
        OUTPUT=outputs/posterior_as_label/$(basename $folder)
        python run.py --model_name_or_path roberta-base --train_file $TRAIN_FILE --validation_file $VALIDATION_FILE --do_train --do_eval \
        --per_device_train_batch_size 4 --learning_rate 2e-5 --num_train_epochs 5 --output_dir $OUTPUT --local_data_name chaosnli \
        --problem_type distribution_matching --evaluation_strategy steps --save_strategy steps --eval_steps 20 --logging_steps 20 --load_best_model_at_end --metric_for_best_model eval_macro_F1 --overwrite_output_dir
    fi
done