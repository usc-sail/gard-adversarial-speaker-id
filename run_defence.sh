#!/bin/bash
# note: we are using same epsilon in train and test
# vary it if needed

stage=0
MODEL_TYPE=cnn
ATTACK=ProjectedGradientDescent 
EPS=0.002 #this is EPS-train, vary it duing testing
#note: EPS_step is harcoded to EPS/5
RATIO=0.5
EPOCHS=30 #fixed for all experiments
BATCH_SIZE=128
opt=adam
augment=0 #gaussian noise augmentation to normal samples
MAX_ITER=10 #vary it during testing (5, 10, 20, 30)

phrase=libri-model-${MODEL_TYPE}
subdir=${ATTACK}-eps_${EPS}-ratio_${RATIO}-epochs_${EPOCHS}-batch_${BATCH_SIZE}-opt_${opt}-aug_${augment}
model_dir=defence_models/ensemble/${subdir}/${phrase}/
MODEL_CKPT=${model_dir}/${phrase}.pt
OUTPUT_DIR=defence_outputs/ensemble/${subdir}/${phrase}/
mkdir -p $OUTPUT_DIR
mkdir -p $model_dir

#adversarial training 
if [ $stage -le 0 ]; then
    python train_adversarial_libri.py \
        --model_type $MODEL_TYPE \
        --model_ckpt $MODEL_CKPT \
        --wav_length 80000 \
        --num_epochs $EPOCHS \
        --epsilon $EPS \
        --ratio $RATIO \
        --optimizer $opt \
        --augment $augment \
        --attack_max_iter $MAX_ITER \
        --attack $ATTACK \
        --log $OUTPUT_DIR/results_train.txt
fi

#continue here if only training -- to save time, we can test later
#continue

if [ $stage -le 1 ]; then
    if [ ! -f "$MODEL_CKPT" ]; then
        echo $MODEL_CKPT "does not exist...skipping..."
        continue
    fi
    echo "-----------------------------------------------"
    echo "Testing with adversarial samples"
    echo "Model to be tested: "$MODEL_CKPT
    echo "-----------------------------------------------"

    for TEST_ATTACK in "ProjectedGradientDescent"; do
        MAX_ITER_test=100
        TEST_OUTPUT_DIR=${OUTPUT_DIR}/${TEST_ATTACK}_maxiter_${MAX_ITER_test}

        for EPS_test in 0.0005 0.002 0.0035 0.005; do
            ## Testing
            REPORT=${TEST_OUTPUT_DIR}/$TEST_ATTACK-untargeted-EPS_${EPS_test}.md  # a file to report the performance
            python test_libri.py \
                --model_type $MODEL_TYPE \
                --model_ckpt $MODEL_CKPT \
                --output_dir $TEST_OUTPUT_DIR \
                --attack $TEST_ATTACK \
                --epsilon $EPS_test \
                --attack_max_iter $MAX_ITER_test \
                --report $REPORT \
                --log $TEST_OUTPUT_DIR/${TEST_ATTACK}_results_test_untargeted_EPS_${EPS_test}.txt
        done
    done

fi

