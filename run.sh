#!/bin/bash

stage=0 # you can set it to 1 to jump to testing (assuming training is done)
MODEL_TYPE=cnn
augment=0
batch_size=128
# Specify either of N_EPOCHS or N_ITERS
N_EPOCHS=30     # Number of epochs, specify optionally
#N_ITERS=500000 # Number of gradient updates, default in 500k

phrase=libri-model-${MODEL_TYPE}
model_dir=models/${phrase}-augment_${augment}-epochs_${N_EPOCHS}-batch_${batch_size}/
MODEL_CKPT=${model_dir}/${phrase}.pt
OUTPUT_DIR=outputs/${phrase}-augment_${augment}-epochs_${N_EPOCHS}-batch_${batch_size}/
mkdir -p $OUTPUT_DIR
mkdir -p $model_dir

# Stage 0: Training
if [ $stage -le 0 ]; then
    python train_libri.py \
        --model_type $MODEL_TYPE \
        --model_ckpt $MODEL_CKPT \
        --batch_size $batch_size \
        --num_workers 30 \
        --n_epochs $N_EPOCHS \
        --log $OUTPUT_DIR/results_train.txt
fi

# Stage 1: Testing with adversarial attacks
if [ $stage -le 1 ]; then
    ## Testing
    if [ ! -f "$MODEL_CKPT" ]; then
        echo $MODEL_CKPT "does not exist...skipping..."
        exit
    fi
    echo "-----------------------------------------------"
    echo "Testing with adversarial samples"
    echo "Model to be tested: "$MODEL_CKPT
    echo "-----------------------------------------------"

    ATTACK=ProjectedGradientDescent 
    EPS=0.002 # 0.002 gives ~30dB SNR for FGSM
    ATTACK_MAX_ITER=10 # i.e., PGD-10 attack at eps=0.002
    
    OUTPUT_DIR_ATTACK=${OUTPUT_DIR}/${ATTACK}-untargeted-eps_${EPS}
    mkdir -p $OUTPUT_DIR_ATTACK
    REPORT=${OUTPUT_DIR_ATTACK}/$ATTACK-untargeted-eps_${EPS}.md  # a file to report the performance
    
    python test_libri.py \
        --model_type $MODEL_TYPE \
        --model_ckpt $MODEL_CKPT \
        --output_dir $OUTPUT_DIR_ATTACK \
        --attack $ATTACK \
        --epsilon $EPS \
        --report $REPORT \
        --attack_max_iter $ATTACK_MAX_ITER \
        --log $OUTPUT_DIR_ATTACK/results_test.txt

fi

