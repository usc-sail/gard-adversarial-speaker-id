#!/bin/bash

#for iter in {1..10}; do  # For multiple random runs, if needed
for iter in 1; do
    augment=0
    opt=adam
    batch_size=128
    swap=0 # swap=0 means src_model=cnn, tgt_model=tdnn. swap=1 is the opposite.

    OUTPUT_DIR=outputs_transferability/1dnn_to_tdnn/iter_${iter}
    if [ $swap -eq 1 ]; then
        OUTPUT_DIR=outputs_transferability/tdnn_to_cnn/iter_${iter}
    fi
    mkdir -p $OUTPUT_DIR

    phrase=libri-model-cnn
    source_model_dir=models/${phrase}-augment_${augment}-epochs_30-batch_${batch_size}-opt_${opt}/iter_${iter}
    SRC_MODEL_CKPT=${source_model_dir}/${phrase}.pt

    phrase=libri-model-tdnn
    target_model_dir=models_tdnn/${phrase}-augment_${augment}-epochs_10-batch_${batch_size}-opt_${opt}/iter_${iter}
    TGT_MODEL_CKPT=${target_model_dir}/${phrase}.pt

    #swap source and target
    if [ $swap -eq 1 ]; then
        echo "Swapping source and target"
        tmp=$SRC_MODEL_CKPT
        SRC_MODEL_CKPT=$TGT_MODEL_CKPT
        TGT_MODEL_CKPT=$tmp
        echo "src: "${SRC_MODEL_CKPT}
        echo "tgt: "${TGT_MODEL_CKPT}
    fi
    #ATTACK=FastGradientMethod  
    #ATTACK=ProjectedGradientDescent
    EPS=0.002 # 0.002 gives ~30dB SNR

    for ATTACK  in "FastGradientMethod" "ProjectedGradientDescent"; do
    
        MAX_ITER=100

        OUTPUT_DIR_ATTACK=${OUTPUT_DIR}/${ATTACK}-untargeted-eps_${EPS}
        mkdir -p $OUTPUT_DIR_ATTACK
        REPORT=${OUTPUT_DIR_ATTACK}/$ATTACK-untargeted-eps_${EPS}.md  # a file to report the performance
        
        if [ -f "$REPORT" ]; then
            echo "Alread done for iter = "${iter}" test attack = "${ATTACK}" eps = "${EPS}
            echo "Skipping..."
            continue
        fi

        python transfer_test.py \
                    --model_ckpt $SRC_MODEL_CKPT \
                    --target_model_ckpt $TGT_MODEL_CKPT \
                    --output_dir $OUTPUT_DIR_ATTACK \
                    --attack $ATTACK \
                    --epsilon $EPS \
                    --attack_max_iter $MAX_ITER \
                    --save_wav 0 \
                    --report $REPORT \
                    --log $OUTPUT_DIR_ATTACK/results_test.txt
            
    done


done
