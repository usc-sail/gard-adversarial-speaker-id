#!/bin/bash

# this for loop for multiple random runs, if needed
for iter in {1..10}; do

    stage=0
    MODEL_TYPE=cnn
    opt=adam
    batch_size=64
    ALR_WEIGHT=1
    N_ITERS=500000

    phrase=libri-model-ALR-${MODEL_TYPE}
    model_dir=models_ALR/${phrase}-batch_${batch_size}-opt_${opt}-alrwt_${ALR_WEIGHT}-it_${N_ITERS}/iter_${iter}
    MODEL_CKPT=${model_dir}/${phrase}.pt
    OUTPUT_DIR=outputs_ALR/${phrase}-batch_${batch_size}-opt_${opt}-alrwt_${ALR_WEIGHT}-it_${N_ITERS}/iter_${iter}
    mkdir -p $OUTPUT_DIR
    mkdir -p $model_dir

    if [ $stage -le 0 ]; then
        python train_libri.py \
            --model_type $MODEL_TYPE \
            --model_ckpt $MODEL_CKPT \
            --optimizer $opt \
            --batch_size $batch_size \
            --alr_weight $ALR_WEIGHT \
            --n_iters $N_ITERS \
            --num_workers 32 \
            --log $OUTPUT_DIR/results_train.txt
    fi


    if [ $stage -le 1 ]; then
        ## Testing
        if [ ! -f "$MODEL_CKPT" ]; then
            echo $MODEL_CKPT "does not exist...skipping..."
            continue
        fi
        echo "-----------------------------------------------"
        echo "Testing with adversarial samples"
        echo "Model to be tested: "$MODEL_CKPT
        echo "-----------------------------------------------"

        TEST_ATTACK=FastGradientMethod  
        #TEST_ATTACK=ProjectedGradientDescent
        #EPS=0.002 # 0.002 gives ~30dB SNR
        MAX_ITER_test=100
        for EPS_test in 0.0005 0.002 0.0035 0.005; do
            TEST_OUTPUT_DIR=${OUTPUT_DIR}/${TEST_ATTACK}
            mkdir -p $TEST_OUTPUT_DIR
            REPORT=${TEST_OUTPUT_DIR}/$TEST_ATTACK-untargeted-EPS_${EPS_test}.md  # a file to report the performance

            if [ -f $REPORT ]; then
                echo "Already done: iter = "${iter}" test att = "${TEST_ATTACK}" eps = "${EPS_test}
                continue
            fi

            python test_libri.py \
                --model_type $MODEL_TYPE \
                --model_ckpt $MODEL_CKPT \
                --output_dir $TEST_OUTPUT_DIR \
                --attack $TEST_ATTACK \
                --attack_max_iter $MAX_ITER_test \
                --epsilon $EPS_test \
                --report $REPORT \
                --log $TEST_OUTPUT_DIR/${TEST_ATTACK}_results_test_untargeted_EPS_${EPS_test}.txt
        done
   
    fi

done
