#!/bin/bash

# input arguments
DATA="${1-PROTEINS}"  # MUTAG, ENZYMES, NCI1, NCI109, DD, PTC, PROTEINS, COLLAB, IMDBBINARY, IMDBMULTI
fold=${2-0}  # which fold as testing data
test_number=${3-0}  # if specified, use the last test_number graphs as test data

# general settings
gm=DGCNN  # model
gpu_or_cpu=gpu
GPU=6  # select the GPU number
CONV_SIZE="32-48-64"
sortpooling_k=0.6  # If k <= 1, then k is set to an integer so that k% of graphs have nodes less than this integer
FP_LEN=0  # final dense layer's input dimension, decided by data
n_hidden=128  # final dense layer's hidden size
bsize=30  # batch size
dropout=True

# dataset-specific settings
case ${DATA} in
DD)
  num_epochs=200
  learning_rate=0.0005
  sortpooling_k=0.7
  ;;
PROTEINS)
  num_epochs=200
  learning_rate=0.001
  ;;
IMDBBINARY)
  num_epochs=500
  learning_rate= 0.0001
  sortpooling_k=0.8
  ;;
IMDBMULTI)
  num_epochs=500
  learning_rate=0.0005
  sortpooling_k=0.9
  ;;
*)
  num_epochs=500
  learning_rate=0.00001
  ;;
esac

if [ ${fold} == 0 ]; then
  rm acc_result.txt
  echo "Running 10-fold cross validation"
  start=`date +%s`
  for i in $(seq 1 10)
  do
    CUDA_VISIBLE_DEVICES=${GPU} python3 main.py \
        -seed 1 \
        -data $DATA \
        -fold $i \
        -learning_rate $learning_rate \
        -num_epochs $num_epochs \
        -hidden $n_hidden \
        -latent_dim $CONV_SIZE \
        -sortpooling_k $sortpooling_k \
        -out_dim $FP_LEN \
        -batch_size $bsize \
        -gm $gm \
        -mode $gpu_or_cpu \
        -dropout $dropout
  done
  stop=`date +%s`
  echo "End of cross-validation"
  echo "The total running time is $[stop - start] seconds."
  echo "The accuracy results for ${DATA} are as follows:"
  cat acc_result.txt
  echo "Average accuracy is"
  cat acc_result.txt | awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; }'
else
  CUDA_VISIBLE_DEVICES=${GPU} python3 main.py \
      -seed 1 \
      -data $DATA \
      -fold $fold \
      -learning_rate $learning_rate \
      -num_epochs $num_epochs \
      -hidden $n_hidden \
      -latent_dim $CONV_SIZE \
      -sortpooling_k $sortpooling_k \
      -out_dim $FP_LEN \
      -batch_size $bsize \
      -gm $gm \
      -mode $gpu_or_cpu \
      -dropout $dropout \
      -test_number ${test_number}
fi
