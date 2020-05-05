#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model convnet \
    --kernel-size 1 \
    --hidden-dim 10 \
    --epochs 5 \
    --weight-decay 0.95 \
    --momentum 0.0 \
    --batch-size 512 \
    --lr 0.01 | tee convnet.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
