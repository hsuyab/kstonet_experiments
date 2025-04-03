#!/bin/bash

# Script to run MNIST experiment with Joint K-StoNet

# Create directories for results
mkdir -p results
mkdir -p analysis

# Set parameters
HIDDEN_DIMS="100 50"  # Two hidden layers with 100 and 50 units
N_COMPONENTS=512  # Number of random Fourier features
GAMMA=0.01  # RBF kernel parameter
DROPOUT_RATE=0.2  # Dropout rate
BATCH_SIZE=128
EPOCHS=30
LEARNING_RATE=0.001
L1_LAMBDA=0.0001  # L1 regularization coefficient similar to LASSO
WEIGHT_DECAY=0.0001  # L2 regularization

# Display parameters
echo "Running Joint K-StoNet MNIST experiment with parameters:"
echo "Hidden dimensions: ${HIDDEN_DIMS}"
echo "RFF components: ${N_COMPONENTS}"
echo "Gamma: ${GAMMA}"
echo "Dropout: ${DROPOUT_RATE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Epochs: ${EPOCHS}"
echo "Learning rate: ${LEARNING_RATE}"
echo "L1 lambda: ${L1_LAMBDA}"
echo "Weight decay: ${WEIGHT_DECAY}"
echo "------------------------"

# Train the model
echo "Training model..."
python main.py \
    --hidden-dims $HIDDEN_DIMS \
    --n-components $N_COMPONENTS \
    --gamma $GAMMA \
    --dropout-rate $DROPOUT_RATE \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --l1-lambda $L1_LAMBDA \
    --weight-decay $WEIGHT_DECAY \
    --save-dir ./results/mnist_exp1

# Analyze the model
echo "Analyzing model..."
python analyze_model.py \
    --model-path ./results/mnist_exp1/best_model.pt \
    --save-dir ./analysis/mnist_exp1

# Additional experiment with different parameters (higher regularization)
echo "------------------------"
echo "Running additional experiment with higher regularization..."

L1_LAMBDA=0.001  # Higher L1 regularization
WEIGHT_DECAY=0.001  # Higher L2 regularization

echo "L1 lambda: ${L1_LAMBDA}"
echo "Weight decay: ${WEIGHT_DECAY}"
echo "------------------------"

# Train the model with higher regularization
python main.py \
    --hidden-dims $HIDDEN_DIMS \
    --n-components $N_COMPONENTS \
    --gamma $GAMMA \
    --dropout-rate $DROPOUT_RATE \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --l1-lambda $L1_LAMBDA \
    --weight-decay $WEIGHT_DECAY \
    --save-dir ./results/mnist_exp2_high_reg

# Analyze the model with higher regularization
python analyze_model.py \
    --model-path ./results/mnist_exp2_high_reg/best_model.pt \
    --save-dir ./analysis/mnist_exp2_high_reg

# Compare the results
echo "------------------------"
echo "Experiment complete!"
echo "Check out the results in ./results and analysis in ./analysis"