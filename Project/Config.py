# Training
epochs = 100
batch_size = 32
lr = 0.001
label_smoothing = 0.1

num_classes = 10
img_size = 32
channel = 3

# Regularization intensities for ablation
dropout_rate = 0.3
heavy_dropout_rate = 0.5
l2_lambda = 1e-4
heavy_l2_lambda = 1e-3

# Reproducibility
seed = 42

#----Data_size----
train_size = 5000
val_size   = 1000
test_size  = 1000

# Output
RESULTS_DIR = './results'
FIGURES_DIR = './results/figures'

# To run in terminal : C:\Users\mk122\anaconda3\envs\tf_env\python.exe experiments.py