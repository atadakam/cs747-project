import os
curr_dir = os.getcwd()
annotation_path = os.path.join(curr_dir,"state-farm-distracted-driver-detection", "driver_imgs_list.csv")
train_dir = os.path.join(curr_dir, "state-farm-distracted-driver-detection", "imgs", "train")
test_dir = os.path.join(curr_dir, "state-farm-distracted-driver-detection", "imgs", "test")
batch_size = 32
n_classes = 10
num_epochs = 2
learning_rate = 0.0001
T = 1