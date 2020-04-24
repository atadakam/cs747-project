import os
curr_dir = os.getcwd()
annotation_path = os.path.join(curr_dir,"state-farm-distracted-driver-detection", "driver_imgs_list.csv")
train_dir = os.path.join(curr_dir, "state-farm-distracted-driver-detection", "imgs", "train")
test_dir = os.path.join(curr_dir, "state-farm-distracted-driver-detection", "imgs", "test")

T = 5
n_classes = 10
