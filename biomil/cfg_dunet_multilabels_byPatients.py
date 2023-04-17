# import the necessary packages
import torch
import os
from pathlib import Path
from glob import glob

# base path of the dataset
DATASET_PREP = "Raw"
DATASET_PATH = os.path.join("dataset", DATASET_PREP)
DATASET_LENGTH = int(len(glob(f"{DATASET_PATH}/*")))

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading    
PIN_MEMORY = True if DEVICE == "cuda:0" else False

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.00015
NUM_EPOCHS = 30
BATCH_SIZE = 32

# define the input image dimensions
INPUT_IMAGE_WIDTH = 192
INPUT_IMAGE_HEIGHT = 192

# define threshold to filter weak predictions
THRESHOLD = 0.5

# define the path to the base output directory
BASE_OUTPUT = "output"
Path(BASE_OUTPUT).mkdir(exist_ok=True)
# BASE_OUTPUT = "output_DUnet_multilabels_DiceBased_byPatients"

# define the path to the output serialized model, model training
# plot, and testing image paths
BASE_MODEL_NAME = f"dunet_DL_{NUM_EPOCHS}epoch_{BATCH_SIZE}_{DATASET_PREP}"
MODEL_PATH = os.path.join(BASE_OUTPUT, (BASE_MODEL_NAME + ".pth"))
TEST_PATIENT_NUMS = os.path.sep.join([BASE_OUTPUT, ("test_patient_num_" + BASE_MODEL_NAME + ".txt")])
PLOT_PATH_LOSS = os.path.sep.join([BASE_OUTPUT, ("plotL_" + BASE_MODEL_NAME+ ".png")])
PLOT_PATH_DICE = os.path.sep.join([BASE_OUTPUT, ("plotD_" + BASE_MODEL_NAME+ ".png")])
TEST_T1_PATHS = os.path.sep.join([BASE_OUTPUT, ("test_t1_" + BASE_MODEL_NAME + ".txt")])
TEST_T2_PATHS = os.path.sep.join([BASE_OUTPUT, ("test_t2_" + BASE_MODEL_NAME + ".txt")])
TEST_FLAIR_PATHS = os.path.sep.join([BASE_OUTPUT, ("test_flair_" + BASE_MODEL_NAME + ".txt")])
TEST_T1CE_PATHS = os.path.sep.join([BASE_OUTPUT, ("test_t1ce_" + BASE_MODEL_NAME + ".txt")])
TEST_MASK_PATHS = os.path.sep.join([BASE_OUTPUT, ("test_mask_" + BASE_MODEL_NAME + ".txt")])

TRAIN_T1_PATHS = os.path.sep.join([BASE_OUTPUT, ("train_t1_" + BASE_MODEL_NAME + ".txt")])
TRAIN_T2_PATHS = os.path.sep.join([BASE_OUTPUT, ("train_t2_" + BASE_MODEL_NAME + ".txt")])
TRAIN_FLAIR_PATHS = os.path.sep.join([BASE_OUTPUT, ("train_flair_" + BASE_MODEL_NAME + ".txt")])
TRAIN_T1CE_PATHS = os.path.sep.join([BASE_OUTPUT, ("train_t1ce_" + BASE_MODEL_NAME + ".txt")])
TRAIN_MASK_PATHS = os.path.sep.join([BASE_OUTPUT, ("train_mask_" + BASE_MODEL_NAME + ".txt")])