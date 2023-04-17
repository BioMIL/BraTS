# USAGE
# python predict.py

# import the necessary packages
from biomil import cfg_dunet_multilabels_byPatients as config
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import torch
import torch
import cv2
import os

def dice_coef2(y_true, y_pred): 
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    if union==0: 
        return 1
    intersection = np.sum(y_true_f * y_pred_f)
    return 2. * intersection / union    

def prepare_plot(predMask, count):
    # predMask = predMask.argmax(axis=-1)
    # initialize our figure
    #figure, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))

    # plot the original image, its mask, and the predicted mask
    #ax[0, 0].imshow(t1, cmap = "gray")
    #ax[0, 1].imshow(t2, cmap = "gray")
    #ax[0, 2].imshow(flair, cmap = "gray")
    #ax[0, 3].imshow(t1ce, cmap = "gray")
    #ax[1, 0].imshow(origMask[..., 1:4])
    plt.imshow(predMask[..., 1:4])
    #ax[1, 2].imshow(~origMask[..., 0], cmap = "gray")
    #ax[1, 3].imshow(~predMask[..., 0], cmap = "gray")	
    # ax[1, 1].imshow(((predMask==1)*255).astype("uint8"))
    # ax[1, 2].imshow(((predMask==2)*255).astype("uint8"))
    # ax[1, 3].imshow(((predMask==3)*255).astype("uint8"))

    # set the titles of the subplots
   
    # ax[1,1].set_title("NCR/NET")
    # ax[1,2].set_title("edema")
    # ax[1,3].set_title("ET")

    # set the layout of the figure and display it
    #figure.tight_layout()
    #figure.show()
    #plt.show()
    plt.savefig(os.path.join("D:\\Vin\\BraTS22_TrainingData", "out_png", ("DL_aug0_600_ff_"+ str(count) + ".png")))
    plt.close("all")
	

def make_predictions_3d(model, t1Paths, t2Paths, flairPaths, t1cePaths, maskPaths, transforms):
	predAll = []
	predWT = []
	predL0 = []
	predL1 = []
	predL2 = []
	predL4 = []
	gtAll = []
	gtWT = []
	gtL0 = []
	gtL1 = []
	gtL2 = []
	gtL4 = []	
	# set model to evaluation mode
	model.eval()
	count = 0

	for imagePaths in zip(t1Paths, t2Paths, flairPaths, t1cePaths, maskPaths):
	# turn off gradient tracking
		with torch.no_grad():
			# load the image from disk, swap its color channels, cast it
			# to float data type, and scale its pixel values
			(t1Path, t2Path, flairPath, t1cePath, maskPath) = imagePaths
			t1_img = cv2.imread(t1Path, 0)
			t2_img = cv2.imread(t2Path, 0)
			flair_img = cv2.imread(flairPath, 0)
			t1ce_img = cv2.imread(t1cePath, 0)
			
			image = np.stack([t1_img, t2_img, flair_img, t1ce_img], axis=-1)
			mask = cv2.imread(maskPath, 0)
			label0_mask = (mask==0)
			label1_mask = (mask==1)			
			label2_mask = (mask==2)
			label4_mask = (mask==4)

			mask = np.stack((label0_mask*255, label1_mask*255, label2_mask*255, label4_mask*255), axis=-1).astype("uint8")
			# mask = np.stack((label1_mask*255, label2_mask*255, label4_mask*255), axis=-1).astype("uint8")

			image = transforms(image)
			mask = transforms(mask)

			# make the channel axis to be the leading one, add a batch
			# dimension, create a PyTorch tensor, and flash it to the
			# current device
			image = torch.unsqueeze(image, axis=0)
			image = image.to(config.DEVICE)

			# make the prediction, pass the results through the sigmoid
			# function, and convert the result to a NumPy array
			predMask = model(image).squeeze()

			predMask = F.log_softmax(predMask, dim=0) 
			predMask = (predMask.permute(1, 2, 0).cpu().numpy()).argmax(axis=-1)
			label0_pred = (predMask==0)
			label1_pred = (predMask==1)			
			label2_pred = (predMask==2)
			label4_pred = (predMask==3)

			predMask = np.stack((label0_pred*255, label1_pred*255, label2_pred*255, label4_pred*255), axis=-1).astype("uint8")        				

			mask = (mask.permute(1, 2, 0).numpy()).argmax(axis=-1)     
			label0_mask = (mask==0)
			label1_mask = (mask==1)			
			label2_mask = (mask==2)
			label4_mask = (mask==3)

			mask = np.stack((label0_mask*255, label1_mask*255, label2_mask*255, label4_mask*255), axis=-1).astype("uint8")	

			predAll.append(predMask>0)
			gtAll.append(mask>0)
			predWT.append(predMask.argmax(axis=-1)>0)
			gtWT.append(mask.argmax(axis=-1)>0)
			predL0.append(label0_pred>0)	
			predL1.append(label1_pred>0) 
			predL2.append(label2_pred>0)
			predL4.append(label4_pred>0)
			gtL0.append(label0_mask>0)    
			gtL1.append(label1_mask>0)
			gtL2.append(label2_mask>0)
			gtL4.append(label4_mask>0)	

			# prepare a plot for visualization
			predMask = cv2.resize(predMask, (240, 240))        
			prepare_plot(predMask, count)
			count += 1

	predAll = np.array(predAll).flatten()
	gtAll = np.array(gtAll).flatten()
	predWT = np.array(predWT).flatten()
	gtWT = np.array(gtWT).flatten()
	predL0 = np.array(predL0).flatten()
	predL1 = np.array(predL1).flatten()
	predL2 = np.array(predL2).flatten()
	predL4 = np.array(predL4).flatten()
	gtL0 = np.array(gtL0).flatten()
	gtL1 = np.array(gtL1).flatten()
	gtL2 = np.array(gtL2).flatten()
	gtL4 = np.array(gtL4).flatten()			

	dice_WT = dice_coef2(gtWT, predWT) # label1 && label2 && label3
	dice_BG = dice_coef2(gtL0, predL0) # only BG label
	dice_all = dice_coef2(gtAll, predAll)	# calculate all				
	diceL1 = dice_coef2(gtL1, predL1)
	diceL2 = dice_coef2(gtL2, predL2)
	diceL4 = dice_coef2(gtL4, predL4)
	tn, fp, fn, tp = confusion_matrix(gtWT, predWT).ravel()
	tnl1, fpl1, fnl1, tpl1 = confusion_matrix(gtL1, predL1).ravel()
	tnl2, fpl2, fnl2, tpl2 = confusion_matrix(gtL2, predL2).ravel()
	tnl4, fpl4, fnl4, tpl4 = confusion_matrix(gtL4, predL4).ravel()

	return dice_WT, dice_BG, dice_all, diceL1, diceL2, diceL4, tn, fp, fn, tp, tnl1, fpl1, fnl1, tpl1, tnl2, fpl2, fnl2, tpl2, tnl4, fpl4, fnl4, tpl4
# load the image paths in our testing file and randomly select 10
# image paths


BASE_OUTPUT = "output"
BASE_MODEL_NAME = "dunet_raw_DL_60epoch_aug0_1251_ff_"
MODEL_PATH = os.path.join(BASE_OUTPUT, (BASE_MODEL_NAME + ".pth"))

print(f"[INFO]: Model being used: {MODEL_PATH}")
print(f"[INFO] DATASET BEING USED : {config.DATASET_PATH}")
print("[INFO] loading up test image paths...")
# testing dataset
t1Paths = open(config.TEST_T1_PATHS).read().strip().split("\n")
t2Paths = open(config.TEST_T2_PATHS).read().strip().split("\n")
flairPaths = open(config.TEST_FLAIR_PATHS).read().strip().split("\n")
t1cePaths = open(config.TEST_T1CE_PATHS).read().strip().split("\n")
maskPaths = open(config.TEST_MASK_PATHS).read().strip().split("\n")

# training dataset
# t1Paths = open(config.TRAIN_T1_PATHS).read().strip().split("\n")
# t2Paths = open(config.TRAIN_T2_PATHS).read().strip().split("\n")
# flairPaths = open(config.TRAIN_FLAIR_PATHS).read().strip().split("\n")
# t1cePaths = open(config.TRAIN_T1CE_PATHS).read().strip().split("\n")
# maskPaths = open(config.TRAIN_MASK_PATHS).read().strip().split("\n")

print(f"Total {len(t1Paths)} images in testing set...")
# imagePaths = np.random.choice(imagePaths, size=10)

# define transformations
T = transforms.Compose([transforms.ToPILImage(),
    transforms.CenterCrop(192),
    # transforms.Resize((config.INPUT_IMAGE_HEIGHT,
    # 	config.INPUT_IMAGE_WIDTH)),
    transforms.ToTensor()])

# load our model from disk and flash it to the current device
print("[INFO] load up model...")
dunet = torch.load(MODEL_PATH).to(config.DEVICE)
dice_WT, dice_BG, dice_all, diceL1, diceL2, diceL4,\
	 tn, fp, fn, tp, tnl1, fpl1, fnl1, tpl1,\
		  tnl2, fpl2, fnl2, tpl2, tnl4, fpl4, fnl4, tpl4 = make_predictions_3d(dunet,
                                                                                    t1Paths,
																					t2Paths,
																					flairPaths,
																					t1cePaths,
																					maskPaths,
																					T)




print(f"dice_all = {dice_all}")
print(f"dice_WT = {dice_WT}")
print(f"dice_BG dice = {dice_BG}")
print(f"NCR/NET dice = {diceL1}")
print(f"edema dice = {diceL2}")
print(f"ET dice = {diceL4}", end="\n\n")

# Confusion Matrix TC
print(f"WT accracy = {(tn+tp) / (tn+fp+fn+tp)}") 
print(f"WT recall = {tp / (fn+tp)}")
print(f"WT precision = {tp / (fp+tp)}")
print(f"WT Confusion Matrix: \nTP = {tp}\nTN = {tn}\nFP = {fp}\nFN = {fn}", end="\n\n")
# print(f"TP = {tp/(tp+tn+fp+fn)}\nTN = {tn/(tp+tn+fp+fn)}\nFP = {fp/(tp+tn+fp+fn)}\nFN = {fn/(tp+tn+fp+fn)}")

# Confusion Matrix TC
print(f"TC accracy = {(tnl1+tpl1) / (tnl1+fpl1+fnl1+tpl1)}") 
print(f"TC recall = {tpl1 / (fnl1+tpl1)}")
print(f"TC precision = {tpl1 / (fpl1+tpl1)}")
print(f"TC Confusion Matrix: \nTP = {tpl1}\nTN = {tnl1}\nFP = {fpl1}\nFN = {fnl1}", end="\n\n")

# Confusion Matrix Edema
print(f"Edema accracy = {(tnl2+tpl2) / (tnl2+fpl2+fnl2+tpl2)}") 
print(f"Edema recall = {tpl2 / (fnl2+tpl2)}")
print(f"Edema precision = {tpl2 / (fpl2+tpl2)}")
print(f"Edema Confusion Matrix: \nTP = {tpl2}\nTN = {tnl2}\nFP = {fpl2}\nFN = {fnl2}", end="\n\n")

# Confusion Matrix TC
print(f"ET accracy = {(tnl4+tpl4) / (tnl4+fpl4+fnl4+tpl4)}") 
print(f"ET recall = {tpl4 / (fnl4+tpl4)}")
print(f"ET precision = {tpl4 / (fpl4+tpl4)}")
print(f"ET Confusion Matrix: \nTP = {tpl4}\nTN = {tnl4}\nFP = {fpl4}\nFN = {fnl4}")

