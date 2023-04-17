# import the necessary packages
from biomil import cfg_dunet_multilabels_byPatients as config
from torch.utils.data import Dataset
import os
import numpy as np
import random
import cv2

class SegmentationDataset(Dataset):
	def __init__(self, t1_Paths, t2_Paths, flair_Paths, t1ce_Paths, mask_Paths, torch_transforms, aug_transforms):
		# store the image and mask filepaths, and augmentation transforms
		self.t1_Paths = t1_Paths
		self.t2_Paths = t2_Paths
		self.flair_Paths = flair_Paths
		self.t1ce_Paths = t1ce_Paths
		self.mask_Paths = mask_Paths
		self.torch_transforms = torch_transforms
		self.aug_transforms = aug_transforms

	def __len__(self):
		# return the number of total samples contained in the dataset
		return len(self.t1_Paths)

	def __getitem__(self, idx):
		# load the image from disk, swap its channels from BGR to grayscale,
		# and read the associated mask from disk in grayscale mode
        # load image in grayscale mode -> cv2.imread(path, 0)
		t1_img = cv2.imread(self.t1_Paths[idx], 0)
		t2_img = cv2.imread(self.t2_Paths[idx], 0)
		flair_img = cv2.imread(self.flair_Paths[idx], 0)
		t1ce_img = cv2.imread(self.t1ce_Paths[idx], 0)
        
		image = np.stack([t1_img, t2_img, flair_img, t1ce_img], axis=-1)
		mask = cv2.imread(self.mask_Paths[idx], 0)

		label0 = (mask==0)
		label1 = (mask==1)			
		label2 = (mask==2)
		label4 = (mask==4)
		mask = (np.stack((label0*255, label1*255, label2*255, label4*255), axis=-1)).astype("uint8")          

		# check to see if we are applying any transformations
		if self.aug_transforms is not None:
			# apply the transformations to both image and its mask
			augmented = self.aug_transforms(image=image, mask=mask)
			image = augmented['image']
			mask = augmented['mask']

		if self.torch_transforms is not None:
			# apply the transformations to both image and its mask
			image = self.torch_transforms(image)
			mask = self.torch_transforms(mask)

		# return a tuple of the image and its mask
		return (image, mask)

class BraTS2020loader:
    def __init__(self, dataset):    
        self.paths = dataset
        self.test_patient_nums = list(random.sample(range(1, config.DATASET_LENGTH), k=int(np.floor(0.15*(config.DATASET_LENGTH)))))
        print(config.DATASET_LENGTH)
        print("self.test_patient_nums = {}".format(self.test_patient_nums))
        self.t1_paths = []
        self.t2_paths = []
        self.flair_paths = []
        self.t1ce_paths = []
        self.gt_paths = []
        self.t1_paths_test = []
        self.t2_paths_test = []
        self.flair_paths_test = []
        self.t1ce_paths_test = []
        self.gt_paths_test = []        
        self.data_t1 = []
        self.data_t2 = []
        self.data_flair = []
        self.data_t1ce = []
        self.gt = []
        self.NCR_NET = []
        self.edema = []
        self.ET = []
        self.idx = None
        self.labels = {'t1':'1', 't2':'2', 'flair':'3', 't1ce':'4', 'seg':'5'}
        
    def load(self):
        self.get_paths_classes()
        for imagePath in self.t1_paths:
            img = cv2.imread(imagePath)
            self.data_t1.append(img)

        for imagePath in self.t2_paths:
            img = cv2.imread(imagePath)
            self.data_t2.append(img)

        for imagePath in self.flair_paths:
            img = cv2.imread(imagePath)
            self.data_flair.append(img)

        for imagePath in self.t1ce_paths:
            img = cv2.imread(imagePath)
            self.data_t1ce.append(img)

        for imagePath in self.gt_paths:
            img = cv2.imread(imagePath)
            
            self.NCR_NET.append((img==1)*255)
            self.edema.append((img==2)*255)
            self.ET.append((img==4)*255)
            self.gt.append(img)
        
    def get_paths_classes(self):
        imagePaths = list(list_images(self.paths))
        for imagePath in imagePaths:
            label = imagePath.split(os.path.sep)[-1].split(".")[0].split("_")[-2]
            patient_num = imagePath.split(os.path.sep)[-1].split(".")[0].split("_")[-3]
            
            if int(patient_num) in self.test_patient_nums:
                if label == self.labels["seg"]:
                    self.gt_paths_test.append(imagePath)               
                elif label == self.labels["t1"]: 
                    self.t1_paths_test.append(imagePath)
                elif label == self.labels["t2"]:  
                    self.t2_paths_test.append(imagePath)
                elif label == self.labels["flair"]:  
                    self.flair_paths_test.append(imagePath)
                else:
                    self.t1ce_paths_test.append(imagePath)
            else:                    
                if label == self.labels["seg"]:
                    self.gt_paths.append(imagePath)               
                elif label == self.labels["t1"]: 
                    self.t1_paths.append(imagePath)
                elif label == self.labels["t2"]:  
                    self.t2_paths.append(imagePath)
                elif label == self.labels["flair"]:  
                    self.flair_paths.append(imagePath)
                else:
                    self.t1ce_paths.append(imagePath)   
                    
        self.gt_paths_test = self.sort(self.gt_paths_test)  
        self.t1_paths_test = self.sort(self.t1_paths_test)
        self.t2_paths_test = self.sort(self.t2_paths_test)
        self.flair_paths_test = self.sort(self.flair_paths_test)
        self.t1ce_paths_test = self.sort(self.t1ce_paths_test)
        
        self.gt_paths = self.sort(self.gt_paths)  
        self.t1_paths = self.sort(self.t1_paths)
        self.t2_paths = self.sort(self.t2_paths)
        self.flair_paths = self.sort(self.flair_paths)
        self.t1ce_paths = self.sort(self.t1ce_paths)
    
    # make sure the image order -> not 1,10,11,..,100,..,2,20,21,..,3,30,..
    def sort(self, imagePaths):
        tmp = []
        for i, imagePath in enumerate(imagePaths):
            img_num = imagePath.split(os.path.sep)[-1].split(".")[0].split("_")[-1]
            tmp.append((int(img_num), i))
        tmp.sort()
        img_num_sorted = np.array(tmp)[:,1]    
        paths_sorted = np.array(imagePaths.copy())
        
        return list(paths_sorted[img_num_sorted])    

# make list of image with diff categories/ avoid .txt ect
def list_images(basePath, contains=None):
    image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    # return the set of files that are valid
    return list_files(basePath, validExts=image_types, contains=contains)

def list_files(basePath, validExts=None, contains=None):
    
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename)
                yield imagePath             