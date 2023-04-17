#%%
# USAGE
# import the necessary packages
from biomil.dataset_dunet_multilabels_byPatients import SegmentationDataset, BraTS2020loader
from biomil.dunet_revised import D_Unet
from biomil import cfg_dunet_multilabels_byPatients as config
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch
import time
import os
import albumentations as A


MODEL_NAME = f"dunet_DL_{config.NUM_EPOCHS}epoch_{config.DATASET_PREP}"
ALPHA = 0.25
GAMMA = 2
class DiceLoss(Module):
    def __init__(self, weight = None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.softmax(inputs, dim=1)
        
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class FocalLoss(Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1): 
        
        #first compute cross-entropy 
        CCE = F.cross_entropy(inputs, targets, reduction='mean')
        CCE_EXP = torch.exp(-CCE)
        focal_loss = alpha * (1-CCE_EXP)**gamma * CCE
                       
        return focal_loss

class EnhancedMixingLoss(Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.softmax(inputs, dim=1)       
        
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # first compute binary cross-entropy and then focal loss
        CCE = F.nll_loss(inputs, targets, reduction='mean')
        CCE_EXP = torch.exp(-CCE)
        focal_loss = alpha * (1-CCE_EXP)**gamma * CCE     
        
        # Dice loss
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        # diceloss = 1 - dice
        
        return focal_loss - torch.log(dice)

def dice_coef2(y_true, y_pred): 
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    union = torch.sum(y_true_f) + torch.sum(y_pred_f)
    if union==0: 
        return 1
    intersection = torch.sum(y_true_f * y_pred_f)
    return 2. * intersection / union    

def plot_training_history(H, pathL, pathD):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["test_loss"], label="test_loss")	
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc='center right')
    plt.savefig(pathL)
    plt.close()

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_dice"], label="train_dice")
    plt.plot(H["test_dice"], label="test_dice")	
    plt.title("Training Dice on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Dice")
    plt.legend(loc='center right')
    plt.savefig(pathD)	
    plt.close()

def main():
    print(f"[INFO]: Model being used: {config.MODEL_PATH}")
    # load the image and mask filepaths in a sorted manner
    print(f"[INFO] DATASET BEING USED : {config.DATASET_PATH}")
    braTS2020 = BraTS2020loader(config.DATASET_PATH)
    braTS2020.get_paths_classes()
    test_patient_nums = [] 
    for num in braTS2020.test_patient_nums:
        test_patient_nums.append(str(num))
    
    trainT1s = braTS2020.t1_paths
    trainT2s = braTS2020.t2_paths
    trainFLAIRs = braTS2020.flair_paths
    trainT1CEs = braTS2020.t1ce_paths
    trainMasks = braTS2020.gt_paths

    testT1s = braTS2020.t1_paths_test
    testT2s = braTS2020.t2_paths_test
    testFLAIRs = braTS2020.flair_paths_test
    testT1CEs = braTS2020.t1ce_paths_test
    testMasks = braTS2020.gt_paths_test

    # write the testing image paths to disk so that we can use then
    # when evaluating/testing our model
    print("[INFO] saving training & testing image paths...")
    f = open(config.TEST_PATIENT_NUMS, "w")
    f.write("\n".join([str(num) for num in braTS2020.test_patient_nums]))
    f.close()

    f = open(config.TEST_T1_PATHS, "w")
    f.write("\n".join(testT1s))
    f.close()

    f = open(config.TEST_T2_PATHS, "w")
    f.write("\n".join(testT2s))
    f.close()

    f = open(config.TEST_FLAIR_PATHS, "w")
    f.write("\n".join(testFLAIRs))
    f.close()

    f = open(config.TEST_T1CE_PATHS, "w")
    f.write("\n".join(testT1CEs))
    f.close()

    f = open(config.TEST_MASK_PATHS, "w")
    f.write("\n".join(testMasks))
    f.close()

    f = open(config.TRAIN_T1_PATHS, "w")
    f.write("\n".join(trainT1s))
    f.close()

    f = open(config.TRAIN_T2_PATHS, "w")
    f.write("\n".join(trainT2s))
    f.close()

    f = open(config.TRAIN_FLAIR_PATHS, "w")
    f.write("\n".join(trainFLAIRs))
    f.close()

    f = open(config.TRAIN_T1CE_PATHS, "w")
    f.write("\n".join(trainT1CEs))
    f.close()

    f = open(config.TRAIN_MASK_PATHS, "w")
    f.write("\n".join(trainMasks))
    f.close()				

    # define transformations
    T = transforms.Compose([transforms.ToPILImage(),
        # transforms.CenterCrop(192),
        # transforms.Resize((config.INPUT_IMAGE_HEIGHT,
        # 	config.INPUT_IMAGE_WIDTH)),
        transforms.ToTensor()])

    T_aug = A.Compose([
    # A.CenterCrop(width=192, height=192)
    A.CenterCrop(width=192, height=192),
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.3),
    # A.RandomGamma(p=0.3),
    A.Rotate(limit=30, p=0.15),
    # A.RandomBrightnessContrast(p=0.3),
    # A.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.01)
    ])

    # create the train and test datasets
    trainDS = SegmentationDataset(t1_Paths=trainT1s,
                                    t2_Paths=trainT2s,
                                    flair_Paths=trainFLAIRs,
                                    t1ce_Paths=trainT1CEs,
                                    mask_Paths=trainMasks,
                                    torch_transforms=T,
                                    aug_transforms=T_aug)
    testDS = SegmentationDataset(t1_Paths=testT1s,
                                    t2_Paths=testT2s,
                                    flair_Paths=testFLAIRs,
                                    t1ce_Paths=testT1CEs,
                                    mask_Paths=testMasks,
                                    torch_transforms=T,
                                    aug_transforms=T_aug)
    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(testDS)} examples in the test set...")

    # create the training and test data loaders
    trainLoader = DataLoader(trainDS, shuffle=True,
        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
        num_workers=8 , persistent_workers=True)
    testLoader = DataLoader(testDS, shuffle=False,
        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
        num_workers=8 , persistent_workers=True)

    # dunet = D_Unet(multilabels=True).cuda()
    dunet = (D_Unet(multilabels=True).to(config.DEVICE))

    # initialize loss function and optimizer
    lossFunc = DiceLoss()
    # lossFunc = FocalLoss()
    opt = Adam(dunet.parameters(), lr=config.INIT_LR)

    # calculate steps per epoch for training and test set
    trainSteps = len(trainDS) // config.BATCH_SIZE
    testSteps = len(testDS) // config.BATCH_SIZE

    # initialize a dictionary to store training history
    H = {"train_loss": [], "test_loss": [], "train_dice":[], "test_dice":[]}

    # loop over epochs
    print("[INFO] training the network...")
    startTime = time.time()
    for e in tqdm(range(config.NUM_EPOCHS)):
        # set the model in training mode
        dunet.train()

        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalTestLoss = 0
        totalTrainDice = 0
        totalTestDice = 0

        # loop over the training set
        for (i, (x, y)) in enumerate(trainLoader):
            # send the input to the device
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

            # perform a forward pass and calculate the training loss
            pred = dunet(x)
            loss = lossFunc(pred, y)

            # calculate training dice
            tmp = F.log_softmax(pred, dim=1).argmax(dim=1)
            gt = y.argmax(dim=1)
            dice = dice_coef2(tmp>0, gt>0)

            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            opt.zero_grad()
            loss.backward()
            opt.step()

            # add the loss to the total training loss so far
            totalTrainLoss += loss
            totalTrainDice += dice

        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            dunet.eval()

            # loop over the validation set
            for (x, y) in testLoader:
                # send the input to the device
                (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

                # make the predictions and calculate the validation loss
                pred = dunet(x)
                totalTestLoss += lossFunc(pred, y)

                #calculate testing dice
                tmp = F.log_softmax(pred, dim=1).argmax(dim=1)
                gt = y.argmax(dim=1)				
                totalTestDice += dice_coef2(tmp>0, gt>0)	

        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgTestLoss = totalTestLoss / testSteps
        avgTrainDice = totalTrainDice/ trainSteps
        avgTestDice = totalTestDice / testSteps
        print(f'avgTrainLoss={type(avgTrainLoss)}, avgTestLoss={type(avgTestLoss)}, avgTrainDice={type(avgTrainDice)}, avgTestDice={type(avgTestDice)}')
        print(f'avgTrainLoss={avgTrainLoss}, avgTestLoss={avgTestLoss}, avgTrainDice={avgTrainDice}, avgTestDice={avgTestDice}')

        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
        H["train_dice"].append(avgTrainDice.detach().cpu().numpy())
        H["test_dice"].append(avgTestDice.detach().cpu().numpy())

        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
        print("Train loss: {:.6f}, Test loss: {:.4f}, Train dice: {:.6f}, Test dice: {:.4f}".format(
            avgTrainLoss, avgTestLoss, avgTrainDice, avgTestDice))

        if avgTestDice > 0.95:
            torch.save(dunet.state_dict(), os.path.join(config.BASE_OUTPUT, (MODEL_NAME+str(e+1)+"epoch_95.pth")))
            pathL = os.path.join(config.BASE_OUTPUT, ("plotL_"+MODEL_NAME+str(e+1)+"epoch_95.png"))
            pathD = os.path.join(config.BASE_OUTPUT, ("plotD_"+MODEL_NAME+str(e+1)+"epoch_95.png"))
            plot_training_history(H, pathL, pathD)
        elif avgTestDice > 0.94:
            torch.save(dunet.state_dict(), os.path.join(config.BASE_OUTPUT, (MODEL_NAME+str(e+1)+"epoch_94.pth")))
            pathL = os.path.join(config.BASE_OUTPUT, ("plotL_"+MODEL_NAME+str(e+1)+"epoch_94.png"))
            pathD = os.path.join(config.BASE_OUTPUT, ("plotD_"+MODEL_NAME+str(e+1)+"epoch_94.png"))
            plot_training_history(H, pathL, pathD)
        elif avgTestDice > 0.93:
            torch.save(dunet.state_dict(), os.path.join(config.BASE_OUTPUT, (MODEL_NAME+str(e+1)+"epoch_93.pth")))
            pathL = os.path.join(config.BASE_OUTPUT, ("plotL_"+MODEL_NAME+str(e+1)+"epoch_93.png"))
            pathD = os.path.join(config.BASE_OUTPUT, ("plotD_"+MODEL_NAME+str(e+1)+"epoch_93.png"))
            plot_training_history(H, pathL, pathD)
        elif avgTestDice > 0.92:
            torch.save(dunet.state_dict(), os.path.join(config.BASE_OUTPUT, (MODEL_NAME+str(e+1)+"epoch_92.pth")))
            pathL = os.path.join(config.BASE_OUTPUT, ("plotL_"+MODEL_NAME+str(e+1)+"epoch_92.png"))
            pathD = os.path.join(config.BASE_OUTPUT, ("plotD_"+MODEL_NAME+str(e+1)+"epoch_92.png"))
            plot_training_history(H, pathL, pathD)
        elif avgTestDice > 0.91:
            torch.save(dunet.state_dict(), os.path.join(config.BASE_OUTPUT, (MODEL_NAME+str(e+1)+"epoch_91.pth")))
            pathL = os.path.join(config.BASE_OUTPUT, ("plotL_"+MODEL_NAME+str(e+1)+"epoch_91.png"))
            pathD = os.path.join(config.BASE_OUTPUT, ("plotD_"+MODEL_NAME+str(e+1)+"epoch_91.png"))
            plot_training_history(H, pathL, pathD)
        elif avgTestDice > 0.90:
            torch.save(dunet.state_dict(), os.path.join(config.BASE_OUTPUT, (MODEL_NAME+str(e+1)+"epoch_90.pth")))	
            pathL = os.path.join(config.BASE_OUTPUT, ("plotL_"+MODEL_NAME+str(e+1)+"epoch_90.png"))
            pathD = os.path.join(config.BASE_OUTPUT, ("plotD_"+MODEL_NAME+str(e+1)+"epoch_90.png"))
            plot_training_history(H, pathL, pathD)	
        elif avgTestDice > 0.89:
            torch.save(dunet.state_dict(), os.path.join(config.BASE_OUTPUT, (MODEL_NAME+str(e+1)+"epoch_89.pth")))
            pathL = os.path.join(config.BASE_OUTPUT, ("plotL_"+MODEL_NAME+str(e+1)+"epoch_89.png"))
            pathD = os.path.join(config.BASE_OUTPUT, ("plotD_"+MODEL_NAME+str(e+1)+"epoch_89.png"))
            plot_training_history(H, pathL, pathD)	
        elif avgTestDice > 0.88:
            torch.save(dunet.state_dict(), os.path.join(config.BASE_OUTPUT, (MODEL_NAME+str(e+1)+"epoch_88.pth")))
            pathL = os.path.join(config.BASE_OUTPUT, ("plotL_"+MODEL_NAME+str(e+1)+"epoch_88.png"))
            pathD = os.path.join(config.BASE_OUTPUT, ("plotD_"+MODEL_NAME+str(e+1)+"epoch_88.png"))
            plot_training_history(H, pathL, pathD)	
        elif avgTestDice > 0.87:
            torch.save(dunet.state_dict(), os.path.join(config.BASE_OUTPUT, (MODEL_NAME+str(e+1)+"epoch_87.pth")))
            pathL = os.path.join(config.BASE_OUTPUT, ("plotL_"+MODEL_NAME+str(e+1)+"epoch_87.png"))
            pathD = os.path.join(config.BASE_OUTPUT, ("plotD_"+MODEL_NAME+str(e+1)+"epoch_87.png"))
            plot_training_history(H, pathL, pathD)	
        elif avgTestDice > 0.86:
            torch.save(dunet.state_dict(), os.path.join(config.BASE_OUTPUT, (MODEL_NAME+str(e+1)+"epoch_86.pth")))
            pathL = os.path.join(config.BASE_OUTPUT, ("plotL_"+MODEL_NAME+str(e+1)+"epoch_86.png"))
            pathD = os.path.join(config.BASE_OUTPUT, ("plotD_"+MODEL_NAME+str(e+1)+"epoch_86.png"))
            plot_training_history(H, pathL, pathD)	
        elif avgTestDice > 0.85:
            torch.save(dunet.state_dict(), os.path.join(config.BASE_OUTPUT, (MODEL_NAME+str(e+1)+"epoch_85.pth")))
            pathL = os.path.join(config.BASE_OUTPUT, ("plotL_"+MODEL_NAME+str(e+1)+"epoch_85.png"))
            pathD = os.path.join(config.BASE_OUTPUT, ("plotD_"+MODEL_NAME+str(e+1)+"epoch_85.png"))
            plot_training_history(H, pathL, pathD)	
        elif avgTestDice > 0.84:
            torch.save(dunet.state_dict(), os.path.join(config.BASE_OUTPUT, (MODEL_NAME+str(e+1)+"epoch_84.pth")))
            pathL = os.path.join(config.BASE_OUTPUT, ("plotL_"+MODEL_NAME+str(e+1)+"epoch_84.png"))
            pathD = os.path.join(config.BASE_OUTPUT, ("plotD_"+MODEL_NAME+str(e+1)+"epoch_84.png"))
            plot_training_history(H, pathL, pathD)	
        elif avgTestDice > 0.83:
            torch.save(dunet.state_dict(), os.path.join(config.BASE_OUTPUT, (MODEL_NAME+str(e+1)+"epoch_83.pth")))
            pathL = os.path.join(config.BASE_OUTPUT, ("plotL_"+MODEL_NAME+str(e+1)+"epoch_83.png"))
            pathD = os.path.join(config.BASE_OUTPUT, ("plotD_"+MODEL_NAME+str(e+1)+"epoch_83.png"))
            plot_training_history(H, pathL, pathD)	
        elif avgTestDice > 0.82:
            torch.save(dunet.state_dict(), os.path.join(config.BASE_OUTPUT, (MODEL_NAME+str(e+1)+"epoch_82.pth")))
            pathL = os.path.join(config.BASE_OUTPUT, ("plotL_"+MODEL_NAME+str(e+1)+"epoch_82.png"))
            pathD = os.path.join(config.BASE_OUTPUT, ("plotD_"+MODEL_NAME+str(e+1)+"epoch_82.png"))
            plot_training_history(H, pathL, pathD)	
        elif avgTestDice > 0.81:
            torch.save(dunet.state_dict(), os.path.join(config.BASE_OUTPUT, (MODEL_NAME+str(e+1)+"epoch_81.pth")))
            pathL = os.path.join(config.BASE_OUTPUT, ("plotL_"+MODEL_NAME+str(e+1)+"epoch_81.png"))
            pathD = os.path.join(config.BASE_OUTPUT, ("plotD_"+MODEL_NAME+str(e+1)+"epoch_81.png"))
            plot_training_history(H, pathL, pathD)	
        elif avgTestDice > 0.80:
            torch.save(dunet.state_dict(), os.path.join(config.BASE_OUTPUT, (MODEL_NAME+str(e+1)+"epoch_80.pth")))
            pathL = os.path.join(config.BASE_OUTPUT, ("plotL_"+MODEL_NAME+str(e+1)+"epoch_80.png"))
            pathD = os.path.join(config.BASE_OUTPUT, ("plotD_"+MODEL_NAME+str(e+1)+"epoch_80.png"))
            plot_training_history(H, pathL, pathD)	

    # display the total time needed to perform the training
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["test_loss"], label="test_loss")	
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc='center right')
    plt.savefig(config.PLOT_PATH_LOSS)

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_dice"], label="train_dice")
    plt.plot(H["test_dice"], label="test_dice")	
    plt.title("Training Dice on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Dice")
    plt.legend(loc='center right')
    plt.savefig(config.PLOT_PATH_DICE)

    plt.close("all")

    # serialize the model to disk
    torch.save(dunet.state_dict(), config.MODEL_PATH)

if __name__ == "__main__":
    main()
