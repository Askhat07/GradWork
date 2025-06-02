import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchsummary import summary
import time

if __name__ == "__main__":

    train_directory = "C:/Users/ashat/PycharmProjects/GraduateWork/Plant/Plant Diseases Dataset/Plant Diseases Dataset/train"
    validation_directory = "C:/Users/ashat/PycharmProjects/GraduateWork/Plant/Plant Diseases Dataset/Plant Diseases Dataset/valid"
    diseases = os.listdir(train_directory)
    # Printing the disease names
    print(diseases)

    print("Total classes: {}".format(len(diseases)))

    # Extraction of unique plants and diseases
    plants_array = []
    number_disease = 0
    for plant in diseases:
        if plant.split('___')[0] not in plants_array:
            plants_array.append(plant.split('___')[0])
        if plant.split('___')[1] != 'healthy':
            number_disease += 1

    # Unique plants in the dataset
    print("Unique plants: ", plants_array)
    # Number of unique plants
    print("Number plants: {}".format(len(plants_array)))
    # Number of unique diseases
    print("Numbers disease: {}".format(number_disease))

    # Number of images for each disease
    nums_list = {}
    for disease in diseases:
        nums_list[disease] = len(os.listdir(train_directory + '/' + disease))
    # Converting the nums dictionary to pandas dataframe passing index as plant name and number of
    # images as column
    image_per_class = pd.DataFrame(nums_list.values(), index=nums_list.keys(), columns=["number of images"])
    print(image_per_class)

    # Plotting number of images available for each disease
    index = [n for n in range(38)]
    plt.figure(figsize=(20, 5))
    plt.bar(index, [n for n in nums_list.values()], width=0.65)
    plt.xlabel('Plants/Diseases', fontsize=10)
    plt.ylabel('Number of images', fontsize=10)
    plt.xticks(index, diseases, fontsize=5, rotation=90)
    plt.title('Images for each class of plant diseases')
    plt.show()

    n_train = 0
    for val in nums_list.values():
        n_train += val
    print(f"There are {n_train} images for training")

    # Datasets for validation and training
    train = ImageFolder(train_directory, transform=transforms.ToTensor())
    validation = ImageFolder(validation_directory, transform=transforms.ToTensor())

    image, label = train[0]
    print(image.shape, label)
    # Total number of classes in train set
    print(len(train.classes))
    # For checking some images from training dataset
    def showImage(image, label):
        print("Label: " + train.classes[label] + "(" + str(label) + ")")
        plt.imshow(image.permute(1, 2, 0))
        plt.show()

    # showImage(*train[30000])

    # Setting the seed value
    randomSeed = 7
    print(torch.manual_seed(randomSeed))
    # Setting the batch size
    batchSize = 32

    # DataLoaders for training and validation
    trainDataLoaders = DataLoader(train, batchSize, shuffle=True, num_workers=2, pin_memory=True)
    validationDataLoaders = DataLoader(validation, batchSize, num_workers=2, pin_memory=True)

    # # Helper function to show a batch of training instances
    # def showBatch(data):
    #     for images, labels in data:
    #         fig, ax = plt.subplots(figsize=(30, 30))
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         ax.imshow(make_grid(images, nrow=8).permute(1, 2, 0))
    #         plt.show()
    #         break
    # # Images for first batch of training
    # showBatch(trainDataLoaders)

    # For moving data into GPU
    def getDefaultDevice():
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    # For moving data to device (CPU or GPU)
    def toDevice(data, device):
        if isinstance(data, (list, tuple)):
            return [toDevice(x, device) for x in data]
        return data.to(device, non_blocking=True)

    # For loading in the device (GPU if available else CPU)
    class device_data_loader():
        def __init__(self, dataLoad, device):
            self.dataLoad = dataLoad
            self.device = device

        def __iter__(self):
            for batch in self.dataLoad:
                yield toDevice(batch, self.device)

        def __len__(self):
            return len(self.dataLoad)

    dev = getDefaultDevice()
    print(dev)
    # Moving data into GPU
    trainDataLoaders = device_data_loader(trainDataLoaders, dev)
    validationDataLoaders = device_data_loader(validationDataLoaders, dev)

    # class simple_residual_block(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.conv_one = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
    #         self.relu_one = nn.ReLU()
    #         self.conv_two = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
    #         self.relu_two = nn.ReLU()
    #
    #     def forward(self, x):
    #         out = self.conv_one(x)
    #         out = self.relu_one(out)
    #         out = self.conv_two(out)
    #         return self.relu_two(out) + x

    # For calculating the accuracy
    def accuracy(outputs, labels):
        _, predictions = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(predictions == labels).item() / len(predictions))

    # Base class for the model
    class image_classification_base(nn.Module):
        def trainingStep(self, batch):
            images, labels = batch
            out = self(images)                  # Generate predictions
            loss = F.cross_entropy(out, labels) # Calculate loss
            return loss

        def validationStep(self, batch):
            images, labels = batch
            out = self(images)                  # Generate prediction
            loss = F.cross_entropy(out, labels) # Calculate loss
            accu = accuracy(out, labels)        # Calculate accuracy
            return {"validation_loss": loss.detach(), "validation_accuracy": accu}

        def validationEpochEnd(self, outputs):
            batch_loss = [x["validation_loss"] for x in outputs]
            batch_accu = [x["validation_accuracy"] for x in outputs]
            epoch_loss = torch.stack(batch_loss).mean() # Combine loss
            epoch_accu = torch.stack(batch_accu).mean() # Combine accuracy
            return {"validation_loss": epoch_loss, "validation_accuracy": epoch_accu}

        def epochEnd(self, epoch, res):
            print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, validation_loss: {:.4f}, validation_accuracy: {:.4f}".format(
                epoch, res['lrs'][-1], res['train_loss'], res['validation_loss'], res['validation_accuracy']
            ))

    # Architecture for training

    # Convolution block with BatchNormalization
    def conv_block(in_channels, out_channels, pool=False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool2d(4))
        return nn.Sequential(*layers)

    # ResNet architecture
    class res_net_9(image_classification_base):
        def __init__(self, in_channels, number_diseases):
            super().__init__()

            self.conv_one = conv_block(in_channels, 64)
            self.conv_two = conv_block(64, 128, pool=True) # out_dim: 128 x 64 x 64
            self.res_one = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

            self.conv_three = conv_block(128, 256, pool=True) # out_dim: 256 x 16 x 16
            self.conv_four = conv_block(256, 512, pool=True)  # out_dim: 512 x 4 x 4
            self.res_two = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

            self.classifier = nn.Sequential(nn.MaxPool2d(4), nn.Flatten(), nn.Linear(512, number_diseases))

        def forward(self, xb):  # xb is the loaded batch
            out = self.conv_one(xb)
            out = self.conv_two(out)
            out = self.res_one(out) + out
            out = self.conv_three(out)
            out = self.conv_four(out)
            out = self.res_two(out) + out
            out = self.classifier(out)
            return out

    # Defining the model and moving it to the GPU
    model = toDevice(res_net_9(3, len(train.classes)), dev)
    print(model)

    # getting summary of the model
    Input_Shape = (3, 256, 256)
    print(summary(model.cuda(), (Input_Shape)))

    # For training
    @torch.no_grad()
    def evaluate(model, validation_loader):
        model.eval()
        outputs = [model.validationStep(batch) for batch in validation_loader]
        return model.validationEpochEnd(outputs)

    def getLr(optimizer):
        for paramGroup in optimizer.param_groups:
            return paramGroup['lr']

    def fitOneCycle(epochs, max_lr, model, trainLoader, validationLoader, weight_decay=0,
                    grad_clip=None, opt_func=torch.optim.SGD):
        torch.cuda.empty_cache()
        history = []
        optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
        # Scheduler for one cycle learning rate
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(trainLoader))

        for epoch in range(epochs):
            # Training
            model.train()
            trainLosses = []
            lrs = []
            for batch in trainLoader:
                loss = model.trainingStep(batch)
                trainLosses.append(loss)
                loss.backward()

                # Gradient clipping
                if grad_clip:
                    nn.utils.clip_grad_value_(model.parameters(), grad_clip)

                optimizer.step()
                optimizer.zero_grad()

                # Recording and updating learning rates
                lrs.append(getLr(optimizer))
                scheduler.step()

            # validation
            result = evaluate(model, validationLoader)
            result['train_loss'] = torch.stack(trainLosses).mean().item()
            result['lrs'] = lrs
            model.epochEnd(epoch, result)
            history.append(result)
        return history

    epochs = 2
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam

    startTime = time.time()
    history = [evaluate(model, validationDataLoaders)]
    history += fitOneCycle(epochs, max_lr, model, trainDataLoaders, validationDataLoaders, grad_clip=grad_clip,
                           weight_decay=1e-4, opt_func=opt_func)

    print("Execution time: {:.2f} seconds".format(time.time() - startTime))
    print(history)

    def show_accuracies(history):
        accuracies = [x['validation_accuracy'] for x in history]
        plt.plot(accuracies, '-x')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Accuracy vs Number of epochs')
        plt.show()

    def show_losses(history):
        trainLosses = [x.get('train_loss').cpu().numpy()
                       if isinstance(x.get('train_loss'), torch.Tensor)
                       else x.get('train_loss') for x in history]
        validationLosses = [x['validation_loss'].cpu().numpy()
                            if isinstance(x['validation_loss'], torch.Tensor)
                            else x['validation_loss'] for x in history]
        plt.plot(trainLosses, '-bx')
        plt.plot(validationLosses, '-rx')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Training', 'Validation'])
        plt.title('Loss vs Number of epochs')
        plt.show()

    def show_lrs(history):
        lrs = np.concatenate([x.get('lrs', []) for x in history])
        plt.plot(lrs)
        plt.xlabel('Batch number')
        plt.ylabel('Learning rate')
        plt.title('Learning rate vs Batch number')
        plt.show()

    show_accuracies(history)
    show_losses(history)
    show_lrs(history)


    test_directory = "C:/Users/ashat/PycharmProjects/GraduateWork/Plant/Plant Diseases Dataset/test"
    test = ImageFolder(test_directory, transform=transforms.ToTensor())

    testImages = sorted(os.listdir(test_directory + '/test')) # Since images in test folder are in alphabetical order
    print(testImages)

    # Converts image to array and return the predicted class with highest probability
    def predictImage(img, model):
        # Convert to a batch of 1
        xb = toDevice(img.unsqueeze(0), dev)
        # Get predictions from model
        yb = model(xb)
        # Pick index with highest probability
        _, predictions = torch.max(yb, dim=1)
        # Retrieve the class label
        return train.classes[predictions[0].item()]

    # getting all predictions (actual label vs predicted)
    for i, (img, label) in enumerate(test):
        print('Label:', testImages[i], ', Predicted:', predictImage(img, model))

    # # save/load state_dict
    # state_dict_path = 'C:/Users/ashat/PycharmProjects/GraduateWork/Plant/Plant Diseases Dataset/plant-disease-model.pth'
    # torch.save(model.state_dict(), state_dict_path)
    #
    # # save/load entire model
    # complete_model_path = 'C:/Users/ashat/PycharmProjects/GraduateWork/Plant/Plant Diseases Dataset/plant-disease-model-complete.pth'
    # torch.save(model, complete_model_path)