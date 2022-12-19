import argparse
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import os
import matplotlib.image as mpimg
from sklearn.metrics import f1_score
from skimage import morphology
from PIL import Image

from data_process import *
from mask_to_submission import *
from metrics import *
from model import *
from functions import *


def img_float_to_uint8(img, PIXEL_DEPTH = 255):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg

# to make predictions applied to images to show road and background
def target_newimg(img, predicted_img, PIXEL_DEPTH = 255):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img



def run(args):

    # have a path for results
    
    output_dir = "result/"
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # build model and train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = eval(args.model)().to(device)
    criterion = eval(args.loss)()
    # whether use previous trained model
    if args.use_model:
        
        model.load_state_dict(torch.load(output_dir + args.model + str(args.learning_rate) + ".pkl", map_location=torch.device("cpu")))
    else:
        # load training set and pre-process the data
        data_dir = "training/"
        train_data_filename = data_dir + "images/"
        train_masks_filename = data_dir + "groundtruth/"

        all_files = os.listdir(train_data_filename)
        random.seed(230)
        random.shuffle(all_files)

        training = int(0.9 * len(all_files))
        train_files = all_files[:training]
        valid_files = all_files[training:]

        train_size = len(train_files)
        #print("Loading " + str(train_size) + " train images")
        train_imgs = np.asarray([data_processing(train_data_filename + train_files[i], label=1) for i in range(train_size)])
        train_masks = np.asarray([data_processing(train_masks_filename + train_files[i], label=0) for i in range(train_size)])

        train_imgs = np.reshape(train_imgs, (9 * train_size, 400, 400, 3))
        train_masks = np.reshape(train_masks, (9 * train_size, 400, 400))
        train_imgs = torch.FloatTensor(train_imgs)
        train_masks = torch.FloatTensor(train_masks)

        #print("After data processing, there are " + str(len(train_imgs)) + " train images")

        #print("Loading " + str(len(valid_files)) + " valid images")
        valid_imgs = np.asarray([mpimg.imread(train_data_filename + valid_files[i]) for i in range(len(valid_files))])
        valid_masks = np.asarray([mpimg.imread(train_masks_filename + valid_files[i]) for i in range(len(valid_files))])
        valid_imgs = torch.tensor(valid_imgs)
        valid_masks = torch.tensor(valid_masks)

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))

        # adjust the learning rate to reduce the loss
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.scaler, patience=5,threshold= 0.001, verbose=True)

        print("Start training")
        
        F1_MAX = 0
        Train_losses = []
        Train_F1 = []
        loss = []
        F1score = []
        steps_per_epoch = int(9 * train_size / args.batch_size)
        print("Total number of iterations = " + str(args.epochs * steps_per_epoch))
        training_indices = range(9* train_size)

        for iepoch in range(args.epochs):

            # Permute training indices
            random.seed(230)
            perm_indices = np.random.permutation(training_indices)
            train_losses, train_f1 = 0, 0

            for step in range(steps_per_epoch):
                offset = (step * args.batch_size) % (9 * train_size)
                batch_indices = perm_indices[offset : (offset + args.batch_size)]

                # Compute the offset of the current minibatch in the data.
                # Note that we could use better randomization across epochs.
                batch_data = train_imgs[batch_indices, :, :, :]
                batch_masks = train_masks[batch_indices, :, :].to(device)
                batch_data = batch_data.permute(0, 3, 1, 2).to(device)

                train_pred = model(batch_data)
                train_pred = torch.sigmoid(train_pred).squeeze(1)
                train_loss = criterion(train_pred, batch_masks)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                with torch.no_grad():
                    F1score_train = F1(train_pred, batch_masks, args)
                    train_losses += train_loss.detach() / steps_per_epoch
                    train_f1 += F1score_train / steps_per_epoch

                if (step + 1) % 9 == 0:

                    print("epoch " + str(iepoch + 1) + " step " + str(step + 1) + " loss: " + str(train_loss.cpu().detach().numpy()) + " F1-score: " + str(F1score_train))

            scheduler.step(train_losses)
            Train_losses.append(train_losses.cpu().detach().numpy())
            Train_F1.append(F1score_train)

            # validation
            valid_losses, valid_f1 = 0, 0
            model.eval()
            valid_data = valid_imgs.permute(0, 3, 1, 2).to(device)
            valid_masks = valid_masks.to(device)
            ind = range(len(valid_data))
            size = 9
            total_step = int(len(valid_data) / size)
            for step in range(total_step):
                offset = (step * size) % len(valid_data)
                batch_indices = ind[offset : (offset + size)]

                batch_data = valid_data[batch_indices, :, :, :].to(device)
                batch_masks = valid_masks[batch_indices, :, :].to(device)

                with torch.no_grad():
                    valid_pred = model(batch_data)
                    valid_pred = torch.sigmoid(valid_pred).squeeze(1)
                    valid_loss = criterion(valid_pred, batch_masks)
                    F1score_valid = F1(valid_pred, batch_masks, args)
                    valid_losses += valid_loss.detach() / total_step
                    valid_f1 += F1score_valid / total_step

            print("valid loss: " + str(valid_losses.cpu().detach().numpy()) + " valid F1-score: " + str(valid_f1))
            if valid_f1 > F1_MAX:
                # save model
                F1_MAX = valid_f1
                torch.save(model.state_dict(), output_dir + args.model + str(args.learning_rate) + ".pkl")

            loss.append(valid_losses.cpu().detach().numpy())
            F1score.append(valid_f1)
            pickle.dump({"Loss": loss, "F1_score": F1score}, open(output_dir + args.model + str(args.learning_rate) + "_performance.pkl", "wb"))
    
    if args.prediction:
        print("Running prediction on test set")
        Results = []
        model.eval()
        test_dir = "test_set_images/"
        n = len(os.listdir(test_dir))
        print("Loading " + str(n) + " test images")
        test_imgs = np.asarray([mpimg.imread(test_dir + "test_" + str(i + 1) + "/" + "test_" + str(i + 1) + ".png") for i in range(n)])
        test_imgs = torch.tensor(test_imgs)
        test_data = test_imgs.permute(0, 3, 1, 2).to(device)
        ind = range(n)
        size = 1  # Because of the computational limitation, we only can process one images simultaneously
        total_step = int(n / size)
        for step in range(total_step):
            offset = (step * size) % n
            batch_indices = ind[offset : (offset + size)]

            batch_data = test_data[batch_indices, :, :, :]

            test_pred = model(batch_data)
            test_pred = torch.sigmoid(test_pred).squeeze(1)

            result = test_pred[0].cpu().detach().numpy()

            result[result <= 0.25] = 0
            result = morphology.remove_small_objects(result.astype(bool), 800)
            Results.append(result)
            #print(result.shape)

            new_img = target_newimg(test_imgs[step, :, :, :].cpu().detach().numpy(), result)
            new_img.save(output_dir + str(step) + "new_img.png")

            print("Already running prediction on " + str(step + 1) + " images")
         #print(Results)

        mask_to_submission(output_dir + args.model + str(args.learning_rate) + "submission.csv", Results)
    # plot the performance of different models
    plt.plot(range(args.epochs), Train_losses, label = 'train loss')
    plt.plot(range(args.epochs), Train_F1, label = 'Train_F1')
    plt.plot(range(args.epochs), loss, label = 'Valid loss')
    plt.plot(range(args.epochs), F1score, label = 'Valid_F1')
    plt.ylabel('Performance')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(output_dir + args.model + str(args.learning_rate) + 'train_loss.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    # optimizer args
    parser.add_argument("--learning_rate", type=float, default=7e-4, help="Set Learning rate")
    parser.add_argument("--scaler", type=float, default=0.5, help="Adjust the learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="First order decaying parameter")
    parser.add_argument("--beta2", type=float, default=0.999, help="Second order decaying parameter")
    # model args
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=6, help="Number of batch sizes")
    parser.add_argument("--model", type=str, default="Res_Unet", help="choose the NN model")
    parser.add_argument("--loss", type=str, default="IoULoss", help="choose the loss function")
    parser.add_argument("--use_model", type=bool, default=False, help="whether use previous trained model")
    parser.add_argument("--prediction", type=bool, default=True, help="whether running prediction on test set")

    args = parser.parse_args()
    print(args)
    run(args)
