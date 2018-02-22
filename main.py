import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from datasets import KITTIDataset, RedDotDataset
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
import numpy as np
import cv2
from tqdm import tqdm

IMAGE_SIZE = (128, 128)

class KFNet(nn.Module):

    """
        Allowable modes include:
            'feed_forward'
            'cnn_with_R_likelihood'
            'backprop_kf'
    """
    def __init__(self, mode, args):
        super(KFNet, self).__init__()
        self.mode = mode
        self.build_conv()
        self.build_KF()
        self.args = args
	self.set_grads_for_mode()

    # how do I define a function not on self?
    # Does using this as a self mean that each norm layer will be identical//is that an issue
    def resp_norm(self, input):
        pass

    def change_mode(self, mode):
	self.mode = mode
	self.set_grads_for_mode()

    def set_grads_for_mode():
        if self.mode == 'feed_forward':
            self.fc3_L.weights.requires_grad = False
            self.fc3_L.bias.requires_grad = False
        
        if self.mode == 'feed_forward' or self.mode == 'cnn_with_R_likelihood':
            return

        if self.mode == 'backprop_kf':
            return 

    def build_conv(self):
        # (n - k)/s + 1
        # 128 --> (119)/2 + 1 = 60 --> (60 - 2)/2 + 1 = 30 --> (30 - 9)/2 + 1 = 11 --> (11 - 2)/2 + 1 = 5 * 8 = 40
        self.conv1 = nn.Conv2d(3, 4, 9, stride=2) 
        self.batchnorm1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, 9, stride=2)
        self.batchnorm2 = nn.BatchNorm2d(8)
        self.fc1 = nn.Linear(200, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3_z = nn.Linear(32, 2)
        self.fc3_L = nn.Linear(32, 3)

    def build_KF(self):
        pass
        # Consider single train example first, then batch
        # for each frame in sequence:
            # h'[t+1] = Ah[t]
            # Sig'[t+1] = ASig_x[t]A.T + B_WQB_W.T
            # K[t+1] = Sig'[t+1]CT(CSig'[t+1]CT+R[t+1])^-1
            # set h[t+1] = h'[t+1] + K[t+1](z[t+1] - Ch'[t+1])
            # Sig[t+1] = (I - K[t+1]C)Sig'[t+1]



    def forward(self, x):
        x = x.view(-1, IMAGE_SIZE[0], IMAGE_SIZE[0], 3)
        x = x.permute(0,3,1,2)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2, stride=2)
        x = self.batchnorm1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2, stride=2)
        x = self.batchnorm2(x)
        x = x.view(-1, 200)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        z = self.fc3_z(x)
        z = z.view(self.args.batch_size, -1, 2)

        if self.mode == 'feed_forward':
            return z

        L = self.fc3_L(x)
        L_m = Variable(torch.FloatTensor(L.shape[0], 2, 2).zero_())
        L_m[:,0,0] = L[:,0]
        L_m[:,1,0] = L[:,1]
        L_m[:,1,1] = L[:,2]
        R = torch.bmm(L_m, torch.transpose(L_m, 1, 2))
        R = R.view(self.args.batch_size, -1, 2, 2)

        if self.mode == 'cnn_with_R_likelihood':
            return z, R

    	# Pass entire batch of images through convolutional net
    	# Apply recurrence of KF
    	# Return final state estimate 


    def loss(self, predictions, labels):
        # depending on mode
            # If just training CNN: MSE loss on positions 
            # If also R now: maximize LL
            # End to end: loss is MSE over all timesteps of output observation of timestep - gt_observation 
        if self.mode == 'feed_forward':
            loss = F.mse_loss(predictions.view(-1, 2), labels.contiguous().view(-1, 2))
        elif self.mode == 'cnn_with_R_likelihood':
            z = predictions[0].view(-1, 2)
            R = predictions[1].view(-1, 2)
            labels = labels.contiguous().view(-1, 2)
            #[batch_size][2] * [batch_size][2][2] * [batch_size][2]
            R_inv = [t.inverse() for t in torch.functional.unbind(R)]
            R_inv = torch.functional.stack(R_inv)
            nll = torch.bmm(z, torch.bmm(R, z))
            loss = torch.mean(nll) 
        elif self.mode == 'backprop_kf':
            loss = 0
        
        return loss


def init_model(args, is_cuda, batch_size, model_type):
    model = KFNet(model_type, args)
    if is_cuda:
        model.cuda()
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.Adam(params)

    kwargs = {'num_workers': 1, 'pin_memory': True} if is_cuda else {}

    train_dataset = RedDotDataset(base_dir='redDot/')

    # Parameters for loading KITTI Dataset
    #train_dataset = KITTIDataset(frame_selections_file='sequences.csv',
    #                                           images_dir='dataset/sequences/',
    #                                           poses_dir='dataset/poses/')
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, **kwargs) ## Maybe: num_workers=4
    return model, optimizer, train_loader

def train(model, optimizer, train_loader, epoch, is_cuda, log_interval, save_model):
    model.train()
    for batch_idx, sampled_batch in enumerate(tqdm(train_loader)):
        image_data = sampled_batch['images']
        gt_poses = sampled_batch['gt']

        if is_cuda:
            image_data, gt_poses = image_data.cuda(), gt_poses.cuda()

        image_data, gt_poses = Variable(image_data.float()), Variable(gt_poses.float()) ### TODO: figure out why this exits
        optimizer.zero_grad()
        output = model(image_data)

        loss = model.loss(output, gt_poses[:,0:2,:].squeeze())
        loss.backward()
        optimizer.step()

        #if batch_idx % log_interval == 0:
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #        epoch, batch_idx * image_data.shape[0], len(train_loader.dataset),
        #        100. * batch_idx / len(train_loader), loss.data[0]))
    
    if(save_model):
        torch.save(model, "epoch/" + str(epoch))

def test(epoch, model, loader, is_cuda):
    model.eval()
    loss = []
    for batch_idx, sampled_batch in enumerate(loader):
        image_data = sampled_batch['images']
        gt_poses = sampled_batch['gt']

        if is_cuda:
            image_data, gt_poses = image_data.cuda(), gt_poses.cuda()

        image_data, gt_poses = Variable(image_data.float()), Variable(gt_poses.float()) ### TODO: figure out why this exits
        output = model(image_data)
        loss.append(F.mse_loss(output.view(-1, 2), gt_poses[:,0:2,:].squeeze().contiguous().view(-1, 2)).data[0])
        visualize_result(gt_poses.squeeze().view(-1,4), output.squeeze(), str(epoch) + "_" + str(batch_idx))
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, np.mean(loss)))

def init_image():
    img = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[0], 3))
    img.fill(255)
    return img

def write_traj_on_image(img, traj, color):
    N = traj.shape[1]
    print(traj.shape)
    for i in range(N-1):
        cv2.line(img, (traj[0,i]+64, traj[1,i]+64), (traj[0,i+1]+64, traj[1,i+1]+64), color)
    
def visualize_result(gt_traj, est_traj, idx):
    img = init_image()
    write_traj_on_image(img, gt_traj, (0,255,0))
    write_traj_on_image(img, est_traj, (0,0,0))
    cv2.imwrite("results/" + str(idx) + ".png", img)

def main():
    # Training settings 
    parser = argparse.ArgumentParser(description='Deep Visual Odometry using Backprop KF')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--load_model', action='store_true', default=False, help='turn on model saving')
    parser.add_argument('--model-path', type=str, help='model to load')
    parser.add_argument('--save-model', action='store_true', default=False, help='save model (default False)')
    parser.add_argument('--model-type', type=str, default='feed_forward', help='model type')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists("results"): 
        os.makedirs("results")

    ## Later, add test function from https://github.com/pytorch/examples/blob/master/mnist/main.py
    model, optimizer, train_loader = init_model(args, args.cuda, args.batch_size, args.model_type)
    if args.load_model and args.model_path: model = torch.load(args.model_path)
    for epoch in range(1, args.epochs + 1):
        train(model, optimizer, train_loader, epoch, args.cuda, args.log_interval, args.save_model)
        test(epoch, model, train_loader, args.cuda)

if __name__ == "__main__": main()
