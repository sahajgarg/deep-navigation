import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from datasets import KITTIDataset
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

IMAGE_SIZE = (128, 128)


# Training settings 
parser = argparse.ArgumentParser(description='Deep Visual Odometry using Backprop KF')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

class KFNet(nn.Module):

    """
        Allowable modes include:
            'feed_forward'
            'cnn_with_R_likelihood'
            'backprop_kf'
    """
    def __init__(self, mode):
        super(KFNet, self).__init__()
        self.mode = mode
        build_conv()
        build_KF()

    # how do I define a function not on self?
    # Does using this as a self mean that each norm layer will be identical//is that an issue
    def resp_norm(self, input):
        pass


    def build_conv(self):
        # (n - k)/s + 1
        # 128 --> (119)/2 + 1 = 60 --> (60 - 2)/2 + 1 = 30 --> (30 - 9)/2 + 1 = 11 --> (11 - 2)/2 + 1 = 5 * 8 = 40
        self.conv1 = nn.Conv2d(3, 4, 9, stride=2) 
        self.batchnorm1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, 9, stride=2)
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.fc1 = nn.Linear(40, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3_z = nn.Linear(32, 2)

        if self.mode != 'feed_forward':
            self.fc3_L = nn.Linear(32, 3)

    def build_KF(self):
        if self.mode == 'feed_forward' or 'cnn_with_R_likelihood':
            ## Really should build piecewise KF
            return

        # Consider single train example first, then batch
        # for each frame in sequence:
            # h'[t+1] = Ah[t]
            # Sig'[t+1] = ASig_x[t]A.T + B_WQB_W.T
            # K[t+1] = Sig'[t+1]CT(CSig'[t+1]CT+R[t+1])^-1
            # set h[t+1] = h'[t+1] + K[t+1](z[t+1] - Ch'[t+1])
            # Sig[t+1] = (I - K[t+1]C)Sig'[t+1]



    def forward(self, x):
        if self.mode == 'feed_forward':
            x = x.view(-1, IMAGE_SIZE[0], IMAGE_SIZE[0], 3)
            x = F.max_pool2d(F.relu(self.conv1(x)), 2, stride=2)
            x = self.batchnorm1(x)
            x = F.max_pool2d(F.relu(self.conv2(x)), 2, stride=2)
            x = self.batchnorm2(x)
            x = x.view(-1, 40)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            z = self.fc3_z(x)
            z = z.view(args.batch_size, -1, 2)
            return z


    	# Pass entire batch of images through convolutional net
    	# Apply recurrence of KF
    	# Return final state estimate 

    	pass

    def loss(self, predictions, labels):
        # depending on mode
            # If just training CNN: MSE loss on positions 
            # If also R now: maximize LL
            # End to end: loss is MSE over all timesteps of output observation of timestep - gt_observation 
        if self.mode == 'feed_forward':
            loss = F.mse_loss(predictions.view(-1, 2), labels.view(-1, 2))
        elif self.mode == 'cnn_with_R_likelihood':
            loss = 0
        elif self.mode == 'backprop_kf':
            loss = 0
        
        return loss


model = KFNet()
if args.cuda:
	model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

tran_dataset = RedDotDataset(base_dir='redDot/')

# Parameters for loading KITTI Dataset
#train_dataset = KITTIDataset(frame_selections_file='sequences.csv',
#                                           images_dir='dataset/sequences/',
#                                           poses_dir='dataset/poses/')
train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                        shuffle=True, **kwargs) ## Maybe: num_workers=4


def train(epoch):
    model.train()
    print("starting")
    for batch_idx, sampled_batch in enumerate(train_loader):
        image_data = sampled_batch['images']
        gt_poses = sampled_batch['gt']

        if args.cuda:
            image_data, gt_poses = image_data.cuda(), gt_poses.cuda()

        image_data, gt_poses = Variable(image_data), Variable(gt_poses) ### TODO: figure out why this exits
        optimizer.zero_grad()
        output = model(image_data)

        ## REDEFINE THIS LOSS
        labels = None
        loss = model.loss(labels)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(image_data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.image_data[0]))
            print("log interval")

## Later, add test function from https://github.com/pytorch/examples/blob/master/mnist/main.py
for epoch in range(1, args.epochs + 1):
    train(epoch)
    #test()

