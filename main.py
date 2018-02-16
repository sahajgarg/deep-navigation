import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from datasets import KITTIDataset
from torch.utils.data import Dataset, DataLoader


# Training settings 
parser = argparse.ArgumentParser(description='Deep Visual Odometry using Backprop KF')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
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

    def __init__(self):
        super(Net, self).__init__()
        # Initialize convolutions 
        # Initialize Kalman filter graph structure 

        Pass

    def forward(self, x):
    	# Pass entire batch of images through convolutional net
    	# Apply recurrence of KF
    	# Return final state estimate 

    	pass

model = KFNet()
if args.cuda:
	model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_dataset = KITTIDataset(frame_selections_file='dataset/train_frames.csv',
                                           images_dir='dataset/sequences/',
                                           poses_dir='dataset/poses/')
train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                        shuffle=True, **kwargs) ## Maybe: num_workers=4


def train(epoch):
    model.train()
    for batch_idx, sampled_batch in enumerate(train_loader):
    	image_data = sampled_batch['images']
    	gt_poses = sampled_batch['gt_poses']

        if args.cuda:
            image_data, gt_poses = image_data.cuda(), gt_poses.cuda()

        image_data, gt_poses = Variable(image_data), Variable(gt_poses)
        optimizer.zero_grad()
        output = model(iamge_data)

        ## REDEFINE THIS LOSS
        loss = F.nll_loss(output, gt_poses)
        loss.backward()
        optimizer.step()


        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

## Later, add test function from https://github.com/pytorch/examples/blob/master/mnist/main.py
for epoch in range(1, args.epochs + 1):
    train(epoch)
    #test()

