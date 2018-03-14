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
import pickle
from tqdm import tqdm

IMAGE_SIZE = (128, 128)

class KFNet(nn.Module):
    """
        Allowable modes include:
            'feed_forward: (FF)'
            'r_const: (RC) '
            'r_against_r_const: (RARC)'
            'R_t: (R)'
            'backprop_kf: (BKF)'
            'LSTM w/backprop KF, training only LSTM: (LSTMBKF)'
            'LSTM end to end, training all params: (LSTMBKFE2E)'
    """
    def __init__(self, batch_size, dynamics_path):
        super(KFNet, self).__init__()
        self.hidden_size = 5
        self.build_conv()
        self.build_LSTM()
        self.mode = None
        self.batch_size = batch_size
        self.set_grads_for_mode()
        self.fclr = nn.Linear(128*128*3, 2)
        for name, param in self.named_parameters():
            if "bias" not in name and "batchnorm" not in name: 
                torch.nn.init.xavier_uniform(param)
            elif "bias" in name:
                param.data.zero_()
        self.RC = nn.Parameter(50.0*torch.eye(2).float().cuda(), requires_grad=True)
        with open(dynamics_path, 'rb') as dyn_file:
            dynamics = pickle.load(dyn_file)
            self.A = Variable(torch.Tensor(dynamics['A']).cuda())
            self.AT = Variable(torch.Tensor(dynamics['A'].T).cuda())
            self.B = Variable(torch.Tensor(dynamics['B']).cuda())
            self.BT = Variable(torch.Tensor(dynamics['B'].T).cuda())
            self.C = Variable(torch.Tensor(dynamics['C']).cuda())
            self.CT = Variable(torch.Tensor(dynamics['C'].T).cuda())
            self.Q = Variable(torch.Tensor(dynamics['Q']).cuda())

    # Does using this as a self mean that each norm layer will be identical//is that an issue
    def resp_norm(self, input):
        pass

    def change_mode(self, mode):
        self.mode = mode
        self.set_grads_for_mode()

    def set_grads_for_mode(self):
        if self.mode == 'LR':
            for param in self.parameters():
                param.requires_grad = False
            self.fclr.weight.requires_grad = True
            self.fclr.bias.requires_grad = True
        
        if self.mode == 'FF':
            for param in self.parameters():
                param.requires_grad = True
            self.fc3_L.weight.requires_grad = False
            self.fc3_L.bias.requires_grad = False
            self.RC.requires_grad = False
            # LSTM parameters are not being tuned for any model except the LSTM, by default

        if self.mode == 'RC':
            for param in self.parameters():
                param.requires_grad = False
            self.RC.requires_grad = True

        if self.mode == 'RARC':
            for param in self.parameters():
                param.requires_grad = True #should be False
            self.RC.requires_grad = False #shouldn't need this line once set to false above
            self.fc3_L.weight.requires_grad = True
            self.fc3_L.bias.requires_grad = True

        if self.mode == 'R':
            for param in self.parameters():
                param.requires_grad = True
            self.RC.requires_grad = False
        
        if self.mode == 'BKF':
            for param in self.parameters():
                param.requires_grad = True
            self.RC.requires_grad = False

        if self.mode == 'LSTMBKF':
            for name, param in self.named_parameters():
                if 'LSTM' not in name:
                    param.requires_grad = False
                elif 'bias' not in name:
                    num_blocks = int(param.shape[0] / 5)
                    block_size = int(param.shape[0] / num_blocks)
                    tmp_xav = torch.Tensor(param.shape).float().cuda()
                    torch.nn.init.xavier_normal(tmp_xav)
                    param.data = tmp_xav
                else: continue
            self.RC.requires_grad = False
        
        if self.mode == 'LSTMBKFE2E':
            for param in self.parameters():
                param.requires_grad = True
            self.RC.requires_grad = False

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

    def build_LSTM(self):
        self.LSTM = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
	
	##input is z and R flattened
    ## MB_size x timesteps x 2
    def run_LSTM(self, input_flat):
        out, hidden = self.LSTM(input_flat)
        return out
       
    def run_KF(self, z, R):
        outputs = []
        
        h = Variable(torch.zeros(z.size(0), 4, 1).float().cuda(), requires_grad=False)
        hprime = Variable(torch.zeros(z.size(0), 4, 1).float().cuda(), requires_grad=False)

   	# s is covariance matrix
        s = Variable(torch.zeros(z.size(0), 4, 4).float().cuda(), requires_grad=False)
        sprime = Variable(torch.zeros(z.size(0), 4, 4).float().cuda(), requires_grad=False)

        K_t = Variable(torch.zeros(z.size(0), 4, 2).float().cuda(), requires_grad=False)
        I = Variable(torch.eye(4).float().cuda())

        # Initialize mean and covariance for time step 0
        h[:,0:2,:] = z[:,0,:]
        s[:,0:2,0:2] = R[:,0,:,:]
        s[:,2:4,2:4] = torch.stack([Variable(torch.eye(2).cuda(), requires_grad=False) for i in range(z.size(0))]) 
        outputs += [h.squeeze(2)]

        # for each frame in sequence:
            # h'[t+1] = Ah[t]
            # Sig'[t+1] = ASig_x[t]A.T + B_WQB_W.T
            # K[t+1] = Sig'[t+1]CT(CSig'[t+1]CT+R[t+1])^-1
            # set h[t+1] = h'[t+1] + K[t+1](z[t+1] - Ch'[t+1])
            # Sig[t+1] = (I - K[t+1]C)Sig'[t+1]
        for t in range(z.size(1)-1):
            hprime = torch.matmul(self.A, h)
            sprime = torch.matmul(self.A, s @ self.AT) + self.B @ self.Q @ self.BT
            K_t = torch.matmul(sprime @ self.CT, binv(torch.matmul(self.C, sprime @ self.CT) + R[:,t+1,:,:]))
            alpha = z[:,t+1,:].unsqueeze(2) - torch.matmul(self.C, hprime)
            h = hprime + torch.matmul(K_t, alpha)
            s = torch.matmul((I - K_t @ self.C), sprime)
            outputs += [h.squeeze(2)]
        
        outputs = torch.stack(outputs, 1)
        return outputs 

    def forward(self, x):
        if self.mode == 'LR':
            x = x.view(-1, IMAGE_SIZE[0] * IMAGE_SIZE[0] * 3)
            x = self.fclr(x)
            x = x.view(self.batch_size, -1, 2)
            return x

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
        z = z.view(self.batch_size, -1, 2)

        if self.mode == 'FF' or self.mode == 'RC':
            return z

        L = self.fc3_L(x)
        L_m = Variable(torch.FloatTensor(L.shape[0], 2, 2).zero_()).cuda()
        L_m[:,0,0] = L[:,0]
        L_m[:,1,0] = L[:,1]
        L_m[:,1,1] = L[:,2]
        R = torch.bmm(L_m, torch.transpose(L_m, 1, 2))
        R = R.view(self.batch_size, -1, 2, 2)

        if self.mode == 'R' or self.mode == 'RARC':
            return z, R

        if self.mode == 'BKF':
            return self.run_KF(z, R)
		
        if self.mode == 'LSTMBKF' or self.mode == 'LSTMBKFE2E':
            L = L.view(1, L.shape[0], L.shape[1])
            inp = torch.cat([z,L],2)
            outp = self.run_LSTM(inp)
            outp = 100*outp
            z_new = outp[:,:,0:2]
            L_new = outp[:,:,2::]
            L_new = L_new.squeeze()
            L_m = Variable(torch.FloatTensor(L_new.shape[0], 2, 2).zero_()).cuda()
            L_m[:,0,0] = L_new[:,0]
            L_m[:,1,0] = L_new[:,1]
            L_m[:,1,1] = L_new[:,2]
            R_new = torch.bmm(L_m, torch.transpose(L_m, 1, 2))
            R_new = R_new.view(self.batch_size, -1, 2, 2)
            return self.run_KF(z_new, R_new)

    def loss(self, predictions, labels, epoch):
        # MSE prediction loss on feed foward model
        if self.mode == 'FF' or self.mode == 'LR':
            loss = F.mse_loss(predictions.view(-1, 2), labels.contiguous().view(-1, 2))

        # Log Likelihood assuming constant R
        elif self.mode == 'RC':
            z = predictions.view(-1, 2)
            labels = labels.contiguous().view(-1, 2)
            RC_inv = self.RC.inverse()
            error = z - labels
            error = error.view(error.shape[0], error.shape[1], 1)
            nll_det = 1.0/2.0*torch.log(self.RC[0,0]*self.RC[1,1]-self.RC[0,1]*self.RC[1,0])
            nll = torch.bmm(torch.transpose(error, 2, 1) @ RC_inv, error)
            loss = torch.mean(nll) + nll_det

        # Train R_t to output RC
        elif self.mode == 'RARC':
            R = predictions[1].view(-1,2,2)
            diff = R - self.RC
            return torch.mean((diff**2).sum()) + 3*F.mse_loss(predictions[0].view(-1, 2), labels.contiguous().view(-1, 2))
        
        # Log likelihood assuming variable R_t
        elif self.mode == 'R':
            z = predictions[0].view(-1, 2)
            R = predictions[1].view(-1, 2, 2)
            labels = labels.contiguous().view(-1, 2)
            R_inv = binv(R)
            error = (z - labels).unsqueeze(2)
            nll_det = 1.0/2.0*torch.log(R[:,0,0]*R[:,1,1]-R[:,0,1]*R[:,1,0])
            nll = torch.bmm(torch.transpose(error,2,1), torch.bmm(R_inv, error)) + nll_det
            loss = torch.mean(nll) 

        elif self.mode == 'BKF':
            h = predictions[0].view(-1, 4)
            z = h[:,0:2]
            loss = F.mse_loss(z, labels.contiguous().view(-1, 2))

        # Train LSTMBKF with just LSTM or end to end
        elif self.mode == 'LSTMBKF' or self.mode == 'LSTMBKFE2E':
            h = predictions[0].view(-1, 4)
            z = h[:,0:2]
            loss = F.mse_loss(z, labels.contiguous().view(-1, 2))

        return loss

# Inversion of a batch of tensors x
def binv(x):
    inv = [t.inverse() for t in torch.functional.unbind(x)]
    return torch.functional.stack(inv)

# Build and intialize model, dataloaders
def init_model(args, is_cuda, batch_size, test_batch_size):
    model = KFNet(args.batch_size, "./train/dynamics.pkl")
    if is_cuda:
        model.cuda()

    # 0 workers is about 20% faster than 1 and 10x faster than 4
    kwargs = {'num_workers': 0, 'pin_memory': True} if is_cuda else {}
    train_dataset = RedDotDataset(base_dir='./train/redDot/')
    val_dataset = RedDotDataset(base_dir='./val/redDot/')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, **kwargs) 

    return model, train_loader, val_loader

# Change mode and return new optimizer
def change_mode(model, mode):
    model.change_mode(mode)
    print("Optimizing: ", [name for (name, param) in model.named_parameters() if param.requires_grad])
    params = [param for param in model.parameters() if param.requires_grad]
    lrs = {'FF': 0.001, 'RC': 0.01, 'RARC': 0.004, 'R': 0.0001}
    lr = lrs[mode] if mode in lrs else 0.001
    optimizer = optim.Adam(params, lr=lr)

    return optimizer    

# Perform a single epoch of training
def train(model, optimizer, train_loader, epoch, is_cuda, save_model):
    model.train()
    for batch_idx, sampled_batch in enumerate(tqdm(train_loader)):
        image_data = sampled_batch['images']
        gt_poses = sampled_batch['gt']

        if is_cuda:
            image_data, gt_poses = image_data.cuda(), gt_poses.cuda()
        image_data, gt_poses = Variable(image_data.float()), Variable(gt_poses.float()) 

        optimizer.zero_grad()
        output = model(image_data)
        loss = model.loss(output, torch.transpose(gt_poses[:,0:2,:], 2, 1), epoch)
        loss.backward()
        optimizer.step()
    
    if save_model and epoch % 2 == 0:
        torch.save(model, "epoch/" + str(epoch))

# Perform an evaluation after epoch on the loader
def test(epoch, model, loader, is_cuda, is_vis):
    model.eval()
    print_pixel_loss=False #Set to true to view how pixel prediction loss changes when it's not equal to the model loss
    pixel_loss = []
    model_loss = []
    for batch_idx, sampled_batch in enumerate(loader):
        image_data = sampled_batch['images']
        gt_poses = sampled_batch['gt']

        if is_cuda:
            image_data, gt_poses = image_data.cuda(), gt_poses.cuda()

        image_data, gt_poses = Variable(image_data.float()), Variable(gt_poses.float())
        output = model(image_data)
        model_loss.append(model.loss(output, torch.transpose(gt_poses[:,0:2,:], 2, 1), epoch).data[0])

        if print_pixel_loss:
            if model.mode == 'R' or model.mode == 'RARC' or model.mode == 'RC': 
                output = output[0]
            if model.mode == 'BKF' or model.mode == 'LSTMBKFE2E' or model.mode == 'LSTMBKF':
                output = output[:,:,0:2].contiguous()
            pixel_loss.append(F.mse_loss(output.view(-1, 2), torch.transpose(gt_poses[:,0:2,:],2,1).contiguous().view(-1, 2)).data[0])

        if is_vis: 
            visualize_result(torch.transpose(gt_poses[:,0:2,:],2,1).contiguous().view(-1,2), output.view(-1,2), str(epoch) + "_" + str(batch_idx))
    
    if print_pixel_loss:
        print('Test Epoch: {} \tPixel Loss: {:.6f}'.format(epoch, np.mean(pixel_loss)))

    print('Test Epoch: {} \tModel Loss: {:.6f}'.format(epoch, np.mean(model_loss)))

# Util for initializing image for visualization
def init_image():
    img = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[0], 3))
    img.fill(255)
    return img

# Util for writing trajectory on image
def write_traj_on_image(img, traj, color):
    N = traj.shape[0]
    for i in range(N-1):
        cv2.line(img, (traj[i,0]+64, traj[i,1]+64), (traj[i+1,0]+64, traj[i+1,1]+64), color)
    

# Visualize the result
def visualize_result(gt_traj, est_traj, idx):
    img = init_image()
    write_traj_on_image(img, gt_traj, (0,255,0))
    write_traj_on_image(img, est_traj, (0,0,0))
    cv2.imwrite("results/" + str(idx) + ".png", img)

# Defines the sequence of modes and their lengths in epochs for training
def train_all(args):
    model, train_loader, val_loader = init_model(args, args.cuda, args.batch_size, args.test_batch_size)

    if args.load_model: 
        model = torch.load(args.load_dir + "/{}".format(args.load_model))
        epoch = args.load_model + 1
    else:
        epoch = 1

    mode_lengths = [('FF', 5), ('RC', 5), ('RARC', 40), ('R', 30), ('BKF',20), ('LSTMBKF', 40), ('LSTMBKFE2E',40)]
    modes = []
    total=0
    for mode in mode_lengths:
        total += mode[1]
        modes.append((mode[0], total))
    
    for mode in modes:
        print(mode[0])
        optimizer = change_mode(model, mode[0])
        while epoch <= mode[1]:
            train(model, optimizer, train_loader, epoch, args.cuda, args.save_model)
            print_val = True
            if print_val:
                print("Numbers on val set")
                test(epoch, model, val_loader, args.cuda, args.visualize)
            print_train = False 
            if print_train:
                print("Numbers on training set")
                test(epoch, model, train_loader, args.cuda, args.visualize)
            epoch += 1


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
    parser.add_argument('--load-model', type=int, help='load model from epoch #(int)')
    parser.add_argument('--load-dir', type=str, default="./epoch", help='directory to load model (default ./epoch)')
    parser.add_argument('--save-model', action='store_true', default=False, help='save model (default False)')
    parser.add_argument('--visualize', action='store_true', default=False, help='visualize model (default False)')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists("results"): 
        os.makedirs("results")

    train_all(args)


if __name__ == "__main__": main()
