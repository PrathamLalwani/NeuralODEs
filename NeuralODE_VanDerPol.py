import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
torch.set_default_dtype(torch.float64)
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=100)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=1500)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
# parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--save', type=str, default='./experiment1')
args = parser.parse_args()
args.adjoint = True
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

true_y0 = torch.tensor([[.1, 0.]]).to(device)
t = torch.linspace(0., 2., args.batch_time).to(device)
true_A = torch.tensor([[0, 1.0], [-1.0, 0.]]).to(device)
mu = 1.
M = 2.

class Lambda(nn.Module):

    def forward(self, t, y):
        y_new = torch.stack([y[:,1],mu * (1-y[:,0]**2)*y[:,1]-y[:,0]],axis=1)
        return y_new



def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')


def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def get_random_sample():
    y0s = -M + torch.rand((10, 2))*2*M
    batch_t = t
    true_y = odeint(Lambda(), y0s, batch_t, method = 'dopri5')
    return y0s.to(device),batch_t.to(device),true_y.to(device)
    

def vector_field_error(odefunc, true_odefunc):
    y, x = np.mgrid[-4:4:21j, -4:4:21j]
    dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
    mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
    dydt = (dydt / mag)
    dydt = dydt.reshape(21, 21, 2)
    dydt_true = true_odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
    mag = np.sqrt(dydt_true[:, 0]**2 + dydt_true[:, 1]**2).reshape(-1, 1)
    dydt_true = (dydt_true / mag)
    dydt_true = dydt_true.reshape(21, 21, 2)
    return np.linalg.norm(dydt_true-dydt,axis=-1)

    


# def makedirs(dirname):
#     if not os.path.exists(dirname):
#         os.makedirs(dirname)


if args.viz:
    makedirs(f'png_model_{mu}_{torch.max(t).cpu().numpy()}')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(141, frameon=False)
    ax_phase = fig.add_subplot(142, frameon=False)
    ax_vecfield = fig.add_subplot(143, frameon=False)
    ax_vecfield_true = fig.add_subplot(144, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, true_odefunc, itr):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        ax_vecfield_true.cla()
        ax_vecfield_true.set_title('True Vector Field')
        ax_vecfield_true.set_xlabel('x')
        ax_vecfield_true.set_ylabel('y')


        y, x = np.mgrid[-4:4:21j, -4:4:21j]
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(-4, 4)
        ax_vecfield.set_ylim(-4, 4)

        y, x = np.mgrid[-4:4:21j, -4:4:21j]
        dydt = true_odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield_true.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield_true.set_xlim(-4, 4)
        ax_vecfield_true.set_ylim(-4, 4)

        fig.tight_layout()
        # Create the directory
        output_dir = f'png_model_{mu}_{torch.max(t).cpu().numpy()}'
        os.makedirs(output_dir, exist_ok=True)

        # Save the figure
        plt.savefig(f'{output_dir}/{itr:03d}.png')
        plt.draw()
        plt.pause(0.001)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2,50),
            nn.Tanh(),
            nn.Linear(50,50),
            nn.Tanh(),
            nn.Linear(50, 2),

        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return torch.stack([y[...,1],-y[...,0]],axis=-1) +  self.net(y)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val




if __name__ == '__main__':

    ii = 0
    makedirs(args.save)

    func = ODEFunc().to(device)
    
    optimizer = optim.AdamW(func.parameters(), lr=3e-4)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)
    min_loss = float('inf')
    batch_y0, batch_t, batch_y = get_random_sample()

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        pred_y = odeint(func, batch_y0, batch_t).to(device)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())
        if loss.item()<min_loss:
            torch.save({'state_dict': func.state_dict(), 'args': args}, os.path.join(args.save, f'model_{mu}_{torch.max(t).cpu().numpy()}.pth'))

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_y, pred_y, func,Lambda(), ii)
                ii += 1

        end = time.time()
