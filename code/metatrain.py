# -----------------------------------------------------------------------------
# Description: Meta-training script for Meta-PINN. This code performs 
#              meta-training using a MAML-like framework for PDE-based tasks.
#              It loads training data, performs inner and outer optimization 
#              loops, logs progress, periodically saves checkpoints, and 
#              includes validation and testing steps.
# -----------------------------------------------------------------------------

import  os
import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np
import  argparse
import  scipy.io as sio
from torch.utils.tensorboard import SummaryWriter
from pinnmodel import PINN
from copy import deepcopy
from random import shuffle
import math
from collections import OrderedDict
import random
from losses import PINNLoss, RegLoss
from utils import PositionalEncod, calculate_grad

# Directory for saving checkpoints
dir_checkpoints = './checkpoints/metatrain/'
os.makedirs(dir_checkpoints, exist_ok=True)

def main(args):
    # Set random seeds for reproducibility
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)


    print(args)

    # Set up TensorBoard writer for logging training/validation metrics
    writer = SummaryWriter(comment=f'runs/metatrain/MetaLR_{args.meta_lr}_UpdateLR_{args.update_lr}_Epoch_{args.epoch}_Updatestep_{args.update_step}')

    # Define the training device
    device = torch.device('cuda')


    # pinn model
    # Define input/output dimensions for the model
    input_dim = 3   # Input dimension (e.g., x, z, sx)
    output_dim = 2  # Output dimension (e.g., real and imaginary parts)
    neurons = [input_dim, 256, 256, 128, 128, 64, 64, output_dim]

    # Initialize the meta-training model (MAML-like structure)
    maml = PINN(neurons).to(device)

    # Count the total number of trainable parameters
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print('Total trainable tensors:', num)

    omega = 2*math.pi*args.omega
    torch.cuda.empty_cache()
    # -------------------- Load Training Data --------------------
    # Load support data set for meta-training
    data = sio.loadmat(f'../dataset/metatrain/metatrain.mat')
    x_train = data['x_metatrain']
    sx_train = data['sx_metatrain']
    z_train = data['z_metatrain']
    m_train = data['m_metatrain']
    m0_train = data['m0_metatrain']
    u0_real_train = data['u0_real_metatrain']
    u0_imag_train = data['u0_imag_metatrain']

    # Convert numpy arrays to torch tensors and move them to GPU
    x_train = torch.tensor(x_train, dtype=torch.float32, requires_grad=True).to(device)
    z_train = torch.tensor(z_train, dtype=torch.float32, requires_grad=True).to(device)
    sx_train = torch.tensor(sx_train, dtype=torch.float32).to(device)
    m_train = torch.tensor(m_train, dtype=torch.float32).to(device)
    m0_train = torch.tensor(m0_train, dtype=torch.float32).to(device)
    u0_real_train = torch.tensor(u0_real_train, dtype=torch.float32).to(device)
    u0_imag_train = torch.tensor(u0_imag_train, dtype=torch.float32).to(device)

    # -------------------- Load Validation Data --------------------
    # For validation, load data.
    data = sio.loadmat(f'../dataset/metatest/layer_model.mat')
    x_val = data['x_train']
    sx_val = data['sx_train']
    z_val = data['z_train']
    m_val = data['m_train']
    m0_val = data['m0_train']
    u0_real_val = data['u0_real_train']
    u0_imag_val = data['u0_imag_train']
    x_test = data['x_test']
    sx_test = data['sx_test']
    z_test = data['z_test']
    du_real_test = data['du_real_test']
    du_imag_test = data['du_imag_test']

    # Convert validation data to torch tensors
    x_val = torch.tensor(x_val, dtype=torch.float32, requires_grad=True).to(device)
    z_val = torch.tensor(z_val, dtype=torch.float32, requires_grad=True).to(device)
    sx_val = torch.tensor(sx_val, dtype=torch.float32).to(device)
    m_val = torch.tensor(m_val, dtype=torch.float32).to(device)
    m0_val = torch.tensor(m0_val, dtype=torch.float32).to(device)
    u0_real_val = torch.tensor(u0_real_val, dtype=torch.float32).to(device)
    u0_imag_val = torch.tensor(u0_imag_val, dtype=torch.float32).to(device)

    # Convert test data to torch tensors
    x_test = torch.tensor(x_test, dtype=torch.float32, requires_grad=True).to(device)
    z_test = torch.tensor(z_test, dtype=torch.float32, requires_grad=True).to(device)
    sx_test = torch.tensor(sx_test, dtype=torch.float32).to(device)
    du_real_test = torch.tensor(du_real_test, dtype=torch.float32).to(device)
    du_imag_test = torch.tensor(du_imag_test, dtype=torch.float32).to(device)

    _, data_num = x_train.size()

    # Set up the optimizer for meta-training
    meta_optimizer = torch.optim.AdamW(maml.parameters(), lr=args.meta_lr, weight_decay=4e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(meta_optimizer, step_size=5000, gamma=0.8)

    # Define loss functions
    criterion_pde = PINNLoss()           # PDE residual loss (physics-informed)
    criterion_reg = RegLoss()            # Additional regularization loss

    # Set up the optimizer for validation
    maml_copy = deepcopy(maml)
    test_optimizer = torch.optim.AdamW(maml_copy.parameters(), lr=args.test_lr, weight_decay=4e-5)

    # -------------------- Meta-Training Loop --------------------
    for step in range(args.epoch):
        maml.train()

        # Randomly select tasks for support (inner update) and query (outer update)
        rand_task_select = random.sample(range(data_num), args.ntask * 2)
        rand_task_spt = rand_task_select[:args.ntask]   # support tasks
        rand_task_qry = rand_task_select[args.ntask:]   # query tasks

        outer_loss = torch.tensor(0., device=device)

        # Loop over tasks to compute outer_loss for meta-update
        for i in range(args.ntask):

            # Support set data for the i-th task
            x = x_train[:, rand_task_spt[i]].unsqueeze(1)
            z = z_train[:, rand_task_spt[i]].unsqueeze(1)
            sx = sx_train[:, rand_task_spt[i]].unsqueeze(1)
            m0 = m0_train[:, rand_task_spt[i]].unsqueeze(1)
            m = m_train[:, rand_task_spt[i]].unsqueeze(1)
            u0_real = u0_real_train[:, rand_task_spt[i]].unsqueeze(1)
            u0_imag = u0_imag_train[:, rand_task_spt[i]].unsqueeze(1)

            # Get a copy of the current model parameters for inner updates
            params = OrderedDict(maml.named_parameters())

            # Loop over tasks to compute outer_loss for meta-update
            for k in range(args.update_step):
                input = torch.cat([x,z,sx],-1)

                # Forward with functional parameters
                du_real_pred, du_imag_pred = maml.functional_forward(input, params=params)

                # Compute gradients (second derivatives) for PDE constraints
                du_real_xx, du_real_zz, du_imag_xx, du_imag_zz = calculate_grad(x, z, du_real_pred, du_imag_pred)

                # Compute PDE loss and regularization losses
                loss_pde = criterion_pde(x, z, sx, omega, m, m0, u0_real, u0_imag, du_real_pred, du_imag_pred, du_real_xx, 
                    du_real_zz, du_imag_xx, du_imag_zz)

                loss_reg = criterion_reg(x, z, sx, omega, m0, du_real_pred, du_imag_pred)

                inner_loss = args.loss_scale * (loss_pde + loss_reg)

                # Compute gradients of inner_loss w.r.t. params
                grads = torch.autograd.grad(inner_loss, params.values(), create_graph=not args.first_order)

                # Update params for this task's inner loop (fast adaptation)
                params = OrderedDict(
                        (name, param - args.update_lr * grad)
                        for ((name, param), grad) in zip(params.items(), grads))


            # -------------------- Outer Update Step --------------------
            # Query set data for the i-th task
            x = x_train[:, rand_task_qry[i]].unsqueeze(1)
            z = z_train[:, rand_task_qry[i]].unsqueeze(1)
            sx = sx_train[:, rand_task_qry[i]].unsqueeze(1)
            m0 = m0_train[:, rand_task_qry[i]].unsqueeze(1)
            m = m_train[:, rand_task_qry[i]].unsqueeze(1)
            u0_real = u0_real_train[:, rand_task_qry[i]].unsqueeze(1)
            u0_imag = u0_imag_train[:, rand_task_qry[i]].unsqueeze(1)

            input = torch.cat([x,z,sx],-1)

            # Evaluate on the query set with updated parameters
            du_real_pred, du_imag_pred = maml.functional_forward(input, params=params)

            du_real_xx, du_real_zz, du_imag_xx, du_imag_zz = calculate_grad(x, z, du_real_pred, du_imag_pred)

            # Compute losses on the query set
            loss_pde = criterion_pde(x, z, sx, omega, m, m0, u0_real, u0_imag, du_real_pred, du_imag_pred, du_real_xx, 
                    du_real_zz, du_imag_xx, du_imag_zz)

            loss_reg = criterion_reg(x, z, sx, omega, m0, du_real_pred, du_imag_pred)

            # Accumulate outer loss from all tasks
            outer_loss += args.loss_scale * (loss_pde + loss_reg)

        # Average outer_loss over the number of tasks
        outer_loss = outer_loss / args.ntask

        # Meta-optimizer update on outer_loss
        meta_optimizer.zero_grad()
        outer_loss.backward()
        meta_optimizer.step()
        scheduler.step()

        # Logging to TensorBoard
        writer.add_scalar('Loss/meta_loss', outer_loss.item(), step)
        writer.add_scalar('Loss/loss_pde', loss_pde.item(), step)
        writer.add_scalar('Loss/loss_reg', loss_reg.item(), step)
        writer.add_scalar('Loss/inner_loss', inner_loss.item(), step)

        # Print training status every 100 steps
        if (step + 1) % 100 == 0:
            print(f'step: {step + 1} Training inner loss: {inner_loss.item()}')
            print(f'step: {step + 1} Training meta loss: {outer_loss.item()}')
            print(f'step: {step + 1} Training PDE loss: {loss_pde.item()}')
            print(f'step: {step + 1} Training REG loss: {loss_reg.item()}')

        # Save model checkpoint every 100 steps
        if (step + 1) % 500 == 0:
            torch.save(maml.state_dict(), f'{dir_checkpoints}CP_epoch{step + 1}.pth')

        # Perform validation every 5000 steps
        if (step + 1) % 5000 == 0:
            print('---------------------------------------------------------')
            print('------------------- Validation start --------------------')
            print('---------------------------------------------------------')

            # copy a meta-trained model
            maml_copy.load_state_dict(maml.state_dict())
            maml_copy.train()

            # Fine-tune on validation data
            for k in range(args.update_step_test):
                test_optimizer.zero_grad()

                input = torch.cat([x_val, z_val, sx_val],-1)

                du_real_pred, du_imag_pred = maml_copy(input)

                du_real_xx, du_real_zz, du_imag_xx, du_imag_zz = calculate_grad(x_val, z_val, du_real_pred, du_imag_pred)

                loss_pde = criterion_pde(x_val, z_val, sx_val, omega, m_val, m0_val,  
                    u0_real_val, u0_imag_val, du_real_pred, du_imag_pred, du_real_xx, du_real_zz, du_imag_xx, du_imag_zz)

                loss_reg = criterion_reg(x_val, z_val, sx_val, omega, m0_val, du_real_pred, du_imag_pred)

                test_loss = loss_pde + loss_reg

                test_loss.backward()

                test_optimizer.step()

                if (k + 1) % 100 == 0:
                    print(f'step: {k + 1} Validation loss: {test_loss.item()}')
                    print(f'step: {k + 1} Validation PDE loss: {loss_pde.item()}')
                    print(f'step: {k + 1} Validation REG loss: {loss_reg.item()}')

            # Evaluate on test data after validation
            with torch.no_grad():
                input = torch.cat([x_test, z_test, sx_test], -1)
                du_real_pred, du_imag_pred = maml_copy(input)
                # Mean squared error on test data
                accs_real = (torch.pow((du_real_test - du_real_pred), 2)).mean().item()
                accs_imag = (torch.pow((du_imag_test - du_imag_pred), 2)).mean().item()

            print(f'Test accs real: {accs_real}')
            print(f'Test accs imag: {accs_imag}')

            print('---------------------------------------------------------')
            print('---------------------- Ending Test ----------------------')
            print('---------------------------------------------------------')

    writer.close()

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=50000)
    argparser.add_argument('--ntask', type=int, help='Number of tasks per meta-iteration', default=8)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=2e-3)
    argparser.add_argument('--test_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--omega', type=float, help='frequecy', default=5)
    argparser.add_argument('--loss_scale', type=float, help='task-level inner update learning rate', default=0.1)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=10)
    argparser.add_argument('--update_step_test', type=int, help='test upate step', default=5000)
    argparser.add_argument('--PosEnc', type=int, help='PosEnc', default=2)
    argparser.add_argument('--first_order', type=str, help='whether first order approximation of MAML is used', default=True)

    args = argparser.parse_args()

    main(args)
