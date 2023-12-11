from util import *
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import numpy as np
import random
from pytorch_lightning.utilities import rank_zero_info
from datetime import datetime
import copy
from SimulationDataset import SimulationState

import matplotlib.pyplot as plt
import time
from PointNet import PointNetfeat


'''
Full Network
'''

class CROMnet(pl.LightningModule):
    def __init__(self, data_format, preprop_params, example_input_array, initial_lr, batch_size, lr, epo, lbl, scale_mlp, ks, strides, siren_enc, siren_dec, enc_omega_0, dec_omega_0, verbose, loaded_from=None):
        super(CROMnet, self).__init__()

        #data specific parameters
        self.data_format = data_format
        self.example_input_array = example_input_array
        self.preprop_params = preprop_params

        #Training parameters
        self.verbose = verbose #Print data specific parameters or not
        self.lr = initial_lr
        self.batch_size = batch_size
        
        self.learning_rates, self.accumulated_epochs = generateEPOCHS(lr, epo)
        
        #Updated parameters
        self.loaded_from = loaded_from

        #network structure parameters (things we can change for network)
        self.lbllength = lbl
        self.scale_mlp = scale_mlp
        self.ks = ks
        self.strides = strides
        self.siren_enc = siren_enc
        self.siren_dec = siren_dec
        self.enc_omega_0 = enc_omega_0
        self.dec_omega_0 = dec_omega_0      

        self.criterion = nn.MSELoss()

        self.encoder = NetEnc(data_format, self.lbllength, self.siren_enc, self.enc_omega_0)
                       #NetEnc(data_format, self.lbllength, self.ks, self.strides, self.siren_enc, self.enc_omega_0)
        self.decoder = NetDec(data_format, self.lbllength, self.scale_mlp, self.siren_dec, self.dec_omega_0)
        self.netmap = NetMap(data_format, self.lbllength, self.scale_mlp, self.siren_dec, self.dec_omega_0)

        self.sim_state_list = []
        
        self.xhat_x = []
        self.xhat_y = []
        
        self.start = time.time()
        self.save_hyperparameters()

    def setup(self, stage):
        
        if stage == "fit":

            self.decoder.invStandardizeQ.set_params(self.preprop_params)
            self.netmap.prepare.set_params(self.preprop_params)
            #self.encoder.standardizeQ.set_params(self.preprop_params)
        
        if stage == "test":
            self.path_basename = os.path.split(os.path.dirname(self.loaded_from))[-1]

    def training_step(self, train_batch, batch_idx):
        
        encoder_input = train_batch['encoder_input']
        q = train_batch['q']
        outputs_local, xhat, _, mapped_x= self.forward(encoder_input)
        
        loss = 100000000 * self.criterion(outputs_local, q)#dragon
        
        mapped_x = mapped_x[0].view(mapped_x.shape[1], self.lbllength, -1)
        

        loss2 = 0
        
        #print('loss2 = ', loss2)
        loss2 = loss2 * 10.
        tensorboard_logs = {'train_loss_step': loss, 'sum dot product': loss2}

        self.log_dict(tensorboard_logs, prog_bar=True)
        
        return {'loss': loss + loss2, 'log': tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        self.log('train_loss_epoch', avg_loss)
        self.log('step', torch.tensor(self.current_epoch, dtype=torch.float32))
        end = time.time()
        print(end - self.start)

        return 

    def test_step(self, test_batch, batch_idx):

        encoder_input = test_batch['encoder_input']
        q = test_batch['q']
        outputs_local, labels, _, _ = self.forward(encoder_input)
        loss = 100000000 * self.criterion(outputs_local, q)
        self.log('test_loss', loss)

        labels = labels.detach().cpu().numpy()
        
        batch_size_local = encoder_input.size(0)

        output_regular = outputs_local.detach().cpu().numpy()
        x_regular = test_batch['x'].detach().cpu().numpy().astype(float)
        filenames = test_batch['filename']
        times = test_batch['time'].detach().cpu().numpy()

        # specific to fem dataset
        if 'faces' in test_batch:
            faces = test_batch['faces'].detach().cpu().numpy()

        for i in range(batch_size_local):
            label = labels[i, :]
            input_q = output_regular[i,:,:]
            input_x = x_regular[i,:,:]
            input_t = np.array([[times[i]]])
            filename = filenames[i]
            filename_out = convertInputFilenameIntoOutputFilename(filename, self.path_basename)
            sim_state = SimulationState(filename_out, False, input_x, input_q, input_t, label=label)
            # specific to fem dataset
            
            sim_state.q = sim_state.x + sim_state.q # assume displacenement training data; generalize later
            self.sim_state_list.append(sim_state)
            #print(label, filename)
            #self.xhat_x.append(label[0])
            #self.xhat_y.append(label[1])

        return loss

    def forward(self, x):

        x = x

        #encoder input
        state = x[:,:, :self.data_format['o_dim']]

        #x0 for decoder
        x0 = x[:, :, self.data_format['o_dim']:]

        x_map_input = x0.clone()
        x_map_input = x_map_input.view(x0.size(0) * x0.size(1), x0.size(2))
        mapped_x = self.netmap.forward(x_map_input)
        
        mapped_x = mapped_x.view(x0.shape[0], x0.shape[1], -1)

        #encoder -> xhat
        #print("state shape = ", state.shape)
        state_enc = state.clone()
        state_enc = state_enc[:,:2500,:]
        state_x0_enc = x0.clone()
        state_x0_enc = state_x0_enc[:,:2500,:]

        state_enc = torch.cat((state_x0_enc, state_enc), 2)
        xhat = self.encoder.forward(state_enc)
        
        #Store label
        label = xhat.view(xhat.size(0), xhat.size(2))

        #decoder input
        xhat = xhat.expand(xhat.size(0), self.data_format['npoints'], xhat.size(2))
        #x = torch.cat((xhat, x0), 2)
        #mapped_x = mapped_x.expand(xhat.size(0), mapped_x.size(1), mapped_x.size(2))

        x = torch.cat((xhat, mapped_x), 2)
        
        #store original shape for later & reshape for decoder
        batch_size_local = x.size(0)
        x = x.view(x.size(0)*x.size(1), x.size(2))

        #Store decoder for computing Jacobian
        decoder_input = x

        #decoder -> x
        x = self.decoder.forward(decoder_input)

        #return to original shape
        x = x.view(batch_size_local, -1, x.size(1))

        return x, label, decoder_input, mapped_x


    def print_hyperparameters(self):

        rank_zero_info('\n\n---Data Info---')
        rank_zero_info('data_path: ' + self.data_format['data_path'])
        rank_zero_info('# of points per file: ' + str(self.data_format['npoints']))
        rank_zero_info('# of i_dim: ' + str(self.data_format['i_dim']))
        rank_zero_info('# of o_dim: ' + str(self.data_format['o_dim']))
        rank_zero_info('x mean: ' + str(self.preprop_params['mean_x']))
        rank_zero_info('x std: ' + str(self.preprop_params['std_x']))
        rank_zero_info('x min: ' + str(self.preprop_params['min_x']))
        rank_zero_info('x max: ' + str(self.preprop_params['max_x']))
        rank_zero_info('q mean: ' + str(self.preprop_params['mean_q']))
        rank_zero_info('q std: ' + str(self.preprop_params['std_q']))

        rank_zero_info('\n---Network Info---')

        rank_zero_info('lbllength: ' + str(self.lbllength))
        rank_zero_info('scale_mlp: ' + str(self.scale_mlp))
        rank_zero_info('siren enc: ' + str(self.siren_enc))
        if self.siren_enc:
            rank_zero_info('\tomega_0: ' + str(self.enc_omega_0))
        rank_zero_info('siren_dec: ' + str(self.siren_dec))
        if self.siren_dec:
            rank_zero_info('\tomega_0: ' + str(self.dec_omega_0))
        #rank_zero_info('lambda_f: ' + str(self.lambda_f))
        rank_zero_info("")

    def adaptiveLRfromRange(self, epoch):
        for idx in range(len(self.accumulated_epochs)-1):
            do = self.accumulated_epochs[idx]
            up = self.accumulated_epochs[idx+1]
            if do <= epoch < up:
                return self.learning_rates[idx]
        if epoch == self.accumulated_epochs[-1]: #last epoch
            return self.learning_rates[-1]
        else:
            exit('invalid epoch for adaptiveLRfromRange')

    def configure_optimizers(self):

        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = LambdaLR(optimizer, lr_lambda=self.adaptiveLRfromRange)
        #scheduler = {'scheduler': ReduceLROnPlateau(optimizer, 'min'), 'monitor': 'train_loss_epoch'}
        return [optimizer], [scheduler]

    def on_train_start(self,):
        if self.verbose:
            self.print_hyperparameters()
    
    def on_test_start(self,):
        if self.verbose:
            self.print_hyperparameters()
    
    def test_epoch_end(self, outputs):
        
        pass


'''
Decoder Network
'''

class NetDec(pl.LightningModule):
    def __init__(self, data_format, lbllength, scale_mlp, siren, omega_0):
        super(NetDec, self).__init__()

        self.lbllength = lbllength
        self.scale_mlp = scale_mlp        
        self.siren = siren
        self.omega_0 = omega_0

        #self.dec0 = nn.Linear(data_format['o_dim'] * 5, data_format['o_dim'], bias = False)
        

        #self.act = Activation(self.siren, self.omega_0)
        self.invStandardizeQ = invStandardizeQ(data_format)
        
        self.layers = []
        
        #self.layers.append(self.dec0)
        #self.layers.append(self.invStandardizeQ)

        self.init_weights()
        self.init_grads()
    
    def init_weights(self):
        with torch.no_grad():
            for layer in self.layers:
                if layer.__class__.__name__ == 'Linear':

                    random.seed(datetime.now())
                    seed_number = random.randint(0, 100)
                    random.seed(0)
                    torch.manual_seed(seed_number)
                    
                    if self.siren:
                        layer.weight.uniform_(-np.sqrt(6 / layer.in_features) / self.omega_0, 
                                                np.sqrt(6 / layer.in_features) / self.omega_0)
                    else:
                        nn.init.xavier_uniform_(layer.weight)

            #nn.init.xavier_uniform_(self.dir.weight)

            #if self.siren:
            #    self.dec0.weight.uniform_(-1 / self.dec0.in_features, 
            #                                 1 / self.dec0.in_features)

    def init_grads(self):
        #self.dir.grad_func = make_linear_grad_of(self.dir.weight)
        for layer in self.layers:
            if layer.__class__.__name__ == 'Linear':
                layer.grad_func = make_linear_grad_of(layer.weight)
            elif layer.__class__.__name__ == 'Activation':
                if self.siren:
                    layer.grad_func = make_siren_grad_of(self.omega_0)
                else:
                    layer.grad_func = make_elu_grad_of(1)
            elif layer.__class__.__name__ == 'invStandardizeQ':
                pass
            elif layer.__class__.__name__ == 'Prepare':
                pass
            else:
                print(layer.__class__.__name__)
                exit('invalid grad layer')

    def forward(self, x):
        

        lbl = x[:, :self.lbllength].view(-1, 1, self.lbllength)
        geo = x[:, self.lbllength:].view(x.shape[0], self.lbllength, -1)
        
        
        x = lbl.matmul(geo)
        
        x = x.view(x.shape[0], -1)
        
        for layer in self.layers:
                #grad = torch.matmul(layer.grad_func(y), grad)
                #y = layer(y)
                x = layer(x)

        #print("x shape = ", x.shape) 
        return x
    
    def computeJacobianFullAnalytical(self, x):
        grad = None
        
        lbl = x[:, :self.lbllength].view(-1, 1, self.lbllength)
        geo = x[:, self.lbllength:].view(x.shape[0], self.lbllength, -1)
            
       
        
        x = lbl.matmul(geo)
        
        x = x.view(x.shape[0], -1)
        
        #print("geo = ", geo)
        grad = geo.transpose(1, 2)
        
        
        for layer in self.layers:
                grad = torch.matmul(layer.grad_func(y), grad)
                y = layer(y)
                x = layer(x)
        
        return grad, x
        
        
        

    


class NetMap(pl.LightningModule):
    def __init__(self, data_format, lbl, scale_mlp, siren, omega_0):
        super(NetMap, self).__init__()

        #scale_mlp = scale_mlp * 2 
        self.scale_mlp = scale_mlp        
        self.siren = siren
        self.omega_0 = omega_0

        self.dec0 = nn.Linear(data_format['i_dim'], data_format['i_dim'] * scale_mlp)
        self.dec1 = nn.Linear(data_format['i_dim'] * scale_mlp,
                              data_format['i_dim'] * scale_mlp)
        self.dec2 = nn.Linear(data_format['i_dim'] * scale_mlp,
                              data_format['i_dim'] * scale_mlp)
        self.dec3 = nn.Linear(data_format['i_dim'] * scale_mlp,
                              data_format['i_dim'] * scale_mlp)
        self.dec4 = nn.Linear(data_format['i_dim'] * scale_mlp,
                              data_format['i_dim'] * scale_mlp)
        self.enc = nn.Linear(data_format['i_dim'] * scale_mlp,
                              data_format['o_dim'] * lbl)

        self.act = Activation(self.siren, self.omega_0)
        self.prepare = Prepare(0, self.siren, data_format)

        self.layers = []
        
        self.layers.append(self.prepare)
        self.layers.append(self.dec0)
        self.layers.append(self.act)
        self.layers.append(self.dec1)
        self.layers.append(self.act)
        self.layers.append(self.dec2)
        self.layers.append(self.act)
        self.layers.append(self.dec3)
        self.layers.append(self.act)
        self.layers.append(self.dec4)
        self.layers.append(self.act)
        self.layers.append(self.enc)
        self.layers.append(self.act)
        

        self.init_weights()
        self.init_grads()
    
    def init_weights(self):
        with torch.no_grad():
            for layer in self.layers:
                if layer.__class__.__name__ == 'Linear':

                    random.seed(datetime.now())
                    seed_number = random.randint(0, 100)
                    random.seed(0)
                    torch.manual_seed(seed_number)
                    
                    if self.siren:
                        layer.weight.uniform_(-np.sqrt(6 / layer.in_features) / self.omega_0, 
                                                np.sqrt(6 / layer.in_features) / self.omega_0)
                    else:
                        nn.init.xavier_uniform_(layer.weight)

            if self.siren:
                self.dec0.weight.uniform_(-1 / self.dec0.in_features, 
                                             1 / self.dec0.in_features)

    def init_grads(self):
        for layer in self.layers:
            if layer.__class__.__name__ == 'Linear':
                layer.grad_func = make_linear_grad_of(layer.weight)
            elif layer.__class__.__name__ == 'Activation':
                if self.siren:
                    layer.grad_func = make_siren_grad_of(self.omega_0)
                else:
                    layer.grad_func = make_elu_grad_of(1)
            elif layer.__class__.__name__ == 'Prepare':
                pass
            else:
                print(layer.__class__.__name__)
                exit('invalid grad layer')

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        #print(x.shape)
        x_norm = torch.sum(x ** 2, dim=1)
        
        x_norm = torch.sqrt(x_norm).view(-1,1)
        #print(x_norm.shape, x_norm)
        #x = x / x_norm
        x2 = torch.sum(x ** 2, dim=1)
        #print(x2)
        return x



'''
Decoder Gradient Network
'''

class NetDecFuncGrad(pl.LightningModule):
    def __init__(self, netdec):
        super(NetDecFuncGrad, self).__init__()
        self.layers = nn.ModuleList()
        for layer in netdec.layers:
            if not layer.__class__.__name__ == 'Activation':
                layer_copy = copy.deepcopy(layer)
            else:
                layer_copy = Activation(layer.siren, layer.omega_0)
            self.layers.append(layer_copy)
        
        self.lbllength = netdec.lbllength

        for layer in self.layers:
            if layer.__class__.__name__ == 'Linear':
                layer.grad_func = make_linear_grad_of(layer.weight)
            elif layer.__class__.__name__ == 'Activation':
                if layer.siren:
                    layer.grad_func = make_siren_grad_of(layer.omega_0)
                else:
                    layer.grad_func = make_elu_grad_of(1)
            elif layer.__class__.__name__ == 'invStandardizeQ':
                pass
            elif layer.__class__.__name__ == 'Prepare':
                pass
            else:
                print(layer.__class__.__name__)
                exit('invalid grad layer')       
    
    def forward(self, x):
        with torch.inference_mode():
            grad = None
        
            lbl = x[:, :self.lbllength].view(-1, 1, self.lbllength)
            geo = x[:, self.lbllength:].view(x.shape[0], self.lbllength, -1)
            
       
        
            x = lbl.matmul(geo)
        
            x = x.view(x.shape[0], -1)
        
            #print("geo = ", geo)
            grad = geo.transpose(1, 2).contiguous()
            
            
            for layer in self.layers:
                grad = torch.matmul(layer.grad_func(y), grad)
                y = layer(y)
                x = layer(x)
            
            return grad, x

'''
Encoder Network
'''

class NetEnc(pl.LightningModule):
    def __init__(self, data_format, lbllength, siren, omega_0):
        super(NetEnc, self).__init__()

        self.siren = siren
        self.omega_0 = omega_0
        
        self.pointnet = PointNetfeat()
        self.output_layer = nn.Linear(256,
                              lbllength)
        
        #self.standardizeQ = standardizeQ(data_format)
        self.act = Activation(self.siren, self.omega_0)

    

    def forward(self, state):
        #state = self.standardizeQ(state)

        state = torch.transpose(state, 1, 2)
        
        x, _, _ = self.pointnet(state)
        xhat = self.act(self.output_layer(x))
        xhat = xhat.view(xhat.size(0), 1, xhat.size(1))

        return xhat

'''
Sub Modules
'''

class standardizeQ(nn.Module):
    def __init__(self, data_format):
        super(standardizeQ, self).__init__()
        self.register_buffer('mean_q_torch', torch.zeros(data_format['o_dim']))
        self.register_buffer('std_q_torch', torch.zeros(data_format['o_dim']))
    
    def set_params(self, preprop_params):
        self.mean_q_torch = torch.from_numpy(preprop_params['mean_q']).float()
        self.std_q_torch = torch.from_numpy(preprop_params['std_q']).float()

    def forward(self, q):
        return (q - self.mean_q_torch + self.mean_q_torch) / self.std_q_torch

class Activation(nn.Module):
    def __init__(self, siren, omega_0):
        super(Activation, self).__init__()
        self.siren = siren
        self.omega_0 = omega_0

    def forward(self, input):
        if self.siren:
            return self.actSIN(input)
        else:
            return self.actELU(input)
    
    def actELU(self, input):
        return F.elu(input)

    def actSIN(self, input):
        return torch.sin(self.omega_0 * input)

class invStandardizeQ(nn.Module):
    def __init__(self, data_format):
        super(invStandardizeQ, self).__init__()

        self.register_buffer('mean_q_torch', torch.zeros(data_format['o_dim']))
        self.register_buffer('std_q_torch', torch.zeros(data_format['o_dim']))

    def set_params(self, preprop_params):
        self.mean_q_torch = torch.from_numpy(preprop_params['mean_q']).float()
        self.std_q_torch = torch.from_numpy(preprop_params['std_q']).float()

    def forward(self, q_standardized):
        return self.mean_q_torch - self.mean_q_torch + q_standardized  * self.std_q_torch
    
    def grad_func(self, x):
        # `(N, in\_features)`
        assert(len(x.shape)==2)
        grad_batch = self.std_q_torch.expand_as(x)
        grad_batch = torch.diag_embed(grad_batch)
        return grad_batch
    

class Prepare(nn.Module):
    def __init__(self, lbllength, siren, data_format):
        super(Prepare, self).__init__()
        self.siren = siren
        self.lbllength = lbllength

        self.register_buffer('min_x_torch', torch.zeros(data_format['i_dim']))
        self.register_buffer('max_x_torch', torch.zeros(data_format['i_dim']))
        self.register_buffer('mean_x_torch', torch.zeros(data_format['i_dim']))
        self.register_buffer('std_x_torch', torch.zeros(data_format['i_dim']))

    def set_params(self, preprop_params):
        self.min_x_torch = torch.from_numpy(preprop_params['min_x']).float()
        self.max_x_torch = torch.from_numpy(preprop_params['max_x']).float()
        self.mean_x_torch = torch.from_numpy(preprop_params['mean_x']).float()
        self.std_x_torch = torch.from_numpy(preprop_params['std_x']).float()

    def forward(self, x):
        
        
        x = self.prep(x)
        
        
        return x

    def clipX(self, x):
        return 2 * (x - self.min_x_torch) / (self.max_x_torch - self.min_x_torch) - 1

    def standardizeX(self, x):
        return (x - self.mean_x_torch) / self.std_x_torch

    def prep(self, x):
        if self.siren:
            return self.clipX(x)
        else:
            return self.standardizeX(x)
    


