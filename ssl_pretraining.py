import logging
import argparse
import numpy as np
import os
import json 
import datetime
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from torchvision.models import resnet18, resnet50

from models import EmbModel
import datasets as ds

from utils import nt_xent,triplet_loss,linear_eval_all,AverageMeter

import PIL
import sys
PIL.Image.MAX_IMAGE_PIXELS = 933120000

def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))
    
        
def train_ssl(model, args, train_loader, optimizer, scheduler, epoch):

    model.train()
    loss_meter = AverageMeter(args['train_loss'])
    train_bar = tqdm(train_loader)
    for data in train_bar:

        optimizer.zero_grad()
        x = torch.cat((data['im_t1'], data['im_t2']), 0).to(args['device'])
        b_size = x.shape[0]
        op = model(x)
        
        if args['return_context']:
            data['con'] = data['con'].to(args['device'])
            
        if args['store_embeddings']:
            model.update_memory(data['id'].to(args['device']), op['emb'][:b_size//2, :])
        
        ### Self-supervised training losses
       

        if args['train_loss'] == 'simclr':            
            loss = nt_xent(op['emb'][:b_size//2, :], op['emb'][b_size//2:, :], args)
        elif args['train_loss'] == 'triplet':
            loss = triplet_loss(op['emb'], args, margin=args['triplet_margin'])

        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_meter.update(loss.item(), x.shape[0])
        train_bar.set_description("Train epoch {}, loss: {:.4f}".format(epoch, loss_meter.avg))

    return loss_meter.avg

        
def select_train_items(model, args, train_loader):
    # samples new positive items based on distance in context and embedding space
    
    
    if args['pos_type'] == 'context_sample':
        # distance in context space
        context = train_loader.dataset.context.to(args['device'])
        con_dist = torch.cdist(context, context)
        con_dist = torch.softmax(-con_dist / args['con_temp_select'], dim=1)
   
    # sample new positives based on distance matrix
    sample_inds = torch.multinomial(con_dist, 1)[:, 0]
    train_loader.dataset.update_alternative_positives(sample_inds)
    
def main(args):
    assert torch.cuda.is_available()
    cudnn.benchmark = True

    # get datasets
    train_set_ssl, train_set_lin, val_set_lin, test_set_lin, train_inds_lin_1, train_inds_lin_10 = ds.get_dataset(args)

    args['context_size'] = train_set_ssl.context_size
    args['num_train'] = train_set_ssl.num_examples
    print('Running on: ',torch.cuda.get_device_name(torch.cuda.current_device()))  
    

    train_loader = DataLoader(train_set_ssl, batch_size=args['batch_size'], shuffle=True,
                              num_workers=args['workers'], drop_last=False)
            

       
    train_loader_lin = DataLoader(train_set_lin, batch_size=args['batch_size'], 
                                      num_workers=args['workers'], shuffle=False)
    test_loader_lin      = DataLoader(test_set_lin,  batch_size=args['batch_size'], 
                                    num_workers=args['workers'], shuffle=False)    
  

    if args['pretrained_model'] != '':
        args['pretext_finetune'] = True
    
    # initialize model
    base_encoder = eval(args['backbone'])
    model = EmbModel(base_encoder, args).to(args['device'])
    
    if args['pretrained_model'] != '':
        # need to exlude projector as it will be a different size for supervised
        print('Loading pretrained', args['pretrained_model'])
        state_dict = torch.load(args['pretrained_model'])['state_dict']
        state_dict = {k: v for k, v in state_dict.items() if 'projector' not in k}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg, '\n')
    
    
    # if burn in period, freeze the backbone weights for the first few epochs
    if args['burn_in'] > 0:
        for param in model.enc.parameters():
            param.requires_grad = False

    optimizer = torch.optim.SGD(
        model.parameters(),
        args['learning_rate'],
        momentum=args['momentum'],
        weight_decay=args['weight_decay'])

    # lr decay schedule
    if args['schedule'] == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer, args['epochs'] * len(train_loader))
    

    # main train loop
    res = [] 
    for epoch in range(1, args['epochs'] + 1):  
        if args['burn_in'] == epoch:
            for param in model.enc.parameters():
                param.requires_grad = True

        if args['pos_type'] == 'context_sample':
            # choose positives
            if epoch > args['burn_in_select']:
                select_train_items(model, args, train_loader)
    
        loss_avg = train_ssl(model, args, train_loader, optimizer, scheduler, epoch)  
        

    print('\nLinear evaluation')
    train_inds = [np.array(train_inds_lin_1), np.array(train_inds_lin_10), np.arange(len(train_set_lin))]    
    train_split_perc = [1, 10,100]
    if args['eval_ssl']:
        res = linear_eval_all(model, train_loader_lin, test_loader_lin, args, train_inds, train_split_perc, False)
    else:
        res = {}
                    
    #op['state_dict'] = model.state_dict()
    #torch.save(op, args['op_file_name'])
    #args['save_output'] = False
    if args['save_output']:
        op = {}
        op['args'] = args 
        op['epoch'] = args['epochs']
        op['results'] = res
            
        with open(args['op_res_name'], 'w') as da:
            json.dump(op, da, indent=2)
    
        op['state_dict'] = model.state_dict()
        torch.save(op, args['op_file_name'])
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Self-Supervised with Context')

    parser.add_argument('--dataset', choices=['ucf101','food101','sun397','stanford_cars','fgvc-aircraft','caltech-101','eurosat','oxford_pets','oxford_flowers','dtd',
                                              'kenya','cct20','serengeti','icct','fmow','oct'], type=str)
    
    parser.add_argument('--train_loss', default='simclr', 
                        choices=['simclr', 'triplet' ],help='type of self-supervised learning loss', type=str)
    
    parser.add_argument('--backbone', default='resnet18',choices=['resnet18','resnet50'],help='cnn backbone for ssl pretraining', type=str)
    parser.add_argument('--no_evaluation', dest='eval_ssl', action='store_false')
    parser.add_argument('--not_cached_images', dest='cache_images', action='store_false')  # default for cache_images will be True
    parser.add_argument('--not_pretrained', dest='pretrained', action='store_false')  # default for pretrained will be True
    parser.add_argument("--output_dir",default='ssl_pretraining/',help="directory to store model weights learnt from pretext task and linear evaluation results", type=str)

    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--learning_rate_mult', default=0.03, type=float)
    parser.add_argument('--im_res', default=112, choices=[112, 224], type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--projection_dim', default=128, type=int)
    parser.add_argument('--supervised_amt', default=1, choices=[1, 10, 100], type=int)
    parser.add_argument('--seed', default=2001, type=int)
    
    parser.add_argument("--model_type", default='ssl', help="type of pretraining, used to differentiate for the stage we are in (ssl or adaptation)", type=str)
    parser.add_argument('--pos_type', default='augment_self',choices=['augment_self', 'context_sample'],help="the way positives are rerieved during self-supervised pretext task. By default, self-augmentations are used.", type=str)
    parser.add_argument('--con_temp_select', default=0.05, type=float)
    parser.add_argument('--burn_in_select', default=1, type=int)  
    parser.add_argument('--pretrained_model', default='', type=str)
    parser.add_argument('--exp_name', default='',help='experiment name in case we want to categorize the different runs', type=str)
    
    # turn the args into a dictionary
    args = vars(parser.parse_args())


    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    
    args['return_context'] = False
    if 'context' in args['pos_type']:
        args['return_context'] = True

    args['data_dir'] = os.path.join('data/{}/'.format(args['dataset']))    
    args['metadata'] = os.path.join(args['data_dir'],'{}_meta.csv'.format(args['dataset']))

    args['learning_rate'] = args['learning_rate_mult']*args['batch_size']/256
    args['momentum'] = 0.9
    args['weight_decay'] = 0.0005
    args['schedule'] = 'cosine'
    args['workers'] = 6
    args['burn_in'] = 0  # if > 0, the backbone will be frozen for "burn_in" epochs
    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    args['lin_max_iter'] = 1000  # number of iterations in the linear evaluation  
    
    args['triplet_margin'] = 0.3
    args['temperature'] = 0.5
            

    args['return_alt_pos'] = False
    args['store_embeddings'] = False

    args['cache_images'] = False
    args['use_clip'] = False
        
    
    # setup how positive images are selected 
    if args['pos_type'] == 'augment_self':
        pass
    elif args['pos_type'] == 'context_sample':
        args['store_embeddings'] = True
        args['return_alt_pos'] = True
        
    args['save_output'] = True
    args['op_dir'] = os.path.join(args['output_dir'],'results/') 
    args['op_dir_mod'] =os.path.join(args['output_dir'],'models/') 
    
    if not os.path.isdir(args['op_dir']):
        os.makedirs(args['op_dir'])
    if not os.path.isdir(args['op_dir_mod']):
        os.makedirs(args['op_dir_mod'])
        
    cur_time = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    args['cur_time'] = cur_time
    op_str = args['dataset'] + '_' + args['backbone'] + '_' + args['train_loss'] + '_' + args['cur_time'] 
    args['op_file_name'] = args['op_dir_mod'] + op_str + '.pt'
    args['op_res_name'] = args['op_dir'] + op_str + '.json'
    args['op_im_name'] = args['op_dir'] + op_str + '_' + str(args['epochs']) + '.png'
    
    #sys.stdout = open('/home/omi/projects/camera_traps_self_supervised/{}.txt'.format(op_str),'wt')


    if 'imagenet' in args['dataset']:
        args['pretrained'] = False
   
    print('\n**********************************')
    print('Experiment :', args['exp_name'])
    print('Dataset    :', args['dataset'])
    print('Train loss :', args['train_loss'])
    print('Pos type   :', args['pos_type'])
    print('Backbone   :', args['backbone'])
    print('Pretrained :', args['pretrained'])
    print('Cached ims :', args['cache_images'])
    print('Evaluating after SSL :', args['eval_ssl'])     
    print('Op file    :', args['op_res_name'])
    
    main(args)
