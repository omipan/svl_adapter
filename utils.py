import sys
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from tqdm import tqdm



### Losses for Self-Supervised Learning

'''
Code acquired from https://github.com/omipan/camera_traps_self_supervised/blob/main/losses.py
'''
def triplet_loss(emb, args, dist_type='cosine', margin=0.3):
    # NOTE currently just randomly selects indices as negatives 

    b_size = emb.shape[0]
    inds = torch.randint(0, b_size, (b_size//2, ))
    mask = (inds != torch.arange(b_size//2)).float().cuda()            
    
    if dist_type == 'l2':
        loss = (mask*F.triplet_margin_loss(emb[:b_size//2, :], emb[b_size//2:, :], 
                                           emb[inds, :], margin=margin, reduction='none')).mean()
                                            
    elif dist_type == 'cosine':
        pos_dist = (-F.cosine_similarity(emb[:b_size//2, :], emb[b_size//2:, :], dim=1) + 1)/2
        neg_dist = (-F.cosine_similarity(emb[:b_size//2, :], emb[inds, :], dim=1) + 1)/2
        hinge_dist = torch.clamp(margin + pos_dist - neg_dist, min=0.0)
        loss = (mask*hinge_dist).mean()
    
    return loss


'''
Code acquired from https://github.com/omipan/camera_traps_self_supervised/blob/main/losses.py
'''
def nt_xent(x1, x2, args):
    # assumes that the input data is stacked i.e. 
    # x1 (B1 C H W) + x2 (B2 C H W) - B1 typically == B2
    
    x = torch.cat((x1, x2), 0)
    x = F.normalize(x, dim=1)
    x_scores = x @ x.t()
    x_scale = x_scores / args['temperature']   # scale with temperature

    # (2N-1)-way softmax without the score of i-th entry itself.
    # Set the diagonals to be large negative values, which become zeros after softmax.
    x_scale = x_scale - torch.eye(x_scale.size(0)).to(x_scale.device) * 1e5
    
    # targets 2N elements.
    if x1.shape[0] == 1:
        # last element is the target i.e. all should be the same
        targets = torch.zeros(x.shape[0], device=x.device).long() 
        x_scale[0,0] = 1.0 / args['temperature'] 
    else: 
        # data is stacked in two halves
        targets = torch.arange(x.shape[0], device=x.device).long()
        targets[:x.shape[0]//2] += x.shape[0]//2
        targets[x.shape[0]//2:] -= x.shape[0]//2
        
    return F.cross_entropy(x_scale, targets)




def adapter_train(train_loader,optimizer,adapter,criterion,args):
    '''
    Utility function that trains the MLP adapter with few-shot samples on top of frozen features.
    '''
    adapter.train()
    for epoch in range(args['epochs']):
        losses = []
        for batch_num, input_data in enumerate(train_loader):
            optimizer.zero_grad()
            x, y = input_data
            x = x.to(args['device']).float()
            y = y.to(args['device'])

            output = adapter(x)
            loss = criterion(output, y)
            loss.backward()
            losses.append(loss.item())

            optimizer.step()

            if batch_num % 40 == 0:
                print('\tEpoch %d | Batch %d | Loss %6.2f' % (epoch, batch_num, loss.item()))
        #print('Epoch %d | Loss %6.2f' % (epoch, sum(losses)/len(losses)))

def adapter_predict(adapter,loader,args,return_logits=False):
    '''
    Utility function that used the tuned MLP adapter to make predictions.
    '''
    adapter.eval()
    preds = []
    all_logits = torch.Tensor().to(args['device'])

    with torch.no_grad():
        for input_data in loader:
            if args['finetune_type']!="end_to_end":
                x, y = input_data
            else:
                x = input_data['im']
                y = input_data['target']
            x = x.to(args['device']).float()
            y = y.to(args['device'])
            logits = adapter(x)
            all_logits=torch.concat((all_logits,logits))
            output = logits.argmax(dim=1)
            preds = preds+list(output.cpu().numpy())
    if return_logits:
        return all_logits
    else:
        return preds

    
'''
Code acquired from https://github.com/omipan/camera_traps_self_supervised/blob/main/utils.py
'''
def linear_eval_all(model, train_loader, test_loader, args, inds, amts, grid_search=False, target_type='target'):
    '''
    Utility function that can be used to linearly evaluate the quality of the representations learnt by the self-supervised learning pretext task.
    '''
    # extract train and test features - only do this once
    x_train_o, y_train_o, ids_train = get_features(model, train_loader, args, target_type)
    x_test_o, y_test_o, ids_test = get_features(model, test_loader, args, target_type)

    # loop over the different data splits 
    res = {}
    for ii in range(len(inds)): 
        # select subset of data
        x_train = x_train_o[inds[ii], :]
        y_train = y_train_o[inds[ii]]

        # make sure the labels are consistent and range from 0 to C-1   
        _, inv_labels = np.unique(np.hstack((y_train, y_test_o)), return_inverse=True)
        y_train = inv_labels[:y_train.shape[0]]
        y_test = inv_labels[y_train.shape[0]:]
                
        # perform linear evaluation
        test_acc, test_acc_bal = train_linear(x_train, y_train, x_test_o, y_test, args['lin_max_iter'], grid_search) 
        amt = str(amts[ii])
        res['test_acc_' + amt] = test_acc
        res['test_acc_bal_' + amt] = test_acc_bal
        print('Linear eval ' + (amt+'%').rjust(4) + ': acc {:.2f},  bal acc {:.2f}'.format(test_acc, test_acc_bal))
            
    return res

'''
Code acquired from https://github.com/omipan/camera_traps_self_supervised/blob/main/utils.py
'''
def get_features(model, loader, args, target_type, op_type='feat', standard_backbone=False): 
    '''
    Return features given pretrained encoder 
    '''
    
    # extract features from the model 
    if op_type == 'feat':
        only_feats = True
    else:
        only_feats = False 
        
    model.eval()
    features = []
    targets = []    
    ids = []
    with torch.no_grad():
        for data in loader:
            data['im'] = data['im'].to(args['device'])
            
            if args['use_clip']:
                features.append(model.encode_image(data['im']).data.cpu().numpy())
            elif standard_backbone:
                features.append(model(data['im']).data.cpu().numpy())
            else:

                op = model(data['im'], only_feats=only_feats)
                features.append(op[op_type].data.cpu().numpy())


            targets.append(data[target_type].cpu().numpy())
            ids.append(data['id'].cpu().numpy())
    
    return np.vstack(features), np.hstack(targets), np.hstack(ids)

'''
Code acquired from https://github.com/omipan/camera_traps_self_supervised/blob/main/utils.py
'''
def train_linear(x_train_ip, y_train, x_test_ip, y_test, max_iter, grid_search):
    '''
    Utility used within linear evaluation routine (linear_eval_all.py)
    '''
    x_train = x_train_ip.astype(np.float32).copy()
    x_test = x_test_ip.astype(np.float32).copy()
    
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    x_train = normalize(x_train, norm='l2')
    x_test = normalize(x_test, norm='l2')

    rseed = 0    
    if grid_search:
        parameters = {'C' : [0.001, 0.01, 0.1, 1, 10, 100]}
        cls = LogisticRegression(random_state=rseed, tol=1e-4, multi_class='multinomial', C=1., dual=False, max_iter=max_iter)       
        clf = GridSearchCV(cls, parameters, n_jobs=-1, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=rseed), refit=True)        
        clf.fit(x_train, y_train)
    else:
        clf = LogisticRegression(random_state=rseed, tol=1e-4, multi_class='multinomial', C=1., dual=False, max_iter=max_iter, n_jobs=-1).fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    acc = accuracy_score(y_test, y_pred)*100
    bal_acc = balanced_accuracy_score(y_test, y_pred)*100

    return acc, bal_acc

def accuracy(output, targets, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = targets.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res  


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

'''
Code acquired from https://github.com/gaopengcuhk/Tip-Adapter/blob/main/tip_adapter_ImageNet.py
'''
def zeroshot_classifier(classnames, templates, model):
    import clip
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights