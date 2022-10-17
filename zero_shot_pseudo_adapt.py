import os
import ast
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, resnet50
import clip
from utils import end_to_end_train
from datasets import FeatureDataset
from models import AdapterMLP
from utils import adapter_train,adapter_predict
from utils import zeroshot_classifier,accuracy_from_logits

dataset_to_template = {'cct20':'a photo of a {}.',
                      'kenya':'a photo of a {}.',
                      'icct':'a photo of a {}.',
                      'serengeti':'a photo of a {}.',
                      'fgvc_aircraft':'a photo of a {}, a type of aircraft.',
                      'caltech_101':'a photo of a {}.',
                      'eurosat':'a centered satellite photo of {}.',
                      'oxford_pets':'a photo of a {}, a type of pet.',
                      'oxford_flowers':'a photo of a {}, a type of flower.',
                      'dtd':'{} texture.',
                      'ucf101':'a photo of a person doing {}.',
                      'food101':'a photo of {}, a type of food.',
                      'sun397':'a photo of a {}.',
                      'stanford_cars':'a photo of a {}.',
                      'fmow':'a photo of a {}.',
                      'oct':'an OCT scan of {} retina.',
                      }


def main(args):    
    
    pretrained_model = args['pretrained_model']
    if any([x in args['pretrained_model'] for x in ['simclr','triplet']]): ## if model is trained with ssl
        pretrained_model = pretrained_model+'_'+args['pos_type']
  
    pretrained_feature_folder = pretrained_model+'_feat'

    pseudolabel_type = '{}_pseudolabels_{}'.format(args['pseudolabel_model'],args['pseudo_conf']) #clip_RN50_pseudolabels_70

    train_file = np.load(os.path.join(args['feature_path'],pretrained_feature_folder,'{}_train_features_{}.npz'.format(pretrained_model,pseudolabel_type)))
    train_feature, train_label = train_file["feature_list"], train_file["label_list"]
    val_file = np.load(os.path.join(args['feature_path'],pretrained_feature_folder,'{}_val_features_{}.npz'.format(pretrained_model,pseudolabel_type)))
    val_feature, val_label = val_file["feature_list"], val_file["label_list"]
    test_file = np.load(os.path.join(args['feature_path'],pretrained_feature_folder,'{}_test_features_{}.npz'.format(pretrained_model,pseudolabel_type)))
    test_feature, test_label = test_file["feature_list"], test_file["label_list"]

    num_shot=0
    if args['finetune_type'] in ['mlp','mlp_adapter']:
        mlp_adaptation=', MLP'
      
        num_classes = len(set(train_label))
        input_size = train_feature.shape[1]
        train_set = FeatureDataset(train_feature,train_label)
        val_set  = FeatureDataset(val_feature,val_label)
        test_set  = FeatureDataset(test_feature,test_label)
        train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True)
        val_loader  = DataLoader(val_set,  batch_size=args['batch_size'], shuffle=False)
        test_loader  = DataLoader(test_set,  batch_size=args['batch_size'], shuffle=False)
        model = AdapterMLP(num_classes,input_size,args['hidden_size']).to(args['device'])
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        print(model)

        adapter_train(train_loader,optimizer,model,criterion,args)
        
        pred = adapter_predict(model,test_loader,args)
        if args['finetune_type'] == 'mlp_adapter':
                
                logits = adapter_predict(model,test_loader,args,return_logits=True)
                # test_file = np.load(os.path.join(dataset_path,pretrained_feature_folder, "{}_test_features.npz".format(pretrained_model)))
                # test_feature, test_label = test_file["feature_list"], test_file["label_list"]
                

                clip_test_file = np.load(os.path.join(args['feature_path'],'clip_{}_feat'.format(args['clip_fusion_model']),'clip_{}_test_features.npz'.format(args['clip_fusion_model'])))
                clip_test_feature, clip_test_label = clip_test_file["feature_list"], clip_test_file["label_list"]
                clip_test_feature = torch.from_numpy(clip_test_feature).to(args['device']).half()
                clip_test_features = clip_test_feature # / clip_test_feature.norm(dim=-1,keepdim=True)
                clip_test_labels = torch.from_numpy(clip_test_label).to(args['device']).half()
                ### Fusion of CLIP Zero-shot and adapted image features
                data_dir = '{}{}/'.format(args['root_data_dir'],args['dataset'])
                metadir = data_dir+'{}_meta.csv'.format(args['dataset'])
                meta = pd.read_csv(metadir,index_col=0)
                meta['category_id'] = meta.category_id.astype(int)
                if args['dataset'] in['cct20','kenya','icct','serengeti']:
                    #meta['label'] = meta.species.copy()
                    ## shoats in kenya corresponds to sheeps or goats
                    meta['label'] = meta['label'].apply(lambda x: x.replace('shoats','sheep or goat'))
                    # In Serengeti make the labels clip readable
                    meta['label'] = meta['label'].apply(lambda x:   x.replace('guineaFowl','guineafowl')
                                                                    .replace('lionFemale','lion female')
                                                                    .replace('gazelleThomsons','gazelle thomsons')
                                                                    .replace('vervetMonkey','vervet monkey')
                                                                    .replace('lionMale','lion male')
                                                                    .replace('gazelleGrants','gazelle grants')
                                                                    .replace('otherBird','other bird')
                                                                    .replace('koriBustard','kori bustard')
                                                                    .replace('dikDik','dik dik')
                                                                    .replace('batEaredFox','bat-eared fox')
                                                                    .replace('secretaryBird','secretary bird')
                                                                    .replace('hyenaSpotted','hyena spotted')
                                                                    .replace('hyenaStriped','hyena striped')
                                                                    .replace('secretaryBird','secretary bird'))
                meta['label'] = meta['label'].apply(lambda x: x.replace('_',' '))
                if args['dataset']=="oct":
                    meta['label'] = meta.label.apply(lambda x: x.replace('CNV','Choroidal Neovascularization').replace('DME','Diabetic Macular Edema')
                                                    .replace('DRUSEN','Drusen').replace('NORMAL','Healthy'))
                classes = list(meta.drop_duplicates('category_id')[['category_id','label']].sort_values('category_id').label.values)
                templates = [dataset_to_template[args['dataset']]]
                    ### Load CLIP model
                
                with torch.no_grad():
                    clip_model, _ = clip.load('RN50', device=args['device'])
                    zeroshot_weights = zeroshot_classifier(classes, templates, clip_model)
                    old_logits = 100. * clip_test_features @ zeroshot_weights
                
                ## select alpha based on clip confidence
                clip_conf = old_logits.softmax(dim=-1).max(dim=-1).values.mean().item()
                alpha = 1-clip_conf
                mlp_adaptation =', SVL_ADAPTER*, Alpha: {}'.format(np.round(alpha,3))
                ### Given fusion calculate old logits again with normalized clip_features
                clip_test_features = clip_test_features / clip_test_features.norm(dim=-1,keepdim=True)
                old_logits = 100. * clip_test_features @ zeroshot_weights
                # new_logits = alpha_multiplier*logits + (1-alpha_multiplier)*old_logits
                new_logits = alpha*logits + (1-alpha)*old_logits
                pred = new_logits.argmax(dim=1).cpu().numpy()
                #test_acc =accuracy_score(test_label, new_logits.argmax(dim=1).cpu().numpy())*100

    test_acc = accuracy_score(test_label, pred)*100
    test_bal_acc = balanced_accuracy_score(test_label, pred)*100
    save_line = "{},{} pretraining,{} pseudolabels, {} Shot, Test acc stat: {:.2f}, Bal acc stat: {:.2f}{}\n".format(args['dataset'],pretrained_model,pseudolabel_type, num_shot, test_acc, test_bal_acc,mlp_adaptation)

    with open("report/zero_shot_results","a+",) as writer:
            writer.write(save_line)
    print(save_line, flush=True)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_data_dir", type=str, default='data/')
    parser.add_argument('--dataset', choices=['ucf101','food101','sun397','stanford_cars','fgvc-aircraft','caltech-101','eurosat','oxford_pets','oxford_flowers','dtd',
                                              'kenya','cct20','serengeti','icct','fmow','oct'], type=str)
    parser.add_argument("--feature_path", type=str, default='',help="directory with extracted features")
    
    parser.add_argument("--pretrained_model", type=str, choices=['simclr_RN50','triplet_RN50', 'clip_RN50','imagenet_RN50','clip_ViT-L14'],help='type of pretrained model')
    parser.add_argument("--pos_type", type=str, choices=['context_sample','augment_self', ''],default='augment_self',help='type of positives used during pretraining, if features from ssl')

    parser.add_argument("--return_single_image",action="store_true",default=True)
    parser.add_argument("--pseudo_conf",type=str,default='')
    parser.add_argument("--pseudolabel_model",type=str,choices=['clip_RN50','clip_ViT-L14'],default='clip_RN50')
    parser.add_argument("--clip_fusion_model",type=str,choices=['RN50','ViT-L_14','ViT-L_14'],default='RN50',help='which clip model to fuse ssl adapter with')

    parser.add_argument("--finetune_type", type=str, choices=['linear_probe','mlp','mlp_adapter','end_to_end'],default='linear_probe',help='type of low-shot finetuning')
    parser.add_argument("--confidence_alpha",action="store_true",default=False)
    parser.add_argument("--batch_size", type=int,help='batch size if finetuning is mlp',default=32)
    parser.add_argument("--epochs", type=int,help='number of adapter tuning epochs if finetuning is mlp',default=50)
    parser.add_argument("--hidden_size", type=int,help='hidden layer dimensions of mlp if finetuning is mlp',default=256)
    parser.add_argument("--device", type=str, default='cuda')
    
    args = vars(parser.parse_args())
    #ip_args = vars(parser.parse_args())

    main(args)