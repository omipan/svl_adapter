
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import FeatureDataset
from models import AdapterMLP
from utils import adapter_train,adapter_predict
from sklearn.decomposition import PCA
import clip
from utils import zeroshot_classifier

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
                      'oct':'an OCT scan of {} retina.'}

def main(args):
    
    ### Load extracted features
    os.makedirs('report',exist_ok=True)
    val_shot_list = {1: 1, 2: 2, 4: 4, 8: 4, 16: 4}
    num_step = 8
    num_run = 3 
    
    pretrained_model = args['pretrained_model']
    if any([x in args['pretrained_model'] for x in ['simclr','triplet']]): ## if model is trained with ssl
        pretrained_model = pretrained_model+'_'+args['pos_type']

  
    pretrained_feature_folder = pretrained_model+'_feat'

    print('Pretrained model {}'.format(pretrained_model))
    train_file = np.load(os.path.join(args['feature_path'],pretrained_feature_folder, "{}_train_features.npz".format(pretrained_model)))
    train_feature, train_label = train_file["feature_list"], train_file["label_list"]

    val_file = np.load(os.path.join(args['feature_path'],pretrained_feature_folder, "{}_val_features.npz".format(pretrained_model)))
    val_feature, val_label = val_file["feature_list"], val_file["label_list"]
    test_file = np.load(os.path.join(args['feature_path'],pretrained_feature_folder, "{}_test_features.npz".format(pretrained_model)))
    test_feature, test_label = test_file["feature_list"], test_file["label_list"]
    
    if args['reduce_dims']:
        dim_reducer = PCA(n_components=1024)
        dim_reducer.fit(np.concatenate((train_feature,val_feature,test_feature)))
        train_feature = dim_reducer.transform(train_feature)
        val_feature = dim_reducer.transform(val_feature)
        test_feature = dim_reducer.transform(test_feature)


    for num_shot in [1, 2, 4, 8, 16]:
        test_acc_step_list = np.zeros([num_run, num_step])
        for seed in range(1, num_run+1):
            np.random.seed(seed)
            print(f"-- Seed: {seed} --------------------------------------------------------------")
            # Sampling
            all_label_list = np.unique(train_label)
            selected_idx_list = []
            for label in all_label_list:
                label_collection = np.where(train_label == label)[0]
                
                selected_idx = np.random.choice(label_collection, size=min(len(label_collection),num_shot), replace=False)
                selected_idx_list.extend(selected_idx)
                

            fewshot_train_feature = train_feature[selected_idx_list]
            fewshot_train_label = train_label[selected_idx_list]


            val_num_shot = val_shot_list[num_shot]
            val_selected_idx_list = []
            for label in all_label_list:
                label_collection = np.where(val_label == label)[0]
                selected_idx = np.random.choice(label_collection, size=min(len(label_collection),val_num_shot), replace=False)
                val_selected_idx_list.extend(selected_idx)

            fewshot_val_feature = val_feature[val_selected_idx_list]
            fewshot_val_label = val_label[val_selected_idx_list]

            ### finetune MLP or Linear Probe
            # search initialization
            if args['finetune_type'] in ['mlp','mlp_adapter']:
                mlp_adaptation=', MLP'

                num_classes = len(all_label_list)
                input_size = fewshot_train_feature.shape[1]

                train_set = FeatureDataset(fewshot_train_feature,fewshot_train_label)
                val_set  = FeatureDataset(fewshot_val_feature,fewshot_val_label)
                test_set  = FeatureDataset(test_feature,test_label)
                train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True)
                val_loader  = DataLoader(val_set,  batch_size=args['batch_size'], shuffle=False)
                test_loader  = DataLoader(test_set,  batch_size=args['batch_size'], shuffle=False)
                model = AdapterMLP(num_classes,input_size,args['hidden_size']).to(args['device'])
                optimizer = torch.optim.Adam(model.parameters())

                criterion = nn.CrossEntropyLoss()
                print(model)

                adapter_train(train_loader,optimizer,model,criterion,args)

                


                pred = adapter_predict(model,test_loader,args,return_logits=False)
                test_acc = accuracy_score(test_label, pred)*100
                print("Test Accuracy before fusing with ZS CLIP: {:.2f}".format(test_acc), flush=True)

                
                if args['finetune_type'] == 'mlp_adapter':
                    logits = adapter_predict(model,test_loader,args,return_logits=True)
                                
                    ## clip val feature, useful for alpha tuning in residual adapter
                    clip_val_file = np.load(os.path.join(args['feature_path'],'clip_{}_feat'.format(args['clip_fusion_model']),'clip_{}_val_features.npz'.format(args['clip_fusion_model'])))
                    clip_val_feature, _ = clip_val_file["feature_list"], clip_val_file["label_list"]
                    fewshot_clip_val_feature = clip_val_feature[val_selected_idx_list]                   
                    fewshot_clip_val_feature = torch.from_numpy(fewshot_clip_val_feature).to(args['device']).half()
                    fewshot_clip_val_features = fewshot_clip_val_feature / fewshot_clip_val_feature.norm(dim=-1,keepdim=True)
    


                    clip_test_file = np.load(os.path.join(args['feature_path'],'clip_{}_feat'.format(args['clip_fusion_model']),'clip_{}_test_features.npz'.format(args['clip_fusion_model'])))
                    clip_test_feature, clip_test_label = clip_test_file["feature_list"], clip_test_file["label_list"]
                    clip_test_feature = torch.from_numpy(clip_test_feature).to(args['device']).half()

                    clip_test_features = clip_test_feature  
                    if args['clip_fusion_model'] == 'ViT-L_14':
                        clip_test_features = clip_test_feature / clip_test_feature.norm(dim=-1,keepdim=True)
                    clip_test_labels = torch.from_numpy(clip_test_label).to(args['device']).half()
                    
                    ### Fusion of CLIP Zero-shot and adapted image features
                    data_dir =   '{}{}/'.format(args['root_data_dir'],args['dataset'])
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
                    clip_model, _ = clip.load(args['clip_fusion_model'].replace('_','/'), device=args['device'])
                    
                    zeroshot_weights = zeroshot_classifier(classes, templates, clip_model)

                    old_logits = 100. * clip_test_features @ zeroshot_weights
                    
                    if args['confidence_alpha']:
                        clip_conf = old_logits.softmax(dim=-1).max(dim=-1).values.mean().item()
                        alpha = 1-clip_conf
                        mlp_adaptation =', SVL_ADAPTER*, Alpha: {}'.format(np.round(alpha,3))
                       
                    elif args['tune_alpha']:
                        alpha = 0.5  ###INIT ALPHA, for comparison purposes
                        mlp_adaptation =', SVL_ADAPTER'
                    else:
                        alpha = args['adapter_alpha']
                        mlp_adaptation=', {}'.format(args['finetune_type'].upper())+'_ALPHA_{}'.format(args['adapter_alpha'])
                    if args['clip_fusion_model']!='RN50':
                        mlp_adaptation=mlp_adaptation+'_{}'.format(args['clip_fusion_model'].replace('/',''))
                    
                    ### Given fusion calculate old logits again with normalized clip_features
                    clip_test_features = clip_test_feature / clip_test_feature.norm(dim=-1,keepdim=True)
                    old_logits = 100. * clip_test_features @ zeroshot_weights

                   
                    #### TUNE ALPHA
                    if args['tune_alpha']:
                        print('Test Accuracy before tuning alpha(0.5): {}'.format(test_acc)) 
                        best_alpha = 0
                        best_val_acc = 0
                        alpha_list = [al/20 for al in range(0,20+1)] #20 values in 0-1 range
                        for alpha in alpha_list:
                            old_val_logits = 100. * fewshot_clip_val_features @ zeroshot_weights

                            val_logits = adapter_predict(model,val_loader,args,return_logits=True)
                            new_val_logits = alpha*val_logits + (1-alpha)*old_val_logits


                            val_acc =accuracy_score(fewshot_val_label, new_val_logits.argmax(dim=1).cpu().numpy())*100
                            if val_acc>best_val_acc:
                                best_val_acc = val_acc
                                best_alpha = alpha
                                print('New best setting, alpha: {}, val accuracy: {}'.format(best_alpha,best_val_acc))
                        
                        new_logits = best_alpha*logits + (1-best_alpha)*old_logits
                        test_acc =accuracy_score(test_label, new_logits.argmax(dim=1).cpu().numpy())*100
                        print('Test Accuracy after tuning alpha: {}'.format(test_acc))
                    
                    else:
                        new_logits = alpha*logits + (1-alpha)*old_logits
                        test_acc =accuracy_score(test_label, new_logits.argmax(dim=1).cpu().numpy())*100

                print("Test Accuracy: {:.2f}".format(test_acc), flush=True)
                test_acc_step_list[seed - 1, num_step-1] = test_acc
                
        # save results of last step
        test_acc_list = test_acc_step_list[:, -1]
        acc_mean = np.mean(test_acc_list)
        acc_std = np.std(test_acc_list)
 
       
        save_line = "{},{}, {} Shot, Test acc stat: {:.2f} ({:.2f}){}\n".format(args['dataset'],pretrained_model, num_shot, acc_mean, acc_std,mlp_adaptation)
        print(save_line, flush=True)
        with open(
            "report/low_shot_results.txt",
            "a+",
        ) as writer:
            writer.write(save_line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_data_dir",type=str,default='data/')
    parser.add_argument('--dataset', choices=['ucf101','food101','sun397','stanford_cars','fgvc-aircraft','caltech-101','eurosat','oxford_pets','oxford_flowers','dtd',
                                              'kenya','cct20','serengeti','icct','fmow','oct'], type=str)
   
    parser.add_argument("--feature_path", type=str, default='',help="directory with extracted features")
    parser.add_argument("--pretrained_model", type=str, choices=['simclr_RN50','triplet_RN50', 'clip_RN50','imagenet_RN50','clip_ViT-L14'],help='type of pretrained model')
    parser.add_argument("--pos_type", type=str, choices=['context_sample','augment_self', ''],default='augment_self',help='type of positives used during pretraining, if features from ssl')
    parser.add_argument("--return_single_image",action="store_true",default=True)
    parser.add_argument("--reduce_dims", action='store_true',default=False, help="reduce dimensions of features, default is 1024")
    
    parser.add_argument("--finetune_type", type=str, choices=['linear_probe','mlp','mlp_adapter','end_to_end'],default='linear_probe',help='type of low-shot finetuning')
    parser.add_argument("--adapter_alpha", type=float,default=0.2,help='residual feature parameter. The higher the value the more low-shot knowledge impacts the prediction')
    parser.add_argument("--confidence_alpha",action="store_true",default=False)
    parser.add_argument("--tune_alpha",action="store_true",default=False)
    parser.add_argument("--batch_size", type=int,help='batch size if finetuning is mlp',default=32)
    parser.add_argument("--epochs", type=int,help='number of adapter tuning epochs if finetuning is mlp',default=50)
    parser.add_argument("--hidden_size", type=int,help='hidden layer dimensions of mlp if finetuning is mlp',default=256)
    
    parser.add_argument("--clip_fusion_model",type=str,choices=['RN50','ViT-L_14','ViT-L_14'],default='RN50',help='which clip model to fuse ssl adapter with')

    parser.add_argument("--device", type=str, default='cuda')
    args = vars(parser.parse_args())
   
    main(args)

                    
                    

