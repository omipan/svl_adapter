import ast
import torch
import clip
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir =   '{}{}/'.format(args['root_data_dir'],args['dataset'])
    print(data_dir)
   
    metadir = data_dir+'{}_meta.csv'.format(args['dataset'])
    meta = pd.read_csv(metadir,index_col=0)
    meta['category_id'] = meta.category_id.astype(int)
    if args['dataset'] in ['cct20','kenya','icct','serengeti']:
        meta['label'] = meta['label'].apply(lambda x: x.replace('shoats','sheep or goat')) ## shoats in kenya corresponds to sheeps or goats, rename accordingly
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

    split_list = ['train']
    sampled = meta.loc[meta.img_set.isin(split_list)]
    label_mapper = meta.drop_duplicates(subset=['label'])['label'].reset_index()
    label_mapper.drop(columns=['index'],inplace=True)
    label_mapper['clip_label'] = label_mapper.label.apply(lambda x: x.replace('_',' '))
    if args['dataset'] == 'kenya':
        label_mapper['clip_label'] = label_mapper['clip_label'].apply(lambda x: 'a photo of {}.'.format(x))
    elif args['dataset']=="cct20":
        label_mapper['clip_label'] = label_mapper['clip_label'].apply(lambda x: 'a photo of {}.'.format(x))
    elif args['dataset']=="icct":
        label_mapper['clip_label'] = label_mapper['clip_label'].apply(lambda x: 'a photo of {}.'.format(x))
    elif args['dataset']=="serengeti":
        label_mapper['clip_label'] = label_mapper['clip_label'].apply(lambda x: 'a photo of {}.'.format(x))
    elif args['dataset']=="fgvc_aircraft":
        label_mapper['clip_label'] = label_mapper['clip_label'].apply(lambda x: 'a photo of a {}, a type of aircraft.'.format(x))
    elif args['dataset']=="caltech_101":
        label_mapper['clip_label'] = label_mapper['clip_label'].apply(lambda x: 'a photo of a {}.'.format(x))
    elif args['dataset']=="eurosat":
        label_mapper['clip_label'] = label_mapper['clip_label'].apply(lambda x: 'a centered satellite photo of {}.'.format(x))
    elif args['dataset']=="oxford_pets":
        label_mapper['clip_label'] = label_mapper['clip_label'].apply(lambda x: 'a photo of a {}, a type of pet.'.format(x))
    elif args['dataset']=="oxford_flowers":
        label_mapper['clip_label'] = label_mapper['clip_label'].apply(lambda x: 'a photo of a {}, a type of flower.'.format(x))
    elif args['dataset']=="dtd":
        label_mapper['clip_label'] = label_mapper['clip_label'].apply(lambda x: '{} texture.'.format(x))
    elif args['dataset']=="ucf101":
        label_mapper['clip_label'] = label_mapper['clip_label'].apply(lambda x: 'a photo of a person doing {}.'.format(x))
    elif args['dataset']=="food101":
        label_mapper['clip_label'] = label_mapper['clip_label'].apply(lambda x: 'a photo of {}, a type of food.'.format(x))
    elif args['dataset']=="sun397":
        label_mapper['clip_label'] = label_mapper['clip_label'].apply(lambda x: 'a photo of a {}.'.format(x))
    elif args['dataset']=="stanford_cars":
        label_mapper['clip_label'] = label_mapper['clip_label'].apply(lambda x: 'a photo of a {}.'.format(x))
    elif args['dataset']=="fmow":
        label_mapper['clip_label'] = label_mapper['clip_label'].apply(lambda x: 'a satellite photo of {}.'.format(x))
    elif args['dataset']=="oct":
        label_mapper['clip_label'] = label_mapper['clip_label'].apply(lambda x: 'an OCT scan of {} retina.'.format(x))

    clip_to_target = dict(label_mapper[['clip_label','label']].values)
    ## label to pretrained model id mapping
    grouped_label_to_clip_ids = pd.DataFrame(clip_to_target.values()).reset_index().rename(columns={'index':'clip_id',0:'dataset_label'}).groupby('dataset_label').clip_id.apply(list)
    N = sampled.shape[0]
    
    ### Load CLIP model
    model, preprocess = clip.load(args['model_subtype'], device=device)


    image_links = list(sampled.img_path)
    targets = list(sampled.label)
    clip_labels = list(clip_to_target.keys())
    
    pred_df = pd.DataFrame()
    correct_list = []
    print('DATASET : {}'.format(args['dataset']))
    for i in range(N):
        if i%100==0:
            print('{}% done'.format(i/N))
        img = Image.open(os.path.join(data_dir,image_links[i]))
        image =  preprocess(img).unsqueeze(0).to(device)

        text = clip.tokenize(clip_labels).to(device)
        with torch.no_grad():
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1) #.cpu().numpy()
        
        label_probs = grouped_label_to_clip_ids.apply(lambda x: torch.index_select(probs.cpu(), 1, torch.tensor(x)).sum().item())
        if len(clip_labels)>2:
            prob1,prob2,prob3 = label_probs.sort_values(ascending=False)[0:3].values
            pred1,pred2,pred3 = label_probs.sort_values(ascending=False)[0:3].index
        else:
            ## Datasets with binary classification, less than 3 potential labels
            prob1,prob2 = label_probs.sort_values(ascending=False)[0:2].values
            pred1,pred2 = label_probs.sort_values(ascending=False)[0:2].index
            prob3 = np.nan
            pred3 = np.nan

        rest_of_the_predictions = label_probs.sort_values(ascending=False)[1:].index
        rest_of_the_prediction_probs =label_probs.sort_values(ascending=False)[1:].values
        
        pred_correct =  targets[i]==pred1
        correct_list.append(pred_correct)
    
        df_temp = pd.DataFrame({'dataset':args['dataset'],
                        'img_path':image_links[i],
                        'pred1': [pred1],
                        'pred2': [pred2],
                        'pred3': [pred3],
                        'prob1':prob1,
                        'prob2':prob2,
                        'prob3':prob3,
                        'rest_of_pred':[list(rest_of_the_predictions)],
                        'rest_of_pred_probs':[list(rest_of_the_prediction_probs)],
                        'correct':pred_correct,
                        'target': [targets[i]]})
        pred_df = pred_df.append(df_temp,ignore_index=True)
    num_shot=0
    acc_mean = 100*np.mean(correct_list)

    clip_model = 'clip_'+args['model_subtype'].replace('/','').replace('-','_')
    save_line = "{},{}, {} Shot, Test acc stat: {:.2f} ()\n".format(args['dataset'],clip_model, num_shot, acc_mean, '')
    print(save_line, flush=True)

    pred_df['img_path_trimmed'] = pred_df['img_path'].apply(lambda x: x.replace(data_dir,''))
    label_to_category = dict(meta[['label','category_id']].drop_duplicates().values)
 
    predicted_labels = set(pred_df.pred1)
    sub_df = pred_df.copy()
    sub_df = sub_df.rename(columns={'target':'label'})
    images_left =sub_df.shape[0]
    unique_pred = sub_df.pred1.nunique()

    pseudo_df = pd.DataFrame()
    for pred_label in predicted_labels:
        sub_label_df = pred_df.loc[(pred_df.pred1==pred_label) & (pred_df.prob1>=args['confidence_lower_bound'])]
        sub_label_df = sub_label_df.sort_values('prob1',ascending=False).iloc[0:args['imgs_per_label']]
        pseudo_df = pd.concat((pseudo_df,sub_label_df))

    pseudo_full = pseudo_df.rename(columns={'target':'label'}).copy()
    pseudo_full.drop_duplicates(subset='img_path',inplace=True)

    ## Check whether every label has representation in the pseudolabel space
    list_of_classes_without_pseudolabel = set(pred_df.target)-predicted_labels

    if len(list_of_classes_without_pseudolabel)>0:
        print(args['dataset']+' NEEDED EXTRA GUESSES') ## if no 1st choice pseudolabels for a certain category, keep some of the 2nd guesses
        rare_df = pd.DataFrame()
        for rare_label in list_of_classes_without_pseudolabel:
            temp_df = pred_df.copy()
            indices = temp_df.rest_of_pred.apply(lambda x: x.index(rare_label) if rare_label in x else -1).index.to_list()
            order_of_pred = temp_df.rest_of_pred.apply(lambda x: x.index(rare_label) if rare_label in x else -1).values
            for idx in indices:
                if order_of_pred[idx]!=-1:
                    temp_df.loc[idx,'rare_pred'] = temp_df.loc[idx].rest_of_pred[order_of_pred[idx]]
                    temp_df.loc[idx,'rare_pred_prob'] = temp_df.loc[idx].rest_of_pred_probs[order_of_pred[idx]]

            if len(rare_df)>0:
                temp_df = temp_df.loc[~temp_df.img_path.isin(set(rare_df.img_path))]
            rare_df = pd.concat((rare_df,temp_df.dropna(subset=['rare_pred']).sort_values('rare_pred_prob',ascending=False).head(1)))
            rare_df['pred1'] = rare_df['rare_pred']

        pseudo_full = pseudo_full.loc[~pseudo_full.img_path.isin(set(rare_df.img_path))]


        pseudo_full = pd.concat((pseudo_full,rare_df))

    meta_train_replace = meta.loc[meta.img_path.isin(set(pseudo_full.img_path_trimmed))]
    pseudo_full.sort_values('img_path_trimmed',inplace=True)
    pseudo_full['pseudolabel'] = pseudo_full['pred1']

    meta_train_replace.sort_values('img_path',inplace=True)
    ### Replace label and the corresponding category id with the predicted ones(pseudolabels)
    meta_train_replace['label'] = pseudo_full['pseudolabel'].values
    meta_train_replace['category_id'] = meta_train_replace['label'].apply(lambda x: label_to_category[x])

    ### keep val/test as they are for evaluation purposes
    meta_test = meta.loc[~meta.img_set.isin(split_list)].copy()
    ### save file updated with pseudolabels
    meta_new = pd.concat((meta_train_replace,meta_test))
    # reset index
    meta_new.reset_index(inplace=True)
    meta_new.drop(columns=['index'],inplace=True)
    
    meta_new.to_csv('{}/{}_meta_{}_pseudo_clip_{}shot.csv'.format(data_dir,args['dataset'],clip_model,args['imgs_per_label']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_data_dir",type=str,default='data/') 
    parser.add_argument('--dataset', choices=['ucf101','food101','sun397','stanford_cars','fgvc-aircraft','caltech-101','eurosat','oxford_pets','oxford_flowers','dtd',
                                              'kenya','cct20','serengeti','icct','fmow','oct'], type=str)
    parser.add_argument("--model_subtype",type=str, choices=["ViT-B/32", "ViT-B/16","ViT-L/14", "RN50"],default="RN50", help="exact type of clip pretraining backbone")
    parser.add_argument("--confidence_lower_bound", type=float,help='minimum confidence required for a pseudolabel to be kept',default=0.0)
    parser.add_argument("--imgs_per_label", type=int,help='the amount of pseudolabels to keep for each of the predicted labels (ranked based on their clip confidence, higher first)',default=16)
    # turn the args into a dictionary
    args = vars(parser.parse_args())
    main(args)

