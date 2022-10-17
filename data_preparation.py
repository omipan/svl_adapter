import os
import json
import numpy as np
import pandas as pd
from ast import literal_eval
from PIL import Image
import argparse


data_dict = {'eurosat':{'img_dir_name':'2750/','split_file':'split_zhou_EuroSAT.json','split_names':['train','val','test']},
             'dtd':{'img_dir_name':'images/','split_file':'split_zhou_DescribableTextures.json','split_names':['train','val','test']},
            'caltech-101':{'img_dir_name':'101_ObjectCategories/','split_file':'split_zhou_Caltech101.json','split_names':['train','val','test']},
            'oxford_flowers':{'img_dir_name':'jpg/','split_file':'split_zhou_OxfordFlowers.json','split_names':['train','val','test']},
            'oxford_pets':{'img_dir_name':'images/','split_file':'split_zhou_OxfordPets.json','split_names':['train','val','test']},
            'sun397':{'img_dir_name':'SUN397/','split_file':'split_zhou_SUN397.json','split_names':['train','val','test']},
            'ucf101':{'img_dir_name':'UCF-101-midframes/','split_file':'split_zhou_UCF101.json','split_names':['train','val','test']},
            'food101':{'img_dir_name':'images/','split_file':'split_zhou_Food101.json','split_names':['train','val','test']},
            'stanford_cars':{'img_dir_name':'','split_file':'split_zhou_StanfordCars.json','split_names':['train','val','test']},
            'oct':{'img_dir_name':'','split_file':'split_OCT.json','split_names':['train','val','test']},
            'fmow':{'img_dir_name':'','split_file':'split_FMOW.json','split_names':['train','val','test']},
            'kenya':{'img_dir_name':'','split_file':'split_KENYA.json','split_names':['train','val','test']},
            'cct20':{'img_dir_name':'','split_file':'split_CCT20.json','split_names':['train','val','test']},
            'serengeti':{'img_dir_name':'','split_file':'split_SERENGETI.json','split_names':['train','val','test']}, 
            'icct':{'img_dir_name':'','split_file':'split_ICCT.json','split_names':['train','val','test']}}


def main(args):
    datadir = '{}{}/'.format(args['root_data_dir'],args['dataset'])
    print('Preprocessing data for {}'.format(args['dataset']))
    dataset_df = pd.DataFrame()
    if args['dataset'] =='fgvc-aircraft':
        ## crop images given existing boxes
        crop_dir = os.path.join(datadir,'data/image_boxes/')
        os.makedirs(crop_dir,exist_ok=True)
        for split in ['train','val','test']:
            images =  list(np.genfromtxt(datadir+'data/images_{}.txt'.format(split),dtype='str'))
            temp_df = pd.DataFrame()
            temp_df['img_id'] = images
            temp_df['img_set'] = split
            dataset_df = pd.concat((dataset_df,temp_df))
            
        variant_dict = {}
        variant_dict.update({f.split(' ')[0]:f.split(' ')[1] for f in np.genfromtxt(datadir+'data/images_variant_train.txt',dtype='str',delimiter='\n')})
        variant_dict.update({f.split(' ')[0]:f.split(' ')[1] for f in np.genfromtxt(datadir+'data/images_variant_val.txt',dtype='str',delimiter='\n')})
        variant_dict.update({f.split(' ')[0]:f.split(' ')[1] for f in np.genfromtxt(datadir+'data/images_variant_test.txt',dtype='str',delimiter='\n')})

        colist = ['xmin','ymin','xmax','ymax']
        for i,col in enumerate(colist):
            img_to_col = {f.split(' ')[0]:f.split(' ')[i+1] for f in np.genfromtxt(datadir+'data/images_box.txt',dtype='str',delimiter='\n')}
            dataset_df[col] = dataset_df.img_id.apply(lambda x: int(img_to_col[x]) )

        dataset_df['variant'] = dataset_df.img_id.apply(lambda x: variant_dict[x])
        dataset_df['original_img_path'] = os.path.join(datadir,'data/images/')+dataset_df['img_id']+'.jpg'

        dataset_df['img_width'] = 0
        dataset_df['img_height'] = 0
        dataset_df['img_path'] = ''
        dataset_df.reset_index(inplace=True)
        print('Crops getting extracted for {} ...'.format(args['dataset']))
        for i,row in dataset_df.iterrows():
            img = Image.open(row.original_img_path)
            width, height = img.size

            dataset_df.loc[i,'img_width']=width
            dataset_df.loc[i,'img_height']=height
            left = row.xmin
            top = row.ymin
            right = row.xmax
            bottom = min(height-20,row.ymax)
            #crop img
            crop_img = img.crop((left, top, right, bottom))
            dst_path = crop_dir+row.img_id+'.jpg'
            dataset_df.loc[i,'img_path'] = 'data/image_boxes/'+row.img_id+'.jpg'
            crop_img.save(dst_path, 'JPEG')
        print('Processing meta dataframe ...')

        label = 'variant'
        dataset_df['label'] = dataset_df[label]

        label_to_category_id = dict(dataset_df['label'].drop_duplicates().reset_index().drop(columns=['index']).reset_index().sort_values('index')[['label','index']].values)
        dataset_df['category_id'] = dataset_df['label'].apply(lambda x: label_to_category_id[x])
        dataset_df['width'] = dataset_df['xmax']-dataset_df['xmin']
        dataset_df['height'] = dataset_df['ymax']-dataset_df['ymin']
        dataset_df.rename(columns={'xmin':'x1','ymin':'y1','xmax':'x2','ymax':'y2'},inplace=True)

            
    else:
        with open('{}{}'.format(datadir,data_dict[args['dataset']]['split_file'])) as f:
            split_dictionary = json.load(f)
        img_dir_name = data_dict[args['dataset']]['img_dir_name']
        print('Processing meta dataframe ...')
        for split in data_dict[args['dataset']]['split_names']: #['train','val','test']
            dataset_split = split_dictionary[split]
            for data_tuple in dataset_split:

                row = {'img_id':data_tuple[0].replace('.jpg',''),
                    'img_name':data_tuple[0],
                    'img_path':img_dir_name+data_tuple[0],
                    'category_id':data_tuple[1],
                    'label':data_tuple[2],
                    'img_set':split}
                dataset_df = dataset_df.append(row,ignore_index=True)

                
    meta_dir = os.path.join(datadir,'{}_meta.csv'.format(args['dataset']))
    dataset_df.to_csv(meta_dir)
    print('Data processing done, the metafile is saved in {}.'.format(meta_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_data_dir",type=str,default='data/') 
    parser.add_argument('--dataset', choices=['ucf101','food101','sun397','stanford_cars','fgvc-aircraft','caltech-101','eurosat','oxford_pets','oxford_flowers','dtd',
                                              'kenya','cct20','serengeti','icct','fmow','oct'], type=str)
    
    
    # turn the args into a dictionary
    args = vars(parser.parse_args())
    main(args)