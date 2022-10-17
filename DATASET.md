# How to install datasets

To start with, you can download the following datasets and store them under `$DATA` directory. In our case we use `data/` as the default and if you want to use a different path, just make sure you define it in the arguments of the python scripts that store or load from there.


The file structure looks like:

```
data/
|–– caltech-101/
|–– eurosat/
|–– cct20/
```

Datasets list:
- [Caltech101](#caltech101)
- [OxfordPets](#oxfordpets)
- [StanfordCars](#stanfordcars)
- [Flowers102](#flowers102)
- [Food101](#food101)
- [FGVCAircraft](#fgvcaircraft)
- [SUN397](#sun397)
- [DTD](#dtd)
- [EuroSAT](#eurosat)
- [UCF101](#ucf101)
- [FMoW](#fmow)
- [OCT](#oct)
- [CCT20](#cct20)
- [ICCT](#icct)
- [Serengeti](#serengeti)
- [MMCT](#mmct)


The instructions to prepare each dataset are detailed below. To ensure reproducibility and fair comparison for future work, we provide fixed train/val/test splits. The fixed splits are either from the original datasets (if available) or created by the authors of CoOp and ourselves (last 6 datasets in the list).

## Standard Datasets

### Caltech101
- Create a folder named `caltech-101/` under `$DATA`.
- Download `101_ObjectCategories.tar.gz` from http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz and extract the file under `$DATA/caltech-101`.
- Download `split_zhou_Caltech101.json` from this [link](https://drive.google.com/file/d/1hyarUivQE36mY6jSomru6Fjd-JzwcCzN/view?usp=sharing) and put it under `$DATA/caltech-101`. 

The directory structure should look like
```
caltech-101/
|–– 101_ObjectCategories/
|–– split_zhou_Caltech101.json
```

### OxfordPets
- Create a folder named `oxford_pets/` under `$DATA`.
- Download the images from https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz.
- Download the annotations from https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz.
- Download `split_zhou_OxfordPets.json` from this [link](https://drive.google.com/file/d/1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs/view?usp=sharing). 

The directory structure should look like
```
oxford_pets/
|–– images/
|–– annotations/
|–– split_zhou_OxfordPets.json
```

### StanfordCars
- Create a folder named `stanford_cars/` under `$DATA`.
- Download the train images http://ai.stanford.edu/~jkrause/car196/cars_train.tgz.
- Download the test images http://ai.stanford.edu/~jkrause/car196/cars_test.tgz.
- Download the train labels https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz.
- Download the test labels http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat.
- Download `split_zhou_StanfordCars.json` from this [link](https://drive.google.com/file/d/1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT/view?usp=sharing).

The directory structure should look like
```
stanford_cars/
|–– cars_test\
|–– cars_test_annos_withlabels.mat
|–– cars_train\
|–– devkit\
|–– split_zhou_StanfordCars.json
```

### Flowers102
- Create a folder named `oxford_flowers/` under `$DATA`.
- Download the images and labels from https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz and https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat respectively.
- Download `cat_to_name.json` from [here](https://drive.google.com/file/d/1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0/view?usp=sharing). 
- Download `split_zhou_OxfordFlowers.json` from [here](https://drive.google.com/file/d/1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT/view?usp=sharing).

The directory structure should look like
```
oxford_flowers/
|–– cat_to_name.json
|–– imagelabels.mat
|–– jpg/
|–– split_zhou_OxfordFlowers.json
```

### Food101
- Download the dataset from https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/ and extract the file `food-101.tar.gz` under `$DATA`, resulting in a folder named `$DATA/food-101/`.
- Download `split_zhou_Food101.json` from [here](https://drive.google.com/file/d/1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl/view?usp=sharing).

The directory structure should look like
```
food-101/
|–– images/
|–– license_agreement.txt
|–– meta/
|–– README.txt
|–– split_zhou_Food101.json
```

### FGVCAircraft
- Download the data from https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz.
- Extract `fgvc-aircraft-2013b.tar.gz` and keep only `data/`.
- Move `data/` to `$DATA` and rename the folder to `fgvc_aircraft/`.

The directory structure should look like
```
fgvc_aircraft/
|–– images/
|–– ... # a bunch of .txt files
```

### SUN397
- Create a folder named  `sun397/` under `$DATA`.
- Download the images http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz.
- Download the partitions https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip.
- Extract these files under `$DATA/sun397/`.
- Download `split_zhou_SUN397.json` from this [link](https://drive.google.com/file/d/1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq/view?usp=sharing).

The directory structure should look like
```
sun397/
|–– SUN397/
|–– split_zhou_SUN397.json
|–– ... # a bunch of .txt files
```

### DTD
- Download the dataset from https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz and extract it to `$DATA`. This should lead to `$DATA/dtd/`.
- Download `split_zhou_DescribableTextures.json` from this [link](https://drive.google.com/file/d/1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x/view?usp=sharing).

The directory structure should look like
```
dtd/
|–– images/
|–– imdb/
|–– labels/
|–– split_zhou_DescribableTextures.json
```

### EuroSAT
- Create a folder named `eurosat/` under `$DATA`.
- Download the dataset from http://madm.dfki.de/files/sentinel/EuroSAT.zip and extract it to `$DATA/eurosat/`.
- Download `split_zhou_EuroSAT.json` from [here](https://drive.google.com/file/d/1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o/view?usp=sharing).

The directory structure should look like
```
eurosat/
|–– 2750/
|–– split_zhou_EuroSAT.json
```

### UCF101
- Create a folder named `ucf101/` under `$DATA`.
- Download the zip file `UCF-101-midframes.zip` from [here](https://drive.google.com/file/d/10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O/view?usp=sharing) and extract it to `$DATA/ucf101/`. This zip file contains the extracted middle video frames.
- Download `split_zhou_UCF101.json` from this [link](https://drive.google.com/file/d/1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y/view?usp=sharing).

The directory structure should look like
```
ucf101/
|–– UCF-101-midframes/
|–– split_zhou_UCF101.json
```

<br>

---

<br>


## Challenging Datasets

The following datasets make up the challenging datasets used in the [SVL-Adapter paper](https://arxiv.org/abs/2210.03794). We provide the splits we used in a .json file for the shake of benchmarking and comparison. When available, the train/test splits follow the ones provided by the original curators of each dataset.

### FMoW
The Functional Map of the World (FMoW) dataset presented in [this paper](https://arxiv.org/abs/1711.07846) contains thousands of satellite images which are labeled based on the functional purpose of the building or land they contain. We use the fMoW-rgb version of the dataset and keep a subset of the data (defined in split_FMOW.json) for efficiency.

- Create a folder named `fmow/` under `$DATA`.
- Download the images along with bounding box and annotations https://github.com/fMoW/dataset for both train/val and test subsets.
- Extract these files under `$DATA/fmow/`.
- Download `split_FMOW.json` from this [link](https://drive.google.com/file/d/1ogM545px5VT2RVw9s9wC5wB2wupKPeZb/view?usp=sharing).


The directory structure should look like:
```
fmow/
|–– train/
|–– test/
|–– split_FMOW.json
```

### OCT
This dataset contains thousands of validated Optical Coherence Tomography (OCT) described and analyzed in [this paper](https://www.sciencedirect.com/science/article/pii/S0092867418301545?via%3Dihub). The images are split into a training set and a testing set of independent patients with images having labels from 1 of the 4 following categories: CNV, DME, DRUSEN, and NORMAL.


- Create a folder named `oct/` under `$DATA`.
- Download the images from https://data.mendeley.com/datasets/rscbjbr9sj/3 (the image labels are indicated by the name of the folder they are in).
- Extract these files under `$DATA/oct/`.
- Download `split_OCT.json` from this [link](https://drive.google.com/file/d/1QztEKqFzDP4EpcthSwMZLUyuoMIZ3ujS/view?usp=sharing).

The directory structure should look like:
```
oct/
|–– train/
|–– test/
|–– split_OCT.json
```

## Camera Trap datasets
A large repository of camera trap data can be found at [lila.science](http://lila.science/), including [Caltech Camera Traps (CCT20)](https://beerys.github.io/CaltechCameraTraps/), [Island Conservation Camera Traps (ICCT)](https://lila.science/datasets/island-conservation-camera-traps/) and [Snapshot Serengeti](https://lila.science/datasets/snapshot-serengeti) datasets which were used for to evaluate SVL-Adapter across challenging task. For each camera trap dataset examined, we extract the bounding boxes around the object of interest given their availability.  Note: If bounding box annotations are not available for a camera trap dataset, regions around animals can be extracted pretty accurately by utilizing the [MegaDetector].

### CCT20
- Create a folder named `cct20/` under `$DATA`.
- Download the images along with bounding box and annotations https://beerys.github.io/CaltechCameraTraps/.
- Extract these files under `$DATA/cct20/`.
- Download `split_CCT20.json` from this [link](https://drive.google.com/file/d/1Rt-vn_MfQEQ4wyhvNcCiiIVmGNHIxTCt/view?usp=sharing).

The directory structure should look like:
```
cct20/
|–– train_images/
|–– cis_val_images/
|–– cis_test_images/
|–– trans_val_images/
|–– trans_test_images/
|–– split_CCT20.json
```

### ICCT
- Create a folder named `icct/` under `$DATA`.
- Download the images along with bounding box and annotations https://lila.science/datasets/island-conservation-camera-traps/.
- Extract these files under `$DATA/icct/`.
- Download `split_ICCT.json` from this [link](https://drive.google.com/file/d/1wnw_jQF9y6W-58WUfWO1ZscsFe-hPsRx/view?usp=sharing).

The directory structure should look like:
```
icct/
|–– train_images/
|–– cis_val_images/
|–– cis_test_images/
|–– trans_val_images/
|–– trans_test_images/
|–– split_ICCT.json
```

### Serengeti
Note: We use a subset of this dataset (defined in split_SERENGETI.json).

- Create a folder named `serengeti/` under `$DATA`.
- Download the images that have bounding box information and annotations available https://lila.science/datasets/snapshot-serengeti.
- Extract these files under `$DATA/serengeti/`.
- Download `split_SERENGETI.json` from this [link](https://drive.google.com/file/d/12NpA-GVT5HCHwOvC7iFan2aBc0OV9WTo/view?usp=sharing).

The directory structure should look like:
```
serengeti/
|–– train/
|–– test/
|–– split_SERENGETI.json
|–– 
```

### MMCT
- This dataset is collected in the Maasai Mara region in Kenya for the [Biome Health Project](https://www.biomehealthproject.com/) which is funded by WWF UK. The dataset not public yet. We will add a link to the data and the splits used as soon as it becomes available.
