# VQGRaD
Visual Question Generation (VQG) from images is a rising research topic in both fields of natural language processing and computer vision. Although there are some recent efforts towards generating questions from images in the open domain, the VQG task in the medical domain has not been well-studied so far due to the lack of labeled data. In this paper, we introduce a goal-driven VQG approach for radiology images called VQGRaD that generates questions targeting specific image aspects such as modality and abnormality. In particular, we study generating natural language questions based on the visual content of the image and on additional information such as the image caption and the question category. VQGRaD encodes the dense vectors of different inputs into two latent spaces, which allows generating, for a specific question category, relevant questions about the images, with or without their captions. We also explore the impact of domain knowledge incorporation (e.g., medical entities and semantic types) and data augmentation techniques on visual question generation in the medical domain. Experiments performed on the VQA-RAD dataset of clinical visual questions showed that VQGRaD achieves 61.86% BLEU score and outperforms strong baselines. We also performed a blinded human evaluation of the grammaticality, fluency, and relevance of the generated questions. The human evaluation demonstrated the better quality of VQGRaD outputs and showed that incorporating medical entities improves the quality of the generated questions. Using the test data and evaluation process of the ImageCLEF 2020 VQA-Med challenge, we found that relying on the proposed data augmentation technique to generate new training samples by applying different kinds of transformations, can mitigate the lack of data, avoid overfitting, and bring a substantial improvement in medical VQG.

![VQGRaD model](https://www.google.com/search?q=VQGRaD&rlz=1C1GCEA_enUS961US961&oq=VQGRaD&aqs=chrome..69i57.7839j0j1&sourceid=chrome&{google:instantExtendedEnabledParameter}ie=UTF-8#imgrc=1TkksPJic0txoM)

## Requirements
gensim==3.0.0\
nltk==3.4.5\
numpy==1.12.1\
Pillow==6.2.0\
progressbar2==3.34.3\
h5py==2.8.0\
torch==0.4.0\
torchvision==0.2.0\
torchtext==0.2.3\
jupyter==1.0.0

install Python requirements:
```
pip install -r requirements.txt
```
## Downloads and Setup
Once you clone this repo, run the vocab.py, store_dataset.py, train_vqgrad.py and evaluate_vqgrad.py file to process the dataset, to train and evaluate the model.
```shell
$ python vocab.py
$ python store_dataset.py
$ python train_vqgrad.py
$ python evaluate_vqgrad.py
```

## Citation
If you are using this repository or a part of it, please cite our paper:

```
@Article{info12080334,
AUTHOR = {Sarrouti, Mourad and Ben Abacha, Asma and Demner-Fushman, Dina},
TITLE = {Goal-Driven Visual Question Generation from Radiology Images},
JOURNAL = {Information},
VOLUME = {12},
YEAR = {2021},
NUMBER = {8},
ARTICLE-NUMBER = {334},
URL = {https://www.mdpi.com/2078-2489/12/8/334},
ISSN = {2078-2489}
}




```

## Contact
For more information, please contact me on sarrouti.mourad[at]gmail.com.


