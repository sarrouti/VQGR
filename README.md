# VQGR
VQGR, a visual question generation about radiology images, is based on the variational auto-encoders architecture and designed so that it can take a radiology image to generate a related question.

![VQGR model](https://www.google.com/search?q=VQGRaD&rlz=1C1GCEA_enUS961US961&oq=VQGRaD&aqs=chrome..69i57.7839j0j1&sourceid=chrome&{google:instantExtendedEnabledParameter}ie=UTF-8#imgrc=1TkksPJic0txoM)

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
Once you clone this repo, run the vocab.py, store_dataset.py, train.py and evaluate.py file to process the dataset, to train and evaluate the model.
```shell
$ python vocab.py
$ python store_dataset.py
$ python train.py
$ python evaluate.py
```

## Citation
If you are using this repository or a part of it, please cite our paper:

```
@inproceedings{sarrouti-etal-2020-visual,
    title = "Visual Question Generation from Radiology Images",
    author = "Sarrouti, Mourad  and
      Ben Abacha, Asma  and
      Demner-Fushman, Dina",
    booktitle = "Proceedings of the First Workshop on Advances in Language and Vision Research",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.alvr-1.3",
    doi = "10.18653/v1/2020.alvr-1.3",
    pages = "12--18",
    abstract = "Visual Question Generation (VQG), the task of generating a question based on image contents, is an increasingly important area that combines natural language processing and computer vision. Although there are some recent works that have attempted to generate questions from images in the open domain, the task of VQG in the medical domain has not been explored so far. In this paper, we introduce an approach to generation of visual questions about radiology images called VQGR, i.e. an algorithm that is able to ask a question when shown an image. VQGR first generates new training data from the existing examples, based on contextual word embeddings and image augmentation techniques. It then uses the variational auto-encoders model to encode images into a latent space and decode natural language questions. Experimental automatic evaluations performed on the VQA-RAD dataset of clinical visual questions show that VQGR achieves good performances compared with the baseline system. The source code is available at https://github.com/sarrouti/vqgr.",
}





```

## Contact
For more information, please contact me on sarrouti.mourad[at]gmail.com.


