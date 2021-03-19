# VQGR
VQGR, a visual question generation about radiology images, is based on the variational auto-encoders architecture and designed so that it can take a radiology image as input and generate a natural question as output.

![VQGR model](https://github.com/sarrouti/VQGR/blob/master/vqgri.jpg)

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
$ python train_vqa.py
$ python evaluate_vqa.py
```

## Citation
If you are using this repository or a part of it, please cite our paper:

```
@inproceedings{Sarrouti2020,
    title = "Visual Question Generation from Radiology Images",
    author = "Sarrouti, Mourad  and Ben Abacha, Asma  and Demner-Fushman, Dina",
    booktitle = "Proceedings of the First Workshop on Advances in Language and Vision Research",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.alvr-1.3",
    pages = "12--18"
}
```

## Contact
For more information, please contact me on sarrouti.mourad[at]gmail.com.


