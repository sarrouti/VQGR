"""
Created on Tue Jun 23 20:15:11 2020

@author: sarroutim2
"""

"""Transform all the IQ VQA dataset into a hdf5 dataset.
"""

from PIL import Image
from torchvision import transforms

import argparse
import json
import h5py
import numpy as np
import os
import progressbar

from train_utils import Vocabulary
from vocab import load_vocab
from vocab import process_text


def create_questions_images_ids(questions_captions):
    """


    Returns:
        questions: set of question ids.
        image_ids: Set of image ids.
    """
    questions = set()
    image_ids = set()
    for q in questions_captions:
        question_id = q['question_id']
        questions.add(question_id)
        image_ids.add(q['image_id'])

    return questions, image_ids


def save_dataset(image_dir, questions, vocab, output,
                 im_size=224, max_q_length=20,
                 with_answers=False):
    """Saves the Visual RAD images and the questions in a hdf5 file.

    Args:
        image_dir: Directory with all the images.
        questions: Location of the questions and images.
        vocab: Location of the vocab file.
        output: Location of the hdf5 file to save to.
        im_size: Size of image.
        max_q_length: Maximum length of the questions.
    """
    # Load the data.
    vocab = load_vocab(vocab)

    with open(questions) as f:
        questions = json.load(f)

    # Get question ids and image ids.
    qids, image_ids = create_questions_images_ids(questions)
    total_questions = len(qids)
    total_images = len(image_ids)
    print ("Number of images to be written: %d" % total_images)
    print ("Number of QAs to be written: %d" % total_questions)

    h5file = h5py.File(output, "w")
    d_questions = h5file.create_dataset(
        "questions", (total_questions, max_q_length), dtype='i')
    d_indices = h5file.create_dataset(
        "image_indices", (total_questions,), dtype='i')
    d_images = h5file.create_dataset(
        "images", (total_images, im_size, im_size, 3), dtype='f')


    # Create the transforms we want to apply to every image.
    transform = transforms.Compose([
        transforms.Resize((im_size, im_size))])

    # Iterate and save all the questions and images.
    bar = progressbar.ProgressBar(maxval=total_questions)
    i_index = 0
    q_index = 0
    done_img2idx = {}
    for entry in questions:
        image_id = entry['image_id']
        question_id = entry['question_id']
        if image_id not in image_ids:
            continue
        if question_id not in qids:
            continue
        if image_id not in done_img2idx:
            try:
                path = image_id
                image = Image.open(os.path.join(image_dir, path+".jpg")).convert('RGB')
            except IOError:
                path = image_id
                image = Image.open(os.path.join(image_dir, path)).convert('RGB')
            image = transform(image)
            d_images[i_index, :, :, :] = np.array(image)
            done_img2idx[image_id] = i_index
            i_index += 1
        q, length = process_text(entry['question'], vocab,
                                 max_length=max_q_length)
        d_questions[q_index, :length] = q

        d_indices[q_index] = done_img2idx[image_id]
        q_index += 1
        bar.update(q_index)
    h5file.close()
    print ("Number of images written: %d" % i_index)
    print ("Number of QAs written: %d" % q_index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Inputs.
    parser.add_argument('--image-dir', type=str, default='/home/sarroutim2/PosDoc NLM/VQA/Codes/IQ/iq-master-med-mine/data/vqa/train_images_VQ{RAD+Med}',
                        help='directory for resized images')
    parser.add_argument('--questions', type=str,
                        default='data/vqg/'
                        'train_questions_RAD_captions__titles_v1.json',
                        help='Path for train annotation file.')
    parser.add_argument('--vocab-path', type=str,
                        default='data/processed/vocab_vqg.json',
                        help='Path for saving vocabulary wrapper.')

    # Outputs.
    parser.add_argument('--output', type=str,
                        default='data/processed/vqg_dataset.hdf5',
                        help='directory for resized images.')
    # Hyperparameters.
    parser.add_argument('--im_size', type=int, default=224,
                        help='Size of images.')
    parser.add_argument('--max-q-length', type=int, default=20,
                        help='maximum sequence length for questions.')
    args = parser.parse_args()

    save_dataset(args.image_dir, args.questions, args.vocab_path,
                 args.output, im_size=args.im_size,
                 max_q_length=args.max_q_length)
    print('Wrote dataset to %s' % args.output)
    # Hack to avoid import errors.
    Vocabulary()
