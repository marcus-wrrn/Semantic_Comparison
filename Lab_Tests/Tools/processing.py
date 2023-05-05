import pywt
from sentence_transformers import SentenceTransformer, util
import numpy as np
import csv
import time
from os.path import isfile
import pandas as pd
import re
import spacy
import matplotlib.pyplot as plt

model = SentenceTransformer('all-mpnet-base-v2')

def plot(tensor, tensor2=None, title='Similarity scores', secondTensor=False):
  x = np.arange(len(tensor))
  y = tensor
  plt.plot(x,y)
  if secondTensor:
    plt.plot(x, tensor2)
  plt.xlabel('Sample Number')
  plt.ylabel('Similarity')
  plt.title(title)
  plt.show()


# Create sentance embeddings
def create_embeddings(model : SentenceTransformer, text : list[str]):
    # Create and save the encodings to the specified file
    print("Hello")
    return np.array(model.encode(text, normalize_embeddings=True))


def find_similarity(model, embeddings, text):
  question_embedding = model.encode(text)
  print(len(question_embedding))
  print(len(embeddings))
  return util.dot_score(embeddings, question_embedding)

def generate_all_text_emeddings(model, text: list[str]):
   return [model.encode(val) for val in text]


nlp = spacy.load('en_core_web_sm')
def split_sentances(text: str):
  #text = text.replace('\n', '').replace('?', '').replace('.', '')

  # Split the string into sentences using regular expression
  return [sent.text for sent in nlp(text).sents]


def denoize(arr):
    coeffs = pywt.wavedec(arr, 'db4', level=6)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(arr)))
    coeffs = [pywt.threshold(c, threshold, 'soft') for c in coeffs]
    return pywt.waverec(coeffs, 'db4')


def plot_denoized(arr, title=None):
    x_denoised = denoize(np.abs(arr))
    if title != None:
        plot(x_denoised, title=title)
    else:
        plot(x_denoised)


def denoise(data, wavelet='db4', level=None, threshold_type='soft'):
    """
    Denoise a 1D wave using wavelet denoising.

    Parameters:
        data (array-like): Input data (1D wave) to be denoised.
        wavelet (str): Wavelet type, default is 'db4' (Daubechies wavelet).
        level (int): Decomposition level. If None, the maximum level will be used.
        threshold_type (str): Threshold type, either 'soft' or 'hard'. Default is 'soft'.

    Returns:
        array-like: Denoised data.
    """

    # Convert input data to a NumPy array
    data = np.asarray(data)

    # Calculate the maximum decomposition level if not specified
    if level is None:
        level = pywt.dwt_max_level(data_len=len(data), filter_len=pywt.Wavelet(wavelet).dec_len)

    # Perform wavelet decomposition
    coeffs = pywt.wavedec(data, wavelet, level=level)

    # Calculate the universal threshold
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(data)))

    # Apply thresholding
    if threshold_type == 'soft':
        coeffs[1:] = [pywt.threshold(c, value=threshold, mode='soft') for c in coeffs[1:]]
    elif threshold_type == 'hard':
        coeffs[1:] = [pywt.threshold(c, value=threshold, mode='hard') for c in coeffs[1:]]
    else:
        raise ValueError("Invalid threshold type. Choose either 'soft' or 'hard'.")

    # Perform wavelet reconstruction
    denoised_data = pywt.waverec(coeffs, wavelet)

    return denoised_data

def estimate_completion(iteration, total_iterations, start_time):
    elapsed_time = time.time() - start_time
    time_per_iteration = elapsed_time / iteration
    remaining_iterations = total_iterations - iteration
    estimated_completion_time = remaining_iterations * time_per_iteration
    return estimated_completion_time
