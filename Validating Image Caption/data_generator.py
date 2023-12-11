import os
import numpy as np
from keras.utils import Sequence
from keras.preprocessing.sequence import pad_sequences
from image_encoding_model import image_encoder
 
class DataGenerator(Sequence):
    def __init__(self, captions_df, images_path, images_txt_path, word_to_indices, max_length):
        self.captions_df = captions_df
        self.images_path = images_path
        self.images_filenames = self.read_file(images_txt_path)
        self.word_to_indices = word_to_indices
        self.max_sen_length = max_length

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, index):
        image_features = []
        caption_indices = []
        output_labels = []

        img_filename = self.images_filenames[index]
        row = self.captions_df[self.captions_df['image_file'] == img_filename].iloc[0]
        positive_captions = row['positive_captions']
        negative_captions = row['negative_captions']

        positive_captions_indices = self.get_indices_of_captions(positive_captions)
        negative_captions_indices = self.get_indices_of_captions(negative_captions)

        image_feature = image_encoder(os.path.join(self.images_path, img_filename))
        image_features = np.array([image_feature for i in range(10)])
        caption_indices = np.array(positive_captions_indices + negative_captions_indices)
        output_labels = np.array([1 for i in range(5)] + [0 for i in range(5)])

        return ([image_features, caption_indices], output_labels)
        

    def get_indices_of_captions(self, captions):
        caps_indices_padded = []
        for cap in captions:
            #print(cap)
            cap_indices = [self.word_to_indices[w] for w in cap.split(' ') if w in self.word_to_indices]
            cap_indices_padded = pad_sequences([cap_indices], maxlen=self.max_sen_length, padding='post')[0]
            caps_indices_padded.append(cap_indices_padded)
        return caps_indices_padded

    def read_file(self, image_file):
        file = open(image_file, 'r')
        lines = file.readlines()
        lines = [line.strip() for line in lines if line.strip()]
        return lines


