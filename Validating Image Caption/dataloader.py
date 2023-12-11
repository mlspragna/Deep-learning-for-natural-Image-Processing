import os
import pandas as pd
import re
import random
import ast
import string

def process_captions(captions_list):
	replace_punctuation = str.maketrans('', '', string.punctuation)
	return [caption.lower().translate(replace_punctuation) for caption in captions_list]

def load_captions_into_dictonary(token_file):
	image_to_captions = {}
	file = open(token_file, 'r')
	lines = file.readlines()
	for line in lines:
		line = line.strip()
		pattern = r'(#\d+\t*\s*)'
		image_file,_,caption = re.split(pattern, line)
		if image_file in image_to_captions:
			image_to_captions[image_file].append(caption)
		else:
			image_to_captions[image_file] = [caption]

	for image in image_to_captions:
		image_to_captions[image] = process_captions(image_to_captions[image])

	return image_to_captions

def generate_negative_samples(image_to_captions):
	dataset = {'image_file':[], 'positive_captions':[], 'negative_captions':[]}
	image_files = set(image_to_captions.keys())
	for image_file in image_to_captions:
		five_image_files = random.sample(image_files - {image_file}, 5)
		five_negative_captions = [random.sample(image_to_captions[img_file],1)[0] for img_file in five_image_files]
		dataset['image_file'].append(image_file)
		dataset['positive_captions'].append(image_to_captions[image_file])
		dataset['negative_captions'].append(five_negative_captions)
	dataset_df = pd.DataFrame(dataset)
	return dataset_df
