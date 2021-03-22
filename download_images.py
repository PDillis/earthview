import os
from typing import Union

import shutil
import operator
import json

import urllib
import urllib.request
import urllib.error

import cv2  			# pip install opencv-python

import click  			# pip install click
from tqdm import tqdm  	# pip install tqdm

from parser import get_latest_json_multi_thread


# ===========================================================================================


# Helper function to easily make a zip file from a directory of images. Adapted from Sean Behan's blog
# http://www.seanbehan.com/how-to-use-python-shutil-make_archive-to-zip-up-a-directory-recursively-including-the-root-folder/
def make_zip_file(
		parent_path_to_zip: Union[str, os.PathLike],
		folder_to_zip: Union[str, os.PathLike],
		zip_filename: str,
		path_to_save_zip: Union[str, os.PathLike]) -> None:
	"""
	Auxiliary function to make it easy to create a ZIP file with the contents of a directory.

	:param parent_path_to_zip: Parent path containing the folder we wish to ZIP
	:param folder_to_zip: Folder within the parent_path_to_zip that will be compressed
	:param zip_filename: Name of the ZIP file (without .zip extension)
	:param path_to_save_zip: Path where we will then move the ZIP file
	:return: None, the saved ZIP file at the directory path_to_save_zip
	"""
	zip_path = shutil.make_archive(
		base_name=zip_filename,
		format='zip',  # for reference, options available: 'zip', 'tar', 'gztar', 'bztar', 'xztar', if so desired
		root_dir=parent_path_to_zip,
		base_dir=folder_to_zip)
	# Make the save dir if it doesn't exist
	if not os.path.exists(path_to_save_zip):
		os.mkdir(path_to_save_zip)
	# Move the ZIP file
	res = shutil.move(zip_path, path_to_save_zip)
	print(f'ZIP file saved at "{res}"!')


def download_static_json(json_path: Union[str, os.PathLike] = os.getcwd()) -> None:
	"""
	Auxiliary function to download the static JSON file if so needed.

	:param json_path: Path where we will save the static JSON file (default will be the current directory)
	:return: Saved file at the desired directory
	"""
	static_url = 'https://raw.githubusercontent.com/PDillis/earthview/master/earthview.json'
	res = urllib.request.urlopen(static_url).read()
	# Get the data and image URLs
	data = json.loads(res)

	with open(os.path.join(json_path, 'earthview.json'), 'w') as f:
		json.dump(data, f, sort_keys=True, indent=4)

# ===========================================================================================


def get_img_urls_static_json() -> list:
	"""
	Auxiliary function to get the image URLs using the static JSON file that can be found in the original repository.
	The downside is that more images may become available, and this JSON file won't be updated as often.

	:return: List of image URLs.
	"""
	# Load the json and get the image urls (2069 images in total, as of mid-March 2021)
	static_url = 'https://raw.githubusercontent.com/PDillis/earthview/master/earthview.json'
	res = urllib.request.urlopen(static_url).read()
	# Get the data and image URLs
	data = json.loads(res)

	# We will get, for each dict in data, the image url and remove any duplicates using set
	# I apologize if this is too obscure, as there are other simpler ways to do this
	img_urls = list(set(map(operator.itemgetter('image'), data)))

	return img_urls


def get_img_urls_local(
		processes_per_cpu: int = 8,
		max_index: int = 20000,
		json_path: Union[str, os.PathLike] = os.getcwd()) -> list:
	"""
	Auxiliary function to get the image URLs that are stored in the local JSON file. If it doesn't exist, then we will
	use the static JSON file found in the "earthview" repository.

	:param processes_per_cpu: Number of processes to run in parallel
	:param max_index: Maximum image url to try; try higher number as time progresses
	:param json_path: Path to the JSON file (by default saved in the current directory).
	:return: List of image URLs
	"""
	# If JSON file doesn't exist, then generate it
	if not os.path.isfile(os.path.join(json_path, 'earthview.json')):
		print(f'Local JSON file at "{json_path}" does not exist, creating a new one...')
		try:
			# Save some time and get the static JSON file
			download_static_json(json_path=json_path)
		except urllib.error.HTTPError:  # Error 404, static JSON no longer exists
			get_latest_json_multi_thread(processes_per_cpu=processes_per_cpu, max_index=max_index, json_path=json_path)

	# Load the json and get the image urls
	with open(os.path.join(json_path, 'earthview.json')) as json_file:
		data = json.load(json_file)

	# We will get, for each dict in data, the image url and remove any duplicates using set
	img_urls = list(set(map(operator.itemgetter('image'), data)))

	return img_urls


# ===========================================================================================


@click.group()
def main():
	pass


# ===========================================================================================


def test_image(
		img_save_path: Union[str, os.PathLike],
		expected_height: int = 1200,
		expected_width: int = 1800,
		expected_channels: int = 3) -> None:
	"""
	Test the saved image to see if it was correctly saved (otherwise, it needs to be downloaded again).
	:param img_save_path: Path to the image (including the image file name and extension, '.jpg')
	:param expected_height: Expected height in pixels of the image; will be the same throughout: 1200 pixels
	:param expected_width: Expected width in pixels of the image; will be the same throughout: 1800 pixels
	:param expected_channels: Number of channels of the image; expected an RGB image, so 3
	:return: None, checks will be conducted
	"""
	assert cv2.haveImageReader(img_save_path), f'Image "{img_save_path}" was incorrectly saved!'

	img = cv2.imread(img_save_path)
	h, w, c = img.shape
	msg = f'Image "{img_save_path}" has unexpected dimensions: ({h}, {w}, {c})'
	assert (h, w, c) == (expected_height, expected_width, expected_channels), msg


def download_all(
		img_urls: list,
		save_path: Union[str, os.PathLike]) -> None:
	"""
	Auxiliary function to download all the images to the save_path.

	:param img_urls: list containing all the urls of the images
	:param save_path: Path to save the images at
	:return: None, the images will be saved at the desired path
	"""
	for img_url in tqdm(img_urls, desc='Downloading images...', unit='images'):
		img_name = os.path.basename(img_url)  # 'https://www.gstatic.com/prettyearth/assets/full/1003.jpg' -> '1003.jpg'
		img_save_path = os.path.join(save_path, img_name)
		# If image exists, open it, and if there's no error, skip
		if cv2.haveImageReader(img_save_path):
			continue

		# Get and download the image
		urllib.request.urlretrieve(img_url, img_save_path)
		# Check it was correctly saved and has the expected dimensions
		test_image(img_save_path)


@main.command(name='download-all')
@click.option('-sj', '--static-json', is_flag=True, help='', )
@click.option('-jp', '--json-path', type=click.Path(), help='', default=os.getcwd(), show_default=True)
@click.option('-sp', '--img-save-path', type=click.Path(), help='', default=os.path.join(os.getcwd(), 'images'), show_default=True)
@click.option('-z', '--make-zip', is_flag=True, help='', )
def download_images(
		static_json: bool,
		json_path: Union[str, os.PathLike],
		img_save_path: Union[str, os.PathLike],
		make_zip: bool):
	"""
	Download all the images into the local machine without grouping.

	Images will be saved at full resolution (1800x1200) in JPEG format in the path './images/all/full_resolution'
	as they won't be classified by country.

	:param static_json: Use the static JSON file found in the Github repository (in case the local one is corrupted or lost).
	:param json_path: Path to the local JSON file with the image URLs. Used only if not using the static JSON file.
	:param img_save_path: Root path where we will save our images at.
	:param make_zip: Make a ZIP file with all the images; to be saved at './images/zip_files'
	:return: Images will be saved at the specified path in full resolution.
	"""
	# Set the final save path for the images
	save_path = os.path.join(img_save_path, 'all', 'full_resolution')
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	# Get the image urls
	if static_json:
		print('Using static JSON file...')
		try:
			img_urls = get_img_urls_static_json()
		except urllib.error.HTTPError:  # Error 404, static JSON no longer exists
			print('Static JSON not found! Using local JSON instead...')
			img_urls = get_img_urls_local(json_path=json_path)
	else:
		print('Using local JSON...')
		img_urls = get_img_urls_local(json_path=json_path)
	# Download all images (keeping their original names)
	download_all(img_urls, save_path)

	# If user wishes to zip the images (easier to move around this way)
	if make_zip:
		print(f'Making ZIP file...')
		make_zip_file(
			parent_path_to_zip=os.path.join(img_save_path, 'all'),
			folder_to_zip='full_resolution',
			zip_filename='all_imgs_full_resolution',
			path_to_save_zip=os.path.join(os.getcwd(), 'images', 'zip_files'))


# ===========================================================================================


def get_img_urls_by_country_static() -> list:
	"""
	Auxiliary function to get the image URLs and respective countries using the static JSON file that can be found in
	the original repository. The downside is that more images may become available, and this JSON file won't be updated
	as often.

	:return: List of tuples of image URLs and respective countries
	"""
	# Load the json and get the image urls (2069 images in total, as of mid-March 2021)
	static_url = 'https://raw.githubusercontent.com/PDillis/earthview/master/earthview.json'
	res = urllib.request.urlopen(static_url).read()
	data = json.loads(res)

	# Get a tuple of (country, img_url) for each dict in data, and remove any duplicates using set
	imgs_by_country = list(set(map(operator.itemgetter('image', 'country'), data)))

	return imgs_by_country


def get_img_urls_by_country_local(
		processes_per_cpu: int = 8,
		max_index: int = 20000,
		json_path: Union[str, os.PathLike] = os.getcwd()) -> list:
	"""
	Auxiliary function to get the image URLs that are stored in the local JSON file. If it doesn't exist, then we will
	use the static JSON file found in the "earthview" repository.

	:param processes_per_cpu: Number of processes to run in parallel
	:param max_index: Maximum image url to try; try higher number as time progresses
	:param json_path: Path to the JSON file (by default saved in the current directory).
	:return: List of image URLs
	"""
	# If JSON file doesn't exist, then generate it
	if not os.path.isfile(os.path.join(json_path, 'earthview.json')):
		print(f'Local JSON file at "{json_path}" does not exist, creating a new one...')
		try:
			# Save some time and get the static JSON file
			download_static_json(json_path=json_path)
		except urllib.error.HTTPError:  # Error 404, static JSON no longer exists
			get_latest_json_multi_thread(processes_per_cpu=processes_per_cpu, max_index=max_index, json_path=json_path)

	# Load the json and get the image urls
	with open(os.path.join(json_path, 'earthview.json')) as json_file:
		data = json.load(json_file)
	# We will get, for each dict in data, the image url and remove any duplicates using set
	imgs_by_country = list(set(map(operator.itemgetter('image', 'country'), data)))

	return imgs_by_country


def download_by_country(
		imgs_by_country: list,
		save_path: Union[str, os.PathLike]) -> None:
	"""
	Auxiliary function to download all the images to the save_path.

	:param imgs_by_country: list of tuples containing all the urls of the images and their respective country
	:param save_path: Path to save the images at
	:return: None, the images will be saved at the desired path
	"""
	for img_url, country in tqdm(imgs_by_country, desc='Downloading images...', unit='images'):
		# If the image doesn't belong to any country, rename to "None"
		if country == '':
			country = 'None'
		# Make the country dir if it doesn't exist
		if not os.path.isdir(os.path.join(save_path, country)):
			os.makedirs(os.path.join(save_path, country))

		img_name = os.path.basename(img_url)  # 'https://www.gstatic.com/prettyearth/assets/full/1003.jpg' -> '1003.jpg'
		img_country_save_path = os.path.join(save_path, country, img_name)
		# If image exists, open it, and if there's no error, skip
		if cv2.haveImageReader(img_country_save_path):
			continue

		# On the other hand, if it's already been downloaded in the './images/all/full_resolution' directory, copy it
		all_imgs_path = os.path.join('images', 'all', 'full_resolution', img_name)
		if cv2.haveImageReader(all_imgs_path):
			res = shutil.copyfile(src=all_imgs_path, dst=img_country_save_path)
			continue

		# Get and download the image
		urllib.request.urlretrieve(img_url, img_country_save_path)
		# Check it was correctly saved and has the expected dimensions
		test_image(img_country_save_path)


@main.command(name='download-by-country')
@click.option('-sj', '--static-json', is_flag=True, help='', default=False, show_default=True)
@click.option('-jp', '--json-path', type=click.Path(), help='', default=os.getcwd(), show_default=True)
@click.option('-sp', '--img-save-path', type=click.Path(), help='', default=os.path.join(os.getcwd(), 'images'), show_default=True)
@click.option('-z', '--make-zip', is_flag=True, help='', )
def download_images_by_country(
		static_json: bool,
		json_path: Union[str, os.PathLike],
		img_save_path: Union[str, os.PathLike],
		make_zip: bool):
	"""
	Download all the images into the local machine by grouping them by country.

	Images will be saved at full resolution (1800x1200) in JPEG format in the path './images/countries/full_resolution'.

	:param static_json: Use the static JSON file found in the Github repository (in case the local one is corrupted or lost).
	:param json_path: Path to the local JSON file with the image URLs. Used only if not using the static JSON file.
	:param img_save_path: Root path where we will save our images at.
	:param make_zip:
	:return: Images will be saved at the specified path in full resolution (classified by country).
	"""
	# Set the final save path for the images
	save_path = os.path.join(img_save_path, 'countries', 'full_resolution')

	if not os.path.exists(save_path):
		os.makedirs(save_path)

	# Get the image URLs and respective countries
	if static_json:
		print('Using static JSON file...')
		try:
			imgs_by_country = get_img_urls_by_country_static()
		except urllib.error.HTTPError:  # Error 404, static JSON no longer exists
			print('Static JSON not found! Using local JSON instead...')
			imgs_by_country = get_img_urls_by_country_local(json_path=json_path)
	else:
		print('Using local JSON...')
		imgs_by_country = get_img_urls_by_country_local(json_path=json_path)

	# Download all images in respective dirs by country (keeping their original names)
	download_by_country(imgs_by_country, save_path)

	# If user wishes to zip the images (easier to move around this way)
	if make_zip:
		print(f'Making ZIP file...')
		make_zip_file(
			parent_path_to_zip=os.path.join(img_save_path, 'countries'),
			folder_to_zip='full_resolution',
			zip_filename='imgs_by_country_full_resolution',
			path_to_save_zip=os.path.join(os.getcwd(), 'images', 'zip_files'))


# ===========================================================================================


if __name__ == '__main__':
	main()


# ===========================================================================================
