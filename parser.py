import os
from typing import Union

import click			# pip install click
from tqdm import tqdm	# pip install tqdm

import urllib
import urllib.request
import urllib.error

import json
from bs4 import BeautifulSoup

from multiprocessing import Pool

# ===========================================================================================


def get_single_data(url: str) -> Union[dict, None]:
	"""
	Get the metadata of a single url.

	:param url: URL of Earth View image
	:return: dictionary containing region, country, google maps and image url
	"""
	num = os.path.basename(url)  # https://earthview.withgoogle.com/1003 -> 1003
	try:
		response = urllib.request.urlopen(url)
		html = BeautifulSoup(response.read(), features="html.parser")
		# We will only save region, country, Google maps url, and image_url per url
		region = html.find("div", class_="location__region").text
		country = html.find("div", class_="location__country").text
		everything = html.find("a", href=True)
		gmaps_url = everything['href']
		image = f'https://www.gstatic.com/prettyearth/assets/full/{num}.jpg'
		return {'region': region, 'country': country, 'map': gmaps_url, 'image': image}
	except urllib.error.HTTPError:  # Error 404: Not found -> skip
		return None


@click.command()
@click.option('-p', '--processes-per-cpu', type=click.INT, help='Number of processes to run in parallel per cpu', default=8, show_default=True)
@click.option('-idx', '--max-index', type=click.INT, help='Max url index to try (increase as time progresses)', default=20000, show_default=True)
@click.option('-pth', '--save-path', type=click.Path(), help='Path to save the JSON file', default=os.getcwd(), show_default=True)
def get_latest_json_multi_thread(
		processes_per_cpu: int,
		max_index: int, save_path:
		Union[str, os.PathLike]):
	"""
	Get the latest JSON file by going through all the urls of the images found in Earth View.
	
	:param processes_per_cpu: Number of processes per cpu to run in parallel (8 works good enough for me)
	:param max_index: Maximum image url to try; try higher number as time progresses, though current highest is 14793
	:param save_path: Path where the JSON file will be saved at (current directory is the default)
	:return: (None) JSON file will be saved at the save_path
	"""
	urls = [f'https://earthview.withgoogle.com/{x}' for x in range(max_index)]

	processes = os.cpu_count() * processes_per_cpu
	pool = Pool(processes=processes)
	# Thanks to Elmar Pruesse on which map to use with tqdm (otherwise it's silent and you question your sanity):
	# https://github.com/tqdm/tqdm/issues/484#issuecomment-461998250
	with pool as p:
		results = list(tqdm(p.imap(func=get_single_data, iterable=urls),
							desc='Fetching...', total=len(urls), unit='urls'))
	# Remove None entries (404 error)
	results = list(filter(None, results))
	print(f'Found {len(results)} images!')
	print(f'Saving the JSON file at "{save_path}"...')
	if not os.path.isdir(save_path):
		os.makedirs(save_path)
	with open(os.path.join(save_path, 'earthview.json'), 'a') as f:
		json.dump(results, f, sort_keys=True, indent=4)


# ===========================================================================================


if __name__ == '__main__':
	get_latest_json_multi_thread()


# ===========================================================================================
