from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint
import os, time, sys

# APIキー

key = "c20d739de80b8ddda202c79d32706b2f"
secret = "acba611c35532931"
wait_time = 0.2

#保存先フォルダの指定
animalname = sys.argv[1]
savedir = "./" + animalname

flickr = FlickrAPI(key, secret, format='parsed-json')
resurl = flickr.photos.search(
   text = animalname,
   per_page = 400,
   media = 'photos',
   sort = 'relevance',
   safe_search = 1,
   extras = 'url_q, licence'
)

photos = resurl['photos']
#返り値を取得する
#pprint(photos)

for i, photo in enumerate(photos['photo']):
    url_q = photo['url_q']
    filepath = savedir + '/' + photo['id'] + '.jpg'
    if os.path.exists(filepath): continue
    urlretrieve(url_q,filepath)
    time.sleep(wait_time)
