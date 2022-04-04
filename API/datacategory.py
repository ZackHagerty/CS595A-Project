import i2v
from PIL import Image
import os
import shutil

illust2vec = i2v.make_i2v_with_chainer("illust2vec_tag_ver200.caffemodel", "tag_list.json")

for images in os.listdir("./archive/animefaces256cleaner"):

    image = Image.open("./archive/animefaces256cleaner/" + images)

    tags = illust2vec.estimate_specific_tags([image], ["aqua hair", "black eyes", "black hair",
    "blonde hair", "blue eyes", "blue hair", "brown eyes", "brown hair", "drill hair", "glasses",
    "grey hair", "green eyes", "green hair", "hat", "long hair", "open mouth", "orange hair",
    "pink hair", "ponytail", "purple eyes", "purple hair", "red eyes", "red hair", "ribbon",
    "short hair", "silver hair", "smile", "twin drills", "white hair", "yellow eyes"])

    tags = sorted(tags[0].keys(), key=tags[0].get, reverse = True)


    if tags[0] == 'yellow eyes':
        shutil.move("./archive/animefaces256cleaner/" + images, "./archive/animefaces256cleaner/yellow eyes")

    if tags[0] == 'white hair':
        shutil.move("./archive/animefaces256cleaner/" + images, "./archive/animefaces256cleaner/white hair")

    if tags[0] == 'twin drills':
        shutil.move("./archive/animefaces256cleaner/" + images, "./archive/animefaces256cleaner/twin drills")

    if tags[0] == 'smile':
        shutil.move("./archive/animefaces256cleaner/" + images, "./archive/animefaces256cleaner/smile")

    if tags[0] == 'silver hair':
        shutil.move("./archive/animefaces256cleaner/" + images, "./archive/animefaces256cleaner/silver hair")

    if tags[0] == 'short hair':
        shutil.move("./archive/animefaces256cleaner/" + images, "./archive/animefaces256cleaner/short hair")

    if tags[0] == 'ribbon':
        shutil.move("./archive/animefaces256cleaner/" + images, "./archive/animefaces256cleaner/ribbon")

    if tags[0] == 'red hair':
        shutil.move("./archive/animefaces256cleaner/" + images, "./archive/animefaces256cleaner/red hair")

    if tags[0] == 'red eyes':
        shutil.move("./archive/animefaces256cleaner/" + images, "./archive/animefaces256cleaner/red eyes")

    if tags[0] == 'purple hair':
        shutil.move("./archive/animefaces256cleaner/" + images, "./archive/animefaces256cleaner/purple hair")

    if tags[0] == 'purple eyes':
        shutil.move("./archive/animefaces256cleaner/" + images, "./archive/animefaces256cleaner/purple eyes")

    if tags[0] == 'ponytail':
        shutil.move("./archive/animefaces256cleaner/" + images, "./archive/animefaces256cleaner/ponytail")

    if tags[0] == 'pink hair':
        shutil.move("./archive/animefaces256cleaner/" + images, "./archive/animefaces256cleaner/pink hair")

    if tags[0] == 'orange hair':
        shutil.move("./archive/animefaces256cleaner/" + images, "./archive/animefaces256cleaner/orange hair")

    if tags[0] == 'open mouth':
        shutil.move("./archive/animefaces256cleaner/" + images, "./archive/animefaces256cleaner/open mouth")

    if tags[0] == 'long hair':
        shutil.move("./archive/animefaces256cleaner/" + images, "./archive/animefaces256cleaner/long hair")

    if tags[0] == 'hat':
        shutil.move("./archive/animefaces256cleaner/" + images, "./archive/animefaces256cleaner/hat")

    if tags[0] == 'green hair':
        shutil.move("./archive/animefaces256cleaner/" + images, "./archive/animefaces256cleaner/green hair")

    if tags[0] == 'green eyes':
        shutil.move("./archive/animefaces256cleaner/" + images, "./archive/animefaces256cleaner/green eyes")

    if tags[0] == 'grey hair':
        shutil.move("./archive/animefaces256cleaner/" + images, "./archive/animefaces256cleaner/grey hair")

    if tags[0] == 'glasses':
        shutil.move("./archive/animefaces256cleaner/" + images, "./archive/animefaces256cleaner/glasses")

    if tags[0] == 'aqua hair':
        shutil.move("./archive/animefaces256cleaner/" + images, "./archive/animefaces256cleaner/aqua hair")

    if tags[0] == 'black eyes':
        shutil.move("./archive/animefaces256cleaner/" + images, "./archive/animefaces256cleaner/black eyes")

    if tags[0] == 'black hair':
        shutil.move("./archive/animefaces256cleaner/" + images, "./archive/animefaces256cleaner/black hair")

    if tags[0] == 'blonde hair':
        shutil.move("./archive/animefaces256cleaner/" + images, "./archive/animefaces256cleaner/blonde hair")

    if tags[0] == 'blue eyes':
        shutil.move("./archive/animefaces256cleaner/" + images, "./archive/animefaces256cleaner/blue eyes")

    if tags[0] == 'blue hair':
        shutil.move("./archive/animefaces256cleaner/" + images, "./archive/animefaces256cleaner/blue hair")

    if tags[0] == 'brown eyes':
        shutil.move("./archive/animefaces256cleaner/" + images, "./archive/animefaces256cleaner/brown eyes")

    if tags[0] == 'brown hair':
        shutil.move("./archive/animefaces256cleaner/" + images, "./archive/animefaces256cleaner/brown hair")

    if tags[0] == 'drill hair':
        shutil.move("./archive/animefaces256cleaner/" + images, "./archive/animefaces256cleaner/drill hair")

    print("Transfer success")

    