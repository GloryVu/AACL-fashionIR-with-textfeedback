import json
import os
images = os.listdir('data/resized_images')
images = [i.replace('.jpg', '') for i in images]
# print(image)
dress = json.load(open('data/captions/cap.dress.val.json'))
dress = [{**d,
          'type': 'dress'} for d in dress]
shirt = json.load(open('data/captions/cap.shirt.val.json'))
shirt = [{**d,
          'type': 'shirt'} for d in shirt]
toptee = json.load(open('data/captions/cap.toptee.val.json'))
toptee = [{**d,
           'type': 'toptee'} for d in toptee]
all = dress+shirt+toptee
all = [sample for sample in all if sample['target'] in images and sample['candidate'] in images]
json.dump(all, open('data/captions/all.val.json', 'w'))
json.dump(all[:32], open('data/captions/all.val.abatch.json', 'w'))
image = set([sample['target'] for sample in all[:32]]+[sample['candidate'] for sample in all[:32]])
image = list(image)

json.dump(image, open('data/image_splits/all.val.abatch.json', 'w'))
dress = json.load(open('data/captions/cap.dress.train.json'))
dress = [{**d,
          'type': 'dress'} for d in dress]
shirt = json.load(open('data/captions/cap.shirt.train.json'))
shirt = [{**d,
          'type': 'shirt'} for d in shirt]
toptee = json.load(open('data/captions/cap.toptee.train.json'))
toptee = [{**d,
           'type': 'toptee'} for d in toptee]
all = dress+shirt+toptee
all = [sample for sample in all if sample['target'] in images and sample['candidate'] in images]

json.dump(all, open('data/captions/all.train.json', 'w'))
json.dump(all[:32], open('data/captions/all.train.abatch.json', 'w'))
image = set([sample['target'] for sample in all[:32]]+[sample['candidate'] for sample in all[:32]])
image = list(image)
json.dump(image, open('data/image_splits/all.train.abatch.json', 'w'))
dress = json.load(open('data/captions/cap.dress.test.json'))
dress = [{**d,
          'type': 'dress'} for d in dress]
shirt = json.load(open('data/captions/cap.shirt.test.json'))
shirt = [{**d,
          'type': 'shirt'} for d in shirt]
toptee = json.load(open('data/captions/cap.toptee.test.json'))
toptee = [{**d,
           'type': 'toptee'} for d in toptee]
all = dress+shirt+toptee
all = [sample for sample in all if sample['candidate'] in images]

json.dump(image, open('data/image_splits/all.test.abatch.json', 'w'))
json.dump(all, open('data/captions/all.test.json', 'w'))
json.dump(all[:32], open('data/captions/all.test.abatch.json', 'w'))
image = set([sample['candidate'] for sample in all[:32]])
image = list(image)
# dress = json.load(open('data/image_splits/split.dress.train.json'))
# shirt = json.load(open('data/image_splits/split.shirt.train.json'))
# toptee = json.load(open('data/image_splits/split.toptee.train.json'))
# all = dress+shirt+toptee
# all = [sample for sample in all if sample in image]

# json.dump(all, open('data/image_splits/all.train.json', 'w'))
# dress = json.load(open('data/image_splits/split.dress.val.json'))
# shirt = json.load(open('data/image_splits/split.shirt.val.json'))
# toptee = json.load(open('data/image_splits/split.toptee.val.json'))
# all = dress+shirt+toptee
# all = [sample for sample in all if sample in image]
# json.dump(all, open('data/image_splits/all.val.json', 'w'))
# dress = json.load(open('data/image_splits/split.dress.test.json'))
# shirt = json.load(open('data/image_splits/split.shirt.test.json'))
# toptee = json.load(open('data/image_splits/split.toptee.test.json'))
# all = dress+shirt+toptee
# all = [sample for sample in all if sample in image]
# json.dump(all, open('data/image_splits/all.test.json', 'w'))