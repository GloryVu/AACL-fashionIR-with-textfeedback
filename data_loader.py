import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import nltk
from PIL import Image
import json


class Dataset(data.Dataset):

    def __init__(self, root, data_file_name, transform=None, return_target=True):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            data: index file name.
            transform: image transformer.
            vocab: pre-processed vocabulary.
        """
        self.root = root
        with open(data_file_name, 'r') as f:
            self.data = json.load(f)
        self.ids = range(len(self.data))
        self.transform = transform
        self.return_target = return_target

    def __getitem__(self, index):
        """Returns one data pair (image and concatenated captions)."""
        data = self.data
        id = self.ids[index]

        candidate_asin = data[id]['candidate']
        candidate_img_name = candidate_asin + '.jpg'
        candidate_image = Image.open(os.path.join(self.root, candidate_img_name)).convert('RGB')
        if self.transform is not None:
            candidate_image = self.transform(candidate_image)

        if self.return_target:
            target_asin = data[id]['target']
            target_img_name = target_asin + '.jpg'
            target_image = Image.open(os.path.join(self.root, target_img_name)).convert('RGB')
            if self.transform is not None:
                target_image = self.transform(target_image)
        else:
            target_image = candidate_image
            target_asin = ''

        caption_texts = data[id]['captions']
        # Convert caption (string) to word ids.
        text_feedback = ' '.join([data[id]['type'], str(caption_texts[0]), 'and', str(caption_texts[1])])

        return target_image, candidate_image, text_feedback, {'target': target_asin, 'candidate': candidate_asin, 'caption': caption_texts}

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of images.
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    target_images, candidate_images, captions, meta = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    target_images = torch.stack(target_images, 0)
    candidate_images = torch.stack(candidate_images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    
    return target_images, candidate_images, captions, meta


def get_loader(root, data_file_name, transform, batch_size, shuffle, return_target, num_workers):
    """Returns torch.utils.data.DataLoader for custom dataset."""
    # relative caption dataset
    dataset = Dataset(root=root,
                      data_file_name=data_file_name,
                      transform=transform,
                      return_target=return_target)
    
    # Data loader for the dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn,
                                              timeout=60)

    return data_loader
