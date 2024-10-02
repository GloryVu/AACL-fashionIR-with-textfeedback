import argparse
import os
import time
import torch
import torch.nn as nn
from torchvision import transforms
from utils import create_exp_dir, Ranker
from data_loader import get_loader
from models import AACL

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


CAPT = 'data/captions/all.{}.json'
IMAGE_ROOT = 'data/resized_images/'
SPLIT = 'data/image_splits/all.{}.json'

triplet_avg = nn.TripletMarginLoss(reduction='elementwise_mean', margin=1)

def eval_batch(data_loader, aacl, ranker):
    # ranker.update_emb(aacl.image_encoder)
    rankings = []
    loss = []
    for i, (target_images, candidate_images, captions, meta_info) in enumerate(data_loader):
        with torch.no_grad():
            target_images = target_images.to(device)
            o_y = aacl.image_encoder(target_images)
            candidate_images = candidate_images.to(device)
            o_xt = aacl(candidate_images,captions)
            # random select negative examples
            m = target_images.size(0)
            random_index = [m - 1 - n for n in range(m)]
            random_index = torch.LongTensor(random_index)
            negative_ft = o_y[random_index]

            loss.append(triplet_avg(anchor=o_xt,
                                    positive=o_y, negative=negative_ft))

            # target_asins = [meta_info[j]['target'] for j in range(len(meta_info))]
            # rankings.append(ranker.compute_rank(o_xt, target_asins))

    metrics = {}
    # rankings = torch.cat(rankings, dim=0)
    # metrics['score'] = 1 - rankings.mean().item() / ranker.data_emb.size(0)
    metrics['loss'] = torch.stack(loss, dim=0).mean().item()
    return metrics


def train(args):
    # wandb.init(project="image-caption-training", config=args)
    
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    transform_dev = transforms.Compose([
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    

    data_loader = get_loader(IMAGE_ROOT.format(args.data_set),
                             CAPT.format('train'),
                             transform,
                             args.batch_size, shuffle=True, return_target=True, num_workers=args.num_workers)

    data_loader_dev = get_loader(IMAGE_ROOT.format(args.data_set),
                                 CAPT.format('val'),
                                 transform_dev,
                                 args.batch_size, shuffle=False, return_target=True, num_workers=args.num_workers)
    ranker = Ranker(root=IMAGE_ROOT.format(args.data_set), image_split_file=SPLIT.format('val'),
                    transform=transform_dev, num_workers=args.num_workers)
    save_folder = '{}/'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    # create_exp_dir(save_folder, scripts_to_save=['models.py', 'data_loader.py', 'train.py', 'build_vocab.py', 'utils.py'])


    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            with open(os.path.join(save_folder, 'log.txt'), 'a+') as f_log:
                f_log.write(s + '\n')
    aacl=AACL(device)
    aacl.to(device)
    current_lr = args.learning_rate
    optimizer = torch.optim.SGD(aacl.parameters(), lr=current_lr)

    cur_patient = 0
    best_score = 100# float('-inf')
    stop_train = False
    total_step = len(data_loader)
    for epoch in range(60):

        for i, (target_images, candidate_images, captions, meta_info) in enumerate(data_loader):

            target_images = target_images.to(device)
            
            candidate_images = candidate_images.to(device)

            o_xt=aacl(candidate_images,captions)
            o_y=aacl.image_encoder(target_images)


            # random select negative examples
            m = target_images.size(0)
            random_index = [m - 1 - n for n in range(m)]
            random_index = torch.LongTensor(random_index)
            negative_ft = o_y[random_index]

            loss = triplet_avg(anchor=o_xt,
                               positive=o_y, negative=negative_ft)

            aacl.zero_grad()
            loss.backward()
            optimizer.step()

            if i % args.log_step == 0:
                logging(
                    '| epoch {:3d} | step {:6d}/{:6d} | lr {:06.6f} | train loss {:8.3f}'.format(epoch, i, total_step,
                                                                                                 current_lr,
                                                                                                 loss.item()))
                # wandb.log({"train_loss": loss.item(), "epoch": epoch, "step": i, "learning_rate": current_lr})

        aacl.eval()
        logging('-' * 77)
        metrics = eval_batch(data_loader_dev, aacl, ranker)
        logging('| eval loss: {:8.3f} | '.format(
            metrics['loss']))
        logging('-' * 77)
        # wandb.log({"eval_loss": metrics['loss'], "eval_score": metrics['score'], "epoch": epoch})

        aacl.train()

        dev_score = metrics['loss']
        if dev_score < best_score:
            best_score = dev_score
            # save best model
            # resnet = image_encoder.delete_resnet()
            # swin = image_encoder
            torch.save(aacl, os.path.join(
                save_folder,
                'best.th'.format(args.embed_size)))
            # image_encoder.load_resnet(resnet)
            # image_encoder.load_swin_transformer(swin)

            cur_patient = 0
        else:
            cur_patient += 1
        if epoch%10==0:
            current_lr *=0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            # if current_lr < args.learning_rate * 1e-3:
            #     stop_train = True
            #     break

        if stop_train:
            break
    logging('best_dev_score: {}'.format(best_score))
    # wandb.log({"best_dev_score": best_score})



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='models',
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')

    parser.add_argument('--data_set', type=str, default='dress')
    parser.add_argument('--log_step', type=int, default=3,
                        help='step size for printing log info')
    parser.add_argument('--patient', type=int, default=3,
                        help='patient for reducing learning rate')

    # Model parameters
    parser.add_argument('--embed_size', type=int , default=768,
                        help='dimension of word embedding vectors')
    # parser.add_argument('--embed_size', type=int , default=512,
    #                     help='dimension of word embedding vectors')
    # Learning parameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    args = parser.parse_args()

    train(args)

