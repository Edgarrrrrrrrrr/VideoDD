import argparse
import torch
import wandb
from utils import get_dataset, get_network, evaluate_synset, get_time


def main(args):
    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, test_loader = get_dataset(
        args.dataset, args.data_path, args.batch_train, args.num_workers,
        preload=args.preload
    )

    wandb.init(
        sync_tensorboard=False,
        project="Baseline_HMDB51",
        job_type="Baseline",
        config=args,
        name=f'{args.dataset}_Baseline_{args.model}_{get_time()}'
    )

    net = get_network(args.model, num_classes=51, im_size=(112, 112), frames=args.frames)


    _, final_train_acc, final_test_acc = evaluate_synset(
        it_eval=0, net=net,
        train_loader=train_loader, test_loader=test_loader,
        args=args, mode='none'
    )

    wandb.log({'Final Train Accuracy': final_train_acc, 'Final Test Accuracy': final_test_acc})
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Full Dataset Baseline on HMDB51')

    parser.add_argument('--data_path', type=str, default='/root/autodl-tmp/Baseline/dataset/hmdb_112/HMDB51', help='Dataset base path')
    parser.add_argument('--dataset', type=str, default='HMDB51', help='Dataset name')
    parser.add_argument('--model', type=str, default='ConvNet3D', help='Model architecture')
    parser.add_argument('--epoch_eval_train', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_train', type=int, default=128, help='Batch size')
    parser.add_argument('--lr_net', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader workers')
    parser.add_argument('--frames', type=int, default=16, help='Frames per video')
    parser.add_argument('--preload', action='store_true', help='Whether to preload all dataset images into memory')

    args = parser.parse_args()
    main(args)
