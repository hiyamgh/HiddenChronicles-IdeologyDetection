import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
# import torchvision
# import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--ip_address', type=str)
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--backend', type=str, default='nccl')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = args.ip_address
    print('ip address is: {}, node id is: {}'.format(args.ip_address, os.environ.get("SLURM_NODEID")))
    print('world size = {} x {} = {}'.format(args.nodes, args.gpus, args.world_size))
    os.environ['MASTER_PORT'] = '9999'
    mp.spawn(train, nprocs=args.gpus, args=(args,))


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def train(gpu, args):
    print('gpu:', gpu)
    print('node:', os.environ.get("SLURM_NODEID"))
    rank = int(os.environ.get("SLURM_NODEID")) * args.gpus + gpu
    print('rank:', rank)
    print(f'setting up rank={rank} (with world_size={args.world_size})')
    dist.init_process_group(backend=args.backend, init_method='env://', world_size=args.world_size, rank=rank)
    print(f'--> done setting up rank={rank}')
    torch.manual_seed(0)
    model = ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    dist.destroy_process_group()


    # Data loading code
    # train_dataset = torchvision.datasets.MNIST(root='./data',
    #                                            train=True,
    #                                            transform=transforms.ToTensor(),
    #                                            download=True)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
    #                                                                 num_replicas=args.world_size,
    #                                                                 rank=rank)
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    #                                            batch_size=batch_size,
    #                                            shuffle=False,
    #                                            num_workers=0,
    #                                            pin_memory=True,
    #                                            sampler=train_sampler)

    # start = datetime.now()
    # total_step = len(train_loader)
    # for epoch in range(args.epochs):
    #     for i, (images, labels) in enumerate(train_loader):
    #         images = images.cuda(non_blocking=True)
    #         labels = labels.cuda(non_blocking=True)
    #         # Forward pass
    #         outputs = model(images)
    #         loss = criterion(outputs, labels)
    #
    #         # Backward and optimize
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         if (i + 1) % 100 == 0 and gpu == 0:
    #             print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step,
    #                                                                      loss.item()))
    # if gpu == 0:
    #     print("Training complete in: " + str(datetime.now() - start))


if __name__ == '__main__':
    main()