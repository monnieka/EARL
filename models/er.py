 
import torch

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_noise_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
 

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_noise_args(parser)
    return parser

class Er(ContinualModel):
    NAME = 'er'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Er, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def observe(self, inputs, labels, not_aug_inputs, true_labels, indexes, epoch):

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_tl= self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))
            true_labels = torch.cat((true_labels, buf_tl))

        outputs = self.net(inputs)


        loss_ext = self.loss(outputs, labels, reduction='none')
        
        loss = loss_ext.mean()
        loss.backward()
        self.opt.step()
        
        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size],
                             true_labels=true_labels[:real_batch_size])

        return loss.item()