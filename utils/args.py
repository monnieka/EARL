
from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models
import os

def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())

    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')

    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')

    parser.add_argument('--n_epochs', type=int,
                        help='Batch size.')
    parser.add_argument('--n_warmup_epochs', type=int,
                        help='epochs before computing persample losses and lip values')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size.')

    parser.add_argument('--distributed', type=str, default='no', choices=['no', 'dp', 'ddp', 'post_bt', 'post_bt_ddp'])
    parser.add_argument('--num_workers', type=int, 
                        help='Number of workers for main dataloaders.')
    
def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--non_verbose', default=1, choices=[0, 1], type=int, help='Make progress bars non verbose')
    parser.add_argument('--disable_log', default=0, choices=[0, 1], type=int, help='Enable csv logging')

    parser.add_argument('--validation', default=0, choices=[0, 1], type=int,
                        help='Test on the validation set')
    parser.add_argument('--ignore_other_metrics', default=1, choices=[0, 1], type=int,
                        help='disable additional metrics')
    parser.add_argument('--debug_mode', type=int, default=0, help='Run only a few forward steps per epoch')
    parser.add_argument('--checkpoint', type=str, default=None, help="path to checkpoints file")
    parser.add_argument('--job_id', type=int, default=os.environ['SLURM_JOBID'] if 'SLURM_JOBID' in os.environ else 0, help='Job id')

    parser.add_argument('--savecheck', default=0, choices=[0, 1], type=int, help='Save checkpoints')

def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int,
                        help='The batch size of the memory buffer.')
    
def add_noise_args(parser: ArgumentParser) -> None:
    parser.add_argument('--noise_type', type = str, choices= ['clean', 'aggre','worst','rand1', 'rand2','rand3','clean100','noisy100'], default=None,
                        help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100')
    parser.add_argument('--noise', type=str, default=None,choices=['sym','asym'])
    parser.add_argument('--noise_rate', type=float)
    parser.add_argument('--start_with_replay', default=0, type=int, choices=[0,1])    

    parser.add_argument('--warmup_buffer_fitting_epochs', type=int, default=10)
    parser.add_argument('--buffer_fitting_epochs', type=int, default=0)
    parser.add_argument('--buffer_fitting_lr', type=float, default=0.05)
    parser.add_argument('--enable_cutmix', type=int, default=0, choices=[0,1])

    parser.add_argument('--lnl_mode', type=str, default=None, choices=[None, 'coteaching', 'dividemix', 'mixmatch', 'puridiver'])

    parser.add_argument('--mixmatch_alpha_buffer_fitting', type=float, default=0.01)
    parser.add_argument('--mixmatch_lambda_buffer_fitting', type=float, default=0.5)
    parser.add_argument('--mixmatch_naug_buffer_fitting', type=int, default=3)
    parser.add_argument('--buffer_fitting_batch_size', type=int, default=32)
    parser.add_argument('--restert_after_buffer_fit', default=0, type=int, choices=[0,1])

    parser.add_argument('--cutmix_prob', type=float, default=0.5)

    parser.add_argument('--use_hard_transform_buffer_fitting', type=int, default=0, choices=[0,1])