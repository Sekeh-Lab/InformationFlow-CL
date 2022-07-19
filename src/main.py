"""Main entry point for doing all pruning-related stuff. Adapted from https://github.com/arunmallya/packnet/blob/master/src/main.py"""
from __future__ import division, print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import json
import warnings
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import dataset
import networks as net
import utils as utils
from prune import SparsePruner
import time
from torch.optim.lr_scheduler  import MultiStepLR

# To prevent PIL warnings.
warnings.filterwarnings("ignore")

###General flags
FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--arch', choices=['vgg16', 'vgg16bn', 'resnet50', 'densenet121'], help='Architectures')
FLAGS.add_argument('--mode', choices=['finetune','maskfinetune', 'connmaskfinetune', 'prune', 'check', 'eval', 'conns'], help='Run mode')
FLAGS.add_argument('--num_outputs', type=int, default=-1, help='Num outputs for dataset')
FLAGS.add_argument('--train_biases', action='store_true', default=False, help='use separate biases or not')
FLAGS.add_argument('--train_bn', default=False, help='train batch norm or not')
FLAGS.add_argument('--preprune', type=int, default=1, help='Whether the current task has been pruned already')

FLAGS.add_argument('--dataset', type=str, default='', help='Name of dataset')
FLAGS.add_argument('--cuda', action='store_true', default=True, help='use CUDA')
FLAGS.add_argument('--init_dump', action='store_true', default=False, help='Initial model dump.')
FLAGS.add_argument('--cores', type=int, default=4, help='Number of CPU cores.')
FLAGS.add_argument('--task_num', type=int, default=1, help='Current task number.')
# Other.
# Optimization options.
FLAGS.add_argument('--lr', type=float, help='Learning rate')
FLAGS.add_argument('--lr_decay_every', type=int, help='Step decay every this many epochs')
FLAGS.add_argument('--lr_decay_factor', type=float, help='Multiply lr by this much every step of decay')
FLAGS.add_argument('--finetune_epochs', type=int, help='Number of initial finetuning epochs')
FLAGS.add_argument('--batch_size', type=int, default=32, help='Batch size')
FLAGS.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
FLAGS.add_argument('--Milestones', nargs='+', type=float, default=[100,150,200])
FLAGS.add_argument('--Gamma', type=float, default=0.2)   
# Paths.
FLAGS.add_argument('--train_path', type=str, default='', help='Location of train data')
FLAGS.add_argument('--test_path', type=str, default='', help='Location of test data')
FLAGS.add_argument('--save_prefix', type=str, default='../checkpoints/', help='Location to save model')
FLAGS.add_argument('--save_conn_name', type=str, default='../checkpoints/conns', help='Location to save conns')                   
FLAGS.add_argument('--loadname', type=str, default='', help='Location to save model')
# Pruning options.
FLAGS.add_argument('--prune_method', type=str, default='sparse', choices=['sparse'], help='Pruning method to use')
FLAGS.add_argument('--prune_perc_per_layer', type=float, default=0.5, help='% of neurons to prune per layer')
FLAGS.add_argument('--post_prune_epochs', type=int, default=0, help='Number of epochs to finetune for after pruning')
FLAGS.add_argument('--disable_pruning_mask', action='store_true', default=False, help='use masking or not')            
FLAGS.add_argument('--freeze_perc', type=float, default=0.02)                   
FLAGS.add_argument('--num_freeze_layers', type=int, default=4)     
FLAGS.add_argument('--freeze_order', choices=['top','bottom', 'random'], help='Order of selection for layer freezing, by connectivity')

    
### An object which handles training, pruning, and testing operations
class Manager(object):
    """Handles training and pruning."""

    ### Data loading and makes a pruner instance
    def __init__(self, args, model, all_task_masks, previous_masks, dataset2idx, dataset2biases, conns, conn_aves, ckpt_epoch, ckpt_accuracy, ckpt_errors):
        self.args = args
        self.cuda = args.cuda
        self.model = model
        self.modelckpt = model
        self.ckpt_epoch = ckpt_epoch
        self.ckpt_accuracy = ckpt_accuracy
        self.ckpt_errors = ckpt_errors
        self.dataset2idx = dataset2idx
        self.dataset2biases = dataset2biases
        self.logsoftmax = nn.LogSoftmax()

        if args.mode != 'check':
            ### Set up data loader, criterion, and pruner.
            train_loader = dataset.train_loader
            test_loader = dataset.test_loader

            self.train_data_loader = train_loader(args.dataset, args.train_path, args.batch_size, pin_memory=args.cuda)
            self.test_data_loader = test_loader(args.dataset, args.test_path, args.batch_size, pin_memory=args.cuda)
            # self.criterion = nn.CrossEntropyLoss()
            
            self.pruner = SparsePruner(
                self.args, self.model, all_task_masks, previous_masks, dataset2idx,
                self.train_data_loader, self.test_data_loader, conns, conn_aves)



    ### Train the model for the current task, using all past frozen weights as well
    def train(self, epochs, preprune, save=True, savename='', best_accuracy=0):
        """Performs training."""
        best_model_acc = best_accuracy
        error_history = []

        if self.args.cuda:
            self.model = self.model.cuda()

        # Get optimizer with correct params.
        params_to_optimize = self.model.parameters()
        optimizer = optim.SGD(params_to_optimize, lr=self.args.lr, momentum=0.9, weight_decay=self.args.weight_decay, nesterov=True)

        scheduler = MultiStepLR(optimizer, milestones=self.args.Milestones, gamma=self.args.Gamma)    

        for epoch in range(epochs):
            running_loss = 0.0
            print('Epoch: ', epoch)
            
            if self.args.train_bn:
                self.model.train()
            else:
                self.model.train_nobn()

            for batch, label in tqdm(self.train_data_loader, desc='Epoch: %d ' % (epoch)):
                x_input, y_label = batch, label
                ########################### Data Loader + Training ##################################
                one_hot                                       = np.zeros((y_label.shape[0], self.args.num_outputs))
                one_hot[np.arange(y_label.shape[0]), y_label] = 1
                y_label                                       = torch.Tensor(one_hot) 
        
        
                x_input, y_label = x_input.cuda(), y_label.cuda()
        
                # Set grads to 0.
                self.model.zero_grad()

                # Do forward-backward.
                output = self.model(x_input)
                loss    = torch.mean(torch.sum(-y_label * self.logsoftmax(output), dim=1))
                loss.backward()

                # Set frozen param grads to 0.
                if not self.args.disable_pruning_mask:
                    self.pruner.make_grads_zero()
        
                optimizer.step()
        
                running_loss += loss.item()
                        
                if np.isnan(running_loss):
                    import pdb; pdb.set_trace()
        
                # Set pruned weights to 0.
                if not self.args.disable_pruning_mask:
                    self.pruner.make_pruned_zero()
                            
            scheduler.step()
            epoch_acc = 100.*self.accuracy(self.model, self.test_data_loader)
            print('Accuracy of the network on the 10000 test images: %f %%\n' % (epoch_acc))

            errors = self.eval(self.pruner.current_dataset_idx, preprune = preprune)
            error_history.append(errors)

            # Save performance history and stats.
            with open(savename + '.json', 'w') as fout:
                json.dump({
                    'error_history': error_history,
                    'args': vars(self.args),
                }, fout)

            # Save best model, if required.
            if save and best_model_acc < epoch_acc:
                print('Best model so far, Accuracy: %0.2f%% -> %0.2f%%' %
                      (best_model_acc, epoch_acc))

                best_model_acc = epoch_acc
                self.save_model(epoch, best_model_acc, errors, savename)

        print('Finished finetuning...')
        print('Best error/accuracy: %0.2f%%, %0.2f%%' %
              (100 - best_model_acc, best_model_acc))
        print('-' * 16)


    ### Saves a checkpoint of the model
    def save_model(self, epoch, best_accuracy, errors, savename):
        """Saves model to file."""
        base_model = self.model

        # Prepare the ckpt.
        self.dataset2idx[self.args.dataset] = self.pruner.current_dataset_idx
        self.dataset2biases[self.args.dataset] = self.pruner.get_biases()
        ckpt = {
            'args': self.args,
            'epoch': epoch,
            'accuracy': best_accuracy,
            'errors': errors,
            'dataset2idx': self.dataset2idx,
            'previous_masks': self.pruner.current_masks,
            'all_task_masks': self.pruner.all_task_masks,
            'conns' : self.pruner.conns,
            'conn_aves' : self.pruner.conn_aves,
            'model': base_model,
        }
        if self.args.train_biases:
            ckpt['dataset2biases'] = self.dataset2biases

        # Save to file.
        torch.save(ckpt, savename + '.pt')




    ### Collect activations, calculate connectivities for filters and layers, then saves them in the checkpoint
    def calc_conns(self, savename = ""):
        """Calculating Connectivities."""
        ### Since no pruning or training will occur when called in "conn" mode, need to explicitly set the current masks
        self.pruner.current_masks = self.pruner.previous_masks
        
        self.pruner.calc_conns()
        ### Commented out to save storage space, but these will save the connectivity data as files
        # np.save(self.args.save_conn_name, self.pruner.conns)
        # np.save('./conn_aves.npy', self.pruner.conn_aves)
        
        self.save_model(self.ckpt_epoch, self.ckpt_accuracy, self.ckpt_errors, savename)



    ### Call for the pruner to prune the model
    def prune(self):
        """Perform pruning."""
        print('Pre-prune eval:')
        ### Preprune reflects whether or not pruning has occured yet for the given task
        self.eval(self.pruner.current_dataset_idx, preprune = True)

        self.pruner.prune()
        self.check(True)

        print('\nPost-prune eval:')
        errors = self.eval(self.pruner.current_dataset_idx, preprune = False)

        accuracy = 100 - errors  # Top-1 accuracy.
        self.save_model(-1, accuracy, errors,
                        self.args.save_prefix + '_postprune')

        # Do final finetuning to improve results on pruned network.
        if self.args.post_prune_epochs:
            print('Doing some extra finetuning...')
            self.train(self.args.post_prune_epochs, save=True,
                       savename=self.args.save_prefix + '_final', best_accuracy=accuracy, preprune = False)

        print('-' * 16)
        print('Pruning summary:')
        self.check(True)
        print('-' * 16)
        print("\n\n\n\n\n\n\n")

    ### Just checks how many parameters per layer are now 0 post-pruning
    def check(self, verbose=False):
        """Makes sure that the layers are pruned."""
        print('Checking...')
        for layer_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                num_params = weight.numel()
                num_zero = weight.view(-1).eq(0).sum()
                if verbose:
                    print('Layer #%d: Pruned %d/%d (%.2f%%)' %
                          (layer_idx, num_zero, num_params, 100 * num_zero / num_params))




    ### Perform inference
    def eval(self, dataset_idx, biases=None, preprune = True):
        """Performs evaluation."""
        print("Current idx: " + str(self.pruner.current_dataset_idx))

        if not self.args.disable_pruning_mask:
            print("Applying dataset mask for dataset: ", dataset_idx)
            self.pruner.apply_mask(dataset_idx, preprune)
        if biases is not None:
            print("biases not none")
            self.pruner.restore_biases(biases)

        self.model.eval()
        print('Performing eval...')

        acc = 100.*self.accuracy(self.model, self.test_data_loader)
        errors = 100-acc
        print('Accuracy is: ', acc)
        
        if self.args.train_bn:
            self.model.train()
        else:
            self.model.train_nobn()
        return errors


    def accuracy(self, net, testloader):
        correct = 0
        total   = 0
        net.eval()
    
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                
                outputs      = net(images, labels=True)
                _, predicted = torch.max(outputs.data, 1)
    
                total   += labels.size(0)
                correct += (predicted == labels).sum().item()
        return float(correct) / total







### Produces an initialized model and initial shared mask prior to beginning training
def init_dump(arch):
    """Dumps pretrained model in required format."""
    if arch == 'vgg16':
        model = net.ModifiedVGG16()
    else:
        raise ValueError('Architecture type not supported.')

    previous_masks = {}
    all_task_masks = {}
    task_mask = {}
    
    for module_idx, module in enumerate(model.shared.modules()):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            print("appending conv or linear layer")
            mask = torch.ByteTensor(module.weight.data.size()).fill_(1)
            task = torch.ByteTensor(module.weight.data.size()).fill_(0)
            
            if 'cuda' in module.weight.data.type():
                mask = mask.cuda()
                task = task.cuda()
                
            previous_masks[module_idx] = mask
            task_mask[module_idx] = task
    

    all_task_masks[0] = [task_mask]
    
    torch.save({
        'dataset2idx': {'CIFAR10': 1},
        'previous_masks': previous_masks,
        'all_task_masks': all_task_masks,
        'epoch': 0,
        'accuracy': 0,
        'errors': 0,        
        'model': model,
        'conns' : {},
        'conn_aves' : {},
    }, '../checkpoints/CIFAR10/%s.pt' % (arch))



######################################################################################################################################################################
###
###     Main function
###
######################################################################################################################################################################




def main():
    ### Early termination conditions
    args = FLAGS.parse_args()
    if args.init_dump:
        print("init dump",flush = True)
        init_dump(args.arch)
        return
    if args.prune_perc_per_layer <= 0:
        print("non-positive prune perc",flush = True)
        return

    torch.cuda.set_device(0)

    
    # Set default train and test path if not provided as input.
    if not args.train_path:
        print("Default training path used")
        args.train_path = '../data/%s/train' % (args.dataset)
    if not args.test_path:
        print("Default testing path used")
        args.test_path = '../data/%s/test' % (args.dataset)


    
    ###################
    ##### Load Ckpt
    ###################
    
    ### For final eval, loadname is final task model, dataset is eval task
    print("Loading from: " + str(args.loadname))
    print("Dataset: " + str(args.dataset))

    ### Note this differs from Packnet, which will initialize a model if no loadname is provided. I explicitly require it to be pre-initialized with initdump to ensure consistency
    ckpt = torch.load(args.loadname)
    model = ckpt['model']
    epoch = ckpt['epoch']
    accuracy = ckpt['accuracy']
    errors = ckpt['errors']
    previous_masks = ckpt['previous_masks']
    ### Note: all_task_masks isn't from PackNet. Initially previous_masks stored a single composite non-binary mask for all tasks,
    ###     We now also store individual binary masks for each task, which allows us to store information about which weights are shared with a given task
    ###     Although the experiments for the publication don't utilize this functionality (we always share all past frozen weights), I've left it in for
    ###     future use by ourselves or anyone looking to modify the code with selective weight sharing.
    all_task_masks = ckpt['all_task_masks']
    dataset2idx = ckpt['dataset2idx']

    if 'dataset2biases' in ckpt:
        print("Biases in checkpoint", flush = True)
        dataset2biases = ckpt['dataset2biases']
    else:
        dataset2biases = {}

    if 'conns' in ckpt:
        conns = ckpt['conns']
    else:
        conns = {}        

    if 'conn_aves' in ckpt:
        conn_aves = ckpt['conn_aves']
    else:
        conn_aves = {}

    print("#######################################################################")
    print("Finished Loading Checkpoint")
    print("Epoch is: ", epoch)
    print("Accuracy is: ", accuracy)
    print("Errors are: ", errors)
    print("Previous Masks keys: ", previous_masks.keys())
    print("All task Masks keys: ", all_task_masks.keys())
    print("Conns keys: ", conns.keys())
    print("Conn_aves keys: ", conn_aves.keys())
    print("Dataset2idx is: ", dataset2idx)
    print("#######################################################################")

    ### This is for producing and setting the classifier layer for a given task's # classes
    model.add_dataset(args.dataset, args.num_outputs)
    model.set_dataset(args.dataset)

    if args.cuda:
        model = model.cuda()

    # Create the manager object.
    manager = Manager(args, model, all_task_masks, previous_masks, dataset2idx, dataset2biases, conns, conn_aves, epoch, accuracy, errors)
    print("Manager created")


    # Train the network
    if args.mode == 'finetune':
        print("finetuning", flush = True)
        # Make pruned params available for new dataset.

        ### Changed this function to also create the new task mask during the allocation phase
        manager.pruner.make_finetuning_mask(args.task_num)

        manager.train(args.finetune_epochs,
                      save=True, savename=args.save_prefix, preprune = True)
                                  
    ### Calculate Connectivities, for our proposed method
    elif args.mode == 'conns':
        print("Calculate Connectivities")
        manager.calc_conns(savename=args.save_prefix)

    # Perform pruning, modified to account for connectivity information
    elif args.mode == 'prune':
        print("Pruning", flush = True)
        print("Prune percent is: ", args.prune_perc_per_layer)
        print("Freeze percent is: ", args.freeze_perc)
        print("Number of layers to freeze: ", args.num_freeze_layers)
        print("Freezing order: ", args.freeze_order)
    
        manager.prune()
        
    # Load model and make sure everything is fine.
    elif args.mode == 'check':
        print("Checking")
        manager.check(verbose=True)
        
    # Perform Inference
    elif args.mode == 'eval':
        print("Evaluating Accuracy")
        print("args.preprune: " + str(args.preprune))
        if args.preprune == 0:
            args.preprune=False
        else:
            args.preprune=True
        # Just run the model on the eval set.
        biases = None
        if 'dataset2biases' in ckpt:
            biases = ckpt['dataset2biases'][args.dataset]
        print("Using ckpt dataset2idx:", ckpt['dataset2idx'][args.dataset])
        manager.eval(ckpt['dataset2idx'][args.dataset], biases, preprune = args.preprune)


if __name__ == '__main__':
    main()
