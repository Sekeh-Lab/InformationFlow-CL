"""
Handles all the pruning and connectivity. Pruning steps are adapted from: https://github.com/arunmallya/packnet/blob/master/src/prune.py
Connectivity steps and implementation of connectivity into the pruning steps are part of our contribution
"""
from __future__ import print_function

import collections
import time
import copy
import torch
import random
import multiprocessing
import torch.nn as nn
import numpy             as np

# Custom imports
from utils                   import activations, corr


class SparsePruner(object):
    """Performs pruning on the given model."""
    ### Relavent arguments are moved to the pruner to explicitly show which arguments are used by it
    def __init__(self, args, model, all_task_masks, previous_masks, current_dataset_idx,
                        train_loader, test_loader, conns, conn_aves):
        self.args = args
        self.model = model 
        self.prune_perc = args.prune_perc_per_layer
        self.freeze_perc = args.freeze_perc
        self.num_freeze_layers = args.num_freeze_layers
        self.freeze_order = args.freeze_order
        self.train_bias = args.train_biases
        self.train_bn = args.train_bn 
        
        self.trainloader = train_loader 
        self.testloader = test_loader 
        
        self.conns = conns
        self.conn_aves = conn_aves
        
        self.current_masks = None
        self.previous_masks = previous_masks
        self.all_task_masks = all_task_masks 


        self.current_dataset_idx = max(list(current_dataset_idx.values()))
        print("current index is: " + str(self.current_dataset_idx))
    
    
    
     
    """
    ###########################################################################################
    #####
    #####  Connectivity Functions
    #####
    #####  Use: Gets the connectivity between each pair of convolutional or linear layers. 
    #####       The primary original code for our published connectivity-based freezing method
    #####
    ###########################################################################################
    """

      
    def calc_conns(self):
        self.task_conns = {}
        self.task_conn_aves = {}        
        
        
        ### Record the indices of all adjacent pairs of layers in the shared network
        ### This numbering method reflects the "layer index" in Figs. 2-4 of the accompanying paper
        parents = []
        children = []
        i = 0
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if (i == 0):
                    parents.append(module_idx)
                    i += 1
                else:
                    parents.append(module_idx)
                    children.append(module_idx)

        for key_id in range(0,len(children)): 
            self.task_conn_aves[parents[key_id]], self.task_conns[parents[key_id]] = self.calc_conn([parents[key_id]], [children[key_id]], key_id)
            
        self.conn_aves[self.args.task_num] = self.task_conn_aves
        self.conns[self.args.task_num] = self.task_conns


   
    
    def calc_conn(self, parent_key, children_key, key_id):
        self.model.eval()
    
        # Obtain Activations
        print("----------------------------------")
        print("Collecting activations from layers")
    
        p1_op = {}
        p1_lab = {}
        c1_op = {}
        c1_lab = {}    

        unique_keys = np.unique(np.union1d(parent_key, children_key)).tolist()
        act         = {}
        lab         = {}
    

        ### Get activations and labels from the function in utils prior to calculating connectivities
        for item_key in unique_keys:
            act[item_key], lab[item_key] = activations(self.testloader, self.model, self.args.cuda, item_key)

        for item_idx in range(len(parent_key)):
            p1_op[str(item_idx)] = copy.deepcopy(act[parent_key[item_idx]]) 
            p1_lab[str(item_idx)] = copy.deepcopy(lab[parent_key[item_idx]]) 
            c1_op[str(item_idx)] = copy.deepcopy(act[children_key[item_idx]])
            c1_lab[str(item_idx)] = copy.deepcopy(lab[children_key[item_idx]])
        
        del act, lab

       
        print("----------------------------------")
        print("Begin Execution of conn estimation")
    
        parent_aves = []
        p1_op = np.asarray(list(p1_op.values())[0])
        p1_lab = np.asarray(list(p1_lab.values())[0])
        c1_op = np.asarray(list(c1_op.values())[0])
        c1_lab = np.asarray(list(c1_lab.values())[0])
    
        task = self.args.task_num


        for c in list(np.unique(np.asarray(p1_lab))):
            p1_op[p1_lab == c] -= np.mean(p1_op[p1_lab == c])
            p1_op[p1_lab == c] /= np.std(p1_op[p1_lab == c])

            c1_op[c1_lab == c] -= np.mean(c1_op[c1_lab == c])
            c1_op[c1_lab == c] /= np.std(c1_op[c1_lab == c])

        """
        Code for averaging conns by parent prior by layer
        """
        
        parent_class_aves = []

        parents_by_class = []
        parents_aves = []
        conn_aves = []
        parents = []
        
        for c in list(np.unique(np.asarray(p1_lab))):
            p1_class = p1_op[np.where(p1_lab == c)]
            c1_class = c1_op[np.where(c1_lab == c)]

            pool = multiprocessing.Pool(self.args.cores)

            ### Parents is a 2D list of all of the connectivities of parents and children for a single class
            parents = pool.starmap(corr, [(p1_class[:,p], c1_class[:,:]) for p in list(range(len(p1_op[0])))], chunksize = 8)

            pool.close()
            pool.join()

            ### This is a growing list of each p-c connectivity for all activations of a given class
            ###     The dimensions are (class, parent, child)
            parents_by_class.append(parents)

        ### This is the final array of appended class sets of parent-child connectivities
        ### Shape should be 10x64x64 for layer 1 in cifar10
        parents_by_class = np.asarray(parents_by_class)
        
        ### Averages all classes, since all class priors are the same for cifar10 and 100
        conn_aves = np.mean(parents_by_class[:], axis=0)
        
        ### Then average over the parents and children to get the layer-layer connectivity
        layer_ave = np.mean(conn_aves[:])

        return layer_ave, conn_aves



    
    
    
    
    
    
    
    """
    ##########################################################################################################################################
    Pruning Functions
    ##########################################################################################################################################
    """
    ### Goes through and calls prune_mask for each layer and stores the results
    ### Then applies the masks to the weights
    def prune(self):
        """Gets pruning mask for each layer, based on previous_masks.
           Sets the self.current_masks to the computed pruning masks.
        """
            
        print('Pruning for dataset idx: %d' % (self.current_dataset_idx))

        assert not self.current_masks, 'Current mask is not empty? Pruning twice?'
        self.current_masks = {}
        

        print('Pruning each layer by removing %.2f%% of values' %
              (100 * self.prune_perc))
    

        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                ### Get the pruned mask for the current layer
                mask = self.pruning_mask(module.weight.data, self.previous_masks[module_idx], module_idx)
                self.current_masks[module_idx] = mask.cuda()
                # Set pruned weights to 0.
                weight = module.weight.data
                weight[self.current_masks[module_idx].eq(0)] = 0.0
                ### Reflect pruned weights in the task-specific mask as well
                for i in range(self.current_dataset_idx):
                    self.all_task_masks[i][0][module_idx][mask.eq(0)] = 0
                # self.all_task_masks[self.current_dataset_idx - 1][0][module_idx][mask.eq(0)] = 0
                
                
        print("\nFOR TASK %d:", self.current_dataset_idx)
        for layer_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                num_params = weight.numel()
                num_frozen = weight[self.current_masks[layer_idx].eq(self.current_dataset_idx)].numel()
                print('Layer #%d: Frozen %d/%d (%.2f%%)' %
                      (layer_idx, num_frozen, num_params, 100 * num_frozen / num_params))                

      
 
 
 
 
    def pruning_mask(self, weights, previous_mask, layer_idx):
        """
        Ranks filters by magnitude. Sets all below kth to 0.
        Returns pruned mask.
        """

        previous_mask = previous_mask.cuda()

        filter_weights = weights
        filter_previous_mask = previous_mask.eq(self.current_dataset_idx)
        tensor = weights[filter_previous_mask]

        abs_tensor = tensor.abs()


        """
        Code for increasing the freezing percent of a given layer based on connectivity
        """

        prune_rank = round(self.prune_perc * abs_tensor.numel())
        connlist = self.conn_aves[self.args.task_num]


        max_n_layers_indices = np.argsort(list(connlist.values()))
        max_n_keys = np.asarray(list(connlist.keys()))[list(max_n_layers_indices)]
        random_idxs = np.copy(max_n_keys)
        np.random.shuffle(random_idxs)
        
        ### Apply freezin if the index is selected, otherwise prune at the baseline rate.
        if self.freeze_order == "top" and (layer_idx in max_n_keys[-self.num_freeze_layers:]):
            prune_rank = round((self.prune_perc - self.freeze_perc) * abs_tensor.numel())
        elif self.freeze_order == "bottom" and (layer_idx in max_n_keys[:self.num_freeze_layers]):
            prune_rank = round((self.prune_perc - self.freeze_perc) * abs_tensor.numel())
        elif self.freeze_order == "random" and (layer_idx in random_idxs[:self.num_freeze_layers]):
            prune_rank = round((self.prune_perc - self.freeze_perc) * abs_tensor.numel())
        
        prune_value = abs_tensor.view(-1).cpu().kthvalue(prune_rank)[0]

        remove_mask = torch.zeros(weights.shape)

        remove_mask[filter_weights.abs().le(prune_value)]=1
        remove_mask[previous_mask.ne(self.current_dataset_idx)]=0

        previous_mask[remove_mask.eq(1)] = 0
        mask = previous_mask
        
        print('Layer #%d, pruned %d/%d (%.2f%%) (Total in layer: %d)' %
              (layer_idx, mask.eq(0).sum(), tensor.numel(),
              100 * mask.eq(0).sum() / tensor.numel(), weights.numel()))

        return mask
        
        
        
        
    
        
        
    
    """
    ##########################################################################################################################################
    Update Functions
    ##########################################################################################################################################
    """


    ### This is unaffected in the shared masks since shared weights always have the current index unless frozen
    def make_grads_zero(self):
        """Sets grads of fixed weights to 0."""
        assert self.current_masks

        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = self.current_masks[module_idx]
                # Set grads of all weights not belonging to current dataset to 0.
                if module.weight.grad is not None:
                    module.weight.grad.data[layer_mask.ne(self.current_dataset_idx)] = 0
                    if not self.train_bias:
                        # Biases are fixed.
                        if module.bias is not None:
                            module.bias.grad.data.fill_(0)
            elif 'BatchNorm' in str(type(module)):
                # Set grads of batchnorm params to 0.
                if not self.train_bn:
                    module.weight.grad.data.fill_(0)
                    module.bias.grad.data.fill_(0)

    ### This is just a prune() but with pre-calculated masks
    def make_pruned_zero(self):
        """Makes pruned weights 0."""
        assert self.current_masks
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = self.current_masks[module_idx]
                module.weight.data[layer_mask.eq(0)] = 0.0


    ### Applies appropriate mask to recreate task model for inference
    def apply_mask(self, dataset_idx, preprune=True):
        """To be done to retrieve weights just for a particular dataset"""
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                mask = -100

                if preprune == True:
                    mask = self.previous_masks[module_idx].cuda()
                else:
                    for i in range(dataset_idx):
                        if i == 0:
                            mask = self.all_task_masks[i][0][module_idx].cuda()
                        else:
                            mask = mask.logical_or(self.all_task_masks[i][0][module_idx].cuda())
                        
                weight[mask.eq(0)] = 0.0
                
                if preprune == True:
                    weight[mask.gt(dataset_idx)] = 0.0

    def restore_biases(self, biases):
        """Use the given biases to replace existing biases."""
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if module.bias is not None:
                    module.bias.data.copy_(biases[module_idx])

    ### Standard getter function
    def get_biases(self):
        """Gets a copy of the current biases."""
        biases = {}
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if module.bias is not None:
                    biases[module_idx] = module.bias.data.clone()
        return biases


    ###  This allocates weights masked to 0 (pruned) to the current task index
    def make_finetuning_mask(self, tasknum):
        """Turns previously pruned weights into trainable weights for
           current dataset.
        """
        assert self.previous_masks
        ### This is to allow finetuning on non-pretrained architectures without prematurely incrementing current_dataset_idx
        if(tasknum != 0):
            self.current_dataset_idx += 1
            
            ### Creates the task-specific mask during the initial weight allocation
            task_mask = {}
            for module_idx, module in enumerate(self.model.shared.modules()):
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    task = torch.ByteTensor(module.weight.data.size()).fill_(0)
                    
                    if 'cuda' in module.weight.data.type():
                        task = task.cuda()
                    task_mask[module_idx] = task
            self.all_task_masks[self.current_dataset_idx - 1] = [task_mask]
  
        
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                mask = self.previous_masks[module_idx]
                task = self.all_task_masks[self.current_dataset_idx - 1][0][module_idx]
                
                mask[mask.eq(0)] = self.current_dataset_idx
                ### All shared and newly allocated weights are initially put in the task-specific mask
                task[mask.eq(self.current_dataset_idx)] = 1
                ### This option just initializes the task mask to all 1s, so that they only get pruned during prune()
                # task[mask.ne(0)] = 1
        self.current_masks = self.previous_masks
        print("Exiting finetuning mask")
