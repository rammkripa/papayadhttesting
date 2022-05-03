import torch
import numpy as np
import math
import random

class PapayaClientDistributed:
  # Has a model
  # Has some data as a tensor
  # Has some labels as a tensor
  # Has 
  # Has some optimizer
  # Has loss_function
    #nodes = {}
    class PartnerWeightManager :
        def __init__(self, num_partners, initial_partner_weight) :
            self.num_partners = num_partners
            self.initial_weight = initial_partner_weight
            self.weights = {}
        def get_weight(self, partner_id) :
            if partner_id not in self.weights.keys() :
                self.weights[partner_id] = torch.tensor(self.initial_weight, requires_grad = True)
            return self.weights[partner_id]



    def __init__(self, dat, labs, batch_sz, num_partners, model_class, loss_fn, optimizer_fn=torch.optim.SGD, initial_partner_weight=None):
        ##################################
        ##### MODEL TRAINING THINGS ######
        ##################################
        self.data = dat
        self.labels = labs
        self.batch_size = batch_sz
        # Model
        self.model = model_class()
        # Model Class
        self.model_class = model_class
        # Optimizer
        self.optimizer = optimizer_fn
        # Loss Function
        self.loss = loss_fn()
        self.epochs_trained = 0
        self.current_partners = {}
        self.logs = {}
        #################################
        ######### PARTNER THINGS ########
        #################################
        self.node_id = round(num_partners * 1000 * np.random.rand()) ## change to ip based later
        #self.nodes[self.node_id] = self ## change to distributed hash table
        #################################
        ######### PARTNER WEIGHTS #######
        #################################
        if initial_partner_weight is None :
            initial_partner_weight = (1/(num_partners + 1))
        self.partner_weight_manager = self.PartnerWeightManager(num_partners, initial_partner_weight)
        #################################


    def model_train_epoch(self):
        '''
        Train the model to one epoch
        '''
        ####### SETUP #######
        num_samples = self.data.shape[0]
        num_batches = math.ceil(num_samples / self.batch_size)
        last_loss = 0
        current_optimizer = self.optimizer(self.model.parameters(), lr=0.01)
        ######################
        ### TRAINING LOOP ####
        ######################
        for i in range(num_batches) :
            current_optimizer.zero_grad()
            start_idx = i * self.batch_size
            end_idx = min((i+1) * self.batch_size, num_samples)
            curr_data = self.data[start_idx:end_idx]
            curr_labels = self.labels[start_idx:end_idx]
            output = self.model(curr_data)
            curr_loss = self.loss(output, curr_labels)
            last_loss = curr_loss.item()
            curr_loss.backward()
            current_optimizer.step()
        #######################
        ### LOGGING ###########
        #######################
        if "stringy" not in self.logs.keys() :
            self.logs["stringy"] = {}
        if "epochs" not in self.logs.keys() :
            self.logs["epochs"] = {}
        self.logs["stringy"][self.epochs_trained] = "node" + str(self.node_id) + "epoch " + str(self.epochs_trained) + " loss " + str(last_loss)
        self.logs["epochs"][self.epochs_trained] = last_loss
        self.epochs_trained += 1

    def select_partners(self, num_to_select) :
        '''
        select partners for the averaging process
        LATER : Use Distributed Hash Table to get these values
        '''
        nodes_to_select_from = self.nodes.values()
        neighbours = [i for i in nodes_to_select_from if i is not self]
        assert num_to_select <= len(neighbours)
        self.current_partners = {}
        for n in random.sample(neighbours, num_to_select) :
            self.current_partners[n.node_id] = n.model.state_dict()

    def update_partner_weights(self) :
        '''
        Train the weights for each of the peers chosen to participate in the update
        '''
        ###############################
        ## assign weights and models ##
        ###############################
        weights = {}
        models = {}
        weights_optimizers = {}
        for p in self.current_partners.keys() :
            weights[p] = self.partner_weight_manager.get_weight(p).item()
            weights_optimizers[p] = torch.optim.SGD([self.partner_weight_manager.get_weight(p)], lr=0.0001)
            models[p] = self.model_class()
            models[p].load_state_dict(self.current_partners[p])
        # Calculate weight sum
        weight_sum = sum(list(weights.values()))
        # Do one epoch of stuff
        ###############################
        ##### SETUP ###################
        ###############################
        num_samples = self.data.shape[0]
        num_batches = math.ceil(num_samples / self.batch_size)
        last_loss = 0
        ###############################
        ##### Training Loop ###########
        ###############################
        for i in range(num_batches) :
            for p in self.current_partners.keys() :
                weights_optimizers[p].zero_grad()
            start_idx = i * self.batch_size
            end_idx = min((i+1) * self.batch_size, num_samples)
            curr_data = self.data[start_idx:end_idx]
            curr_labels = self.labels[start_idx:end_idx]
            output = self.model(curr_data) * (1 - weight_sum)
            for p in self.current_partners.keys() :
                curr_output = models[p](curr_data)
                output += self.partner_weight_manager.get_weight(p) * curr_output
            curr_loss = self.loss(output, curr_labels)
            last_loss = curr_loss.item()
            curr_loss.backward()
            for p in self.current_partners.keys() :
                weights_optimizers[p].step()
        ###############################
        ########## LOG ################
        ###############################
        if "mixing" not in self.logs.keys() :
            self.logs["mixing"] = []
        self.logs["mixing"].append(" MIXING UPDATE node " + str(self.node_id) + " loss " + str(last_loss))

    def average_partners(self) :
        '''
        After updating the partner weights, perform the average with partners
        '''
        #assign weights
        weights = {}
        for p in self.current_partners.keys() :
            weights[p] = self.partner_weight_manager.get_weight(p).item()
        # Calculate weight sum for doing the parameter averaging
        weight_sum = sum(list(weights.values()))
        sd_curr = self.model.state_dict()
        for key in sd_curr :
            sd_curr[key] *= (1 - weight_sum)
            for p in self.current_partners.keys() :
                sd_curr[key] += self.current_partners[p][key] * weights[p]
        self.model = self.model_class()
        self.model.load_state_dict(sd_curr)
