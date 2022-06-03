import io
import argparse
import logging
import asyncio
import socket
from papayaclientdistributed import PapayaClientDistributed
import torch 
import torchvision
from model import TheModel
import numpy as np
from zlib import compress, decompress


from kademlia.network import Server

handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
log = logging.getLogger('kademlia')
log.addHandler(handler)
log.setLevel(logging.DEBUG)

server = Server()


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Optional arguments
    parser.add_argument("-i", "--ip", help="IP address of existing node", type=str, default=None)
    parser.add_argument("-p", "--port", help="port number of existing node", type=int, default=None)
    parser.add_argument("-pp", "--presentport", help="port number of present node", type=int, default=None)
    parser.add_argument("-np", "--numpartners", help="number of partners", type=int, default=None)
    parser.add_argument("-pn", "--partnernumber", help="number of present partner", type=int, default=None)
    return parser.parse_args()


def connect_to_bootstrap_node(loop, args):
    loop.run_until_complete(server.listen(args.presentport))
    bootstrap_node = (args.ip, int(args.port))
    loop.run_until_complete(server.bootstrap([bootstrap_node]))

def listen(loop, args) :
    loop.run_until_complete(server.listen(args.presentport))

def create_bootstrap_node(loop, args):
    loop.run_until_complete(server.listen(args.presentport))

def put(loop, key, value) :
    loop.run_until_complete(server.set(key, value))

def get(loop, key) :
    return loop.run_until_complete(server.get(key))

def bytes_to_state_dict(raw_bytes) :
    decompressed_bytes = decompress(raw_bytes) # decompression
    buffer = io.BytesIO(decompressed_bytes)
    state_dict = torch.load(buffer)
    #print('state dict is  ' + str(state_dict))
    return state_dict
    
def strings_to_state_dict(raw_string) :
    buffer = io.StringIO(raw_string)
    state_dict = torch.load(buffer)
    return state_dict

def main():
    
    args = parse_arguments()
    loop = asyncio.get_event_loop()
    loop.set_debug(True)

    if args.ip and args.port:
        connect_to_bootstrap_node(loop, args)
        ####### GRAB THE DATA: TO BE CHANGED
        
        mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
        mnist_testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
        ###########################################
        ########## Set up for training
        
        batch_size_train = 60000 // (args.numpartners + 1)
        batch_size_test = 500
        train_loader = torch.utils.data.DataLoader(mnist_trainset,batch_size=batch_size_train, shuffle=True)
        test_loader = torch.utils.data.DataLoader(mnist_testset,batch_size=batch_size_test, shuffle=True)
        ######### Hacky way of setting data ########
        i = 0
        client = None
        for batchno, (ex_data, ex_labels) in enumerate(train_loader):
            if i == args.partnernumber :
                client = PapayaClientDistributed(dat = ex_data,
                                            labs = ex_labels.float(),
                                            batch_sz = 500,
                                            num_partners = args.numpartners,
                                            model_class = TheModel,
                                            loss_fn = torch.nn.MSELoss)
            i+=1
        ##############################################
        num_epochs_total = 20
        num_epochs_per_swap = 5
        num_times = (num_epochs_total // num_epochs_per_swap)
        key = args.ip + "X" + str(args.presentport)
        
        #############################################
        ####### Training loop #######################
        
        keys = []
        for i in server.bootstrappable_neighbors() :
            ip2 = i[0]
            port2 = i[1]
            if ip2 != args.ip and port2 != args.port :
                keys.append(str(ip2) + "X" + str(port2))
                print(str(ip2) + "X" + str(port2))
        
        for i in range(0, num_times):
            ### Train model to five epochs  
            for j in range(0, num_epochs_per_swap):
                client.model_train_epoch()
                
            # Put model state dict in hash table
            bytesbuffer = io.BytesIO()
            
            torch.save(client.model.half().state_dict(), bytesbuffer)
            compressed_bytes = compress(bytesbuffer.getvalue(), level = 9)# compression scheme
            print(str(len(compressed_bytes)) + " is the compressed length and the original length is " + str(len(bytesbuffer.getvalue())))
            
            put(loop, key, compressed_bytes)
            client.model.float()
            
            
            #############################
            ###### partner averaging ####
            #############################
            
            if i > 1 and i < num_times - 1 :
                # LATER: select random sample of keys
                #num_to_select = 3
                #partners = random.sample(keys, num_to_select)
                print(str(keys) + " is keyz")
                partners = keys
                client.current_partners = {}
                for partner_key in partners :
                    client.current_partners[partner_key] = bytes_to_state_dict(get(loop, partner_key))
                for j in range(0, 4) :
                    randomthing = j + 2
                    client.update_partner_weights()
                client.average_partners()
            
            ## PRINT THE WHOLE HASH TABLE of neighbours
            print(" PRINTING NEIGHBOURS ")
            for i in server.bootstrappable_neighbors() :
                ip2 = i[0]
                port2 = i[1]
                print (ip2 + " is ip and port is " + str(port2))
                pres_key = str(ip2) + "X" + str(port2)
                if not (ip2 == args.ip and port2 == args.port) and pres_key not in keys:
                    keys.append(pres_key)
                    
        ## MEASURE ACCURACY WRT TEST SET
        accuracies_node = []
        with torch.no_grad():
            for batchno, (ex_data, ex_labels) in enumerate(test_loader) :
                accuracies_node.append(((client.model.forward(ex_data).round() == ex_labels).float().mean()).item())
        print('mean accuracy is ' + str(np.array(accuracies_node).mean()))
        #print(client.model.state_dict())
        
    else:
        create_bootstrap_node(loop, args)
        
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        loop.close() 


if __name__ == "__main__":
    main()