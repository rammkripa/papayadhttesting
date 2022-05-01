import argparse
import logging
import asyncio
import socket

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
    
def main():
    
    args = parse_arguments()
    loop = asyncio.get_event_loop()
    loop.set_debug(True)

    if args.ip and args.port:
        connect_to_bootstrap_node(loop, args)
        while True :
            #listen(loop, args)
            choice = input("Keep adding stuff to the hash table? [y/n]")
            if choice == 'n' :
                break
            value = input('Input a value : ')
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            key = args.ip + "X" + str(args.presentport)
            if key and value:
                put(loop, key, value)
            ## PRINT THE WHOLE HASH TABLE
            
            print(" PRINTING HASH TABLE NEIGHBOURS ")
            for i in server.bootstrappable_neighbors() :
                ip2 = i[0]
                port2 = i[1]
                print (ip2 + " is ip and port is " + str(port2))
                '''
                keyy = str(i[0]) + "X" + str(i[1])
                print(" key is " + keyy + " value is " + get(loop, keyy))
            '''
            ### GET
            get_key = input("Input some key")
            print(" key is " + get_key + " value is " + get(loop, get_key))
        
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