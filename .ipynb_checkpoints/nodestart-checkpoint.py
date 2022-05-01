import argparse
import logging
import asyncio

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
    else:
        create_bootstrap_node(loop, args)
    
    key = input('Input a key : ')
    value = input('Input a value : ')
    
    if key and value:
        put(loop, key, value)
        
    get_key = input('Enter key to get : ')
    
    if get_key :
        print('Got value: ', get(get_key))
        
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        loop.close() 


if __name__ == "__main__":
    main()