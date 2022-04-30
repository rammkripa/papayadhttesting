import btdht

class papayaDHT(btdht.DHT) :
    def on_ping_query(self, query) :
        print("Received Query")
    def on_ping_response(self, query, response) :
        print("Received response")
        if response.addr is not None :
            print ("from", response.addr)

class papayaPartnerWeightManager :
    def __init__(self, byte_id) :
        dht = papayaDHT(id = byte_id)
        
    def ping_node_query(self, node_ip, node_port) :
        msg = btdht.krcp.BMessage()
        
    