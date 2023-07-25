import pysftp

import socket





class Socket_Hepler_Host():

    """Socket_Hepler"""

    """Host: send action"""

    

    def __init__(self, host = '192.168.1.18', port=22):

        

        # Create a socket object

        self.socket = socket.socket()

        self.host = host

        self.port = port

        

    def activate_server(self):

        # connect to the server on local computer

        self.socket.connect((self.host, self.port))

        

    def Transfer_action(self, action):

        

        # Convert integer to string before sending

        scalar_str = str(action)




        # send a thank you message to the client.

        self.socket.send(scalar_str.encode())




        # close the connection

        self.socket.close()

        

        return print("Data Transfered Successfully")

        

class Socket_Hepler_Client():

    """Socket_Hepler"""

    

    def __init__(self, host = '192.168.20.2', port=9999):

        

        # Create a socket object

        self.socket = socket.socket()

        self.host = host

        self.port = port

        

    def activate_listenting(self):

        # connect to the server on local computer

        self.socket.connect((self.host, self.port))

        

        # Bind to the port

        self.socket.bind(('', self.port))

        print("socket binded to %s" %(self.port))




        # put the socket into listening mode

        self.socket.listen(5)

        print("socket is listening")




        while True:




            # Establish connection with client.

            c, addr = self.socket.accept()

            print('Got connection from', addr)




            # receive the data

            scalar_str = c.recv(1024).decode()

            print('Received scalar value:', scalar_str)




            if scalar_str > 0:

                # Close the connection with the client

                c.close()

                break

            

        return scalar_str

if __name__ == '__main__':
    import zmq
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect('tcp://10.162.9.29:5555')
    socket.send_string('client')
    print('send success')
    rec = socket.recv()
    print(rec)
