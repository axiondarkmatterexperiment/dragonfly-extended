#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#1/27/2020
#send message as a byte stream this time...
#heavily influenced by http://eli.thegreenplace.net/2011/08/02/length-prefix-framing-for-protocol-buffers/import socket

import sys
import struct
import binascii

import asteval
import six

import dripline
import dragonfly

import logging
logger = logging.getLogger('dragonfly.custom.jacob_interface')  

class Jacob_Coordinator(dripline.core.Endpoint):
    '''
    intended to be similar to ethernetprovider.py and SAG coordinator but has the special packet header handling required by 
    the cRio controller in the dilution refrigerator 
    
    utf8 encoding hard coded--maybe change later?
    
    '''


#converting global variables to class members
#specifics will be called by respective yaml


    def __init__(self, hostname=None, port=None, header_size = None, timeout=None ...):
        if hostname is None:
            raise ValueError("jacob connections require a configured hostname")
        else:
            self.hostname=hostname
            
        #...<same for port and whatever else)

        self.port = port
        self.header_size = header_size
        self.timeout = timeout
        
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        #change all s.something to self.socket.something?, that's whave I've done for now...

        
    def connect_to_jacob(self)
        try: 
            #s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket #?
        except socket.error:# ",msg" removed
            print('Failed to create socket. Error code: ' + str(ms[0]) + ',Error message : ' + msg[1])
            
            sys.exit();#establish connection
            s.settimeout(timeout) #another hang proctection, in seconds        
            
        self.socket.connect((hostname, port))#turn message into bytestream with 4 byte prefix

    def send_jacob_a_message(self, the_message)
    
        #where sending acutally starts
        msg_bytes = bytearray(the_message,encoding='utf8')
        bytes_len = struct.pack('>L', len(msg_bytes))#and now try to send the thing
    
    
        try: 
            self.socket.sendall(bytes_len + msg_bytes)
        except socket.error:
            print('Send failed')
            sys.exit()   #more socket hang protection#receiving a fixed, n, amout of bytes, code taken from: 
        
    def recvall(sock,n):
        # Helper function to recv n bytes or return None if EOF is hit
            data = ''
        while len(data) < n:
            packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.append(packet)
        return data #determine message length to receive
    
     def listen_to_jacob(self):
        data_len = recvall(self.socket,header_size)  #just receive header_size byte pre fix
        msg_len = struct.unpack('>L', data_len)[0] #turn length into int#receive actual message, i.e. find out message len
        data_bytes = recvall(self.socket,header_size+msg_len) #receive whole message...
        #print(data_bytes)  #is this still needed?
        return data_bytes
        self.socket.close() 

#not sure how to re-structure the s part of this ...
#how to redefine s???  self.socket?
 
#should this be part of send_a_message func???







