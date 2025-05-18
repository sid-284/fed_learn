import socket
import pickle
import struct
import time

def send_model(model, host='localhost', port=9999):
    """
    Send a model to a server.
    
    Args:
        model: The model to send
        host: Server hostname
        port: Server port
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Serialize the model
        serialized_model = pickle.dumps(model)
        
        # Create a socket connection
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            print(f"Connecting to {host}:{port}...")
            s.connect((host, port))
            
            # Send the size of the serialized model first
            s.sendall(struct.pack('!I', len(serialized_model)))
            
            # Send the serialized model
            s.sendall(serialized_model)
            
            # Wait for acknowledgment
            ack = s.recv(3)
            if ack == b'ACK':
                print("Model sent successfully")
                return True
            else:
                print("Failed to receive acknowledgment")
                return False
    
    except Exception as e:
        print(f"Error sending model: {e}")
        return False

def receive_model(port=9999):
    """
    Receive a model from a client.
    
    Args:
        port: Port to listen on
        
    Returns:
        The received model, or None if an error occurred
    """
    try:
        # Create a socket server
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('0.0.0.0', port))
            s.listen(1)
            
            print(f"Listening for connections on port {port}...")
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                
                # Receive the size of the serialized model
                size_data = conn.recv(4)
                size = struct.unpack('!I', size_data)[0]
                
                # Receive the serialized model
                data = b''
                while len(data) < size:
                    packet = conn.recv(min(4096, size - len(data)))
                    if not packet:
                        break
                    data += packet
                
                # Send acknowledgment
                conn.sendall(b'ACK')
                
                # Deserialize the model
                model = pickle.loads(data)
                print(f"Received model of size {len(data)} bytes")
                return model
    
    except Exception as e:
        print(f"Error receiving model: {e}")
        return None

def start_server(port=9999):
    """
    Start a server to receive models.
    
    Args:
        port: Port to listen on
        
    Returns:
        A list of received models
    """
    models = []
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind(('0.0.0.0', port))
        server_socket.listen(5)
        print(f"Server started on port {port}. Waiting for connections...")
        
        server_socket.settimeout(60)  # 60 second timeout
        
        while True:
            try:
                client_socket, address = server_socket.accept()
                print(f"Connection from {address}")
                
                # Receive the size of the serialized model
                size_data = client_socket.recv(4)
                if not size_data:
                    client_socket.close()
                    continue
                    
                size = struct.unpack('!I', size_data)[0]
                
                # Receive the serialized model
                data = b''
                while len(data) < size:
                    packet = client_socket.recv(min(4096, size - len(data)))
                    if not packet:
                        break
                    data += packet
                
                # Send acknowledgment
                client_socket.sendall(b'ACK')
                
                # Deserialize the model
                model = pickle.loads(data)
                models.append(model)
                print(f"Received model {len(models)} of size {len(data)} bytes")
                
                client_socket.close()
                
            except socket.timeout:
                print("Server timeout. No more connections.")
                break
                
    except Exception as e:
        print(f"Server error: {e}")
    
    finally:
        server_socket.close()
        
    return models 