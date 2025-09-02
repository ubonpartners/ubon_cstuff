#!/usr/bin/env python3
import socket
import struct
import json
import time
import threading
import sys
import random

# Message type constants (matching your C++ defines)
MSG_TYPE_SDP = 0x01
MSG_TYPE_NAL = 0x02
MSG_TYPE_JSON = 0x03

class MockStreamManager:
    def __init__(self, host='127.0.0.1', port=8081):
        self.host = host
        self.port = port
        self.clients = {}  # fd -> camera_id mapping
        self.running = False
        self.server_socket = None
        
    def start(self):
        """Start the mock STREAM_MANAGER server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True
            
            print(f"Mock STREAM_MANAGER listening on {self.host}:{self.port}")
            
            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    print(f"New connection from {address}")
                    
                    # Handle each client in a separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address),
                        daemon=True
                    )
                    client_thread.start()
                    
                except Exception as e:
                    if self.running:
                        print(f"Error accepting connection: {e}")
                        
        except Exception as e:
            print(f"Error starting server: {e}")
        finally:
            if self.server_socket:
                self.server_socket.close()
    
    def handle_client(self, client_socket, address):
        """Handle communication with a connected ML_WRAPPER client"""
        buffer = b''
        camera_id = None
        
        try:
            while self.running:
                # Read data from client
                data = client_socket.recv(4096)
                if not data:
                    print(f"Client {address} disconnected")
                    break
                print(f"Received {len(data)} bytes from {address}")
                buffer += data
                
                # Process complete messages
                while len(buffer) >= 5:  # Minimum message size
                    msg_type = buffer[0]
                    print(f"Message type: {msg_type}")

                    if msg_type == MSG_TYPE_JSON:
                        # JSON message: [type:1][length:4][payload]
                        payload_length = struct.unpack('!I', buffer[1:5])[0]
                        print(f"Payload length: {payload_length}")

                        if len(buffer) < 5 + payload_length:
                            break  # Wait for complete message
                            
                        json_payload = buffer[5:5+payload_length].decode('utf-8')
                        buffer = buffer[5+payload_length:]
                        
                        print(f"Received JSON from {address}: {json_payload}")
                        camera_id = self.handle_json_message(client_socket, json_payload)

                    else:
                        print(f"Unknown message type: {msg_type}")
                        break
                        
        except Exception as e:
            print(f"Error handling client {address}: {e}")
        finally:
            client_socket.close()
            if camera_id:
                print(f"Stopped streaming for camera {camera_id}")
    
    def handle_json_message(self, client_socket, json_payload):
        """Handle JSON subscription message from ML_WRAPPER"""
        try:
            data = json.loads(json_payload)
            
            if data.get('action') == 'SUBSCRIBE':
                camera_id = data['body']['camera_id']
                stream_type = data['body']['stream_type']
                
                print(f"ML_WRAPPER subscribed to camera {camera_id} for {stream_type}")
                
                # Send SDP first
                self.send_sdp_message(client_socket, camera_id)
                
                # Start sending NAL units in a separate thread
                nal_thread = threading.Thread(
                    target=self.send_real_nal_stream_simple,
                    args=(client_socket, camera_id),
                    daemon=True
                )
                nal_thread.start()
                
                return camera_id
                
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
        except Exception as e:
            print(f"Error processing JSON: {e}")
            
        return None
    
    def send_sdp_message(self, client_socket, camera_id):
        """Send SDP (Session Description Protocol) message"""
        sdp_content = f"""v=0
o=- {random.randint(1000000000, 9999999999)} {random.randint(1000000000, 9999999999)} IN IP4 127.0.0.1
s=Camera {camera_id} Stream
c=IN IP4 239.255.255.255/255
t=0 0
m=video 5004 RTP/AVP 96
a=rtpmap:96 H264/90000
a=fmtp:96 profile-level-id=42e01e
"""
        
        sdp_bytes = sdp_content.encode('utf-8')
        message = struct.pack('!BI', MSG_TYPE_SDP, len(sdp_bytes)) + sdp_bytes
        
        try:
            client_socket.send(message)
            print(f"Sent SDP for camera {camera_id}")
        except Exception as e:
            print(f"Error sending SDP: {e}")
    
    def send_fake_nal_stream(self, client_socket, camera_id):
        """Send simulated NAL units continuously"""
        print(f"Starting NAL stream for camera {camera_id}")
        
        # Simulate different NAL unit types
        nal_units = [
            # H.264 SPS (Sequence Parameter Set)
            b'\x00\x00\x00\x01\x67\x42\x00\x1e\x9a\x74\x0b\x43\x6c',
            # H.264 PPS (Picture Parameter Set)  
            b'\x00\x00\x00\x01\x68\xce\x3c\x80',
            # H.264 IDR Frame
            b'\x00\x00\x00\x01\x65\x88\x84\x00\x33\xff\xfe\xfd\xfc\xfb\xfa',
            # H.264 P-Frame
            b'\x00\x00\x00\x01\x41\x9a\x24\x4d\x40\x20\x08\x44\x56\x78',
            # Another P-Frame
            b'\x00\x00\x00\x01\x41\x9a\x25\x4d\x41\x21\x09\x45\x67\x89',
        ]
        
        frame_counter = 0
        start_time = time.time()
        
        try:
            while self.running:
                current_time = time.time()
                wall_clock_time = int(current_time * 1000)  # milliseconds
                rtp_time = int((current_time - start_time) * 90000)  # 90kHz timestamp
                
                # Cycle through NAL units
                nal_unit = nal_units[frame_counter % len(nal_units)]
                
                # NAL message format: [type:1][wall_clock:8][rtp_time:8][nal_length:4][nal_unit]
                message = struct.pack('!BQQI', 
                                    MSG_TYPE_NAL,
                                    wall_clock_time,
                                    rtp_time, 
                                    len(nal_unit)) + nal_unit
                
                client_socket.send(message)
                
                frame_counter += 1
                if frame_counter % 30 == 0:  # Log every 30 frames
                    print(f"Sent {frame_counter} NAL units for camera {camera_id}")
                
                # Simulate ~30 FPS (33ms between frames)
                time.sleep(0.033)
                
        except Exception as e:
            if self.running:
                print(f"Error sending NAL stream for camera {camera_id}: {e}")
    
    def send_real_nal_stream(self, client_socket, camera_id, nal_file_path):
        """Send real NAL units from file continuously"""
        print(f"Starting real NAL stream for camera {camera_id} from {nal_file_path}")
        
        try:
            with open(nal_file_path, 'rb') as f:
                # Read all NAL packets from file first
                nal_packets = []
                
                while True:
                    # Read timestamp (8 bytes)
                    timestamp_data = f.read(8)
                    if len(timestamp_data) != 8:
                        break
                        
                    # Read NAL unit length (4 bytes)
                    length_data = f.read(4)
                    if len(length_data) != 4:
                        break
                        
                    # Unpack timestamp and length (assuming big-endian from your file format)
                    rtp_timestamp = struct.unpack('>Q', timestamp_data)[0]  # 8-byte big-endian
                    nal_length = struct.unpack('>I', length_data)[0]        # 4-byte big-endian
                    
                    # Read NAL unit data
                    nal_data = f.read(nal_length)
                    if len(nal_data) != nal_length:
                        print(f"Warning: Expected {nal_length} bytes, got {len(nal_data)}")
                        break
                    
                    nal_packets.append({
                        'rtp_timestamp': rtp_timestamp,
                        'nal_data': nal_data
                    })
                    
                    # Debug first few packets
                    if len(nal_packets) <= 5:
                        print(f"Packet {len(nal_packets)}: RTP={rtp_timestamp}, Length={nal_length}")
                
                print(f"Loaded {len(nal_packets)} NAL packets from file")
                
                if not nal_packets:
                    print("No NAL packets found in file")
                    return
                
                # Start streaming the packets
                start_time = time.time()
                first_timestamp = nal_packets[0]['rtp_timestamp']
                packet_index = 0
                
                while self.running:
                    current_packet = nal_packets[packet_index % len(nal_packets)]
                    current_time = time.time()
                    
                    # Calculate wall clock time
                    wall_clock_time = int(current_time * 1000)  # milliseconds
                    
                    # Use original RTP timestamp from file, but adjust for looping
                    loop_number = packet_index // len(nal_packets)
                    if len(nal_packets) > 1:
                        # Calculate time span of original recording
                        last_timestamp = nal_packets[-1]['rtp_timestamp']
                        recording_duration = last_timestamp - first_timestamp
                        adjusted_rtp_time = current_packet['rtp_timestamp'] + (loop_number * recording_duration)
                    else:
                        adjusted_rtp_time = current_packet['rtp_timestamp']
                    
                    # Send NAL message: [type:1][wall_clock:8][rtp_time:8][nal_length:4][nal_unit]
                    message = struct.pack('!BQQI', 
                                        MSG_TYPE_NAL,
                                        wall_clock_time,
                                        adjusted_rtp_time,
                                        len(current_packet['nal_data'])) + current_packet['nal_data']
                    
                    client_socket.send(message)
                    
                    packet_index += 1
                    if packet_index % 30 == 0:  # Log every 30 frames
                        print(f"Sent {packet_index} real NAL units for camera {camera_id}")
                    
                    # Calculate timing for next frame
                    if packet_index < len(nal_packets):
                        # Use original timing between frames
                        next_packet = nal_packets[packet_index % len(nal_packets)]
                        if packet_index % len(nal_packets) == 0:
                            # Loop back to beginning
                            time_diff = 1.0 / 30.0  # Default 30fps
                        else:
                            # Use time difference from file (RTP is 90kHz)
                            rtp_diff = next_packet['rtp_timestamp'] - current_packet['rtp_timestamp']
                            time_diff = rtp_diff / 90000.0
                            
                        # Ensure reasonable frame rate (clamp between 10fps and 60fps)
                        time_diff = max(1.0/60.0, min(1.0/10.0, time_diff))
                        time.sleep(time_diff)
                    else:
                        # Default timing
                        time.sleep(1.0/30.0)
                    
        except FileNotFoundError:
            print(f"Error: NAL file not found: {nal_file_path}")
        except Exception as e:
            print(f"Error reading NAL file: {e}")
                
        except Exception as e:
            if self.running:
                print(f"Error sending real NAL stream for camera {camera_id}: {e}")

    def send_real_nal_stream_simple(self, client_socket, camera_id):
        """Send real NAL units from file once, then stop"""
        if camera_id == "cam001":
            nal_file_path = "~/ubon_cstuff/raw_nal.nal"
        else:
            nal_file_path = "~/ubon_cstuff/raw_nal_207.nal"
        print(f"Starting real NAL stream for camera {camera_id} from {nal_file_path}")
        
        try:
            with open(nal_file_path, 'rb') as f:
                packet_count = 0
                start_time = time.time()
                
                while self.running:
                    # Read timestamp (8 bytes)
                    timestamp_data = f.read(8)
                    if len(timestamp_data) != 8:
                        print(f"End of file reached for camera {camera_id}. Sent {packet_count} NAL units total.")
                        break  # ✅ Stop when file ends instead of restarting
                        
                    # Read NAL unit length (4 bytes)  
                    length_data = f.read(4)
                    if len(length_data) != 4:
                        print(f"End of file reached for camera {camera_id}. Sent {packet_count} NAL units total.")
                        break  # ✅ Stop when file ends
                        
                    # Unpack length
                    nal_length = struct.unpack('>I', length_data)[0]
                    
                    # Read NAL unit data
                    nal_data = f.read(nal_length)
                    if len(nal_data) != nal_length:
                        print(f"End of file reached for camera {camera_id}. Sent {packet_count} NAL units total.")
                        break  # ✅ Stop when file ends
                    
                    # Generate timestamps
                    current_time = time.time()
                    wall_clock_time = int(current_time * 1000)  # milliseconds
                    rtp_time = int((current_time - start_time) * 90000)
                    
                    # Send NAL message
                    message = struct.pack('!BQQI', 
                                        MSG_TYPE_NAL,
                                        wall_clock_time,
                                        rtp_time,
                                        len(nal_data)) + nal_data
                    
                    client_socket.send(message)
                    
                    packet_count += 1
                    if packet_count % 30 == 0:
                        print(f"Sent {packet_count} real NAL units for camera {camera_id}")
                    
                    # Fixed 30fps timing
                    time.sleep(1.0/30.0)
            
            print(f"Finished sending all NAL units for camera {camera_id}. Stream ended.")
                    
        except FileNotFoundError:
            print(f"Error: NAL file not found: {nal_file_path}")
            # Fall back to fake stream if file doesn't exist
            self.send_fake_nal_stream(client_socket, camera_id)
        except Exception as e:
            if self.running:
                print(f"Error sending real NAL stream for camera {camera_id}: {e}")
    
    def stop(self):
        """Stop the mock server"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()

def main():
    mock_server = MockStreamManager()
    
    try:
        print("Starting Mock STREAM_MANAGER...")
        print("This will simulate camera streams for testing your ML_WRAPPER")
        print("Press Ctrl+C to stop")
        mock_server.start()
    except KeyboardInterrupt:
        print("\nShutting down Mock STREAM_MANAGER...")
        mock_server.stop()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()