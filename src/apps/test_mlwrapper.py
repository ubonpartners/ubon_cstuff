#!/usr/bin/env python3
import socket
import struct
import json
import time
import sys

# Message type constants (matching your C++ defines)
MSG_TYPE_SDP = 0x01
MSG_TYPE_NAL = 0x02
MSG_TYPE_JSON = 0x03

def send_json_message(sock, json_data):
    """Send JSON message: [type:1][length:4][json_payload]"""
    json_str = json.dumps(json_data)
    json_bytes = json_str.encode('utf-8')
    
    message = struct.pack('!BI', MSG_TYPE_JSON, len(json_bytes)) + json_bytes
    sock.send(message)
    print(f"Sent JSON message: {json_str} ({len(json_bytes)} bytes)")

def send_nal_message(sock, nal_data, wall_clock_time=None, rtp_time=None):
    """Send NAL message: [type:1][wall_clock:8][rtp_time:8][nal_length:4][nal_payload]"""
    if wall_clock_time is None:
        wall_clock_time = int(time.time() * 1000)  # milliseconds
    if rtp_time is None:
        rtp_time = int(time.time() * 90000)  # 90kHz timestamp
    
    nal_bytes = nal_data.encode('utf-8') if isinstance(nal_data, str) else nal_data
    
    message = struct.pack('!BQQI', MSG_TYPE_NAL, wall_clock_time, rtp_time, len(nal_bytes)) + nal_bytes
    sock.send(message)
    print(f"Sent NAL message: wall_clock={wall_clock_time}, rtp_time={rtp_time}, nal_length={len(nal_bytes)}")

def send_sdp_message(sock, sdp_data):
    """Send SDP message: [type:1][length:4][sdp_payload]"""
    sdp_bytes = sdp_data.encode('utf-8') if isinstance(sdp_data, str) else sdp_data
    
    message = struct.pack('!BI', MSG_TYPE_SDP, len(sdp_bytes)) + sdp_bytes
    sock.send(message)
    print(f"Sent SDP message: {sdp_data if isinstance(sdp_data, str) else 'binary data'} ({len(sdp_bytes)} bytes)")

def send_multiple_messages_in_one_packet(sock):
    """Test sending multiple messages in a single packet"""
    print("\n=== Testing multiple messages in one packet ===")
    
    # Create multiple messages
    json_msg = json.dumps({"test": "multiple", "id": 1}).encode('utf-8')
    sdp_msg = b"v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\n"
    nal_msg = b"\x00\x00\x00\x01\x67\x42\x00\x1e"  # H.264 SPS NAL unit
    
    # Pack them together
    combined = (
        struct.pack('!BI', MSG_TYPE_JSON, len(json_msg)) + json_msg +
        struct.pack('!BI', MSG_TYPE_SDP, len(sdp_msg)) + sdp_msg +
        struct.pack('!BQQI', MSG_TYPE_NAL, 1234567890, 987654321, len(nal_msg)) + nal_msg
    )
    
    sock.send(combined)
    print(f"Sent {len(combined)} bytes containing 3 messages")

def send_large_message(sock):
    """Test a large JSON message (> 4096 bytes)"""
    print("\n=== Testing large message (>4KB) ===")
    
    large_data = {
        "action": "SUBSCRIBE",
        "body": {
            "camera_id": "cam001",
            "stream_type": 10,
            "large_field": "x" * 5000  # 5KB of data
        },
        "metadata": {
            "timestamp": int(time.time()),
            "client_id": "test_client"
        }
    }
    
    send_json_message(sock, large_data)

def test_fragmented_send(sock):
    """Test sending a message in fragments to test buffer accumulation"""
    print("\n=== Testing fragmented message ===")
    
    json_data = {"fragmented": True, "message": "This message will be sent in pieces"}
    json_str = json.dumps(json_data)
    json_bytes = json_str.encode('utf-8')
    
    # Create the full message
    full_message = struct.pack('!BI', MSG_TYPE_JSON, len(json_bytes)) + json_bytes
    
    # Send in fragments
    chunk_size = 10
    for i in range(0, len(full_message), chunk_size):
        chunk = full_message[i:i+chunk_size]
        sock.send(chunk)
        print(f"Sent fragment {i//chunk_size + 1}: {len(chunk)} bytes")
        time.sleep(0.1)  # Small delay to ensure separate recv() calls

def read_responses(sock, timeout=2):
    """Read and display server responses"""
    sock.settimeout(timeout)
    try:
        while True:
            # Read message header
            header = sock.recv(5)
            if len(header) < 5:
                break
                
            msg_type, length = struct.unpack('!BI', header)
            
            # Read payload
            payload = b''
            while len(payload) < length:
                chunk = sock.recv(length - len(payload))
                if not chunk:
                    break
                payload += chunk
            
            if msg_type == MSG_TYPE_JSON:
                try:
                    json_data = json.loads(payload.decode('utf-8'))
                    print(f"Received JSON response: {json_data}")
                except:
                    print(f"Received JSON response: {payload}")
            else:
                print(f"Received response type {msg_type}: {len(payload)} bytes")
                
    except socket.timeout:
        pass
    finally:
        sock.settimeout(None)

def main():
    HOST = '127.0.0.1'
    PORT = 8080
    
    try:
        # Connect to server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, PORT))
        print(f"Connected to {HOST}:{PORT}")
        
        # Test 1: Simple JSON message
        print("\n=== Testing JSON message ===")
        json_data = {
            "action": "SUBSCRIBE",
            "body": {
                "camera_id": "cam001",
                "stream_type": 10
            }
        }
        send_json_message(sock, json_data)
        read_responses(sock)
        
        time.sleep(10000)
        
#         # Test 2: SDP message
#         print("\n=== Testing SDP message ===")
#         sdp_content = """v=0
# o=- 1234567890 1234567890 IN IP4 192.168.1.100
# s=Test Stream
# c=IN IP4 239.255.255.255/255
# t=0 0
# m=video 5004 RTP/AVP 96
# a=rtpmap:96 H264/90000"""
#         send_sdp_message(sock, sdp_content)
        
#         time.sleep(0.5)
        
#         # Test 3: NAL message
#         print("\n=== Testing NAL message ===")
#         # Simulate H.264 NAL unit (SPS)
#         nal_unit = b"\x00\x00\x00\x01\x67\x42\x00\x1e\x9a\x74\x0b\x43\x6c"
#         send_nal_message(sock, nal_unit, wall_clock_time=1692633600000000, rtp_time=152397000)
        
#         time.sleep(0.5)
        
#         # Test 4: Multiple messages in one packet
#         send_multiple_messages_in_one_packet(sock)
        
#         time.sleep(0.5)
        
#         # Test 5: Large message
#         send_large_message(sock)
#         read_responses(sock)
        
#         time.sleep(0.5)
        
#         # Test 6: Fragmented message
#         test_fragmented_send(sock)
#         read_responses(sock)
        
        print("\n=== All tests completed ===")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sock.close()

if __name__ == "__main__":
    main()