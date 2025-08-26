#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <map>
#include <cstring>
#include <cerrno>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/epoll.h>
#include <json/json.h>
#include <set>
#include <thread>

#include <readerwriterqueue/readerwriterqueue.h>
#include <track.h>
#include "display.h"
#include "jpeg.h"
#include <misc.h>
#include <cuda_stuff.h>
#include <yaml_stuff.h>
#include <profile.h>
#include <functional>

#define MAX_EVENTS 30
#define PORT 8080
#define STREAM_MANAGER_PORT 8081
#define STREAM_MANAGER_IP "127.0.0.1"

#define MSG_TYPE_SDP 0x01
#define MSG_TYPE_NAL 0x02
#define MSG_TYPE_JSON 0x03

#define POISON_PILL 0xFF

#define ACTION_SUBSCRIBE "SUBSCRIBE"

#define STREAM_TYPE_INFERENCE 0x0A
#define STREAM_TYPE_THUMBNAIL 0x0B
#define STREAM_TYPE_BESTFACE 0x0C

static inline uint64_t htonll(uint64_t value) {
#if __BYTE_ORDER == __LITTLE_ENDIAN
    return ((uint64_t)htonl(value & 0xFFFFFFFFULL) << 32) |
           htonl(value >> 32);
#else
    return value;
#endif
}

static inline uint64_t ntohll(uint64_t value) {
#if __BYTE_ORDER == __LITTLE_ENDIAN
    return ((uint64_t)ntohl(value & 0xFFFFFFFFULL) << 32) |
           ntohl(value >> 32);
#else
    return value;
#endif
}

// Structure to hold remote server information for outgoing connections
struct RemoteServer {
    std::string ip;
    int port;
    std::string camera_id; // We make a separate connection request per camera
};

// Structure to hold client information
struct ClientInfo {
    sockaddr_in addr;
    // You can add more client-specific data here
};

// ======================== GLOBAL CONTEXT VARIABLES ==========================
int server_fd, epoll_fd;
std::map<int, ClientInfo> clients;
std::map<int, RemoteServer> servers;
std::map<int, std::vector<char>> client_buffers;
std::map<int, std::vector<char>> server_buffers;
std::map<int, std::queue<std::vector<char>>> server_send_queues;
std::map<int, std::queue<std::vector<char>>> client_send_queues;
std::map<std::string, std::set<std::pair<int, uint8_t>>> camera_subscribers; //map< camera_id, set< client_fd, stream_type > >
std::map<std::string, std::thread> camera_threads;
std::map<std::string, moodycamel::BlockingReaderWriterQueue<std::vector<char>>> camera_queues; // map< camera_id, queue of NAL units >
std::map<std::string, std::map<uint64_t, uint64_t>> camera_timestamp_map; // map< camera_id, map< rtp timestamp, wallclock timestamp > >

// ====================== HELPER FUNCTIONS ======================
bool set_non_blocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags == -1) {
        perror("fcntl(F_GETFL)");
        return false;
    }
    if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) == -1) {
        perror("fcntl(F_SETFL)");
        return false;
    }
    return true;
}

void cleanup_incoming_socket(int fd) {
    std::cout << "Cleaning up incoming socket fd " << fd << std::endl;
    epoll_ctl(epoll_fd, EPOLL_CTL_DEL, fd, nullptr);
    close(fd);
    clients.erase(fd);
    client_buffers.erase(fd);
    client_send_queues.erase(fd);
}

void cleanup_outgoing_socket(int fd) {
    std::cout << "Cleaning up outgoing socket fd " << fd << std::endl;
    epoll_ctl(epoll_fd, EPOLL_CTL_DEL, fd, nullptr);
    close(fd);
    servers.erase(fd);
    server_buffers.erase(fd);
    server_send_queues.erase(fd);
}

int send_complete_message(int fd, std::vector<char>& message) {
    size_t total_sent = 0;
    const char* data = message.data();
    size_t remaining = message.size();
    
    while (remaining > 0) {
        ssize_t bytes_sent = send(fd, data + total_sent, remaining, MSG_NOSIGNAL);
        
        if (bytes_sent == -1) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // Socket buffer full
                std::cout << "Socket buffer full, sent " << total_sent << "/" << message.size() << " bytes" << std::endl;
                break; // Partial send
            }
            perror("send failed");
            return -1;
        }

        message.erase(message.begin(), message.begin() + bytes_sent);
        total_sent += bytes_sent;
        remaining -= bytes_sent;
    }
    std::cout << "Sent " << total_sent << " bytes on fd " << fd << std::endl;
    return message.empty();
}

int connect_to_server(const RemoteServer& server) {
    // check if we are already connected
    for (const auto& [fd, serverInfo] : servers) {
        if (serverInfo.ip == server.ip && serverInfo.port == server.port && serverInfo.camera_id == server.camera_id) {
            return fd;
        }
    }

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket (client)");
        return -1;
    }

    if (!set_non_blocking(server_fd)) {
        close(server_fd);
        return -1;
    }

    struct sockaddr_in remote_addr;
    std::memset(&remote_addr, 0, sizeof(remote_addr));
    remote_addr.sin_family = AF_INET;
    remote_addr.sin_port = htons(server.port);
    if (inet_pton(AF_INET, server.ip.c_str(), &remote_addr.sin_addr) <= 0) {
        perror("inet_pton");
        close(server_fd);
        return -1;
    }

    std::cout << "Attempting connection to " << server.ip << ":" << server.port << " on fd " << server_fd << std::endl;
    int ret = connect(server_fd, (struct sockaddr*)&remote_addr, sizeof(remote_addr));
    if (ret < 0 && errno != EINPROGRESS) {
        perror("connect");
        close(server_fd);
        return -1;
    }

    struct epoll_event event;
    event.data.fd = server_fd;
    event.events = EPOLLIN | EPOLLOUT | EPOLLET; // Monitor for read and write (for connection status)
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, server_fd, &event) == -1) {
        perror("epoll_ctl (add outgoing)");
        close(server_fd);
        return -1;
    }

    servers[server_fd] = server;
    return server_fd;
}

int send_message_to_fd(int fd, std::queue<std::vector<char>>& fd_send_buffer_queue, const std::vector<char>& message){
    if (fd < 0) {
        std::cerr << "Invalid file descriptor" << std::endl;
        return -1;
    }

    fd_send_buffer_queue.push(message);
    // Modify epoll to monitor EPOLLOUT when send buffer is not empty
    struct epoll_event event;
    event.data.fd = fd;
    event.events = EPOLLIN | EPOLLOUT | EPOLLET;
    epoll_ctl(epoll_fd, EPOLL_CTL_MOD, fd, &event);
    std::cout << "Scheduled message of length " << message.size() << " to fd " << fd << std::endl;
    return 0;
}

// ====================== APPLICATION LOGIC ======================
typedef struct state
{
    track_shared_state_t *tss;
    track_stream_t *ts;
    image_t *img;
    double start_time;
    std::string camera_id;
    bool stopped;
} state_t;

static void track_result(void *context, track_results_t *r)
{
    state_t *s=(state_t *)context;
    if (s->stopped)
    {
        return;
    }

    printf("camera %s track_result: result type %d time %f, %d detections\n", s->camera_id.c_str(), r->result_type, r->time, (r->track_dets==0) ? 0 : r->track_dets->num_detections);

    if (camera_timestamp_map[s->camera_id].find(r->time) == camera_timestamp_map[s->camera_id].end())
    {
        std::cout << " ======================== No wallclock timestamp found for rtp timestamp " << r->time << " for camera " << s->camera_id << std::endl;
    }

    if (r->track_dets!=0)
    {
        //detection_list_show(r->track_dets);
        // decode and display "frame jpeg"
        image_t *img=0;
        if (r->track_dets->frame_jpeg!=0)
        {
            size_t jpeg_data_length;
            uint8_t *jpeg_data=jpeg_get_data(r->track_dets->frame_jpeg, &jpeg_data_length);
            img=decode_jpeg(jpeg_data, (int)jpeg_data_length);
        }
        if (img!=0)
        {
            image_t *out_frame_rgb=detection_list_draw(r->track_dets, img);
            if (out_frame_rgb!=0)
            {
                display_image("video", out_frame_rgb);
                image_destroy(out_frame_rgb);
            }
            image_destroy(img);
        }
    }
}

void camera_worker(std::string camera_id) {
    std::cout << "Camera worker thread started for camera_id: " << camera_id << std::endl;

    // ============================== CODE COPED FROM nal_play.c ==============================
    printf("ubon_cstuff version = %s\n", ubon_cstuff_get_version());

    state_t s;
    memset(&s, 0, sizeof(state_t));

    // modify default config so we get 720p30 'frame jpegs'
    // just so we can display these

    const char *basic="main_jpeg:\n"
                       "    enabled: true\n"
                       "    max_width: 1280\n"
                       "    max_height: 720\n"
                       "    min_interval_seconds: 0.03\n";

    const char *config=yaml_merge_string("/mldata/config/track/trackers/uc_reid.yaml", basic);

    s.tss=track_shared_state_create(config);
    s.ts=track_stream_create(s.tss, &s, track_result);
    s.start_time=profile_time();
    s.camera_id = camera_id;
    track_stream_set_minimum_frame_intervals(s.ts, 0.01, 10.0);

    int nalu_buf_size=1024*1024;
    uint8_t *nalu_buf=(uint8_t *)malloc(nalu_buf_size);
    uint64_t first_timestamp=0;
    bool first_nalu=true;
    double actual_start_time=profile_time();

    while (true){
        std::vector<char> nal_packet;
        camera_queues[camera_id].wait_dequeue(nal_packet);

        // Process the NAL packet (call track_stream_add_nalus method in track.h)
        std::cout << "Camera worker thread for camera_id: " << camera_id 
                  << " processing NAL packet of size " << nal_packet.size() << std::endl;

        if (nal_packet.size() < 12){
            // if nal_packet is too short, assume its poison pill
            std::cout << "Camera worker thread for camera_id: " << camera_id << " received poison pill. Exiting." << std::endl;
            s.stopped = true;
            break;
        }

        // NAL packet scheme: [8 bytes wallclock timestamp][8 bytes rtp timestamp][4 bytes nal unit length][nal unit]
        uint64_t wallclock_timestamp = ntohll(*reinterpret_cast<uint64_t*>(nal_packet.data()));
        uint64_t rtp_extended_timestamp = ntohll(*reinterpret_cast<uint64_t*>(nal_packet.data() + 8));
        uint32_t nal_unit_length = ntohl(*reinterpret_cast<uint32_t*>(nal_packet.data() + 16));
        assert(nal_unit_length < 1024 * 1024);

        if (first_nalu)
        {
            first_nalu=false;
            first_timestamp=rtp_extended_timestamp;
        }
        double time=(rtp_extended_timestamp-first_timestamp)/90000.0;

        while(profile_time()-actual_start_time<time) usleep(1000); // try to play in roughly realtime

        nalu_buf[0]=(nal_unit_length>>24)&0xff;
        nalu_buf[1]=(nal_unit_length>>16)&0xff;
        nalu_buf[2]=(nal_unit_length>>8)&0xff;
        nalu_buf[3]=(nal_unit_length>>0)&0xff;

        // add nal_unit into nalu_buf
        std::memcpy(&nalu_buf[4], nal_packet.data() + 20, nal_unit_length);

        // Add the NAL unit to the track stream
        // TODO: READ THE CODEC FROM THE SDP
        // Precision shouldnt be problem unless frames have very similar timestamps
        double rtp_timestamp = rtp_extended_timestamp / 90000.0;
        camera_timestamp_map[camera_id][rtp_timestamp] = wallclock_timestamp;
        track_stream_add_nalus(s.ts, rtp_timestamp, nalu_buf, nal_unit_length + 4, false);
    }
    camera_queues.erase(camera_id);

    track_stream_sync(s.ts);
    const char *stream_stats=track_stream_get_stats(s.ts);
    const char *shared_state_stats=track_shared_state_get_stats(s.tss);
    printf("======== SHARED TRACK STATS ===========\n");
    printf("%s\n\n",shared_state_stats);
    free((void*)shared_state_stats);
    printf("======== STREAM STATS ===========\n");
    printf("%s\n\n", stream_stats);
    free((void*)stream_stats);
    return;
}

void start_camera_thread(const std::string& camera_id) {
    if (camera_threads.find(camera_id) == camera_threads.end()) {
        camera_threads[camera_id] = std::thread(camera_worker, camera_id);
    }
}

void stop_camera_thread(const std::string& camera_id) {
    auto it = camera_threads.find(camera_id);
    if (it != camera_threads.end()) {
        // Send poison pill to the camera queue to signal the thread to exit
        camera_queues[camera_id].enqueue(std::vector<char>(1, POISON_PILL));
        it->second.join();
        camera_threads.erase(it);
        std::cout << "Stopped camera thread for " << camera_id << std::endl;
    }
}

void unsubscribe_to_camera(const std::string camera_id){
    std::cout << "Camera " << camera_id << " unavailable. Disconnecting" << std::endl;
    stop_camera_thread(camera_id);
    std::vector<int> servers_to_cleanup;
    std::vector<int> clients_to_cleanup;


    for (const auto&[client_fd, stream_type] : camera_subscribers[camera_id]) {
        clients_to_cleanup.push_back(client_fd);
    }
    for (auto &[server_fd, server_info] : servers) {
        if (server_info.camera_id == camera_id) {
            servers_to_cleanup.push_back(server_fd);
        }
    }
    
    for (int fd : clients_to_cleanup) {
        cleanup_incoming_socket(fd);
    }
    for (int fd : servers_to_cleanup) {
        cleanup_outgoing_socket(fd);
    }
    camera_subscribers.erase(camera_id);
}

void handle_client_disconnection(int client_fd) {
    std::vector<std::string> cameras_to_unsubscribe;
    for (auto &[camera_id, subscribers] : camera_subscribers) {
        for (auto sub_it = subscribers.begin(); sub_it != subscribers.end(); ) {
            if (sub_it->first == client_fd) {
                sub_it = subscribers.erase(sub_it);
            } else {
                ++sub_it;
            }
        }
        if (subscribers.empty()) {
            cameras_to_unsubscribe.push_back(camera_id);
        }
    }

    for (const auto& camera_id : cameras_to_unsubscribe) {
        unsubscribe_to_camera(camera_id);
    }
    cleanup_incoming_socket(client_fd);
}

void handle_server_disconnection(int server_fd){
    if (servers.count(server_fd)) {
        std::string camera_id = servers[server_fd].camera_id;
        unsubscribe_to_camera(camera_id);
        cleanup_outgoing_socket(server_fd);
    }
}

void subscribe_to_camera(int client_fd, const std::string& camera_id, uint8_t stream_type) {
    if (camera_subscribers[camera_id].empty()) {    
        // Json Message Scheme: [1 byte type][4 bytes payload_length][payload]
        Json::Value json_payload;
        json_payload["action"] = ACTION_SUBSCRIBE;
        json_payload["body"] = Json::Value();
        json_payload["body"]["camera_id"] = camera_id;
        json_payload["body"]["stream_type"] = "nal";
        
        Json::StreamWriterBuilder builder;
        std::string response_str = Json::writeString(builder, json_payload);
        uint32_t payload_length = htonl(response_str.length());
        char* length_bytes = reinterpret_cast<char*>(&payload_length);
        
        std::vector<char> message;
        message.reserve(5 + response_str.length());
        message.push_back(MSG_TYPE_JSON);
        message.insert(message.end(), length_bytes, length_bytes + 4);
        message.insert(message.end(), response_str.begin(), response_str.end());

        int server_fd = connect_to_server({STREAM_MANAGER_IP, STREAM_MANAGER_PORT, camera_id});
        if (server_fd == -1) {
            std::cerr << "Connection to STREAM_MANAGER failed" << std::endl;
            handle_server_disconnection(server_fd);
            return;
        } else if(send_message_to_fd(server_fd, server_send_queues[server_fd], message) == -1) {
            std::cerr << "Failed to schedule message to STREAM_MANAGER for camera " << camera_id << std::endl;
            return;
        }
        start_camera_thread(camera_id);
    }
    camera_subscribers[camera_id].insert({client_fd, stream_type});
    std::cout << "Client " << client_fd << " subscribed to camera " << camera_id
    << " with stream type " << (int)stream_type << std::endl;
}

void handle_json_message(int client_fd, const std::string& json_payload) {
    Json::Value root;
    Json::Reader reader;

    if (!reader.parse(json_payload, root)) {
        std::cerr << "Failed to parse JSON: " << reader.getFormattedErrorMessages() << std::endl;
        return;
    }

    if (root.isMember("action") && root["action"].asString() == ACTION_SUBSCRIBE) {
        std::string camera_id = root["body"]["camera_id"].asString();
        uint8_t stream_type = root["body"]["stream_type"].asUInt();
        subscribe_to_camera(client_fd, camera_id, stream_type);
    } else {
        std::cerr << "Invalid JSON payload: missing or unknown 'action' field." << std::endl;
    }
    
    // Example: Send JSON response
    Json::Value response;
    response["status"] = "success";
    response["message"] = "JSON received";
    Json::StreamWriterBuilder builder;
    std::string response_str = Json::writeString(builder, response);

    uint8_t type_byte = MSG_TYPE_JSON;
    uint32_t length = htonl(response_str.length());

    write(client_fd, &type_byte, 1);
    write(client_fd, &length, 4);
    write(client_fd, response_str.c_str(), response_str.length());
}

void handle_incoming_message(int fd, std::vector<char>& rcv_buffer, void handle_disconnect(int)) {
    char buffer[4096];
    ssize_t bytes_read;
    while ((bytes_read = read(fd, buffer, sizeof(buffer))) > 0) {
    rcv_buffer.insert(rcv_buffer.end(), buffer, buffer + bytes_read);
    }

    if (bytes_read == 0) {
    std::cout << "Connection closed on fd " << fd << " disconnected." << std::endl;
    handle_disconnect(fd);
    } else if (bytes_read == -1 && errno != EAGAIN && errno != EWOULDBLOCK) {
    perror("read error");
    handle_disconnect(fd);
    }

    if (rcv_buffer.empty()){
        return; // No data to process
    }

    int message_processed = 1;
    while (message_processed){
        message_processed = 0;
        
        uint8_t type_byte = rcv_buffer[0];
        switch (type_byte) {
            case MSG_TYPE_JSON:
            // Json Message Scheme: [1 byte type][4 bytes payload_length][payload]
            {
                if (rcv_buffer.size() < 5) {
                    break; // Not enough data for the full message
                } else {
                    uint32_t payload_length;
                    std::memcpy(&payload_length, &rcv_buffer[1], 4);
                    payload_length = ntohl(payload_length);
                    
                    if (rcv_buffer.size() < 5 + payload_length) {
                        break; // Not enough data for the full message
                    }

                    std::string json_payload(rcv_buffer.begin() + 5, rcv_buffer.begin() + 5 + payload_length);
                    handle_json_message(fd, json_payload);
                    rcv_buffer.erase(rcv_buffer.begin(), rcv_buffer.begin() + 5 + payload_length);
                    message_processed = 1;
                    std::cout << "Processed JSON message of length " << payload_length << std::endl;
                }
                break;
            }
            case MSG_TYPE_NAL:
            // Nal Message Scheme: [1 byte type][8 bytes wall clock time][8 bytes rtp time][4 bytes nal unit length][nal unit]
                if (rcv_buffer.size() < 21) {
                    break; // Not enough data for the full message
                } else {
                    uint32_t nal_unit_length;
                    std::memcpy(&nal_unit_length, &rcv_buffer[17], 4);
                    nal_unit_length = ntohl(nal_unit_length);

                    if (rcv_buffer.size() < 21 + nal_unit_length) {
                        break; // Not enough data for the full message
                    }

                    std::vector<char> nal_data(rcv_buffer.begin() + 1, rcv_buffer.begin() + 21 + nal_unit_length);
                    rcv_buffer.erase(rcv_buffer.begin(), rcv_buffer.begin() + 21 + nal_unit_length);
                    message_processed = 1;
                    std::cout << "Processed NAL message of length " << nal_unit_length << std::endl;
                    if (servers.find(fd) != servers.end()) {
                        camera_queues[servers[fd].camera_id].enqueue(nal_data);
                    }
                }
                break;
            case MSG_TYPE_SDP:
            // SDP Message Scheme: [1 byte type][4 bytes payload_length][payload]
                if (rcv_buffer.size() < 5) {
                    break; // Not enough data for the full message
                } else {
                    uint32_t payload_length;
                    std::memcpy(&payload_length, &rcv_buffer[1], 4);
                    payload_length = ntohl(payload_length);
                    
                    if (rcv_buffer.size() < 5 + payload_length) {
                        break; // Not enough data for the full message
                    }

                    std::vector<char> sdp_payload(rcv_buffer.begin() + 5, rcv_buffer.begin() + 5 + payload_length);
                    rcv_buffer.erase(rcv_buffer.begin(), rcv_buffer.begin() + 5 + payload_length);
                    message_processed = 1;
                    std::cout << "Processed SDP message of length " << payload_length << std::endl;
                }
                break;
            default:
                std::cerr << "Unknown message type" << std::endl;
                break;
        }
    }
}

void handle_outgoing_message(int fd, std::queue<std::vector<char>>& send_queue, void handle_disconnection(int)) {
    std::cout << "Handling outgoing messages for fd " << fd << " with send queue size: " << send_queue.size() << std::endl;
    while (!send_queue.empty()) {
        auto& message = send_queue.front();
        int message_sent = send_complete_message(fd, message);
        if (message_sent == 1) {
            send_queue.pop();
        } else if (message_sent == 0) {
            break; // Partial send, try again later
        } else {
            handle_disconnection(fd);
        }
    }
    std::cout << "Completed sending messages on fd " << fd << " with send queue size: " << send_queue.size() << std::endl;
    // If send queue is empty, stop monitoring EPOLLOUT
    if (send_queue.empty()) {
        struct epoll_event event;
        event.data.fd = fd;
        event.events = EPOLLIN | EPOLLET;
        epoll_ctl(epoll_fd, EPOLL_CTL_MOD, fd, &event);
    }
}

int main() {
    init_cuda_stuff();
    image_init();

    // ====================== SERVER SETUP ======================
    struct sockaddr_in server_addr;
    
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed"); exit(EXIT_FAILURE);
    }
    
    int opt = 1;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
        perror("setsockopt"); exit(EXIT_FAILURE);
    }
    
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);
    
    if (bind(server_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind failed"); exit(EXIT_FAILURE);
    }
    
    if (listen(server_fd, 3) < 0) {
        perror("listen"); exit(EXIT_FAILURE);
    }
    
    if (!set_non_blocking(server_fd)) { exit(EXIT_FAILURE); }
    
    std::cout << "Server listening on port " << PORT << std::endl;
    
    // ====================== EPOLL SETUP ======================
    struct epoll_event event, events[MAX_EVENTS];
    if ((epoll_fd = epoll_create1(0)) == -1) {
        perror("epoll_create1"); exit(EXIT_FAILURE);
    }

    event.events = EPOLLIN | EPOLLET;
    event.data.fd = server_fd;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, server_fd, &event) == -1) {
        perror("epoll_ctl"); exit(EXIT_FAILURE);
    }


    // ====================== MAIN EVENT LOOP ======================
    while (true) {
        int num_events = epoll_wait(epoll_fd, events, MAX_EVENTS, -1);
        if (num_events == -1) {
            perror("epoll_wait"); break;
        }

        for (int i = 0; i < num_events; ++i) {
            int current_fd = events[i].data.fd;

            if ((events[i].events & EPOLLERR) || (events[i].events & EPOLLHUP)) {
                std::cerr << "Epoll error on fd " << current_fd << std::endl;
                if (clients.count(current_fd)) {
                    handle_client_disconnection(current_fd);
                } else if (servers.count(current_fd)) {
                    handle_server_disconnection(current_fd);
                }
                continue;
            }

            if (current_fd == server_fd) {
                // =========== 1. HANDLE NEW INCOMING CLIENT CONNECTIONS ===========
                while(true) {
                    struct sockaddr_in client_addr;
                    socklen_t client_addr_len = sizeof(client_addr);
                    int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_addr_len);

                    if (client_fd == -1) {
                        if (errno == EAGAIN || errno == EWOULDBLOCK) break;
                        perror("accept"); break;
                    }

                    set_non_blocking(client_fd);
                    event.events = EPOLLIN | EPOLLET;
                    event.data.fd = client_fd;
                    epoll_ctl(epoll_fd, EPOLL_CTL_ADD, client_fd, &event);

                    clients[client_fd] = {client_addr};

                    char client_ip[INET_ADDRSTRLEN];
                    inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, INET_ADDRSTRLEN);
                    std::cout << "Accepted new connection from " << client_ip << ":" << ntohs(client_addr.sin_port)
                              << " on fd " << client_fd << std::endl;
                }
            } else if (servers.count(current_fd)) {
                // =========== 2. HANDLE SERVER WRITE EVENTS ===========
                if (events[i].events & EPOLLOUT) {
                    std::cout << "Socket ready for writing to server on fd " << current_fd << std::endl;
                    int error = 0;
                    socklen_t len = sizeof(error);
                    if (getsockopt(current_fd, SOL_SOCKET, SO_ERROR, &error, &len) < 0 || error != 0) {
                        if(error != 0) errno = error;
                        perror("Connection to remote server failed");
                        handle_server_disconnection(current_fd);
                    } else {
                        handle_outgoing_message(current_fd, server_send_queues[current_fd], handle_server_disconnection);
                    }
                }
                // =========== 3. HANDLE SERVER READ EVENTS ===========
                if (events[i].events & EPOLLIN) {
                    handle_incoming_message(current_fd, server_buffers[current_fd], handle_server_disconnection);
                }
            } else if (clients.count(current_fd)) {
                // =========== 4. HANDLE CLIENT WRITE EVENTS ===========
                if (events[i].events & EPOLLOUT) {
                    handle_outgoing_message(current_fd, client_send_queues[current_fd], handle_client_disconnection);
                }

                // =========== 5. HANDLE CLIENT READ EVENTS ===========
                if (events[i].events & EPOLLIN) {
                    handle_incoming_message(current_fd, client_buffers[current_fd], handle_client_disconnection);
                }
            }
        }
    }

    close(server_fd);
    close(epoll_fd);
    return 0;
}

// COMPILE SERVER
// g++ ml_wrapper.cpp -I/usr/include/jsoncpp -ljsoncpp -o server