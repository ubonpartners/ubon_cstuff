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
#include <set>
#include <thread>
#include <functional>
#include <optional>

#include <jsoncpp/json/json.h>
#include <readerwriterqueue/readerwriterqueue.h>
#include <argparse/argparse.hpp>


#define EPOLL_MAX_EVENTS 30
#define LISTENING_PORT 8080
#define STREAM_MANAGER_PORT 8081
#define STREAM_MANAGER_IP "127.0.0.1"

#define MSG_TYPE_SDP 0x01
#define MSG_TYPE_NAL 0x02
#define MSG_TYPE_JSON 0x03
#define MSG_TYPE_INFERENCE 0x0A
#define MSG_TYPE_THUMBNAIL 0x0B
#define MSG_TYPE_BESTFACE 0x0C

#define ACTION_SUBSCRIBE "SUBSCRIBE"
#define STREAM_TYPE_INFERENCE 0x0A
#define STREAM_TYPE_THUMBNAIL 0x0B
#define STREAM_TYPE_BESTFACE 0x0C

#define POISON_PILL 0xFF
#define READER_WRITER_QUEUE_SIZE 15 // Unless the consumer(camera_worker) thread is starved, there is no reason to make this larger

#if DO_REAL_INFERENCE
    #include <track.h>
    #include "display.h"
    #include "jpeg.h"
    #include <misc.h>
    #include <cuda_stuff.h>
    #include <yaml_stuff.h>
    #include <profile.h>
#endif

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


struct RemoteServerInfo {
    std::string ip;
    int port;
    std::string camera_id; // We make a separate connection request per camera
};

struct ClientInfo {
    sockaddr_in addr;
};

typedef struct state
{
    #if DO_REAL_INFERENCE
    track_shared_state_t *tss;
    track_stream_t *ts;
    image_t *img;
    #endif
    double start_time;
    std::string camera_id;
    bool stopped;
} state_t;


namespace NetworkManager {
    void initialize_server();
    int establish_server_connection(const RemoteServerInfo& target_server);
    std::optional<RemoteServerInfo> get_server_details(int fd);
    void queue_message_for_fd(int fd, const std::vector<char>& message);
    void terminate_connection(int fd);
    void terminate_connections(std::vector<int> fds);
}

namespace CameraSubscriptions {
    std::vector<int> remove_camera_subscriptions(const std::string& camera_id);
    std::vector<int> remove_client(int client_fd);
    void add_nal_data(int server_fd, const std::vector<char>& nal_data);
    void add_camera_subscription(int client_fd, const std::string& camera_id, uint8_t stream_type);
    std::optional<uint64_t> get_wall_timestamp(const std::string& camera_id, double rtp_time);
    std::set<std::pair<int, uint8_t>> get_camera_clients(const std::string& camera_id);
}

void handle_client_disconnection(int client_fd) {
    auto connections_to_terminate = CameraSubscriptions::remove_client(client_fd);
    NetworkManager::terminate_connections(connections_to_terminate);
    NetworkManager::terminate_connection(client_fd);
}

void handle_server_disconnection(int server_fd){
    auto server_details = NetworkManager::get_server_details(server_fd);
    if (!server_details.has_value()) {
        return;
    }
    auto connections_to_terminate = CameraSubscriptions::remove_camera_subscriptions(server_details->camera_id);
    NetworkManager::terminate_connections(connections_to_terminate);
    NetworkManager::terminate_connection(server_fd);
}

void handle_json_message(int client_fd, const std::string& json_content) {
    Json::Value parsed_json;
    Json::Reader json_parser;

    if (!json_parser.parse(json_content, parsed_json)) {
        std::cerr << "JSON parsing failed: " << json_parser.getFormattedErrorMessages() << std::endl;
        return;
    }

    if (parsed_json.isMember("action") && parsed_json["action"].asString() == ACTION_SUBSCRIBE) {
        std::string camera_id = parsed_json["body"]["camera_id"].asString();
        uint8_t stream_type = parsed_json["body"]["stream_type"].asUInt();
        CameraSubscriptions::add_camera_subscription(client_fd, camera_id, stream_type);
    } else {
        std::cerr << "Invalid JSON content: missing or unknown 'action' field." << std::endl;
    }
}

void handle_incoming_message(int fd, std::vector<char>& receive_buffer, bool is_client) {
    char input_buffer[4096];
    ssize_t bytes_received;
    while ((bytes_received = read(fd, input_buffer, sizeof(input_buffer))) > 0) {
        receive_buffer.insert(receive_buffer.end(), input_buffer, input_buffer + bytes_received);
    }

    if (bytes_received == 0) {
        std::cout << "Connection terminated on fd " << fd << std::endl;
        if (is_client) {
            handle_client_disconnection(fd);
        } else {
            handle_server_disconnection(fd);
        }
    } else if (bytes_received == -1 && errno != EAGAIN && errno != EWOULDBLOCK) {
        perror("read operation failed");
        if (is_client) {
            handle_client_disconnection(fd);
        } else {
            handle_server_disconnection(fd);
        }
    }

    if (receive_buffer.empty()){
        return;
    }

    int message_processed = 1;
    while (message_processed){
        message_processed = 0;
        
        uint8_t message_type = receive_buffer[0];
        switch (message_type) {
            case MSG_TYPE_JSON:
            {
                if (receive_buffer.size() < 5) {
                    break;
                } else {
                    uint32_t content_length;
                    std::memcpy(&content_length, &receive_buffer[1], 4);
                    content_length = ntohl(content_length);
                    
                    if (receive_buffer.size() < 5 + content_length) {
                        break;
                    }

                    std::string json_content(receive_buffer.begin() + 5, receive_buffer.begin() + 5 + content_length);
                    handle_json_message(fd, json_content);
                    receive_buffer.erase(receive_buffer.begin(), receive_buffer.begin() + 5 + content_length);
                    message_processed = 1;
                    std::cout << "Processed JSON message of length " << content_length << std::endl;
                }
                break;
            }
            case MSG_TYPE_NAL:
                if (receive_buffer.size() < 21) {
                    break;
                } else {
                    uint32_t packet_length;
                    std::memcpy(&packet_length, &receive_buffer[17], 4);
                    packet_length = ntohl(packet_length);

                    if (receive_buffer.size() < 21 + packet_length) {
                        break;
                    }

                    std::vector<char> nal_packet(receive_buffer.begin() + 1, receive_buffer.begin() + 21 + packet_length);
                    receive_buffer.erase(receive_buffer.begin(), receive_buffer.begin() + 21 + packet_length);
                    message_processed = 1;
                    std::cout << "Processed NAL message of length " << packet_length << std::endl;
                    CameraSubscriptions::add_nal_data(fd, nal_packet);
                }
                break;
            case MSG_TYPE_SDP:
                if (receive_buffer.size() < 5) {
                    break;
                } else {
                    uint32_t content_length;
                    std::memcpy(&content_length, &receive_buffer[1], 4);
                    content_length = ntohl(content_length);
                    
                    if (receive_buffer.size() < 5 + content_length) {
                        break;
                    }

                    std::vector<char> sdp_content(receive_buffer.begin() + 5, receive_buffer.begin() + 5 + content_length);
                    receive_buffer.erase(receive_buffer.begin(), receive_buffer.begin() + 5 + content_length);
                    message_processed = 1;
                    std::cout << "Processed SDP message of length " << content_length << std::endl;
                }
                break;
            default:
                std::cerr << "Unknown message type encountered" << std::endl;
                break;
        }
    }
}

namespace NetworkManager {
    namespace {
        // Internal state variables
        int main_socket, event_fd;
        std::map<int, ClientInfo> client_registry; //[client_fd, ClientInfo]
        std::map<int, RemoteServerInfo> server_registry; //[server_fd, RemoteServerInfo]
        std::map<int, std::vector<char>> client_incoming_buffers; //[client_fd, buffer]
        std::map<int, std::vector<char>> server_incoming_buffers; //[server_fd, buffer]
        std::map<int, std::queue<std::vector<char>>> server_outgoing_message_queues; //[server_fd, message_queue]
        std::map<int, std::queue<std::vector<char>>> client_outgoing_message_queues; //[client_fd, message_queue]

        bool configure_nonblocking(int socket_fd) {
            int current_flags = fcntl(socket_fd, F_GETFL, 0);
            if (current_flags == -1) {
                perror("fcntl(F_GETFL)");
                return false;
            }
            if (fcntl(socket_fd, F_SETFL, current_flags | O_NONBLOCK) == -1) {
                perror("fcntl(F_SETFL)");
                return false;
            }
            return true;
        }

        void cleanup_client_socket(int fd) {
            if (client_registry.find(fd) == client_registry.end()) {
                return;
            }
            std::cout << "Cleaning up client socket fd " << fd << std::endl;
            epoll_ctl(event_fd, EPOLL_CTL_DEL, fd, nullptr);
            close(fd);
            client_registry.erase(fd);
            client_incoming_buffers.erase(fd);
            client_outgoing_message_queues.erase(fd);
        }

        void cleanup_server_socket(int fd) {
            if (server_registry.find(fd) == server_registry.end()) {
                return;
            }
            std::cout << "Cleaning up server socket fd " << fd << std::endl;
            epoll_ctl(event_fd, EPOLL_CTL_DEL, fd, nullptr);
            close(fd);
            server_registry.erase(fd);
            server_incoming_buffers.erase(fd);
            server_outgoing_message_queues.erase(fd);
        }

        int transmit_full_message(int fd, std::vector<char>& message) {
            size_t bytes_transmitted = 0;
            const char* message_data = message.data();
            size_t bytes_remaining = message.size();
            
            while (bytes_remaining > 0) {
                ssize_t bytes_sent = send(fd, message_data + bytes_transmitted, bytes_remaining, MSG_NOSIGNAL);
                
                if (bytes_sent == -1) {
                    if (errno == EAGAIN || errno == EWOULDBLOCK) {
                        std::cout << "Socket buffer saturated, transmitted " << bytes_transmitted << "/" << message.size() << " bytes" << std::endl;
                        break;
                    }
                    perror("transmission failed");
                    return -1;
                }

                message.erase(message.begin(), message.begin() + bytes_sent);
                bytes_transmitted += bytes_sent;
                bytes_remaining -= bytes_sent;
            }
            std::cout << "Transmitted " << bytes_transmitted << " bytes on fd " << fd << std::endl;
            return message.empty();
        }

        void process_outbound_messages(int fd, std::queue<std::vector<char>>& message_queue) {
            std::cout << "Processing outbound messages for fd " << fd << " with queue size: " << message_queue.size() << std::endl;
            while (!message_queue.empty()) {
                auto& current_message = message_queue.front();
                int transmission_result = transmit_full_message(fd, current_message);
                if (transmission_result == 1) {
                    message_queue.pop();
                } else if (transmission_result == 0) {
                    break;
                } else {
                    if (client_registry.find(fd) != client_registry.end()) {
                        handle_client_disconnection(fd);
                    } else if (server_registry.find(fd) != server_registry.end()) {
                        handle_server_disconnection(fd);
                    }
                }
            }
            std::cout << "Completed processing messages on fd " << fd << " with remaining queue size: " << message_queue.size() << std::endl;
            // If send queue is empty, stop monitoring EPOLLOUT
            if (message_queue.empty()) {
                struct epoll_event event_config;
                event_config.data.fd = fd;
                event_config.events = EPOLLIN | EPOLLET;
                epoll_ctl(event_fd, EPOLL_CTL_MOD, fd, &event_config);
            }
        }

        std::optional<ClientInfo> get_client_details(int fd) {
            if (client_registry.find(fd) != client_registry.end()) {
                return client_registry[fd];
            }
            return std::nullopt;
        }
    }

    // Public interface functions
    void initialize_server(){
        struct sockaddr_in server_address;
        if ((main_socket = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
            perror("socket creation failed"); exit(EXIT_FAILURE);
        }
        int socket_option = 1;
        if (setsockopt(main_socket, SOL_SOCKET, SO_REUSEADDR, &socket_option, sizeof(socket_option))) {
            perror("setsockopt failed"); exit(EXIT_FAILURE);
        }
        server_address.sin_family = AF_INET;
        server_address.sin_addr.s_addr = INADDR_ANY;
        server_address.sin_port = htons(LISTENING_PORT);
        
        if (bind(main_socket, (struct sockaddr *)&server_address, sizeof(server_address)) < 0) {
            perror("bind operation failed"); exit(EXIT_FAILURE);
        }
        if (listen(main_socket, 3) < 0) {
            perror("listen operation failed"); exit(EXIT_FAILURE);
        }
        if (!configure_nonblocking(main_socket)) { exit(EXIT_FAILURE); }
        std::cout << "Network server active on port " << LISTENING_PORT << std::endl;
        
        // Event polling setup
        struct epoll_event event_config, event_list[EPOLL_MAX_EVENTS];
        if ((event_fd = epoll_create1(0)) == -1) {
            perror("epoll_create1 failed"); exit(EXIT_FAILURE);
        }

        event_config.events = EPOLLIN | EPOLLET;
        event_config.data.fd = main_socket;
        if (epoll_ctl(event_fd, EPOLL_CTL_ADD, main_socket, &event_config) == -1) {
            perror("epoll_ctl failed"); exit(EXIT_FAILURE);
        }

        // Main event processing loop
        while (true) {
            int event_count = epoll_wait(event_fd, event_list, EPOLL_MAX_EVENTS, -1);
            if (event_count == -1) {
                perror("epoll_wait failed"); break;
            }

            for (int event_idx = 0; event_idx < event_count; ++event_idx) {
                int active_fd = event_list[event_idx].data.fd;

                if ((event_list[event_idx].events & EPOLLERR) || (event_list[event_idx].events & EPOLLHUP)) {
                    std::cerr << "Epoll error detected on fd " << active_fd << std::endl;
                    if (client_registry.count(active_fd)) {
                        handle_client_disconnection(active_fd);
                    } else if (server_registry.count(active_fd)) {
                        handle_server_disconnection(active_fd);
                    }
                    continue;
                }

                if (active_fd == main_socket) {
                    // Handle new client connections
                    while(true) {
                        struct sockaddr_in client_address;
                        socklen_t client_addr_size = sizeof(client_address);
                        int new_client_fd = accept(main_socket, (struct sockaddr *)&client_address, &client_addr_size);

                        if (new_client_fd == -1) {
                            if (errno == EAGAIN || errno == EWOULDBLOCK) break;
                            perror("accept failed"); break;
                        }

                        configure_nonblocking(new_client_fd);
                        event_config.events = EPOLLIN | EPOLLET;
                        event_config.data.fd = new_client_fd;
                        epoll_ctl(event_fd, EPOLL_CTL_ADD, new_client_fd, &event_config);

                        client_registry[new_client_fd] = {client_address};

                        char client_ip_str[INET_ADDRSTRLEN];
                        inet_ntop(AF_INET, &client_address.sin_addr, client_ip_str, INET_ADDRSTRLEN);
                        std::cout << "New client connected from " << client_ip_str << ":" << ntohs(client_address.sin_port)
                                << " on fd " << new_client_fd << std::endl;
                    }
                } else if (server_registry.count(active_fd)) {
                    // Handle server communication
                    if (event_list[event_idx].events & EPOLLOUT) {
                        std::cout << "Server socket ready for writing on fd " << active_fd << std::endl;
                        int socket_error = 0;
                        socklen_t error_size = sizeof(socket_error);
                        if (getsockopt(active_fd, SOL_SOCKET, SO_ERROR, &socket_error, &error_size) < 0 || socket_error != 0) {
                            if(socket_error != 0) errno = socket_error;
                            perror("Server connection establishment failed");
                            handle_server_disconnection(active_fd);
                        } else {
                            process_outbound_messages(active_fd, server_outgoing_message_queues[active_fd]);
                        }
                    }
                    if (event_list[event_idx].events & EPOLLIN) {
                        handle_incoming_message(active_fd, server_incoming_buffers[active_fd], false);
                    }
                } else if (client_registry.count(active_fd)) {
                    // Handle client communication
                    if (event_list[event_idx].events & EPOLLOUT) {
                        process_outbound_messages(active_fd, client_outgoing_message_queues[active_fd]);
                    }
                    if (event_list[event_idx].events & EPOLLIN) {
                        handle_incoming_message(active_fd, client_incoming_buffers[active_fd], true);
                    }
                }
            }
        }
        close(main_socket);
        close(event_fd);
    }

    int establish_server_connection(const RemoteServerInfo& target_server) {
        // Check for existing connection
        for (const auto& [fd, server_info] : server_registry) {
            if (server_info.ip == target_server.ip && server_info.port == target_server.port && server_info.camera_id == target_server.camera_id) {
                return fd;
            }
        }

        int connection_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (connection_fd < 0) {
            perror("socket creation failed");
            return -1;
        }

        if (!configure_nonblocking(connection_fd)) {
            close(connection_fd);
            return -1;
        }

        struct sockaddr_in target_address;
        std::memset(&target_address, 0, sizeof(target_address));
        target_address.sin_family = AF_INET;
        target_address.sin_port = htons(target_server.port);
        if (inet_pton(AF_INET, target_server.ip.c_str(), &target_address.sin_addr) <= 0) {
            perror("inet_pton failed");
            close(connection_fd);
            return -1;
        }

        std::cout << "Initiating connection to " << target_server.ip << ":" << target_server.port << " on fd " << connection_fd << std::endl;
        int connection_result = connect(connection_fd, (struct sockaddr*)&target_address, sizeof(target_address));
        if (connection_result < 0 && errno != EINPROGRESS) {
            perror("connect failed");
            close(connection_fd);
            return -1;
        }

        struct epoll_event event_config;
        event_config.data.fd = connection_fd;
        event_config.events = EPOLLIN | EPOLLOUT | EPOLLET;
        if (epoll_ctl(event_fd, EPOLL_CTL_ADD, connection_fd, &event_config) == -1) {
            perror("epoll_ctl add failed");
            close(connection_fd);
            return -1;
        }

        server_registry[connection_fd] = target_server;
        return connection_fd;
    }

    std::optional<RemoteServerInfo> get_server_details(int fd) {
        if (server_registry.find(fd) != server_registry.end()) {
            return server_registry[fd];
        }
        return std::nullopt;
    }

    void queue_message_for_fd(int fd, const std::vector<char>& message){
        if (get_client_details(fd)) {
            client_outgoing_message_queues[fd].push(message);
        } else if (get_server_details(fd)) {
            server_outgoing_message_queues[fd].push(message);
        } else {
            std::cerr << "Invalid file descriptor provided" << std::endl;
            return;
        }

        struct epoll_event event_config;
        event_config.data.fd = fd;
        event_config.events = EPOLLIN | EPOLLOUT | EPOLLET;
        epoll_ctl(event_fd, EPOLL_CTL_MOD, fd, &event_config);
        std::cout << "Queued message of " << message.size() << " bytes for fd " << fd << std::endl;
    }

    void terminate_connection(int fd){
        if (get_client_details(fd)) {
            cleanup_client_socket(fd);
        } else if (get_server_details(fd)) {
            cleanup_server_socket(fd);
        } else {
            std::cerr << "Invalid file descriptor provided" << std::endl;
        }
    }

    void terminate_connections(std::vector<int> fd_list){
        for (int fd : fd_list) {
            if (get_client_details(fd)) {
                cleanup_client_socket(fd);
            } else if (get_server_details(fd)) {
                cleanup_server_socket(fd);
            } else {
                std::cerr << "Invalid file descriptor provided" << std::endl;
            }
        }
    }
}

namespace CameraSubscriptions {
    bool dev_mode = false;
    namespace {
        // Internal state
        std::map<std::string, std::set<std::pair<int, uint8_t>>> subscription_registry; //[camera_id, [client_fd, stream_type]]
        std::map<std::string, std::thread> worker_threads; // [camera_id, camera_worker_thread]
        std::map<std::string, moodycamel::BlockingReaderWriterQueue<std::vector<char>>> message_queues; // [camera_id, message_queue]
        std::map<std::string, std::map<uint64_t, uint64_t>> timestamp_mappings; // [camera_id, [rtp_timestamp, wall_timestamp]]
        std::map<int, std::string> camera_fd_to_camera_id; // [server_fd, camera_id]

        void send_mock_inference_data(const std::string& camera_id){
            Json::Value response_data;
            response_data["camera_id"] = camera_id;
            response_data["timestamp"] = 123456789;
            response_data["motion_score"] = 0.85;
            response_data["people_count"] = 2;

            Json::StreamWriterBuilder writer;
            std::string json_string = Json::writeString(writer, response_data);
            std::cout << "Mock response data: " << json_string << std::endl;

            std::vector<char> message;
            message.reserve(5 + json_string.length());
            message.push_back(MSG_TYPE_INFERENCE);
            uint32_t response_length = htonl(json_string.size());
            message.insert(message.end(), (char *)&response_length, (char *)&response_length + 4);
            message.insert(message.end(), json_string.begin(), json_string.end());
            for (const auto& [client_fd, stream_type] : subscription_registry[camera_id]) {
                if (stream_type == STREAM_TYPE_INFERENCE) {
                    NetworkManager::queue_message_for_fd(client_fd, message);
                }
            }
        }

        #if DO_REAL_INFERENCE
        void track_result(void *context, track_results_t *r){
            state_t *processing_state=(state_t *)context;
            if (processing_state->stopped){
                return;
            }

            printf("camera %s track_result: result type %d time %f, %d detections\n", processing_state->camera_id.c_str(), r->result_type, r->time, (r->track_dets==0) ? 0 : r->track_dets->num_detections);

            auto wall_time = CameraSubscriptions::get_wall_timestamp(processing_state->camera_id, r->time); 
            if (!wall_time.has_value()) {
                std::cout << " ======================== No wall timestamp found for rtp time " << r->time << " for camera " << processing_state->camera_id << std::endl;
                return;
            }

            for (auto &[client_fd, stream_type] : CameraSubscriptions::get_camera_clients(processing_state->camera_id)) {
                Json::Value response_data;
                response_data["camera_id"] = processing_state->camera_id;
                response_data["timestamp"] = wall_time.value();
                response_data["motion_score"] = r->motion_score;
                response_data["people_count"] = 0;
                if (r->track_dets != nullptr) {
                    response_data["people_count"] = r->track_dets->num_detections;
                }
                Json::StreamWriterBuilder response_writer;
                std::string response_json = Json::writeString(response_writer, response_data);
                uint32_t response_length = htonl(response_json.size());
                
                std::vector<char> inference_message;
                inference_message.reserve(5 + response_json.length());
                inference_message.push_back(MSG_TYPE_INFERENCE);
                inference_message.insert(inference_message.end(), (char *)&response_length, (char *)&response_length + 4);
                inference_message.insert(inference_message.end(), response_json.begin(), response_json.end());
                NetworkManager::queue_message_for_fd(client_fd, inference_message);
            }

            if (r->track_dets!=0 && dev_mode)
            {
                image_t *current_image=0;
                if (r->track_dets->frame_jpeg!=0)
                {
                    size_t jpeg_size;
                    uint8_t *jpeg_bytes=jpeg_get_data(r->track_dets->frame_jpeg, &jpeg_size);
                    current_image=decode_jpeg(jpeg_bytes, (int)jpeg_size);
                }
                if (current_image!=0)
                {
                    image_t *annotated_frame=detection_list_draw(r->track_dets, current_image);
                    if (annotated_frame!=0)
                    {
                        display_image("video", annotated_frame);
                        image_destroy(annotated_frame);
                    }
                    image_destroy(current_image);
                }
            }
        }
        #endif

        void camera_worker(std::string camera_id) {
            std::cout << "Camera worker started for: " << camera_id << std::endl;
            state_t processing_state;
            memset(&processing_state, 0, sizeof(state_t));
            processing_state.camera_id = camera_id;

            #if DO_REAL_INFERENCE
            const char *default_config="main_jpeg:\n"
                            "    enabled: true\n"
                            "    max_width: 1280\n"
                            "    max_height: 720\n"
                            "    min_interval_seconds: 0.03\n";

            const char *merged_config=yaml_merge_string("/mldata/config/track/trackers/uc_reid.yaml", default_config);

            processing_state.tss=track_shared_state_create(merged_config);
            processing_state.ts=track_stream_create(processing_state.tss, &processing_state, track_result);
            processing_state.start_time=profile_time();
            track_stream_set_minimum_frame_intervals(processing_state.ts, 0.01, 10.0);

            int buffer_size=1024*1024;
            uint8_t *processing_buffer=(uint8_t *)malloc(buffer_size);
            #endif

            while (true){
                std::vector<char> nal_data;
                message_queues[camera_id].wait_dequeue(nal_data);

                std::cout << "Camera worker for: " << camera_id 
                        << " processing packet of size " << nal_data.size() << std::endl;

                if (nal_data.size() < 12){
                    std::cout << "Camera worker for: " << camera_id << " received termination signal." << std::endl;
                    processing_state.stopped = true;
                    break;
                }

                uint64_t wall_timestamp = ntohll(*reinterpret_cast<uint64_t*>(nal_data.data()));
                uint64_t rtp_timestamp = ntohll(*reinterpret_cast<uint64_t*>(nal_data.data() + 8));
                uint32_t packet_length = ntohl(*reinterpret_cast<uint32_t*>(nal_data.data() + 16));
                assert(packet_length < 1024 * 1024);
                double rtp_time_seconds = rtp_timestamp / 90000.0;
                timestamp_mappings[camera_id][rtp_time_seconds] = wall_timestamp;
                
                #if DO_REAL_INFERENCE
                processing_buffer[0]=(packet_length>>24)&0xff;
                processing_buffer[1]=(packet_length>>16)&0xff;
                processing_buffer[2]=(packet_length>>8)&0xff;
                processing_buffer[3]=(packet_length>>0)&0xff;
                std::memcpy(&processing_buffer[4], nal_data.data() + 20, packet_length);
                track_stream_add_nalus(processing_state.ts, rtp_time_seconds, processing_buffer, packet_length + 4, false);
                #endif
                #if DO_REAL_INFERENCE == 0
                send_mock_inference_data(camera_id);
                #endif
            }

            #if DO_REAL_INFERENCE
            track_stream_destroy(processing_state.ts);
            track_shared_state_destroy(processing_state.tss);
            #endif
            return;
        }

        void start_camera_worker(const std::string& camera_id) {
            if (worker_threads.find(camera_id) == worker_threads.end()) {
                message_queues[camera_id] = moodycamel::BlockingReaderWriterQueue<std::vector<char>>(READER_WRITER_QUEUE_SIZE);
                worker_threads[camera_id] = std::thread(camera_worker, camera_id);
            }
        }

        void stop_camera_worker(const std::string& camera_id) {
            auto worker_it = worker_threads.find(camera_id);
            if (worker_it != worker_threads.end()) {
                message_queues[camera_id].enqueue(std::vector<char>(1, POISON_PILL));
                worker_it->second.join();
                std::cout << "Terminated camera worker for " << camera_id << std::endl;
            }
        }

        void camera_cleanup(std::string camera_id){
            subscription_registry.erase(camera_id);
            message_queues.erase(camera_id);
            worker_threads.erase(camera_id);
            timestamp_mappings.erase(camera_id);

            for (auto it = camera_fd_to_camera_id.begin(); it != camera_fd_to_camera_id.end(); ) {
                if (it->second == camera_id) {
                    it = camera_fd_to_camera_id.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }

    // Public interface
    std::vector<int> remove_camera_subscriptions(const std::string& camera_id){
        if (subscription_registry.find(camera_id) == subscription_registry.end()) {
            return {};
        }
        std::cout << "Disconnecting Camera: " << camera_id << std::endl;
        stop_camera_worker(camera_id);

        auto connections_to_terminate = std::vector<int>();
        for (const auto& [client_fd, stream_type] : subscription_registry[camera_id]) {
            connections_to_terminate.push_back(client_fd);
        }
        for (const auto& [server_fd, cam_id] : camera_fd_to_camera_id) {
            if (cam_id == camera_id) {
                connections_to_terminate.push_back(server_fd);
            }
        }
        camera_cleanup(camera_id);
        return connections_to_terminate;
    }

    std::vector<int> remove_client(int client_fd) {
        std::set<std::string> cameras_to_cleanup;
        std::vector<int> connections_to_terminate({client_fd});
        for (auto &[camera_id, client_set] : subscription_registry) {
            for (auto client_it = client_set.begin(); client_it != client_set.end(); ) {
                if (client_it->first == client_fd) {
                    client_it = client_set.erase(client_it);
                } else {
                    ++client_it;
                }
            }
            if (client_set.empty()) {
                cameras_to_cleanup.insert(camera_id);
            }
        }

        for (const auto& camera_id : cameras_to_cleanup) {
            auto connections = remove_camera_subscriptions(camera_id);
            connections_to_terminate.insert(connections_to_terminate.end(), connections.begin(), connections.end());
        }
        return connections_to_terminate;
    }

    void add_camera_subscription(int client_fd, const std::string& camera_id, uint8_t stream_type) {
        if (subscription_registry[camera_id].empty()) {    
            Json::Value request_payload;
            request_payload["action"] = ACTION_SUBSCRIBE;
            request_payload["body"] = Json::Value();
            request_payload["body"]["camera_id"] = camera_id;
            request_payload["body"]["stream_type"] = "nal";
            
            Json::StreamWriterBuilder json_builder;
            std::string json_content = Json::writeString(json_builder, request_payload);
            uint32_t content_length = htonl(json_content.length());
            
            std::vector<char> subscription_message;
            subscription_message.reserve(5 + json_content.length());
            subscription_message.push_back(MSG_TYPE_JSON);
            subscription_message.insert(subscription_message.end(), (char*)&content_length, (char*)&content_length + 4);
            subscription_message.insert(subscription_message.end(), json_content.begin(), json_content.end());

            int server_connection = NetworkManager::establish_server_connection({STREAM_MANAGER_IP, STREAM_MANAGER_PORT, camera_id});
            if (server_connection == -1) {
                std::cerr << "Failed to establish connection to STREAM_MANAGER" << std::endl;
                handle_client_disconnection(client_fd);
                return;
            }
            camera_fd_to_camera_id[server_connection] = camera_id;
            NetworkManager::queue_message_for_fd(server_connection, subscription_message);
            start_camera_worker(camera_id);
        }
        subscription_registry[camera_id].insert({client_fd, stream_type});
        std::cout << "Client " << client_fd << " subscribed to camera " << camera_id
        << " with stream type " << (int)stream_type << std::endl;
    }

    void add_nal_data(int camera_fd, const std::vector<char>& nal_data) {
        if (camera_fd_to_camera_id.find(camera_fd) != camera_fd_to_camera_id.end()) {
            const std::string& camera_id = camera_fd_to_camera_id[camera_fd];
            if (message_queues.find(camera_id) != message_queues.end()) {
                if (!message_queues[camera_id].try_enqueue(nal_data)) {
                    std::cerr << "Failed to enqueue NAL data for camera " << camera_id << std::endl;
                }
            }
        }
    }

    std::optional<uint64_t> get_wall_timestamp(const std::string& camera_id, double rtp_time) {
        if (timestamp_mappings.find(camera_id) != timestamp_mappings.end()) {
            auto& time_map = timestamp_mappings[camera_id];
            if (time_map.find(rtp_time) != time_map.end()) {
                return time_map[rtp_time];
            }
        }
        return std::nullopt;
    }

    std::set<std::pair<int, uint8_t>> get_camera_clients(const std::string& camera_id) {
        if (subscription_registry.find(camera_id) != subscription_registry.end()) {
            return subscription_registry[camera_id];
        }
        return {};
    }
}

/*
TODO
1. Use proper logging instead of std::cerr or std::cout
2. The address and port info should be configurable via command line arguments and config files
3. Create a Cmake/Makefile for the project - make sure to include the necessary external libraries - Currently using Cmake of ubon_cstuff
4. Buffer last seen thumbnail
*/
int main(int argc, char** argv) {
    argparse::ArgumentParser parser("ml_wrapper");
    parser.add_argument("--dev", "-d").help("Run in development mode").default_value(false).implicit_value(true);

    try {
        parser.parse_args(argc, argv);
        CameraSubscriptions::dev_mode = parser.get<bool>("--dev");
    } catch (const std::exception& e) {
        std::cerr << "Error parsing command line arguments: " << e.what() << std::endl;
        return 1;
    }

    #if DO_REAL_INFERENCE
    init_cuda_stuff();
    image_init();
    #endif

    NetworkManager::initialize_server();
}

// JETSON_SOC=1 cmake -DCMAKE_CXX_FLAGS="-DDO_REAL_INFERENCE=1"
// g++ ml_wrapper.cpp -ljsoncpp -o ml_wrapper