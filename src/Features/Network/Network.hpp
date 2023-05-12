#ifndef SFML_ENGINE_NETWORK_HPP
#define SFML_ENGINE_NETWORK_HPP

#include <thread>

#include <SFML/Network.hpp>

class NetworkCore{
    friend class Network;

    private:
        void host(int port);
        void join(const std::string& ip, int port, const std::string& name="");
        void read();
        void write(const std::string& msg);
        void disconnect();
        static NetworkCore* getInstance();
    
    private:
        NetworkCore();
        ~NetworkCore();
        bool m_connected;
        bool m_serverStarted;
        sf::TcpListener m_tcpListener;
        sf::TcpSocket m_communicationSocket;
        std::thread* m_waitConnectionThread;
        std::thread* m_readIncommingMessageThread;
        std::thread* m_joiningServerThread;
        static NetworkCore* m_instance;
};

class Network {
    public:
        static void host(int port);
        static void join(const std::string& ip, int port, const std::string& name="");
        static void send(const std::string& msg);
        static void disconnect();
};
#endif