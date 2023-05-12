#include <Features/Network/Network.hpp>

/////////////////////////////
//
//      NETWORK CORE
//
/////////////////////////////

NetworkCore* NetworkCore::m_instance=nullptr;

NetworkCore::NetworkCore() : m_connected(false), m_serverStarted(false), m_readIncommingMessageThread(nullptr), m_joiningServerThread(nullptr),
m_waitConnectionThread(nullptr) {}

NetworkCore::~NetworkCore() { disconnect(); }

void NetworkCore::host(int port) {
    //GUILogger::log("> Starting server");
    if(m_tcpListener.listen(port) != sf::Socket::Done) {
        //GUILogger::log("> [ERROR] Can't bind socket");
        return;
    }

    m_serverStarted = true;
    sf::Socket::Status status;
    //GUILogger::log("> Listening for connections");
    if( m_tcpListener.accept(m_communicationSocket) != sf::Socket::Done) {
        //GUILogger::log("> [ERROR] Impossible to accept");
    }
    m_connected = true;
    if(m_readIncommingMessageThread == nullptr) m_readIncommingMessageThread = new std::thread(&NetworkCore::read, this);
    //GUILogger::log("> New client is connected: " + m_communicationSocket.getRemoteAddress().toString());
}

void NetworkCore::join(const std::string& ip, int port, const std::string& name) {
    if(m_communicationSocket.connect(ip,port) != sf::Socket::Done) {
        //GUILogger::log("> [ERROR] Connection to the given server is impossible");
        return;
    }
    //GUILogger::log("Connected to: " + m_communicationSocket.getRemoteAddress().toString());
    m_connected = true;
    if(m_readIncommingMessageThread == nullptr) m_readIncommingMessageThread = new std::thread(&NetworkCore::read, this);
}

void NetworkCore::read() {
    int msg_size = 256;
    char data[msg_size];
    std::size_t received; 

    while(m_connected) {
        if(m_communicationSocket.receive(data,256, received) != sf::Socket::Done) {
            //GUILogger::log("> [ERROR] Can't receive the client's message");
        } else {
            std::string str = data;
            //GUILogger::log("> Message received: " + str);
            if(str == "DISCONNECT:") { m_communicationSocket.disconnect(); m_connected=false;}
        }
    }
}

void NetworkCore::write(const std::string& msg) {
    if(m_communicationSocket.send(msg.c_str(), 256) != sf::Socket::Done) {
        //GUILogger::log("> [ERROR] Can't send the message: " + msg);
        return;
    }
    //GUILogger::log("> Message sent");
}

void NetworkCore::disconnect() {
    write("DISCONNECT:");
    m_connected=false;
    m_serverStarted = false;
    m_communicationSocket.disconnect();
    m_tcpListener.close();
    if(m_readIncommingMessageThread != nullptr) { 
        m_readIncommingMessageThread->join();
        delete m_readIncommingMessageThread;
    }
    if(m_waitConnectionThread != nullptr) { 
        m_waitConnectionThread->join();
        delete m_waitConnectionThread;
    }
    //GUILogger::log("> You are disconnected");
}

NetworkCore* NetworkCore::getInstance() {
    if(m_instance == nullptr) m_instance = new NetworkCore();
    return m_instance;
}

/////////////////////////////
//
//      NETWORK
//
/////////////////////////////

void Network::host(int port) { 
    if(!NetworkCore::getInstance()->m_serverStarted) {
        NetworkCore::getInstance()->m_waitConnectionThread = new std::thread(&NetworkCore::host, NetworkCore::getInstance(), port);
    }
}
void Network::join(const std::string& ip, int port, const std::string& name) { 
    if(!NetworkCore::getInstance()->m_connected) {
        NetworkCore::getInstance()->m_joiningServerThread = new std::thread(&NetworkCore::join, NetworkCore::getInstance(), ip, port, name);
    }
}
void Network::send(const std::string& msg) { NetworkCore::getInstance()->write(msg); }
void Network::disconnect() { if(NetworkCore::getInstance()->m_connected) NetworkCore::getInstance()->disconnect(); }