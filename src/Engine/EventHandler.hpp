#ifndef SFML_ENGINE_EVENTHANDLER_HPP
#define SFML_ENGINE_EVENTHANDLER_HPP

#include <SFML/Graphics.hpp>

class EventHandler;

class Observer {
    friend EventHandler;

    public:
        Observer() { AddrForEventHandler=this; }

    private:
        Observer* AddrForEventHandler;
};

class KeyBoardObserver: public Observer {
    public:
        virtual ~KeyBoardObserver() {}
        virtual void notify(sf::Keyboard::Key key, bool pressed) =0;
        virtual void reset() {}
};

class MouseObserver: public Observer {
    public:
        virtual ~MouseObserver() {}
        virtual void notify(sf::Mouse::Button mouse, sf::Vector2i& pos, bool clicked) =0;
        virtual void reset() {}
};

class EventObserver: public Observer {
    public:
        virtual ~EventObserver() {}
        virtual void notify(sf::Event event) =0;
        virtual void reset() {}
};

class EventHandler {
    public: 
        void processEvents(sf::RenderWindow& targetWindow);
        virtual void addKeyBoardObserver(KeyBoardObserver* obj);
        virtual void addMouseObserver(MouseObserver* obj);
        virtual void addEventObserver(EventObserver* obj);
        virtual void delKeyBoardObserver(KeyBoardObserver* obj);
        virtual void delMouseObserver(MouseObserver* obj);
        virtual void delEventObserver(EventObserver* obj);
        static EventHandler* getEventHandler();
        void callResetCommand();

    private:
        ~EventHandler();
        EventHandler();
        std::vector<KeyBoardObserver*> m_VKO;
        std::vector<MouseObserver*> m_VMO;
        std::vector<EventObserver*> m_VEO;
        static EventHandler* _instance;
};

#endif
