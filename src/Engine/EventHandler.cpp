#include <Engine/EventHandler.hpp>

EventHandler* EventHandler::_instance=0;

EventHandler::EventHandler() {}

void EventHandler::processEvents(sf::RenderWindow& targetWindow)
{
    sf::Event m_event;
    while(targetWindow.pollEvent(m_event))
    {
        switch(m_event.type) {
            case sf::Event::KeyPressed:
                for(KeyBoardObserver* ko: m_VKO) ko->notify(m_event.key.code, true);
                break;

            case sf::Event::KeyReleased:
                for(KeyBoardObserver* ko: m_VKO) ko->notify(m_event.key.code, false);
                break;

            case sf::Event::MouseButtonPressed:
                {
                    sf::Vector2i pos= sf::Mouse::getPosition(targetWindow);
                    for(MouseObserver* mo: m_VMO) mo->notify(m_event.mouseButton.button, pos, true);
                }
                break;

            case sf::Event::MouseButtonReleased:
                {
                    sf::Vector2i pos= sf::Mouse::getPosition(targetWindow);
                    for(MouseObserver* mo: m_VMO) mo->notify(m_event.mouseButton.button, pos, false);
                }
                break;

            case sf::Event::MouseMoved:
                {
                    sf::Vector2i pos= sf::Mouse::getPosition(targetWindow);
                    for(MouseObserver* mo: m_VMO) mo->notify(m_event.mouseButton.button, pos, false);
                }
                break;

            case sf::Event::LostFocus:
                callResetCommand();
                break;
            
            case sf::Event::Closed:
                for(EventObserver* eo: m_VEO) eo->notify(m_event);
                break;
            
            case sf::Event::Resized:
                for(EventObserver* eo: m_VEO) eo->notify(m_event);
                break;
        }
    }
}

#include <iostream>

void EventHandler::addKeyBoardObserver(KeyBoardObserver* obj) { 
    for(KeyBoardObserver* kbo: m_VKO) {
        if(kbo == obj) {
            return;
        }
    }
    m_VKO.push_back(obj);
    obj->AddrForEventHandler = obj;
    return;
}

void EventHandler::addMouseObserver(MouseObserver* obj) {
    for(MouseObserver* mo: m_VMO) {
        if(mo == obj) {
            std::cout << "already here" << std::endl;
            return;
        }
    }
    m_VMO.push_back(obj); 
    obj->AddrForEventHandler = obj;
}

void EventHandler::addEventObserver(EventObserver* obj) {
    for(EventObserver* eo: m_VEO) {
        if(eo == obj) {
            return;
        }
    } 
    m_VEO.push_back(obj);
    obj->AddrForEventHandler = obj;
}

void EventHandler::delKeyBoardObserver(KeyBoardObserver* obj) {
    for(int i=0; i< m_VKO.size();i++) {
        if(m_VKO[i] == obj->AddrForEventHandler)  {
            m_VKO.erase(m_VKO.begin()+i);
            return;
        }
    }
    return;
}

void EventHandler::delMouseObserver(MouseObserver* obj) {
    for(int i=0; i< m_VMO.size();i++) {
        if(m_VMO[i] == obj->AddrForEventHandler)  {
            m_VMO.erase(m_VMO.begin()+i);
            return;
        }
    }
    return;
}

void EventHandler::delEventObserver(EventObserver* obj) {
    for(int i=0; i< m_VEO.size();i++) {
        if(m_VEO[i] == obj->AddrForEventHandler)  {
            m_VEO.erase(m_VEO.begin()+i);
            return;
        }
    }
    return;
}

void EventHandler::callResetCommand() {
    for(KeyBoardObserver* ko: m_VKO) ko->reset();
    for(MouseObserver* mo: m_VMO) mo->reset();
    for(EventObserver* eo: m_VEO) eo->reset();
}


EventHandler* EventHandler::getEventHandler() {
    if(_instance == nullptr) _instance = new EventHandler();
    return _instance;
}

EventHandler::~EventHandler() {}
