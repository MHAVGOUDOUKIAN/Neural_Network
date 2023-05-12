#ifndef SFML_ENGINE_ENGINE_HPP
#define SFML_ENGINE_ENGINE_HPP

#include <SFML/Graphics.hpp>
#include <Engine/EventHandler.hpp>
#include <Features/TimeManager/TimeManager.hpp>
#include <Application/App.hpp>

class Engine : public KeyBoardObserver, public EventObserver
{
    public:
        Engine();
        virtual ~Engine();

        void run();
        void update(sf::Time deltaTime);
        void render();

        virtual void notify(sf::Keyboard::Key key, bool pressed);
        virtual void notify(sf::Event m_event);

    private:
        sf::RenderWindow m_window;
        sf::Time m_timePerFrame;

        TimeManager m_timer;
        EventHandler* m_eventHandler;

        App* m_app;
};

#endif
