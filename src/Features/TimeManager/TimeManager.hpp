#ifndef SFML_ENGINE_TIMEMANAGER_HPP
#define SFML_ENGINE_TIMEMANAGER_HPP


#include <SFML/System.hpp>

#include <map>
#include <vector>
#include <iostream>

class TimeManager {
    public:
        TimeManager();
        ~TimeManager();

        void add(std::string const& key);
        sf::Time restart(std::string const& key);
        std::vector<sf::Time> restartAll();
        void erase(std::string const& key);
        void clear();
        void showClock();

    private:
        std::map<std::string, sf::Clock*> m_clockMap;
};

#endif
