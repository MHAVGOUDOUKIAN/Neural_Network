#include "TimeManager.hpp"

using namespace std;

TimeManager::TimeManager() {}

void TimeManager::add(std::string const& key) {
    if(m_clockMap.count(key)==0) m_clockMap.insert(std::pair<std::string, sf::Clock*>(key, new sf::Clock));
}

sf::Time TimeManager::restart(std::string const& key) {
    if(m_clockMap.count(key)!=0) return m_clockMap.find(key)->second->restart();
    else return sf::Time::Zero;
}

std::vector<sf::Time> TimeManager::restartAll() {
    std::vector<sf::Time> all_timer;
    for(std::map<std::string, sf::Clock*>::iterator it=m_clockMap.begin(); it!=m_clockMap.end(); it++) {
        all_timer.push_back(it->second->restart());
    }
    return all_timer;
}

void TimeManager::erase(std::string const& key) {
    m_clockMap.erase(key);
}

void TimeManager::clear() {
    m_clockMap.clear();
}

void TimeManager::showClock() {
    std::cout << "Chronos disponibles :" << std::endl;
    for(std::map<std::string, sf::Clock*>::iterator it=m_clockMap.begin(); it!=m_clockMap.end(); it++) {
        cout << it->first << " - " << it->second->getElapsedTime().asSeconds() << endl;
    }
}

TimeManager::~TimeManager() {}
