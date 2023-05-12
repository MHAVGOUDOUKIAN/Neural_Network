#ifndef PARTICULEGENERATOR_HPP_INCLUDED
#define PARTICULEGENERATOR_HPP_INCLUDED

#include <vector>
#include <SFML/Graphics.hpp>
#include <iostream>
#include <Features/Random/Random.hpp>
#include "Ball.hpp"

class Particule;

class ParticuleGenerator : public sf::Drawable
{
    public:
        ParticuleGenerator(const float X, const float Y);
        ~ParticuleGenerator();

        void add(const float X, const float Y); // génère une particule aux positions de la souris
        void add(); // génère une particule aux positions du générateur

        void emit(sf::Time deltaTime); // génère une particule à chaque intervalle de temps "particulePerSecond"
        void update(sf::Time deltaTime);
        virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const;

        size_t size(void) const { return lParticules.size(); }

        void setPosition(const float X, const float Y) { pos.x = X; pos.y = Y; }
        void setPosition(sf::Vector2f& newPos) { pos = newPos; }
        sf::Vector2f getPosition(void) { return pos; }
        void start(void);
        void stop(void);
        void enableDuration(const bool boolean) { EnableDuration = boolean; lifeTime =0; timeLastParticuleAdded=0; };
        bool isWorking(void) const { return working; };
        void setDuration(const float time) { duration = time; } // Modifie la durée de vie du générateur
        void setParticulePerSecond(const int cadence) { particulePerSecond = cadence; }
        float getParticulePerSecond(void) { return particulePerSecond; }
        void setLifeTimeParticule(const float value) { lifeTimeParticule = value; }
        float getLifeTimeParticule(void) { return lifeTimeParticule; }
        void setTypeParticule(const short value) { typeParticule = value;}
        short getTypeParticule(void) { return typeParticule; }

    private:
        std::vector<Particule*> lParticules;
        sf::Vector2f pos;
        short typeParticule;
        bool working, EnableDuration; // Indique que le générateur est en cours d'utilisation / Indique que le générateur à une durée de fonctionnement
        float duration, lifeTime; // Durée de vie du générateur
        float particulePerSecond; // Pas de limitation pour générer des particules mais attention aux performances
        float timeLastParticuleAdded; /* Utile uniquement pour emit() // Le nombre de particule max que peut fournir un générateur par seconde dépend des performances du système // 0 => Démarrage du générateur retardé de 'particulePerSecond', toujours initialiser à 1 pour un démarrage direct mais augmente particulePerSecond par 1 */
        float lifeTimeParticule; // Temps de vie des particules
        sf::Vector2f particlesOffset;
        sf::Vector2i prevGenPos;
};

#endif // PARTICULEGENERATOR_HPP_INCLUDED
