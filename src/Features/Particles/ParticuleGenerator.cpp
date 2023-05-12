#include "ParticuleGenerator.hpp"
#include "Particule.hpp"
#include <iostream>

ParticuleGenerator::ParticuleGenerator(const float X, const float Y): pos(sf::Vector2f(X,Y)), typeParticule(0), working(false), EnableDuration(false),
duration(2.f), lifeTime(0), particulePerSecond(300), timeLastParticuleAdded(0), lifeTimeParticule(0.5f), prevGenPos(pos), particlesOffset(700,1000) {}

void ParticuleGenerator::update(sf::Time deltaTime)
{
    lifeTime += deltaTime.asSeconds();

    if(isWorking()) { emit(deltaTime); }

    if(EnableDuration && lifeTime > duration) {
        stop();
        lifeTime = 0;
    }

    for(int i=0; i < int(lParticules.size()); i++) {
        if(!lParticules[i]->isDead()) {
            lParticules[i]->update(deltaTime);
        }
        else {
            delete lParticules[i];
            lParticules.erase(lParticules.begin()+i);
            i--;
        }
    }

    //Quand on finit de mettre à jour le générateur on met a jour la variable prevGenPos
    prevGenPos.x = pos.x;
    prevGenPos.y = pos.y;
}

void ParticuleGenerator::add(const float X, const float Y)
{
    if(typeParticule == 0) { lParticules.push_back(new Particule_Ball(X,Y, lifeTimeParticule));}
    else if(typeParticule == 1) { lParticules.push_back(new Particule_Snow(X,Y, lifeTimeParticule));}
    else if(typeParticule == 2) { lParticules.push_back(new Particule_Dot(X,Y, lifeTimeParticule));}
    else if(typeParticule == 3) { lParticules.push_back(new Particule_Fire(X,Y, lifeTimeParticule));}
    else if(typeParticule == 4) { lParticules.push_back(new Particule_Trail(X,Y, lifeTimeParticule));}
    else if(typeParticule == 5) { lParticules.push_back(new Particule_Dot_Fall(X,Y, lifeTimeParticule));}
    lParticules[lParticules.size()-1]->launch(lParticules[lParticules.size()-1]->getPosition().x, lParticules[lParticules.size()-1]->getPosition().y, randomf(-particlesOffset.x, particlesOffset.x), randomf(-particlesOffset.y, particlesOffset.y));
}

void ParticuleGenerator::add()
{
    if(typeParticule == 0) { lParticules.push_back(new Particule_Ball(this->pos.x ,this->pos.y, lifeTimeParticule));}
    else if(typeParticule == 1) { lParticules.push_back(new Particule_Snow(this->pos.x ,this->pos.y, lifeTimeParticule));}
    else if(typeParticule == 2) { lParticules.push_back(new Particule_Dot(this->pos.x ,this->pos.y, lifeTimeParticule));}
    else if(typeParticule == 3) { lParticules.push_back(new Particule_Fire(this->pos.x ,this->pos.y, lifeTimeParticule));}
    else if(typeParticule == 4) { lParticules.push_back(new Particule_Trail(this->pos.x ,this->pos.y, lifeTimeParticule));}
    else if(typeParticule == 5) { lParticules.push_back(new Particule_Dot_Fall(this->pos.x ,this->pos.y, lifeTimeParticule));}
    lParticules[lParticules.size()-1]->launch(lParticules[lParticules.size()-1]->getPosition().x, lParticules[lParticules.size()-1]->getPosition().y, randomf(-particlesOffset.x, particlesOffset.x), randomf(-particlesOffset.y, particlesOffset.y));
}

void ParticuleGenerator::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    for(int i=0; i < int(lParticules.size()) ; i++) {
        target.draw(*lParticules[i]);
    }
}

void ParticuleGenerator::emit(sf::Time deltaTime)
{
    sf::Vector2f deltaGenPos{pos.x-prevGenPos.x, pos.y-prevGenPos.y};
    timeLastParticuleAdded+=deltaTime.asSeconds();
    int numParticlesToAdd = particulePerSecond*timeLastParticuleAdded;
    for(float i=0; i<numParticlesToAdd ;i++) {
        add(prevGenPos.x+(i/numParticlesToAdd)*deltaGenPos.x, prevGenPos.y+(i/numParticlesToAdd)*deltaGenPos.y);
        timeLastParticuleAdded=0;
    }
}

void ParticuleGenerator::start(void)
{
    if(!isWorking()) {
        working = true;
        lifeTime=0;
        timeLastParticuleAdded=0;
    }
}

void ParticuleGenerator::stop(void)
{
    if(isWorking()) {
        working = false;
        lifeTime=0;
        timeLastParticuleAdded=0;
    }
}

ParticuleGenerator::~ParticuleGenerator() {}
