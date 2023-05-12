#ifndef APP_HPP
#define APP_HPP

#include <vector>
#include <iostream>

#include <SFML/Graphics.hpp>
#include <Engine/EventHandler.hpp>
#include <Features/Particles/ParticuleGenerator.hpp>
#include <Application/Matrix.h>

#define NB_ELEMENTS 10.0f
#define EPOCH 1000
#define ZOOM 12.0f
#define LEARNING_RATE 0.01f
#define C1 3
#define C2 1
#define NB_ENTREE 2

class App : public sf::Drawable {
    public:
        App();
        virtual ~App();
        void update(sf::Time deltaTime);
        virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const;

    private:
        void loadDataset(Matrix& X, Matrix& Y);
        void init();
        void drawDataset();
        Matrix X,Y, W1, b1, W2, b2;
        std::vector<sf::Vector3f> l_dataset;
        bool LearningPhase;
        int compt;

};

#endif