#ifndef BALL_HPP_INCLUDED
#define BALL_HPP_INCLUDED

#include <SFML/Graphics.hpp>
#include <Features/Random/Random.hpp>

class Ball
{
    public:
        Ball();
        Ball(const float X,const float Y);
        ~Ball();

        void update(const sf::Time& deltaTime);
        void draw(sf::RenderWindow& window);
        void launch(const float X, const float Y, const float vit_X, const float vit_Y); // Lancer la balle au point (X,Y) à la vitesse V(vit_x, vit_y)
        void processPhysics(void);
        void processCollision(void);
        float dist(const sf::Vector2f& p1, const sf::Vector2f& p2);
    
    public:
        bool m_isDead;
        float m_masse, m_radius;
        sf::Vector2f m_pos, m_vit;
        sf::Vector2f m_posInit, m_vitInit;
        float m_timeElapsedJump;
        sf::CircleShape m_sprite;
        sf::Color m_color;
        float m_lifeTime, m_lifetimeMax;
        sf::VertexArray m_tracer;
};

#endif // BALL_HPP_INCLUDED
