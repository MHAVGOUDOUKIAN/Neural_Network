#include <iostream>
#include <cmath>
#include "Ball.hpp"

#define g 10 * 150
#define h 5
#define PI 3.14159265358979323846

Ball::Ball(): m_isDead(false), m_masse(5), m_radius(60), m_pos(0,0), m_vit(0,0), m_posInit(0,0), m_vitInit(0,0), m_timeElapsedJump(0),
m_lifeTime(0.f), m_lifetimeMax(0.35f), m_tracer(sf::TriangleFan,30)
{
    int color = randomi(1,5);

    if(color == 1) { m_color = sf::Color::White; }
    else if(color == 2) { m_color = sf::Color::Cyan; }
    else if(color == 3) { m_color = sf::Color::Red; }
    else if(color == 4) { m_color = sf::Color::Yellow; }
    else if(color == 5) { m_color = sf::Color::Green; }

    m_sprite.setFillColor(m_color);
    m_sprite.setRadius(m_radius);
}

Ball::Ball(const float X,const float Y): m_isDead(false), m_masse(5), m_radius(60), m_pos(sf::Vector2f(X,Y)), 
m_vit(sf::Vector2f(0,0)), m_posInit(sf::Vector2f(X,Y)), m_vitInit(sf::Vector2f(0,0)), m_timeElapsedJump(0),
m_lifeTime(0.f), m_lifetimeMax(0.35f), m_tracer(sf::TriangleFan, 30)
{
    int color = randomi(1,5);

    if(color == 1) { m_color = sf::Color::White; }
    else if(color == 2) { m_color = sf::Color::Cyan; }
    else if(color == 3) { m_color = sf::Color::Red; }
    else if(color == 4) { m_color = sf::Color::Yellow; }
    else if(color == 5) { m_color = sf::Color::Green; }

    m_sprite.setFillColor(m_color);
    m_sprite.setRadius(m_radius);
}

void Ball::update(const sf::Time& deltaTime)
{
    m_timeElapsedJump += deltaTime.asSeconds();
    m_lifeTime += deltaTime.asSeconds();

    if(m_lifeTime>m_lifetimeMax) { m_isDead = true; }

    processPhysics();

    processCollision();
}

void Ball::processPhysics(void) // Problème de la balle lancée en (x,y) à la vitesse V(v_x,v_y)
{
        m_pos.y = -((-0.5) * g * m_timeElapsedJump * m_timeElapsedJump + m_vitInit.y * m_timeElapsedJump) + m_posInit.y;
        //m_pos.x = m_vitInit.x * m_timeElapsedJump + m_posInit.x; // Cas sans frottements
        m_pos.x = (m_masse/h) * m_vitInit.x * (1-exp(-h * m_timeElapsedJump / m_masse)) + m_posInit.x; // Cas avec frottements (air)

        m_vit.y = (-1) * g * m_timeElapsedJump + m_vitInit.y;
        //m_vit.x = m_vitInit.x; // Cas sans frottements
        m_vit.x = m_vitInit.x * exp((-h*m_timeElapsedJump)/m_masse); // Cas avec frottements (air)
}

void Ball::processCollision(void) // Rebond entre la balle et un support = lancer de la balle à une nouvelle vitesse au point de collision
{
        if(m_pos.y - m_radius < 0) {
            m_pos.y = m_radius;
            m_vit.y = m_vit.y*(-1);
        }
        else if(m_pos.y + m_radius > 1080) {
            if(m_vit.y > 0 && m_vit.y < 5) {// Cas ou la balle ne rebondit plus
                m_pos.y = 1080-m_radius;
                m_vit.y = 0;
            }
            else { // Cas du rebond
                m_pos.y = 1080-m_radius;
                m_vit.y = m_vit.y*(-1) * 0.65 /* Facteur d'Adhérence du sol */;
            }
        }

        if(m_pos.x - m_radius < 0) {
            m_pos.x = m_radius;
            m_vit.x = m_vit.x * (-1);
        } // Rebond sur les cotés
        else if(m_pos.x + m_radius > 1920) {
            m_pos.x = 1920 - m_radius;
            m_vit.x = m_vit.x * (-1);
            
        }
        if(abs(m_vit.x) < 10) {
            m_vit.x = 0;
        }

    launch(m_pos.x, m_pos.y, m_vit.x, m_vit.y);
}

void Ball::launch(const float X, const float Y, const float vit_X, const float vit_Y)
{
    m_pos = sf::Vector2f(X,Y);

    m_posInit = m_pos;

    m_vit = sf::Vector2f(vit_X,vit_Y);

    m_vitInit = m_vit;

    m_timeElapsedJump = 0;

}

void Ball::draw(sf::RenderWindow& window)
{
    m_sprite.setPosition(m_pos.x - m_radius , m_pos.y - m_radius);

    window.draw(m_sprite);
}

float Ball::dist(const sf::Vector2f& p1, const sf::Vector2f& p2) { return sqrtf((p1.x + p2.x)*(p1.x + p2.x) + (p1.y + p2.y)*(p1.y + p2.y)); }

Ball::~Ball() {}
