#include "Particule.hpp"
#include <cmath>
#include <iostream>

#define g 10 * 150
#define h 5
#define PI 3.14159265358979323846



    //////////////////////////////////////////////////////////////
    ////////////// Particule de base /////////////////////////////  ( NE DOIS PAS ÊTRE UTILISÉE ! )
    //////////////////////////////////////////////////////////////

Particule::Particule() : m_isDead(false), m_pos(sf::Vector2f(0,0)), m_vit(sf::Vector2f(0,0)), m_lifeTime(0.f), m_lifeTimeMax(0.2f) {}

Particule::Particule(const float X,const float Y, sf::PrimitiveType type, std::size_t nb_vertex) : m_isDead(false), m_pos(sf::Vector2f(X,Y)), m_vit(sf::Vector2f(0,0)),
m_lifeTime(0.f), m_lifeTimeMax(0.2f), m_forme(type, nb_vertex) {}

Particule::Particule(const float X,const float Y, sf::PrimitiveType type, std::size_t nb_vertex, float dureeVie): m_isDead(false), m_pos(sf::Vector2f(X,Y)),
m_vit(sf::Vector2f(0,0)), m_lifeTime(0.f), m_lifeTimeMax(dureeVie), m_forme(type, nb_vertex) {}

void Particule::draw(sf::RenderTarget& target, sf::RenderStates states) const
{
    target.draw(m_forme);
}

Particule::~Particule() {}



    //////////////////////////////////////////////////////////////
    ////////////// Particule Ball ////////////////////////////////
    //////////////////////////////////////////////////////////////

Particule_Ball::Particule_Ball() : Particule(0,0, sf::Triangles, 30), m_posInit(sf::Vector2f(m_pos.x, m_pos.y)), m_vitInit(sf::Vector2f(m_vit.x, m_vit.y)),
m_timeElapsedJump(0), m_radius(15), m_radiusInit(m_radius), m_masse(10)
{
    m_lifeTimeMax = 2.f;
    int color = randomi(1,5);

    if(color == 1) { m_color = sf::Color::Blue; }
    else if(color == 2) { m_color = sf::Color::Cyan; }
    else if(color == 3) { m_color = sf::Color::Red; }
    else if(color == 4) { m_color = sf::Color::Yellow; }
    else if(color == 5) { m_color = sf::Color::Green; }
}

Particule_Ball::Particule_Ball(const float X, const float Y) : Particule(X,Y,sf::Triangles, 30), m_posInit(sf::Vector2f(m_pos.x, m_pos.y)), m_vitInit(sf::Vector2f(m_vit.x, m_vit.y)),
m_timeElapsedJump(0), m_radius(15), m_radiusInit(m_radius), m_masse(10)
{
    int color = randomi(1,5);
    m_lifeTimeMax = 2.f;

    if(color == 1) { m_color = sf::Color::Blue; }
    else if(color == 2) { m_color = sf::Color::Cyan; }
    else if(color == 3) { m_color = sf::Color::Red; }
    else if(color == 4) { m_color = sf::Color::Yellow; }
    else if(color == 5) { m_color = sf::Color::Green; }
}

Particule_Ball::Particule_Ball(const float X,const float Y, const float dureeVie): Particule(X,Y,sf::Triangles, 30, dureeVie), m_posInit(sf::Vector2f(m_pos.x, m_pos.y)), m_vitInit(sf::Vector2f(m_vit.x, m_vit.y)),
m_timeElapsedJump(0), m_radius(15), m_radiusInit(m_radius), m_masse(10)
{
    int color = randomi(1,5);

    if(color == 1) { m_color = sf::Color::Blue; }
    else if(color == 2) { m_color = sf::Color::Cyan; }
    else if(color == 3) { m_color = sf::Color::Red; }
    else if(color == 4) { m_color = sf::Color::Yellow; }
    else if(color == 5) { m_color = sf::Color::Green; }
}

void Particule_Ball::update(const sf::Time& deltaTime)
{
    m_timeElapsedJump += deltaTime.asSeconds();
    m_lifeTime += deltaTime.asSeconds();

    if(m_lifeTime>m_lifeTimeMax) { m_isDead = true; }

    processPhysics();

    processCollision();

    m_radius = m_radiusInit * (m_lifeTimeMax-m_lifeTime)/m_lifeTimeMax;

    for(int i=0; i < int(m_forme.getVertexCount()) ; i+=3)
    {
        m_forme[i].position = sf::Vector2f(m_pos.x,m_pos.y);

        // cos et sin en radian
                                               /* V  Combien de degré pour chaque part du cercle ? (30/3 parts) donc 360/10 = 36*/
        m_forme[i+1].position = sf::Vector2f(cos(36*(i/3)* PI/180)*m_radius + m_pos.x, sin(36*(i/3)* PI/180)*m_radius + m_pos.y);
        m_forme[i+2].position = sf::Vector2f(cos(36*(i/3+1)* PI/180)*m_radius + m_pos.x, sin(36*(i/3+1)* PI/180)*m_radius + m_pos.y);

        m_forme[i].color = m_color;
        m_forme[i+1].color = m_color;
        m_forme[i+2].color = m_color;
    }
}

void Particule_Ball::processPhysics(void) // Problème de la balle lancée en (x,y) à la vitesse V(v_x,v_y)
{
        m_pos.y = -((-0.5) * g * m_timeElapsedJump * m_timeElapsedJump + m_vitInit.y * m_timeElapsedJump) + m_posInit.y;
        //m_pos.x = m_vitInit.x * m_timeElapsedJump + m_posInit.x; // Cas sans frottements
        m_pos.x = (m_masse/h) * m_vitInit.x * (1-exp(-h * m_timeElapsedJump / m_masse)) + m_posInit.x; // Cas avec frottements (air)

        m_vit.y = (-1) * g * m_timeElapsedJump + m_vitInit.y;
        //m_vit.x = m_vitInit.x; // Cas sans frottements
        m_vit.x = m_vitInit.x * exp((-h*m_timeElapsedJump)/m_masse); // Cas avec frottements de (air)
}

void Particule_Ball::processCollision(void) // Rebond entre la balle et un support = lancer de la balle à une nouvelle vitesse au point de collision
{
    if(m_pos.y - m_radius < 0) {
        m_pos.y = m_radius;
        m_vit.y = m_vit.y*(-1);
    }
    else if(m_pos.y + m_radius > sf::VideoMode::getDesktopMode().height) {
        if(m_vit.y > 0 && m_vit.y < 5) {// Cas ou la balle ne rebondit plus
            m_pos.y = sf::VideoMode::getDesktopMode().height-m_radius;
            m_vit.y = 0;
        }
        else { // Cas du rebond
            m_pos.y = sf::VideoMode::getDesktopMode().height-m_radius;
            m_vit.y = m_vit.y*(-1) * 0.65 /* Facteur d'Adhérence du sol */;
        }
    }

    if(m_pos.x - m_radius < 0) {
        m_pos.x = m_radius;
        m_vit.x = m_vit.x * (-1);
    } // Rebond sur les cotés
    else if(m_pos.x + m_radius > sf::VideoMode::getDesktopMode().width) {
        m_pos.x = sf::VideoMode::getDesktopMode().width - m_radius;
        m_vit.x = m_vit.x * (-1);
    }
    if(abs(m_vit.x) < 10) {
        m_vit.x = 0;
    }

    launch(m_pos.x, m_pos.y, m_vit.x, m_vit.y);
}

void Particule_Ball::launch(const float X, const float Y, const float vit_X, const float vit_Y)
{
    m_pos = sf::Vector2f(X,Y);
    m_posInit = m_pos;

    m_vit = sf::Vector2f(vit_X,vit_Y);
    m_vitInit = m_vit;

    m_timeElapsedJump = 0;
}

Particule_Ball::~Particule_Ball() {}

    //////////////////////////////////////////////////////////////
    ////////////// Particule Snow ////////////////////////////////
    //////////////////////////////////////////////////////////////

Particule_Snow::Particule_Snow() : Particule(0,0,sf::Triangles, 30), m_radius(5), m_radiusInit(m_radius)
{
    m_color = sf::Color::White;
    m_lifeTimeMax = 4.f;
}

Particule_Snow::Particule_Snow(const float X, const float Y) : Particule(X,Y,sf::Triangles, 30), m_radius(5), m_radiusInit(m_radius)
{
    m_color = sf::Color::White;
    m_lifeTimeMax = 4.f;
}

Particule_Snow::Particule_Snow(const float X,const float Y, const float dureeVie) : Particule(X,Y,sf::Triangles, 30, dureeVie), m_radius(5), m_radiusInit(m_radius)
{
    m_color = sf::Color::White;
}

void Particule_Snow::update(const sf::Time& deltaTime)
{
    m_lifeTime += deltaTime.asSeconds();

    if(m_lifeTime>m_lifeTimeMax || m_pos.x < 0 || m_pos.x > 1920 + m_radius) { m_isDead = true; }

    processPhysics();
    processCollision();

    m_radius = m_radiusInit * (m_lifeTimeMax-m_lifeTime)/m_lifeTimeMax;

    for(int i=0; i < int(m_forme.getVertexCount()) ; i+=3)
    {
        m_forme[i].position = sf::Vector2f(m_pos.x,m_pos.y);
        m_forme[i+1].position = sf::Vector2f(cos(36*(i/3)* PI/180)*m_radius + m_pos.x, sin(36*(i/3)* PI/180)*m_radius + m_pos.y);
        m_forme[i+2].position = sf::Vector2f(cos(36*(i/3+1)* PI/180)*m_radius + m_pos.x, sin(36*(i/3+1)* PI/180)*m_radius + m_pos.y);

        m_forme[i].color = m_color;
        m_forme[i+1].color = m_color;
        m_forme[i+2].color = m_color;
    }

}

void Particule_Snow::processPhysics(void) // Problème de la balle lancée en (x,y) à la vitesse V(v_x,v_y)
{
    m_pos.x += m_vit.x;
    m_pos.y += m_vit.y;
}

void Particule_Snow::launch(const float X, const float Y, const float vit_X, const float vit_Y)
{
    m_pos = sf::Vector2f(randomf(0.f, 1920.f),-1);
    m_vit = sf::Vector2f(randomf(-2.f,2.f),randomf(2,7));
}

Particule_Snow::~Particule_Snow() {}

    //////////////////////////////////////////////////////////////
    ////////////// Particule Dot ////////////////////////////////
    //////////////////////////////////////////////////////////////

Particule_Dot::Particule_Dot() : Particule(0,0,sf::Quads, 4), m_taille(2) {
    m_color = sf::Color::Yellow;
    m_lifeTimeMax = 2.f;
}

Particule_Dot::Particule_Dot(const float X, const float Y) : Particule(X,Y,sf::Quads, 4), m_taille(2) {
    m_color = sf::Color::Yellow;
    m_lifeTimeMax = 2.f;
}

Particule_Dot::Particule_Dot(const float X,const float Y, const float dureeVie) : Particule(X,Y,sf::Quads, 4, dureeVie), m_taille(2) {
    m_color = sf::Color::Yellow;
}

void Particule_Dot::update(const sf::Time& deltaTime)
{
    m_lifeTime += deltaTime.asSeconds();

    if(m_lifeTime>m_lifeTimeMax || m_pos.x < 0 || m_pos.x > 1920 || m_pos.y < 0 || m_pos.y > 1080) { m_isDead = true; }

    processPhysics();
    processCollision();

    m_forme[0].position = sf::Vector2f(m_pos.x - m_taille/2, m_pos.y - m_taille/2);
    m_forme[1].position = sf::Vector2f(m_pos.x + m_taille/2, m_pos.y - m_taille/2);
    m_forme[2].position = sf::Vector2f(m_pos.x + m_taille/2, m_pos.y + m_taille/2);
    m_forme[3].position = sf::Vector2f(m_pos.x - m_taille/2, m_pos.y + m_taille/2);

    m_forme[0].color = m_color;
    m_forme[1].color = m_color;
    m_forme[2].color = m_color;
    m_forme[3].color = m_color;
}

void Particule_Dot::processPhysics(void) // Problème de la balle lancée en (x,y) à la vitesse V(v_x,v_y)
{
    m_pos.x += m_vit.x;
    m_pos.y += m_vit.y;
}

void Particule_Dot::launch(const float X, const float Y, const float vit_X, const float vit_Y)
{
    m_pos = sf::Vector2f(X,Y);
    m_vit = sf::Vector2f(randomf(-10.f,10.f),randomf(-10.f,10.f));
}

Particule_Dot::~Particule_Dot() {}

    //////////////////////////////////////////////////////////////
    ////////////// Particule Fire ////////////////////////////////
    //////////////////////////////////////////////////////////////

Particule_Fire::Particule_Fire() : Particule(0,0,sf::Quads, 4), m_taille(5), phase(randomf(0,6)), amplitude(randomf(0.2,0.5)) {
    int color = randomi(1,3);

    if(color == 1) { m_color = sf::Color::Red; }
    else if(color == 2) { m_color = sf::Color::Yellow; }
    else if(color == 3) { m_color = sf::Color(255,128,0); }

    m_lifeTimeMax = 0.5f;
}

Particule_Fire::Particule_Fire(const float X, const float Y) : Particule(X,Y,sf::Quads, 4), m_taille(5), phase(randomf(0,6)), amplitude(randomf(0.2,0.5))  {
    int color = randomi(1,3);

    if(color == 1) { m_color = sf::Color::Red; }
    else if(color == 2) { m_color = sf::Color::Yellow; }
    else if(color == 3) { m_color = sf::Color(255,128,0); }

    m_lifeTimeMax = 0.5f;
}

Particule_Fire::Particule_Fire(const float X,const float Y, const float dureeVie) : Particule(X,Y,sf::Quads, 4, randomf(dureeVie-0.2f,dureeVie+0.2f)), m_taille(5), phase(randomf(0,6)), amplitude(randomf(0.2,0.5))  {
    int color = randomi(1,3);

    if(color == 1) { m_color = sf::Color::Red; }
    else if(color == 2) { m_color = sf::Color::Yellow; }
    else if(color == 3) { m_color = sf::Color(255,128,0); }
}

void Particule_Fire::update(const sf::Time& deltaTime)
{
    m_lifeTime += deltaTime.asSeconds();

    if(m_lifeTime>m_lifeTimeMax) { m_isDead = true; }

    processPhysics();
    processCollision();

    m_forme[0].position = sf::Vector2f(m_pos.x - m_taille/2, m_pos.y - m_taille/2);
    m_forme[1].position = sf::Vector2f(m_pos.x + m_taille/2, m_pos.y - m_taille/2);
    m_forme[2].position = sf::Vector2f(m_pos.x + m_taille/2, m_pos.y + m_taille/2);
    m_forme[3].position = sf::Vector2f(m_pos.x - m_taille/2, m_pos.y + m_taille/2);

    m_forme[0].color = m_color;
    m_forme[1].color = m_color;
    m_forme[2].color = m_color;
    m_forme[3].color = m_color;
}

void Particule_Fire::processPhysics(void) // Problème de la balle lancée en (x,y) à la vitesse V(v_x,v_y)
{
    m_pos.x += amplitude*cos(m_lifeTime*5+phase);
    m_pos.y += m_vit.y*0.5;
}

void Particule_Fire::launch(const float X, const float Y, const float vit_X, const float vit_Y)
{
    m_pos = sf::Vector2f(randomi(X-5,X+5), randomi(Y-5,Y+5));
    m_vit = sf::Vector2f(0,randomf(-2.f,-1.f));
}

Particule_Fire::~Particule_Fire() {}

    //////////////////////////////////////////////////////////////
    ////////////// Particule Trail ///////////////////////////////
    //////////////////////////////////////////////////////////////

Particule_Trail::Particule_Trail() : Particule(0,0,sf::Quads, 4), m_taille(30) {
    m_color = sf::Color(randomi(0,50),randomi(150,255), randomi(0,50));
    m_lifeTimeMax = 0.5f;
}

Particule_Trail::Particule_Trail(const float X, const float Y) : Particule(X,Y,sf::Quads, 4), m_taille(30) {
    m_color = sf::Color(randomi(0,50),randomi(150,255), randomi(0,50));
    m_lifeTimeMax = 0.5f;
}

Particule_Trail::Particule_Trail(const float X,const float Y, const float dureeVie) : Particule(X,Y,sf::Quads, 4, 0.35), m_taille(30) {
    m_color = sf::Color(randomi(0,50),randomi(150,255), randomi(0,50));
}

void Particule_Trail::update(const sf::Time& deltaTime)
{
    m_lifeTime += deltaTime.asSeconds();

    if(m_lifeTime>m_lifeTimeMax) { m_isDead = true; }

    processPhysics();
    processCollision();

    m_forme[0].position = sf::Vector2f(m_pos.x - m_taille * (1-(m_lifeTime/m_lifeTimeMax))/2, m_pos.y - m_taille * (1-(m_lifeTime/m_lifeTimeMax))/2);
    m_forme[1].position = sf::Vector2f(m_pos.x + m_taille * (1-(m_lifeTime/m_lifeTimeMax))/2, m_pos.y - m_taille * (1-(m_lifeTime/m_lifeTimeMax))/2);
    m_forme[2].position = sf::Vector2f(m_pos.x + m_taille * (1-(m_lifeTime/m_lifeTimeMax))/2, m_pos.y + m_taille * (1-(m_lifeTime/m_lifeTimeMax))/2);
    m_forme[3].position = sf::Vector2f(m_pos.x - m_taille * (1-(m_lifeTime/m_lifeTimeMax))/2, m_pos.y + m_taille * (1-(m_lifeTime/m_lifeTimeMax))/2);

    m_forme[0].color = m_color;
    m_forme[1].color = m_color;
    m_forme[2].color = m_color;
    m_forme[3].color = m_color;
}

void Particule_Trail::launch(const float X, const float Y, const float vit_X, const float vit_Y)
{
    m_pos = sf::Vector2f(randomf(X-20, X+20), randomf(Y-20, Y+20));
    m_vit = sf::Vector2f(0,0);
}

Particule_Trail::~Particule_Trail() {}

    //////////////////////////////////////////////////////////////
    ////////////// Particule Dot Falling//////////////////////////
    //////////////////////////////////////////////////////////////

Particule_Dot_Fall::Particule_Dot_Fall() : Particule(0,0,sf::Quads, 4), m_posInit(sf::Vector2f(m_pos.x, m_pos.y)), m_vitInit(sf::Vector2f(m_vit.x, m_vit.y)),
m_timeElapsedJump(0), m_taille(2), m_masse(10) {
    m_color = sf::Color::Yellow;
    m_lifeTimeMax = 2.f;
}

Particule_Dot_Fall::Particule_Dot_Fall(const float X, const float Y) : Particule(X,Y,sf::Quads, 4), m_posInit(sf::Vector2f(m_pos.x, m_pos.y)), 
m_vitInit(sf::Vector2f(m_vit.x, m_vit.y)), m_timeElapsedJump(0), m_taille(2), m_masse(10) {
    m_color = sf::Color::Yellow;
    m_lifeTimeMax = 2.f;
}

Particule_Dot_Fall::Particule_Dot_Fall(const float X,const float Y, const float dureeVie) : Particule(X,Y,sf::Quads, 4, dureeVie), m_posInit(sf::Vector2f(m_pos.x, m_pos.y)),
m_vitInit(sf::Vector2f(m_vit.x, m_vit.y)), m_timeElapsedJump(0), m_taille(2), m_masse(10) {
    m_color = sf::Color::Yellow;
}

void Particule_Dot_Fall::update(const sf::Time& deltaTime)
{
    m_timeElapsedJump += deltaTime.asSeconds();
    m_lifeTime += deltaTime.asSeconds();

    if(m_lifeTime>m_lifeTimeMax || m_pos.x < 0 || m_pos.x > 1920 || m_pos.y < 0 || m_pos.y > 1080) { m_isDead = true; }

    processPhysics();
    processCollision();

    m_forme[0].position = sf::Vector2f(m_pos.x - m_taille/2, m_pos.y - m_taille/2);
    m_forme[1].position = sf::Vector2f(m_pos.x + m_taille/2, m_pos.y - m_taille/2);
    m_forme[2].position = sf::Vector2f(m_pos.x + m_taille/2, m_pos.y + m_taille/2);
    m_forme[3].position = sf::Vector2f(m_pos.x - m_taille/2, m_pos.y + m_taille/2);

    m_forme[0].color = m_color;
    m_forme[1].color = m_color;
    m_forme[2].color = m_color;
    m_forme[3].color = m_color;
}

void Particule_Dot_Fall::processPhysics(void) // Problème de la balle lancée en (x,y) à la vitesse V(v_x,v_y)
{
        m_pos.y = -((-0.5) * g * m_timeElapsedJump * m_timeElapsedJump + m_vitInit.y * m_timeElapsedJump) + m_posInit.y;
        //m_pos.x = m_vitInit.x * m_timeElapsedJump + m_posInit.x; // Cas sans frottements
        m_pos.x = (m_masse/h) * m_vitInit.x * (1-exp(-h * m_timeElapsedJump / m_masse)) + m_posInit.x; // Cas avec frottements (air)

        m_vit.y = (-1) * g * m_timeElapsedJump + m_vitInit.y;
        //m_vit.x = m_vitInit.x; // Cas sans frottements
        m_vit.x = m_vitInit.x * exp((-h*m_timeElapsedJump)/m_masse); // Cas avec frottements de (air)
}

void Particule_Dot_Fall::launch(const float X, const float Y, const float vit_X, const float vit_Y)
{
    m_pos = sf::Vector2f(X,Y);
    m_posInit = m_pos;

    m_vit = sf::Vector2f(randomf(-700.f,700.f),randomf(-50.f,1000.f));
    m_vitInit = m_vit;

    m_timeElapsedJump = 0;
}

Particule_Dot_Fall::~Particule_Dot_Fall() {}