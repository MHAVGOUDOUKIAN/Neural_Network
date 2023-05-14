#include <Application/App.hpp>

void App::loadDataset( Matrix& X, Matrix& Y) {
    // Création du dataset de plantes toxiques ou saine
	// en fonction de la longueur et de la largeur de leurs feuilles

		// X : attributs
		// x1 longueur feuille
		// x2 largeur feuille

		// Y : classes
		// classe 0: plante saine
		// classe 1: plante toxique
    l_dataset.clear();
    float x1,x2, y;
    for(int i=0; i<NB_ELEMENTS; i++) {
        x2 = randomf(0,15);
        x1 = randomf(0,60);
		if(randomi(0,1))
		{
			// Génération de données normalisées pour une plante toxique
            while(true) {
                if(x1>=20 and x1<=40) {
                    if(x2>=5 and x2<=10) {
                        x2 = randomf(0,15);
                        x1 = randomf(0,60);
                    }
                }
                else break;
                if(x2>=5 and x2<=10) {
                    if(x1>=20 and x1<=40) {
                        x2 = randomf(0,15);
                        x1 = randomf(0,60);
                    }
                } else break;
            }

			X.setCoeff(0,i,x1/60.f); // Longueur feuille
			X.setCoeff(1,i,x2/15.f); // Largeur feuille
			
            y=1;
			Y.setCoeff(0,i,y);
		}
		else {
			// Génération de données normalisées pour une plante saine
            while(true) {
                if(x1<20 or x1>40) x1 = randomf(20,40);
                else if(x2<5 or x2>10) x2 = randomf(5,10);
                else break;
            }
			X.setCoeff(0,i,x1/60.f); // Longueur feuille
			X.setCoeff(1,i,x2/15.f); // Largeur feuille
            y=0;
			Y.setCoeff(0,i,y);
		}
        l_dataset.push_back(sf::Vector3f(x1,x2,y));
	}
}

App::App() : X(Matrix(2, NB_ELEMENTS)), 
Y(Matrix(1, NB_ELEMENTS)),
W1(Matrix(C1,NB_ENTREE)),
b1(Matrix(C1,1)),
W2(Matrix(C2,C1)),
b2(Matrix(C2,1)),
LearningPhase(true)
{
    srand(time(NULL));
	loadDataset(X, Y);

    EventHandler::getEventHandler()->addKeyBoardObserver(this);
    std::cout << "Learning phase started" << std::endl;
    std::cout << "Please wait.." << std::endl;
}

void App::update(sf::Time deltaTime) {
    if(LearningPhase) {
        // Forward propagation
        Matrix Z1=W1*X;
        Z1 = BroadCastAdd(Z1,b1);
        Matrix A1=Z1;
        A1.applySigmo();

        Matrix Z2=W2*A1;
        Z2 = BroadCastAdd(Z2,b2);
        Matrix A2=Z2;
        A2.applySigmo();

        float loss=0;
        for(int i=0; i<NB_ELEMENTS; i++) {
            loss+=Y.getCoeff(0,i)*log(A2.getCoeff(0,i))+(1-Y.getCoeff(0,i))*log(1-A2.getCoeff(0,i));
        }
        loss *= (-1/NB_ELEMENTS);
        std::cout << "Loss A2: " << loss << std::endl;

        // Back propagation
        Matrix dZ2= A2;
        dZ2-=Y;

        Matrix dW2 = dZ2*A1.transposee();
        dW2.constMult(1.0f/NB_ELEMENTS);

        Matrix db2 = SumOnCol(dZ2);
        db2.constMult(1.0f/NB_ELEMENTS);

        Matrix dZ1 = W2.transposee()*dZ2;
        Matrix temp = A1;
        temp.constMult(-1.0f);
        temp = BroadCastAdd(temp, Matrix(1,1,1.0f));
        temp = Hadamard(A1, temp);
        dZ1 = Hadamard(dZ1,temp);

        Matrix dW1 = dZ1*X.transposee();
        dW1.constMult(1.0f/NB_ELEMENTS);

        Matrix db1 = SumOnCol(dZ1);
        db1.constMult(1.0f/NB_ELEMENTS);

        //Update NN
        dW1.constMult(LEARNING_RATE);
        dW2.constMult(LEARNING_RATE);
        W1-=dW1;
        W2-=dW2;
        db1.constMult(LEARNING_RATE);
        db2.constMult(LEARNING_RATE);
        b1-=db1;
        b2-=db2;

        compt++;
        if(compt>= EPOCH) {
            float loss;
            for(int i=0; i<C1; i++) {
                loss=0;
                for(int j=0; j<NB_ELEMENTS; j++) {
                    loss+=Y.getCoeff(0,j)*log(A1.getCoeff(i,j))+(1-Y.getCoeff(0,j))*log(1-A1.getCoeff(i,j));
                }
                loss *= (-1/NB_ELEMENTS);
                std::cout << "Final Loss A1_" << i+1 << ": " << loss << std::endl; 
            }

            LearningPhase=false;
            std::cout << "Training Finished" << std::endl;
        }
    }
}

void App::notify(sf::Keyboard::Key key, bool pressed) {
    if(key == sf::Keyboard::Space && !LearningPhase && pressed) {
        // Prédictions
        std::cout << "Predictions:" << std::endl;
        Matrix X_Test = Matrix(2, NB_ELEMENTS);
        Matrix Y_Test = Matrix(1, NB_ELEMENTS);
        loadDataset(X_Test, Y_Test);

        Matrix Z1=W1*X_Test;
        Z1 = BroadCastAdd(Z1,b1);
        Matrix A1=Z1;
        A1.applySigmo();

        Matrix Z2=W2*A1;
        Z2 = BroadCastAdd(Z2,b2);
        Matrix A2=Z2;
        A2.applySigmo();

        std::cout << "X test" << std::endl;
        X_Test.disp();
        std::cout << "Y test" << std::endl;
        Y_Test.disp();
        std::cout << "probability results" << std::endl;
        A2.disp();

        float nb_error=0;
        for(int i=0; i<A2.col(); i++) {
            if(A2.getCoeff(0,i)>0.5) std::cout << "1 / ";
            else std::cout << "0 / ";   
            
            if(A2.getCoeff(0,i)>0.5f and Y_Test.getCoeff(0,i)==0) nb_error++;
            else if(A2.getCoeff(0,i)<=0.5f and Y_Test.getCoeff(0,i)==1) nb_error++;

        }
        std::cout << std::endl << "Nb_error= " << nb_error << std::endl;
    }
}

void App::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    for(sf::Vector3f p: l_dataset) {
        sf::RectangleShape rect;
        rect.setPosition(p.x*ZOOM, p.y*ZOOM);
        rect.setSize(sf::Vector2f(10,10));
        if(p.z) rect.setFillColor(sf::Color::Cyan); else rect.setFillColor(sf::Color::Yellow);
        target.draw(rect);
    }
}

App::~App() {}