#include <iostream>
#include <Matrix.h>
#include <Random.hpp>
#include <cmath>
#include <random>
#include <chrono>

void loadDataset(Matrix& X, Matrix& Y, const int nb_elements) {
// Création du dataset de plantes toxiques ou saine
	// en fonction de la longueur et de la largeur de leurs feuilles

		// X : attributs
		// x1 longueur feuille
		// x2 largeur feuille

		// Y : classes
		// classe 0: plante saine
		// classe 1: plante toxique

	for(int i=0; i<nb_elements; i++) {
		if(randomi(0,1))
		{
			float x1,x2;
			// Génération de données normalisées pour une plante toxique
			if(randomi(0,1)) x1 = randomf(0,20)/60.f;
			else x1 = randomf(40,60)/60.f;
			if(randomi(0,1)) x2 = randomf(0,5)/15.f;
			else x2 = randomf(10,15)/15.f;

			X.setCoeff(0,i,x1); // Longueur feuille
			X.setCoeff(1,i,x2); // Largeur feuille
			
			Y.setCoeff(0,i,1);
		}
		else {
			// Génération de données normalisées pour une plante saine
			X.setCoeff(0,i,randomf(20,40)/60.f); // Longueur feuille
			X.setCoeff(1,i,randomf(5,10)/15.f); // Largeur feuille
			Y.setCoeff(0,i,0);
		}
	}
}

// Réseaux à deux couches
//  
//	w1 ___o
//	   \ / \
//		x	o -> Sortie
//	   / \ /
//	w2 __ o
//

/*
	Dataset: 
	  - 2 classes : toxique et sain
	  - 2 attributs : longueur et largeur feuilles
*/

int main() {
	
	// Phase d'initialisation
	srand(time(NULL));
	Matrix X=Matrix(2, NB_ELEMENTS);
	Matrix Y=Matrix(1, NB_ELEMENTS);
	loadDataset(X, Y, NB_ELEMENTS);

	Matrix W1 = Matrix(2,2, 0.0f);
	Matrix b1 = Matrix(2,X.col(), 1.0f);
	Matrix W2 = Matrix(1,2, 0.0f);
	Matrix b2 = Matrix(1,X.col(), 1.0f);

	for(int i=0; i<W1.row(); i++) {
		for(int j=0; j<W1.col(); j++) {
			W1.setCoeff(i,j, randomf(0,5));
		}	
	}
	
	float val1=randomf(0,5), val2=randomf(0,5);
	for(int i=0; i<b1.row(); i++) {
		for(int j=0; j<b1.col(); j++) {
			if(i) b1.setCoeff(i,j, val1);
			else b1.setCoeff(i,j, val2);
		}
	}

	for(int i=0; i<W2.row(); i++) {
		for(int j=0; j<W2.col(); j++) {
			W2.setCoeff(i,j, randomf(0,5));
		}	
	}

	val1=randomf(0,5);
	for(int j=0; j<b2.col(); j++) {
		b2.setCoeff(0,j, val1);
	}

	W1.disp();
	b1.disp();
	W2.disp();
	b2.disp();

	return 0;
}