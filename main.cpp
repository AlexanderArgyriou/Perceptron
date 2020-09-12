#include <iostream>
#include "Perceptron.h"

int main()
{
	const std::size_t Epochs = 20'000;
	// Fun set based on Rectangle sides or Square sides
	std::vector< std::vector<int> > TS
	{
		{1,1,1,1},	// Square 0
		{2,1,2,1},	// Rectangle 1
		{2,2,2,2},	// Square 0
		{3,1,3,1},	// Rectangle 1
		{4,4,4,4},	// Square 0
		{7,7,7,7},	// Square 0
		{5,2,5,2},	// Rectangle 1
		{10,6,10,6} // Rectangle 1
	};
	std::vector<int> RE{ 0,1,0,1,0,0,1,1 };

	Perceptron<int> P(TS, RE, TS[0].size(), Perceptron<int>::GetRandom(), Epochs);
	P.Train();

	for (std::size_t i = 0; i < TS.size(); i++)
	{
		if( P.Predict(i) )
			std::cout << "Rectangle" << std::endl;
		else
			std::cout << "Square" << std::endl;
	}	// for

	return 0;
}	// main

