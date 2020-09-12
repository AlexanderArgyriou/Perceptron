#include <iostream>
#include <vector>
#include <random>

template<class T>
class Perceptron
{
	private:
		class TrainSet
		{
			public:
				std::vector<std::vector<T>> Inputs;
				float                       Bias;
				std::vector<T>              Results;

				inline TrainSet(const std::vector<std::vector<T>> &newInputs,
					const std::vector<T> &newResults) noexcept;	// Constructor
		};	// TrainSet

		class Weights
		{
			public:
				std::vector<float> Wt;

				inline Weights(std::size_t S) noexcept;	// Constructor
		};	// Weights

		//--- Private Methods ---//
		inline float CalculateWeightedSum(std::size_t Layer) const noexcept;	// CalculateWeightedSum

		//--- Private Data ---//
		TrainSet    TS;
		Weights     W;
		std::size_t Size;
		std::size_t Epochs;
		float       LR;

	public:
		inline static float GetRandom() noexcept;
		Perceptron(const std::vector<std::vector<T>> &In, 
			const std::vector<T> &Re, const std::size_t newSize,
			float newLR, std::size_t newEpochs);             // Constructor
		inline int Predict(std::size_t Layer) const noexcept;    // Predict
		inline void Train() noexcept;                            // Train
};	// Perceptron

// --- TrainSet Class --- //
template<class T>
inline Perceptron<T>::TrainSet::TrainSet(const std::vector<std::vector<T>> &newInputs,
	const std::vector<T> &newResults) noexcept
	: Inputs(newInputs), Results(newResults), Bias(1.0f)
{}	// Constructor

// --- Weights Class --- //
template<class T>
inline Perceptron<T>::Weights::Weights(std::size_t S) noexcept
	: Wt(S)
{
	for (std::size_t i = 0; i < Wt.size(); i++)
		Wt[i] = GetRandom();
}	// Constructor

// --- Perceptron Class, private section --- //
template<class T>
inline float Perceptron<T>::CalculateWeightedSum(std::size_t Layer) const noexcept
{
	float Ws = 0.0f;
	Ws += TS.Bias;

	for (std::size_t i = 0; i < Size; i++)
		Ws += TS.Inputs[Layer][i] * W.Wt[i];
	return Ws;
}	//  CalculateWeightedSum

// --- Perceptron Class, public section --- //
template<class T>
inline float Perceptron<T>::GetRandom() noexcept
{
	std::random_device               RD;
	std::mt19937                     Gen(RD());
	std::uniform_real_distribution<> Dist(0.0f, 1.0f);

	return (float)Dist(Gen);
}	// // GetRandom()

template<class T>
Perceptron<T>::Perceptron(const std::vector<std::vector<T>> &In, 
	const std::vector<T> &Re, 
	const std::size_t newSize,
	float newLR, std::size_t newEpochs)
	: TS(In, Re), W(newSize), Size(newSize), LR(newLR), Epochs(newEpochs)
{}	// Constructor

template<class T>
inline int Perceptron<T>::Predict(std::size_t Layer) const noexcept
{
	// Step
	return this->CalculateWeightedSum(Layer) > 0 ? 1 : 0;
}	// Predict()

template<class T>
inline void Perceptron<T>::Train() noexcept
{
	for (std::size_t i = 0; i < Epochs; i++)
		for (std::size_t j = 0; j < TS.Inputs.size(); j++)
		{
			float Update = LR * (TS.Results[j] - Predict(j)); // LR * Error
			for (std::size_t k = 0; k < W.Wt.size(); k++)
				W.Wt[k] += Update * TS.Inputs[j][k];

			TS.Bias = Update;
		}	// for
}	// Train
