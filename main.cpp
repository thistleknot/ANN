// code is graciously reconstructed from http://www.ai-junkie.com/ann/evolved/nnt1.html

#include <iostream>
#include <vector>
#include <random>
#include <time.h>
#include <math.h> //abs

using namespace std;

//-------------------------------------used for the neural network
  static int    iNumInputs = 3;
  static int    iNumHidden = 2;
  static int    iNeuronsPerHiddenLayer = 3;
  static int    iNumOutputs = 3;

  //for tweeking the sigmoid function
  static double dActivationResponse = 1;
  //bias value
  static double dBias = 1;
//-------------------------------------used for the neural network

double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

struct SNeuron
{
    //the number of inputs into the neuron
    int m_NumInputs;

    //the weights for each input
    vector<double> m_vecWeight;

    //constructor
    SNeuron(int NumInputs);
};

SNeuron::SNeuron(int NumInputs): m_NumInputs(NumInputs+1)
{

    //we need an additional weight for the bias hence the +1
    for (int i=0; i<NumInputs+1; ++i)
    {

        double temp = fRand( -1.0, 1.0);

        //set up the weights with an initial random value
        m_vecWeight.push_back(temp);

        //cout << endl << i << ": " << temp << endl;
    }
}


struct SNeuronLayer
{

    //the number of neurons in this layer
    int m_NumNeurons;
    //the layer of neurons
    vector<SNeuron> m_vecNeurons;
    SNeuronLayer(int NumNeurons, int NumOnputsPerNeuron);

};

SNeuronLayer::SNeuronLayer(int NumNeurons,
                           int NumInputsPerNeuron):	m_NumNeurons(NumNeurons)
{
	for (int i=0; i<NumNeurons; ++i)

		m_vecNeurons.push_back(SNeuron(NumInputsPerNeuron));
}

class CNeuralNet
{
private:
    int m_NumInputs;
    int m_NumOutputs;
    int m_NumHiddenLayers;
    int m_NeuronsPerHiddenLyr;

    //storage for each layer of neurons including the output layer
    vector<SNeuronLayer> m_vecLayers;

public:
    CNeuralNet();

    //have a guess... ;
    void CreateNet();

    //gets weights from the NN
    vector<double> GetWeights()const;

    //returns the total number of weights in the net
    int GetNumberOfWeights()const;

    //replaces the weights with new ones
    void PutWeights(vector<double> &weights);

    //calculates the outputs from a set of inputs
    vector<double> Update(vector<double> &inputs);

    //sigmoid response curve
    inline double Sigmoid(double activation, double response);

};

//************************ methods forCNeuralNet ************************

//------------------------------default ctor ----------------------------
//
//	creates a ANN based on the default values in params.ini
//-----------------------------------------------------------------------
CNeuralNet::CNeuralNet()
{
	m_NumInputs	          =	iNumInputs;
	m_NumOutputs		      =	iNumOutputs;
	m_NumHiddenLayers	    =	iNumHidden;
	m_NeuronsPerHiddenLyr =	iNeuronsPerHiddenLayer;

	CreateNet();

}

//------------------------------createNet()------------------------------
//
//	this method builds the ANN. The weights are all initially set to
//	random values -1 < w < 1
//------------------------------------------------------------------------

void CNeuralNet::CreateNet()
{
	//create the layers of the network
	if (m_NumHiddenLayers > 0)
	{
		//create first hidden layer
	  m_vecLayers.push_back(SNeuronLayer(m_NeuronsPerHiddenLyr, m_NumInputs));

    for (int i=0; i<m_NumHiddenLayers-1; ++i)
    {

			m_vecLayers.push_back(SNeuronLayer(m_NeuronsPerHiddenLyr,
                                         m_NeuronsPerHiddenLyr));
    }

    //create output layer
	  m_vecLayers.push_back(SNeuronLayer(m_NumOutputs, m_NeuronsPerHiddenLyr));
	}

  else
  {
	  //create output layer
	  m_vecLayers.push_back(SNeuronLayer(m_NumOutputs, m_NumInputs));
  }
}

vector<double> CNeuralNet::GetWeights() const
{
	//this will hold the weights
	vector<double> weights;

	//for each layer
	for (int i=0; i<m_NumHiddenLayers + 1; ++i)
	{

		//for each neuron
		for (int j=0; j<m_vecLayers[i].m_NumNeurons; ++j)
		{
			//for each weight
			for (int k=0; k<m_vecLayers[i].m_vecNeurons[j].m_NumInputs; ++k)
			{
				weights.push_back(m_vecLayers[i].m_vecNeurons[j].m_vecWeight[k]);
			}
		}
	}

	return weights;
}

void CNeuralNet::PutWeights(vector<double> &weights)
{
	int cWeight = 0;

	//for each layer
	for (int i=0; i<m_NumHiddenLayers + 1; ++i)
	{

		//for each neuron
		for (int j=0; j<m_vecLayers[i].m_NumNeurons; ++j)
		{
			//for each weight
			for (int k=0; k<m_vecLayers[i].m_vecNeurons[j].m_NumInputs; ++k)
			{
				m_vecLayers[i].m_vecNeurons[j].m_vecWeight[k] = weights[cWeight++];
			}
		}
	}

	return;
}

int CNeuralNet::GetNumberOfWeights() const
{

	int weights = 0;

	//for each layer
	for (int i=0; i<m_NumHiddenLayers + 1; ++i)
	{

		//for each neuron
		for (int j=0; j<m_vecLayers[i].m_NumNeurons; ++j)
		{
			//for each weight
			for (int k=0; k<m_vecLayers[i].m_vecNeurons[j].m_NumInputs; ++k)

				weights++;

		}
	}

	return weights;
}

double CNeuralNet::Sigmoid(double netinput, double response)
{
	return (netinput / (1+(abs(netinput))));
	//return ( 1 / ( 1 + exp(-netinput / response)));
}

vector<double> CNeuralNet::Update(vector<double> &inputs)
{
	//stores the resultant outputs from each layer
	vector<double> outputs;

	int cWeight = 0;

	//first check that we have the correct amount of inputs
	if (inputs.size() != m_NumInputs)
  {
		//just return an empty vector if incorrect.
		return outputs;
  }

	//For each layer....
	for (int i=0; i<m_NumHiddenLayers + 1; ++i)
	{
		if ( i > 0 )
    {
			inputs = outputs;
    }

		outputs.clear();

		cWeight = 0;

		//for each neuron sum the (inputs * corresponding weights).Throw
		//the total at our sigmoid function to get the output.
		for (int j=0; j<m_vecLayers[i].m_NumNeurons; ++j)
		{
			double netinput = 0;

			int	NumInputs = m_vecLayers[i].m_vecNeurons[j].m_NumInputs;

			//for each weight
			for (int k=0; k<NumInputs - 1; ++k)
			{
				//sum the weights x inputs
				netinput += m_vecLayers[i].m_vecNeurons[j].m_vecWeight[k] *
                    inputs[cWeight++];
			}

			//add in the bias
			netinput += m_vecLayers[i].m_vecNeurons[j].m_vecWeight[NumInputs-1] *
                  dBias;

			//we can store the outputs from each layer as we generate them.
      //The combined activation is first filtered through the sigmoid
      //function
			outputs.push_back(Sigmoid(netinput,
                                dActivationResponse));

			cWeight = 0;
		}
	}

	return outputs;
}

int main()
{
    srand(time(NULL));

    /*
    for (int i=0; i<10;++i)
    {
        double temp = fRand(-1.0, 1.0);
        cout << "number: " << temp << endl;
    }
    */

    CNeuralNet * testNet = new CNeuralNet;

    vector<double> inputs;
    vector<double> outputs;


    inputs.push_back(1);
    inputs.push_back(0);
    inputs.push_back(1);

    outputs = testNet->Update(inputs);

    cout << endl << "outputs: " << endl;

    for (int i = 0; i< outputs.size(); i++)
    {
        cout << endl << outputs[i] << endl;
    }

    inputs.clear();
    inputs.push_back(0);
    inputs.push_back(1);
    inputs.push_back(0);

    outputs = testNet->Update(inputs);

    cout << endl << "outputs: " << endl;

    for (int i = 0; i< outputs.size(); i++)
    {
        cout << endl << outputs[i] << endl;
    }

    inputs.clear();

    inputs.push_back(1);
    inputs.push_back(0);
    inputs.push_back(1);

    outputs = testNet->Update(inputs);

    cout << endl << "outputs: " << endl;

    for (int i = 0; i< outputs.size(); i++)
    {
        cout << endl << outputs[i] << endl;
    }


    return 0;
}
