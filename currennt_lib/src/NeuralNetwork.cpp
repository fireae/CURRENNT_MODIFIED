/******************************************************************************
 * Copyright (c) 2013 Johannes Bergmann, Felix Weninger, Bjoern Schuller
 * Institute for Human-Machine Communication
 * Technische Universitaet Muenchen (TUM)
 * D-80290 Munich, Germany
 *
 * This file is part of CURRENNT.
 *
 * CURRENNT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CURRENNT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *
 * You should have received a copy of the GNU General Public License
 * along with CURRENNT.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#include "NeuralNetwork.hpp"
#include "Configuration.hpp"
#include "LayerFactory.hpp"
#include "layers/InputLayer.hpp"
#include "layers/PostOutputLayer.hpp"
#include "layers/FeedBackLayer.hpp"
#include "helpers/JsonClasses.hpp"
#include "MacroDefine.hpp"
#include "helpers/misFuncs.hpp"
#include <vector>
#include <stdexcept>
#include <cassert>

#include <boost/foreach.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>


template <typename TDevice>
NeuralNetwork<TDevice>::NeuralNetwork(
 const helpers::JsonDocument &jsonDoc,
 int parallelSequences, 
 int maxSeqLength,
 int chaDim,
 int maxTxtLength,
 int inputSizeOverride,
 int outputSizeOverride
 )
{
    try {

	//
	const Configuration &config = Configuration::instance();
	
        // check the layers and weight sections
        if (!jsonDoc->HasMember("layers"))
            throw std::runtime_error("Missing section 'layers'");
        rapidjson::Value &layersSection  = (*jsonDoc)["layers"];

        if (!layersSection.IsArray())
            throw std::runtime_error("Section 'layers' is not an array");

        helpers::JsonValue weightsSection;
        if (jsonDoc->HasMember("weights")) {
            if (!(*jsonDoc)["weights"].IsObject())
                throw std::runtime_error("Section 'weights' is not an object");
            weightsSection = helpers::JsonValue(&(*jsonDoc)["weights"]);
        }
	
	int cnt = 0;
	
	// Add 1220, support to the FeedBackLayer
	std::vector<int> feedBacklayerId; // layer Idx for the FeedBackLayer
	feedBacklayerId.clear();
	bool flagMDNOutput = false;
	m_firstFeedBackLayer = -1;
	
        // extract the layers
        for (rapidjson::Value::ValueIterator layerChild = layersSection.Begin(); 
	     layerChild != layersSection.End(); 
	     ++layerChild, cnt++){
	    
            printf("\nLayer (%d)", cnt);
	    
	    // check the layer child type
            if (!layerChild->IsObject())
                throw std::runtime_error("A layer section in the 'layers' array is not an object");

            // extract the layer type and create the layer
            if (!layerChild->HasMember("type"))
                throw std::runtime_error("Missing value 'type' in layer description");
	    
            std::string layerType = (*layerChild)["type"].GetString();
	    printf(" %s ", layerType.c_str());

            // override input/output sizes
            if (inputSizeOverride > 0 && layerType == "input"){
		// with WE 
		if (config.weUpdate() && config.trainingMode())
		    inputSizeOverride += (config.weDim() - 1);
		(*layerChild)["size"].SetInt(inputSizeOverride);
            }
	    
	    /*  Does not work yet, need another way to identify a) postoutput layer (last!) and 
                then the corresponging output layer and type!
		if (outputSizeOverride > 0 && (*layerChild)["name"].GetString() == "output") {
		(*layerChild)["size"].SetInt(outputSizeOverride);
		}
		if (outputSizeOverride > 0 && (*layerChild)["name"].GetString() == "postoutput") {
		(*layerChild)["size"].SetInt(outputSizeOverride);
		}
	    */
	    
            try {
            	layers::Layer<TDevice> *layer;
		
		/* Add 02-24 Wang for Residual Network*/
		/*
                if (m_layers.empty())
		layer = LayerFactory<TDevice>::createLayer(layerType, 
		&*layerChild, weightsSection, parallelSequences, maxSeqLength);
                else
                    layer = LayerFactory<TDevice>::createLayer(layerType, 
		    &*layerChild, weightsSection, 
		    parallelSequences, maxSeqLength, m_layers.back().get()); */
		
		// first layer
                if (m_layers.empty()){
		    
		    if (layerType == "skipadd"           || layerType == "skipini"       || 
			layerType == "skippara_logistic" || layerType == "skippara_relu" || 
			layerType == "skippara_tanh"     || 
			layerType == "skippara_identity")
		    {
			printf("SkipAdd, SkipPara can not be the first hidden layer");
			throw std::runtime_error("Error in network.jsn: layer type error\n");
		    }
		    
		    layer = LayerFactory<TDevice>::createLayer(
				layerType,     &*layerChild, 
				weightsSection, parallelSequences, 
				maxSeqLength,   chaDim,   maxTxtLength);

		}else if(layerType == "skipadd"           || layerType == "skipini"       ||
			 layerType == "skippara_logistic" || layerType == "skippara_relu" || 
			 layerType == "skippara_tanh"     || 
			 layerType == "skippara_identity")
		{

		    // SkipLayers: all the layers that link to the current skip layer
		    //  here, it includes the last skip layer and the previous normal 
		    //  layer connected to this skip layer
		    std::vector<layers::Layer<TDevice>*> SkipLayers;
		    // for skipadd layer:
		    //   no need to check whether the last skiplayer is directly 
		    //   connected to current skiplayer
		    //   in that case, F(x) + x = 2*x, the gradients will be multiplied by 2
		    // for skippara layer:
		    //   need to check, because H(x)*T(x)+x(1-T(x)) = x if H(x)=x
		    //   check it in SkipParaLayer.cu
		    if (m_skipAddLayers.size()>0 && layerType != "skipini") {
			SkipLayers.push_back(m_skipAddLayers.back());
		    }
		    SkipLayers.push_back(m_layers.back().get());
		    
		    if (layerType == "skipadd" || layerType == "skipini")
			layer = LayerFactory<TDevice>::createSkipAddLayer(
				  layerType,     &*layerChild,
				  weightsSection, parallelSequences, 
				  maxSeqLength,   SkipLayers);
		    else
			layer = LayerFactory<TDevice>::createSkipParaLayer(
				  layerType,     &*layerChild,
				  weightsSection, parallelSequences, 
				  maxSeqLength,   SkipLayers);
		    
		    // add the skipadd layer to Network buffer
		    m_skipAddLayers.push_back(layer);
		
		}else{
		    // other layers types
                    layer = LayerFactory<TDevice>::createLayer(
			       layerType,      &*layerChild,
			       weightsSection, parallelSequences, 
			       maxSeqLength,   chaDim, maxTxtLength, 
			       m_layers.back().get());
		    
		    if (layerType == "mdn")
			flagMDNOutput = true;
		}
		
                m_layers.push_back(boost::shared_ptr<layers::Layer<TDevice> >(layer));
		
		// Add 1220
		if (layerType == "feedback"){
		    feedBacklayerId.push_back(cnt);
		    m_firstFeedBackLayer = cnt;
		}
            }
            catch (const std::exception &e) {
                throw std::runtime_error(std::string("Could not create layer: ") + e.what());
            }
        }
	
	
	
        // check if we have at least one input, one output and one post output layer
        if (m_layers.size() < 3)
            throw std::runtime_error("Error in network.jsn: there must be a hidden layer\n");

        // check if only the first layer is an input layer
        if (!dynamic_cast<layers::InputLayer<TDevice>*>(m_layers.front().get()))
            throw std::runtime_error("The first layer is not an input layer");

        for (size_t i = 1; i < m_layers.size(); ++i) {
            if (dynamic_cast<layers::InputLayer<TDevice>*>(m_layers[i].get()))
                throw std::runtime_error("Multiple input layers defined");
        }

        // check if only the last layer is a post output layer
        if (!dynamic_cast<layers::PostOutputLayer<TDevice>*>(m_layers.back().get()))
            throw std::runtime_error("The last layer is not a post output layer");

        for (size_t i = 0; i < m_layers.size()-1; ++i) {
            if (dynamic_cast<layers::PostOutputLayer<TDevice>*>(m_layers[i].get()))
                throw std::runtime_error("Multiple post output layers defined");
        }

        // check if two layers have the same name
        for (size_t i = 0; i < m_layers.size(); ++i) {
            for (size_t j = 0; j < m_layers.size(); ++j) {
                if (i != j && m_layers[i]->name() == m_layers[j]->name())
                    throw std::runtime_error(
			std::string("Error in network.jsn: different layers have the name '") + 
			m_layers[i]->name() + "'");
            }
        }
	
	// Link the target layer with the feedback layer
	if (!feedBacklayerId.empty()){
	    for (size_t i = 0; i<feedBacklayerId.size(); i++){
		m_layers[feedBacklayerId[i]]->linkTargetLayer(*(m_layers.back().get()));
	    }
	    // check the bi-directional rnn
	    for (size_t i = m_firstFeedBackLayer; i < m_layers.size()-1; i++){
		if (m_layers[i]->type()==std::string("brnn") ||
		    m_layers[i]->type()==std::string("blstm"))
		    throw std::runtime_error(
			 std::string("Error in network.jsn.") +
			 std::string("brnn and blstm can't be used with feedback."));
	    }
	}
	
    }
    catch (const std::exception &e) {
        throw std::runtime_error(std::string("Invalid network file: ") + e.what());
    }
}

template <typename TDevice>
NeuralNetwork<TDevice>::~NeuralNetwork()
{
}

template <typename TDevice>
const std::vector<boost::shared_ptr<layers::Layer<TDevice> > >& NeuralNetwork<TDevice>::layers() const
{
    return m_layers;
}

template <typename TDevice>
layers::InputLayer<TDevice>& NeuralNetwork<TDevice>::inputLayer()
{
    return static_cast<layers::InputLayer<TDevice>&>(*m_layers.front());
}

/* Modify 04-08 to tap in the output of arbitary layer */
/*template <typename TDevice>
  layers::TrainableLayer<TDevice>& NeuralNetwork<TDevice>::outputLayer()
  {
    return static_cast<layers::TrainableLayer<TDevice>&>(*m_layers[m_layers.size()-2]);
  }
*/

template <typename TDevice>
layers::Layer<TDevice>& NeuralNetwork<TDevice>::outputLayer(const int layerID)
{
    // default case, the output layer
    int tmpLayerID = layerID;
    if (tmpLayerID < 0)
	tmpLayerID = m_layers.size()-2;
    
    // check
    if (tmpLayerID > (m_layers.size()-1))
	throw std::runtime_error(std::string("Invalid output_tap ID (out of range)"));
    
    return (*m_layers[tmpLayerID]);
}

template <typename TDevice>
layers::SkipLayer<TDevice>* NeuralNetwork<TDevice>::outGateLayer(const int layerID)
{
    // default case, the output
    int tmpLayerID = layerID;
    
    // check
    if (tmpLayerID > (m_layers.size()-2) || tmpLayerID < 0)
	throw std::runtime_error(std::string("Invalid gate_output_tap ID (out of range)"));
    
    return dynamic_cast<layers::SkipLayer<TDevice>*>(m_layers[tmpLayerID].get());
}

template <typename TDevice>
layers::MDNLayer<TDevice>* NeuralNetwork<TDevice>::outMDNLayer()
{
    return dynamic_cast<layers::MDNLayer<TDevice>*>(m_layers[m_layers.size()-1].get());
}

template <typename TDevice>
layers::PostOutputLayer<TDevice>& NeuralNetwork<TDevice>::postOutputLayer()
{
    return static_cast<layers::PostOutputLayer<TDevice>&>(*m_layers.back());
}

template <typename TDevice>
void NeuralNetwork<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction)
{
    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers)
    {
        layer->loadSequences(fraction);
    }
}

template <typename TDevice>
void NeuralNetwork<TDevice>::restoreTarget(const data_sets::DataSetFraction &fraction)
{
    const Configuration &config = Configuration::instance();

    if (config.scheduleSampOpt() == NN_FEEDBACK_SC_SOFT ||
	config.scheduleSampOpt() == NN_FEEDBACK_SC_MAXONEHOT){
        m_layers[m_layers.size()-1]->loadSequences(fraction);
    }
}


template <typename TDevice>
void NeuralNetwork<TDevice>::computeForwardPass(const int curMaxSeqLength,
						const real_t uttCnt)
{
    // |
    // |- No feedback, normal forward and recurrent computation
    // |- Feedback layer exists
    //    |- Case 0: use only ground truth as feedback data
    //    |- Case 1: use schedule uniform initialization ( 1/N )
    //    |- Case 2: use schedule back-off (set to zero)
    //    |- Case 3: use schedule sampling, soft-vector feedback
    //    |- Case 4: use schedule sampling, one-hot feedback
    //
    
    const Configuration &config = Configuration::instance();

    // No feedback, normal forward computation
    if (m_firstFeedBackLayer <= 0){
	BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers)
	    layer->computeForwardPass();

    // Other cases, Feedback exists
    }else {
	
	// prepare random numbers
    	static boost::mt19937 *gen = NULL;
	if (!gen) {
	    gen = new boost::mt19937;
	    gen->seed(config.randomSeed()+98); // any random number
	}
	boost::random::uniform_real_distribution<real_t> dist(0, 1);

	// options for schedule sampling
	int scheduleSampOpt = config.scheduleSampOpt();
	int scheduleSampPara= config.scheduleSampPara();
	
	// Prepare the ground truth 
	layers::MDNLayer<TDevice> *olm;
	olm = outMDNLayer();
	if (olm != NULL){
	    olm->retrieveFeedBackData();
	}else if (scheduleSampOpt > 0){
	    printf("\n\n Schedule sampling is not implemented for non-MDN network\n\n");
	    throw std::runtime_error(std::string("To be implemented"));
	}else{
	    
	}

	//
	int methodCode;
	switch (scheduleSampOpt){
	case NN_FEEDBACK_GROUND_TRUTH:
	    {
		// Case0: use ground truth directly
		BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers)
		    layer->computeForwardPass();
		break;
	    }
	case NN_FEEDBACK_DROPOUT_1N:
	case NN_FEEDBACK_DROPOUT_ZERO:
	    {
		// Case 1 & 2: schedule back-off, using either 1/N (case 1) or zero (case 2)
		real_t threshold = ((real_t)scheduleSampPara)/100;

		Cpu::real_vector randNum;
		randNum.reserve(curMaxSeqLength);
		for (size_t i = 0; i < curMaxSeqLength; ++i){
		    if (dist(*gen) > threshold){
			randNum.push_back(0);
		    }else{
			randNum.push_back(1);
		    }
		}
		
		// specify the method code
		methodCode = scheduleSampOpt;
		
		// drop out the feedback data
		typename TDevice::real_vector temp = randNum;
		olm->retrieveFeedBackData(temp, methodCode);

		// ComputeForward
		BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers)
		    layer->computeForwardPass();
		break;
	    }
	case NN_FEEDBACK_SC_SOFT:
	case NN_FEEDBACK_SC_MAXONEHOT:
	    {
		// Case 3 & 4: use soft vector as feedback (case 3) or one-hot (case 4)
		real_t sampThreshold;
		methodCode = scheduleSampOpt;
		
		// Forward computation for layers below Feedback
		int cnt = 0;
		BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers)
		{
		    if (cnt == m_firstFeedBackLayer) break; 
		    layer->computeForwardPass();
		    cnt++;
		}
		// Determine the threshold 
		if (scheduleSampPara > 0){
		    // randomly use the generated sample
		    // sampThreshold = ((real_t)scheduleSampPara /
		    // (scheduleSampPara + exp((real_t)uttCnt / scheduleSampPara)));
		    // sampThreshold = 1.0 - ((real_t)uttCnt/scheduleSampPara);
		    sampThreshold = pow(scheduleSampPara/100.0, uttCnt);
		    sampThreshold = ((sampThreshold  < NN_FEEDBACK_SCHEDULE_MIN) ?
				     NN_FEEDBACK_SCHEDULE_MIN : sampThreshold);
		}else{
		    sampThreshold = (-1.0 * (real_t)scheduleSampPara / 100.0);
		}

		// printf("%f %f\n", uttCnt, sampThreshold);
		// Forward computation for layer above feedback using schedule sampling
		for (int timeStep = 0; timeStep < curMaxSeqLength; timeStep++){

		    cnt = 0;
		    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers)
		    {
			if (cnt >= m_firstFeedBackLayer){
			    layer->prepareStepGeneration(timeStep); 
			    layer->computeForwardPass(timeStep);    
			}
			cnt++;
		    }
		    
		    // 
		    if (dist(*gen) > sampThreshold){
			//printf("\n %d HIT", timeStep);
			if (olm != NULL){
			    olm->getOutput(timeStep, 0.0001); 
			    olm->retrieveFeedBackData(timeStep, methodCode);
			    
			    /******** Fatal Error *******/
			    // After getOutput, the targets will be overwritten by generated data.
			    // But the target will be used by calculateError and computeBackWard.
			    // Thus, targets of the natural data should be re-written
			    // This is now implemented as this->restoreTarget(frac)
			    
			}    
		    }else{
			//printf("\n %d MISS", timeStep);
		    }
		}

		
		break;
	    }
	}
    }
}

template <typename TDevice>
void NeuralNetwork<TDevice>::computeForwardPassGen(const int curMaxSeqLength, 
						   const real_t generationOpt)
{
    layers::MDNLayer<TDevice> *olm;
    const Configuration &config = Configuration::instance();
    
    // no feedback layer, normal computation
    if (m_firstFeedBackLayer < 0){

	this->computeForwardPass(curMaxSeqLength, -1);
	// if MDN is available, infer the output, or copy the MDN parameter vector
	olm = outMDNLayer();
	if (olm != NULL) olm->getOutput(generationOpt);

    // feedback layer exists
    }else{

	static boost::mt19937 *gen = NULL;
	if (!gen) {
	    gen = new boost::mt19937;
	    gen->seed(config.randomSeed()+98); // any random number
	}
	boost::random::uniform_real_distribution<real_t> dist(0, 1);

	int scheduleSampOpt = config.scheduleSampOpt();
	int scheduleSampPara= config.scheduleSampPara();
	printf("SSAMPOpt: %d, SSAMPPara: %d\n", scheduleSampOpt, scheduleSampPara);
	
	int methodCode;
	real_t sampThreshold;
	int cnt = 0;
	// layers below Feedback, use normal computation
	BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	    if (cnt == m_firstFeedBackLayer) break; 
	    layer->computeForwardPass();
	    cnt++;
	}
	
	// determine the sampling parameter
	switch (scheduleSampOpt){
	case NN_FEEDBACK_GROUND_TRUTH:
	case NN_FEEDBACK_SC_SOFT:
	    // always uses the soft vector (default option)
	    sampThreshold  = 1;
	    methodCode     = NN_FEEDBACK_GROUND_TRUTH;
	    break;
	case NN_FEEDBACK_SC_MAXONEHOT:
	    if (scheduleSampPara > 0){
		sampThreshold = 1;
		methodCode = NN_FEEDBACK_GROUND_TRUTH;
	    }else{
		sampThreshold = (-1.0 * (real_t)scheduleSampPara / 100.0);
		methodCode = NN_FEEDBACK_SC_MAXONEHOT;
	    }
	    
	    // use the one-hot best
	    break;
	case NN_FEEDBACK_DROPOUT_1N:
	    methodCode = NN_FEEDBACK_DROPOUT_1N;
	    sampThreshold = ((real_t)scheduleSampPara)/100;
	    break;					    
	case NN_FEEDBACK_DROPOUT_ZERO:
	    methodCode = NN_FEEDBACK_DROPOUT_ZERO;
	    sampThreshold = ((real_t)scheduleSampPara)/100;
	    break;
	    //
	}
		    
	// layer above feedback
	for (int timeStep = 0, cnt = 0; timeStep < curMaxSeqLength; timeStep ++, cnt = 0){
	    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
		if (cnt >= m_firstFeedBackLayer){
		    layer->prepareStepGeneration(timeStep); // prepare the matrix (for rnn, lstm)
		    layer->computeForwardPass(timeStep);    // compute for 1 frame
		}
		cnt++;
	    }

	    
	    // if the output is MDN, we need to do one step further to get the output
	    olm = outMDNLayer();
	    if (olm != NULL){
		olm->getOutput(timeStep, generationOpt); // infer the output from MDN
		if (dist(*gen) < sampThreshold){
		    olm->retrieveFeedBackData(timeStep, 0);
		}else{
		    olm->retrieveFeedBackData(timeStep, methodCode);
		    printf("%d ", timeStep);
		}
		
	    }
	}
	/*olm = outMDNLayer();
	if (olm != NULL){	
	    olm->getOutput(generationOpt);
	    }*/
    }
}

template <typename TDevice>
void NeuralNetwork<TDevice>::computeBackwardPass()
{
    BOOST_REVERSE_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers) {
        layer->computeBackwardPass();
    //std::cout << "output errors " << layer->name() << std::endl;
    //thrust::copy(layer->outputErrors().begin(), layer->outputErrors().end(), 
    // std::ostream_iterator<real_t>(std::cout, ";"));
    //std::cout << std::endl;
    }
}

template <typename TDevice>
real_t NeuralNetwork<TDevice>::calculateError() const
{
    return static_cast<layers::PostOutputLayer<TDevice>&>(*m_layers.back()).calculateError();
}

template <typename TDevice>
void NeuralNetwork<TDevice>::exportLayers(const helpers::JsonDocument& jsonDoc) const
{
    if (!jsonDoc->IsObject())
        throw std::runtime_error("JSON document root must be an object");

    // create the layers array
    rapidjson::Value layersArray(rapidjson::kArrayType);

    // create the layer objects
    for (size_t i = 0; i < m_layers.size(); ++i)
        m_layers[i]->exportLayer(&layersArray, &jsonDoc->GetAllocator());

    // if the section already exists, we delete it first
    if (jsonDoc->HasMember("layers"))
        jsonDoc->RemoveMember("layers");

    // add the section to the JSON document
    jsonDoc->AddMember("layers", layersArray, jsonDoc->GetAllocator());
}

template <typename TDevice>
void NeuralNetwork<TDevice>::exportWeights(const helpers::JsonDocument& jsonDoc) const
{
    if (!jsonDoc->IsObject())
        throw std::runtime_error("JSON document root must be an object");

    // create the weights object
    rapidjson::Value weightsObject(rapidjson::kObjectType);

    // create the weight objects
    BOOST_FOREACH (const boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers) {
    	layers::TrainableLayer<TDevice> *trainableLayer = 
	    dynamic_cast<layers::TrainableLayer<TDevice>*>(layer.get());
        if (trainableLayer){
            trainableLayer->exportWeights(&weightsObject, &jsonDoc->GetAllocator());
	}else{
	    // Modify 0507 Wang: for mdn PostProcess Layer
	    layers::MDNLayer<TDevice> *mdnlayer = 
		dynamic_cast<layers::MDNLayer<TDevice>*>(layer.get());
	    if (mdnlayer)
		mdnlayer->exportConfig(&weightsObject, &jsonDoc->GetAllocator());
	}
    }

    // if the section already exists, we delete it first
    if (jsonDoc->HasMember("weights"))
        jsonDoc->RemoveMember("weights");

    // add the section to the JSON document
    jsonDoc->AddMember("weights", weightsObject, jsonDoc->GetAllocator());
}

template <typename TDevice>
std::vector<std::vector<std::vector<real_t> > > NeuralNetwork<TDevice>::getOutputs(
    const int layerID, const bool getGateOutput, const real_t mdnoutput)
{
    std::vector<std::vector<std::vector<real_t> > > outputs;
    layers::SkipLayer<TDevice> *olg;
    layers::MDNLayer<TDevice> *olm;
    int tempLayerId;
    unsigned char genMethod;

    enum genMethod {ERROR = 0, GATEOUTPUT, MDNSAMPLING, MDNPARAMETER, MDNEMGEN, NORMAL};
    
    /* specify old, olm, tempLayerId
       -3.0 is chosen for convience.
       
       < -3.0: no MDN generation
       > -3.0 && < -1.5: generating EM-style
       > -1.5 && < 0.0: generate MDN parameters (mdnoutput = -1.0)
       > 0.0 : generate samples from MDN with the variance = variance * mdnoutput */

    if (mdnoutput >= -3.0 && getGateOutput){
	genMethod = ERROR;
	throw std::runtime_error("MDN output and gate output can not be generated together");

    }else if (mdnoutput < -3.0 && getGateOutput){
	olg = outGateLayer(layerID);
	olm = NULL;
	tempLayerId = layerID;
	if (olg == NULL)
	    throw std::runtime_error("Gate output tap ID invalid\n");
	genMethod = GATEOUTPUT;

    }else if (mdnoutput >= -3.0 && !getGateOutput){
	olg = NULL;
	olm = outMDNLayer();
	if (olm == NULL)
	    throw std::runtime_error("No MDN layer in the current network");
	//olm->getOutput(mdnoutput); // Move to computeForward(curMaxSeqLength, generationOpt)
	tempLayerId = m_layers.size()-1;
	genMethod = (mdnoutput < 0.0) ? ((mdnoutput < -1.5) ? MDNEMGEN:MDNPARAMETER):MDNSAMPLING;
	
    }else{
	olg = NULL;
	olm = NULL;
	tempLayerId = layerID;
	genMethod = NORMAL;
    }
    
    // retrieve the output
    layers::Layer<TDevice> &ol  = outputLayer(tempLayerId);
    for (int patIdx = 0; patIdx < (int)ol.patTypes().size(); ++patIdx) {
	switch (ol.patTypes()[patIdx]) {
	case PATTYPE_FIRST:
	    outputs.resize(outputs.size() + 1);
	    
	case PATTYPE_NORMAL:
	case PATTYPE_LAST: {
	    switch (genMethod){
	    case MDNEMGEN:
	    case MDNSAMPLING:
	    case NORMAL:
		{
		    Cpu::real_vector pattern(ol.outputs().begin() + patIdx * ol.size(), 
					     ol.outputs().begin() + (patIdx+1) * ol.size());
		    int psIdx = patIdx % ol.parallelSequences();
		    outputs[psIdx].push_back(std::vector<real_t>(pattern.begin(), pattern.end()));
		    break;
		}
	    case MDNPARAMETER:
		{
		    
		    Cpu::real_vector pattern(
				olm->mdnParaVec().begin()+patIdx*olm->mdnParaDim(), 
				olm->mdnParaVec().begin()+(patIdx+1)*olm->mdnParaDim());
		    int psIdx = patIdx % ol.parallelSequences();
		    outputs[psIdx].push_back(std::vector<real_t>(pattern.begin(), pattern.end()));
		    break;
		}
	    case GATEOUTPUT:
		{
		    Cpu::real_vector pattern(olg->outputFromGate().begin() + patIdx * ol.size(),
					     olg->outputFromGate().begin()+(patIdx+1) * ol.size());
		    int psIdx = patIdx % ol.parallelSequences();
		    outputs[psIdx].push_back(std::vector<real_t>(pattern.begin(), pattern.end()));
		    break;
		}
	    default:
		break;   
	    }
	}
	default:
	    break;
	}
    }

    return outputs;
}


/* Add 16-02-22 Wang: for WE updating */
// Initialization for using external WE bank
// (read in the word embeddings and save them in a matrix)
template <typename TDevice>
bool NeuralNetwork<TDevice>::initWeUpdate(const std::string weBankPath, const unsigned weDim, 
					  const unsigned weIDDim, const unsigned maxLength)
{
    // check if only the first layer is an input layer
    layers::InputLayer<TDevice>* inputLayer = 
	dynamic_cast<layers::InputLayer<TDevice>*>(m_layers.front().get());
    if (!inputLayer)
	throw std::runtime_error("The first layer is not an input layer");
    else if (!inputLayer->readWeBank(weBankPath, weDim, weIDDim, maxLength)){
	throw std::runtime_error("Fail to initialize for we updating");
    }
}

template <typename TDevice>
bool NeuralNetwork<TDevice>::initWeNoiseOpt(const int weNoiseStartDim, const int weNoiseEndDim,
					    const real_t weNoiseDev)
{
    // check if only the first layer is an input layer
    layers::InputLayer<TDevice>* inputLayer = 
	dynamic_cast<layers::InputLayer<TDevice>*>(m_layers.front().get());
    if (!inputLayer)
	throw std::runtime_error("The first layer is not an input layer");
    else if (!inputLayer->initWeNoiseOpt(weNoiseStartDim, weNoiseEndDim, weNoiseDev)){
	throw std::runtime_error("Fail to initialize for we updating");
    }
    return true;
}



// check whether the input layer uses external we bank
template <typename TDevice>
bool NeuralNetwork<TDevice>::flagInputWeUpdate() const
{
    layers::InputLayer<TDevice>* inputLayer = 
	dynamic_cast<layers::InputLayer<TDevice>*>(m_layers.front().get());
    if (!inputLayer){
	throw std::runtime_error("The first layer is not an input layer");
	return false;
    }
    else
	return inputLayer->flagInputWeUpdate();
}

// save the updated we bank in the input layer
template <typename TDevice>
bool NeuralNetwork<TDevice>::saveWe(const std::string weFile) const
{
    layers::InputLayer<TDevice>* inputLayer = 
	dynamic_cast<layers::InputLayer<TDevice>*>(m_layers.front().get());
    if (!inputLayer){
	throw std::runtime_error("The first layer is not an input layer");
	return false;
    }
    else
	return inputLayer->saveWe(weFile);
}

/* Add 0401 Wang: for MSE weight initialization*/
template <typename TDevice>
bool NeuralNetwork<TDevice>::initMseWeight(const std::string mseWeightPath)
{
    
    layers::PostOutputLayer<TDevice>* outputLayer = 
	dynamic_cast<layers::PostOutputLayer<TDevice>*>(m_layers.back().get());
    if (!outputLayer){
	throw std::runtime_error("The output layer is not a postoutput layer");
	return false;
    }
    else
	return outputLayer->readMseWeight(mseWeightPath);
   
}

/* Add 0413 Wang: for weight mask */
template <typename TDevice>
bool NeuralNetwork<TDevice>::initWeightMask(const std::string weightMaskPath,
					    const int         weightMaskOpt)
{
    std::ifstream ifs(weightMaskPath.c_str(), std::ifstream::binary | std::ifstream::in);
    if (!ifs.good())
	throw std::runtime_error(std::string("Fail to open") + weightMaskPath);
    
    // get the number of we data
    std::streampos numEleS, numEleE;
    long int numEle;
    numEleS = ifs.tellg();
    ifs.seekg(0, std::ios::end);
    numEleE = ifs.tellg();
    numEle  = (numEleE-numEleS)/sizeof(real_t);
    ifs.seekg(0, std::ios::beg);

    real_t tempVal;
    std::vector<real_t> tempVec;
    for (unsigned int i = 0; i<numEle; i++){
	ifs.read ((char *)&tempVal, sizeof(real_t));
	tempVec.push_back(tempVal);
    }
    
    printf("Initialize weight mask: %d mask elements in total, ", (int)numEle);
    printf("under the mode %d", weightMaskOpt);
    
    int pos = 0;
    if (weightMaskOpt > 0){
	printf("\n\tRead mask for embedded vectors ");
	layers::InputLayer<TDevice>* inputLayer = 
	    dynamic_cast<layers::InputLayer<TDevice>*>((m_layers[0]).get());
	pos = inputLayer->readWeMask(tempVec.begin());
	printf("(%d elements)", pos);
    }

    if (weightMaskOpt == 0 || weightMaskOpt==2){
	printf("\n\tRead mask for NN weights (");
	BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	    layers::TrainableLayer<TDevice>* weightLayer = 
		dynamic_cast<layers::TrainableLayer<TDevice>*>(layer.get());
	    if (weightLayer){
		if (weightLayer->weightNum()+pos > numEle){
		    throw std::runtime_error(std::string("Weight mask input is not long enough"));
		}else{
		    weightLayer->readWeightMask(tempVec.begin()+pos, 
						tempVec.begin()+pos+weightLayer->weightNum());
		    pos = pos+weightLayer->weightNum();
		}
		printf("%d ", weightLayer->weightNum());
	    }
	}
	printf("elements)");
    }
    printf("\n");
}

template <typename TDevice>
void NeuralNetwork<TDevice>::maskWeight()
{
    // mask the embedded vectors (if applicable)
    layers::InputLayer<TDevice>* inputLayer = 
	dynamic_cast<layers::InputLayer<TDevice>*>((m_layers[0]).get());
    inputLayer->maskWe();

    // mask the weight (always do, as the default mask value is 1.0)
    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	layers::TrainableLayer<TDevice>* weightLayer = 
	    dynamic_cast<layers::TrainableLayer<TDevice>*>(layer.get());
	if (weightLayer){
	    weightLayer->maskWeight();
	}
    }
}

template <typename TDevice>
void NeuralNetwork<TDevice>::notifyCurrentEpoch(const int trainingEpoch)
{
    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	layer->setCurrTrainingEpoch(trainingEpoch);
    }
}

template <typename TDevice>
void NeuralNetwork<TDevice>::reInitWeight()
{
    printf("Reinitialize the weight\n");
    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	layer->reInitWeight();
    }
}

template <typename TDevice>
void NeuralNetwork<TDevice>::initOutputForMDN(
 const data_sets::DataSetMV &datamv)
{
    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer,
		   m_layers){
	layers::MDNLayer<TDevice>* mdnLayer = 
	    dynamic_cast<layers::MDNLayer<TDevice>*>(layer.get());
	if (mdnLayer){
	    mdnLayer->initPreOutput(datamv.outputM(), datamv.outputV());
	    printf("MDN initialization \t");
	    if (datamv.outputM().size()<1)
		printf("using global zero mean and uni variance");
	    else
		printf("using data mean and variance");
	}
    }
}

template <typename TDevice>
void NeuralNetwork<TDevice>::readMVForOutput(
 const data_sets::DataSetMV &datamv)
{
    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer,
		   m_layers){
	layers::PostOutputLayer<TDevice>* outputLayer = 
	    dynamic_cast<layers::PostOutputLayer<TDevice>*>(layer.get());
	if (outputLayer){
	    outputLayer->readMV(datamv.outputM(), datamv.outputV());
	    printf("Read mean and variance into output layer \t");
	}
    }
}


/* importWeights
 * import weights from pre-trained model
 */
template <typename TDevice>
void NeuralNetwork<TDevice>::importWeights(const helpers::JsonDocument &jsonDoc, 
					   const std::string &ctrStr)
{
    try{
	// Read in the control vector, a sequence of 1 0
	Cpu::int_vector tempctrStr;
	tempctrStr.resize(m_layers.size(), 1);
	if (ctrStr.size() > 0 && ctrStr.size()!= m_layers.size()){
	    throw std::runtime_error("Length of trainedParameterCtr unequal #layer.");
	}else if (ctrStr.size()>0){
	    for (int i=0; i<ctrStr.size(); i++)
		tempctrStr[i] = ctrStr[i]-'0';
	}else{
	    // nothing
	}
	
	// Read in the weight parameter as a whole
	helpers::JsonValue weightsSection;
        if (jsonDoc->HasMember("weights")) {
            if (!(*jsonDoc)["weights"].IsObject())
                throw std::runtime_error("Section 'weights' is not an object");
            weightsSection = helpers::JsonValue(&(*jsonDoc)["weights"]);
        }else{
	    throw std::runtime_error("No weight section found");
	}

	// Assign parameter to each layer
	int cnt=0;
	BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers)
	{
	    layers::TrainableLayer<TDevice>* Layer = 
		dynamic_cast<layers::TrainableLayer<TDevice>*>(layer.get());
	    
	    // Read in the parameter for a hidden layer
	    if (Layer && tempctrStr[cnt] > 0){
		printf("\n\t(%d) ", cnt);
		Layer->reReadWeight(weightsSection, Layer->size(), tempctrStr[cnt]);
		/*
		layers::LstmLayerCharW<TDevice>* LstmCharWLayer = 
		    dynamic_cast<layers::LstmLayerCharW<TDevice>*>(layer.get());
		if (LstmCharWLayer){
		    // Because LstmCharWLayer is special
		    Layer->reReadWeight(weightsSection, LstmCharWLayer->lstmSize(), 
					tempctrStr[cnt]);
		}else{
		   Layer->reReadWeight(weightsSection, Layer->size(), tempctrStr[cnt]); 
		}*/
		
	    // Read in the parameter for MDN layer with trainable link
	    }else if(tempctrStr[cnt] > 0){
		layers::MDNLayer<TDevice>* mdnlayer = 
		    dynamic_cast<layers::MDNLayer<TDevice>*>(layer.get());
		if (mdnlayer && mdnlayer->flagTrainable()){
		    printf("\n\t(%d) ", cnt);
		    mdnlayer->reReadWeight(weightsSection, tempctrStr[cnt]);
		}
		
	    // This layer is skipped
	    }else if(Layer){
		printf("\n\t(%d) not read weight for layer %s", cnt, Layer->name().c_str());
	    }else{
		// other cases
	    }
	    cnt++;
	}
	printf("\tdone\n\n");
	
    }catch (const std::exception &e){
	printf("\nTo read weight from another trained network (refer to net2):\n");
	printf("\n\t1. prepare network.jsn (net1). Set the name of the layer to be initialized\n");
	printf("\n\t   as the same name of the source layer in net2. \n");
	printf("\n\t2. set the --trainedModelCtr as the a string of number (0/1/2/3), whose \n");
	printf("\n\t   length is the same as number of layers in the net1.\n");
	printf("\n\t   Please check currennt --help for the mearning of 0/1/2/3. \n");
	printf("\n\t   If the number for one layer in net1 is 0, or its name can not be found \n");
	printf("\n\t   in net2, that layer will not be initilized using the weights in net2.\n");
	throw std::runtime_error(std::string("Fail to read network weight")+e.what());
    }
}


template <typename TDevice>
Cpu::real_vector NeuralNetwork<TDevice>::getMdnConfigVec()
{
    Cpu::real_vector temp;
    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	layers::MDNLayer<TDevice>* mdnLayer = 
	    dynamic_cast<layers::MDNLayer<TDevice>*>(layer.get());
	if (mdnLayer)
	    temp = mdnLayer->getMdnConfigVec();
    }    
    return temp;
}

// PrintWeightMatrix
// print the weight of a network to a binary data
// use ReadCURRENNTWeight(filename,format,swap) matlab function to read the data
template <typename TDevice>
void NeuralNetwork<TDevice>::printWeightMatrix(const std::string weightPath, const int opt)
{
    std::fstream ifs(weightPath.c_str(),
		      std::ifstream::binary | std::ifstream::out);
    if (!ifs.good()){
	throw std::runtime_error(std::string("Fail to open output weight path: "+weightPath));
    }

    // format of the output binary weight
    std::vector<int> weightSize;
    weightSize.clear();
    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	layers::TrainableLayer<TDevice>* Layer = 
	    dynamic_cast<layers::TrainableLayer<TDevice>*>(layer.get());
	
	if (Layer){
	    weightSize.push_back(Layer->weights().size());
	    weightSize.push_back(Layer->size());
	    weightSize.push_back(Layer->precedingLayer().size());
	    weightSize.push_back(Layer->inputWeightsPerBlock());
	    weightSize.push_back(Layer->internalWeightsPerBlock());
	    if (opt==1){
		if (Layer->type()=="feedforward_tanh")
		    weightSize.push_back(0);
		else if (Layer->type()=="feedforward_logistic")
		    weightSize.push_back(1);
		else if (Layer->type()=="feedforward_identity")
		    weightSize.push_back(2);
		else if (Layer->type()=="feedforward_relu")
		    weightSize.push_back(3);		
		else if (Layer->type()=="lstm")
		    weightSize.push_back(4);		
		else if (Layer->type()=="blstm")
		    weightSize.push_back(5);
		else
		    printf("other weight type not implemented\n");
	    }
	}else{
	    layers::MDNLayer<TDevice>* mdnlayer = 
		dynamic_cast<layers::MDNLayer<TDevice>*>(layer.get());
	    if (mdnlayer && mdnlayer -> flagTrainable()){
		weightSize.push_back(mdnlayer->weights().size());
		weightSize.push_back(mdnlayer->weights().size());
		weightSize.push_back(0);  // previous size = 0
		weightSize.push_back(1);  // internal block = 1
		weightSize.push_back(0);  // internal weight = 0
	    }
	}
    }

    printf("Writing network to binary format: \n");
    // macro information
    // Number of layers
    // weight size, layer size, preceding layer size, inputWeightsPerBlock, internalWeightsPerBlock
    real_t tmpPtr;
    tmpPtr = (real_t)weightSize.size()/((opt==1)?6:5);
    ifs.write((char *)&tmpPtr, sizeof(real_t));
    for (int i = 0 ; i<weightSize.size(); i++){
	tmpPtr = (real_t)weightSize[i];
	ifs.write((char *)&tmpPtr, sizeof(real_t));
    }

    // weights
    int cnt = 0;
    real_t *tmpPtr2;
    Cpu::real_vector weightVec;
    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers){
	layers::TrainableLayer<TDevice>* Layer = 
	    dynamic_cast<layers::TrainableLayer<TDevice>*>(layer.get());
	if (Layer){
	    weightVec = Layer->weights();
	    tmpPtr2 = weightVec.data();
	    if (weightVec.size()>0 && tmpPtr2)
		ifs.write((char *)tmpPtr2, sizeof(real_t)*Layer->weights().size());	
	    printf("Layer (%2d) %s with %lu weights\n", cnt, Layer->type().c_str(), weightVec.size());
	}else{
	    layers::MDNLayer<TDevice>* mdnlayer = 
		dynamic_cast<layers::MDNLayer<TDevice>*>(layer.get());
	    if (mdnlayer && mdnlayer -> flagTrainable()){
		weightVec = mdnlayer->weights();
		tmpPtr2 = weightVec.data();
		if (weightVec.size()>0 && tmpPtr2){
		    ifs.write((char *)tmpPtr2, sizeof(real_t)*mdnlayer->weights().size());
		}else{
		    throw std::runtime_error("Fail to output weight. Void pointer");
		}
		printf("Layer (%2d) MDN with %lu weights\n", cnt, weightVec.size());
	    }
	}
	cnt++;
    }
    ifs.close();
    printf("Writing done\n");
}


// explicit template instantiations
template class NeuralNetwork<Cpu>;
template class NeuralNetwork<Gpu>;
