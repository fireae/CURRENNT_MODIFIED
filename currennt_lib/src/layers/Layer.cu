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
 * You should have received a copy of the GNU General Public License
 * along with CURRENNT.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#ifdef _MSC_VER
#   pragma warning (disable: 4244)
#endif

#include "Layer.hpp"
#include "../helpers/misFuncs.hpp"
#include "../helpers/JsonClasses.hpp"

#include <sstream>
#include <stdexcept>


namespace layers {

    
    
    template <typename TDevice>
    typename Layer<TDevice>::real_vector& Layer<TDevice>::_outputs()
    {
        return m_outputs;
    }
    
    /* Add 16-02-22 Wang: for WE updating */
    template <typename TDevice>
    bool& Layer<TDevice>::inputWeUpdate()
    {
	return m_InputWeUpdate;
    }
    template <typename TDevice>
    bool Layer<TDevice>::_setInputWeUpdate(const bool& flag)
    {
	m_InputWeUpdate = flag;
	return true;
    }
    

    template <typename TDevice>
    Layer<TDevice>::Layer(const helpers::JsonValue &layerChild,
			  int parallelSequences,   int maxSeqLength,
			  bool flagTrainingMode, bool createOutputs)
        : m_name             (layerChild->HasMember("name") ? 
			      (*layerChild)["name"].GetString()  : "")
        , m_size             (layerChild->HasMember("size") ? 
			      (*layerChild)["size"].GetInt()     : 0)
	, m_timeResolution   (layerChild->HasMember("resolution") ? 
			      (*layerChild)["resolution"].GetInt() : 1)
        , m_parallelSequences(parallelSequences)
        , m_maxSeqLength     (misFuncs::getResoLength(
				maxSeqLength,
				(layerChild->HasMember("resolution") ? 
				 (*layerChild)["resolution"].GetInt() : 1)))
        , m_curMaxSeqLength  (0)
        , m_curMinSeqLength  (0)
        , m_curNumSeqs       (0)
	, m_InputWeUpdate    (false)
	, m_flagTrainingMode (true)
	, m_flagSaveOutputMemory (false)
    {
        // check if the name and size values exist
        if (!layerChild->HasMember("name"))
            throw std::runtime_error("Missing value 'name' in layer description");
        if (m_name.empty())
            throw std::runtime_error("Empty layer name in layer description");
        if (!layerChild->HasMember("size"))
            throw std::runtime_error("Missing 'size' in layer");

	if (m_timeResolution > 1){
	    printf("\n\tLayer resolution %d ", m_timeResolution);
	}else if (m_timeResolution < 1){
	    throw std::runtime_error("resolution cannot be less than 1");
	}
	
        // allocate memory for output
        if (createOutputs)
            m_outputs = Cpu::real_vector(m_parallelSequences * m_maxSeqLength * m_size);
	
	// allocate memory for time mark
        m_patTypes = Cpu::pattype_vector(m_parallelSequences * m_maxSeqLength);
	
        // allocate memory for gradients buffer
	if (flagTrainingMode)
	    m_outputErrors  = Cpu::real_vector(this->_outputs().size(), (real_t)0);    
	else
	    m_outputErrors.clear();
	m_outputErrorsCopy.clear();
	
	// initialize the training epoch counter
	m_currTrainingEpoch = -1;

	// set the flag
	m_flagTrainingMode  = (flagTrainingMode ? true : false);

	
	m_layerFlag = (layerChild->HasMember("layerFlag") ? 
		       (*layerChild)["layerFlag"].GetString() : "");
	

    }

    template <typename TDevice>
    Layer<TDevice>::~Layer()
    {
    }
    
    template <typename TDevice>
    const std::string& Layer<TDevice>::name() const
    {
        return m_name;
    }

    template <typename TDevice>
    int Layer<TDevice>::size() const
    {
        return m_size;
    }

    template <typename TDevice>
    const std::string& Layer<TDevice>::getLayerFlag()
    {
	return m_layerFlag;
    }

    template <typename TDevice>
    int Layer<TDevice>::parallelSequences() const
    {
        return m_parallelSequences;
    }

    template <typename TDevice>
    int Layer<TDevice>::maxSeqLength() const
    {
        return m_maxSeqLength;
    }

    template <typename TDevice>
    int Layer<TDevice>::curMaxSeqLength() const
    {
        return m_curMaxSeqLength;
    }

    template <typename TDevice>
    int Layer<TDevice>::curMinSeqLength() const
    {
        return m_curMinSeqLength;
    }

    template <typename TDevice>
    int Layer<TDevice>::curNumSeqs() const
    {
        return m_curNumSeqs;
    }

    template <typename TDevice>
    const int& Layer<TDevice>::getResolution()
    {
	return m_timeResolution;
    }
    
    template <typename TDevice>
    const typename Layer<TDevice>::pattype_vector& Layer<TDevice>::patTypes() const
    {
        return m_patTypes;
    }

    template <typename TDevice>
    typename Layer<TDevice>::real_vector& Layer<TDevice>::outputs()
    {
        return m_outputs;
    }

    template <typename TDevice>
    typename Layer<TDevice>::real_vector& Layer<TDevice>::outputErrors()
    {
        return m_outputErrors;
    }

    template <typename TDevice>
    Cpu::real_vector& Layer<TDevice>::outputErrorsCpu()
    {
	if (m_outputErrorsCopy.size() != m_outputErrors.size())
	    m_outputErrorsCopy = m_outputErrors;
	thrust::copy(m_outputErrors.begin(), m_outputErrors.end(),
		     m_outputErrorsCopy.begin());
        return m_outputErrorsCopy;
    }
    
    template <typename TDevice>
    void Layer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction,
				       const int nnState)
    {
	m_curMaxSeqLength = misFuncs::getResoLength(fraction.maxSeqLength(), m_timeResolution);
	m_curMinSeqLength = misFuncs::getResoLength(fraction.minSeqLength(), m_timeResolution);
	m_curNumSeqs      = fraction.numSequences();
	    
	if (m_timeResolution == 1){
	    m_patTypes    = fraction.patTypes();
	}else{

	    int buffPos   = fraction.patTypesLowTimesResPos(m_timeResolution);
	    int buffLen   = fraction.patTypesLowTimesResLen(m_timeResolution);
	    if (buffPos < 0 || buffLen < 0){
		printf(" %s resolution not found in --resolutions", this->name().c_str());
		throw std::runtime_error("Resolution error");
	    }
	    //m_patTypes.resize(buffLen, PATTYPE_NONE);
	    //thrust::fill(m_patTypes.begin(), m_patTypes.end(), PATTYPE_NONE);
	    thrust::copy(fraction.patTypesLowTimeRes().begin() + buffPos,
			 fraction.patTypesLowTimeRes().begin() + buffPos + buffLen,
			 m_patTypes.begin());
	    	
	}
    }
    
    template <typename TDevice>
    void Layer<TDevice>::exportLayer(const helpers::JsonValue &layersArray, 
				     const helpers::JsonAllocator &allocator) const
    {
        if (!layersArray->IsArray())
            throw std::runtime_error("The JSON value is not an array");

        // create and fill the layer object
        rapidjson::Value layerObject(rapidjson::kObjectType);
        layerObject.AddMember("name", name().c_str(), allocator);
        layerObject.AddMember("type", type().c_str(), allocator);
        layerObject.AddMember("size", size(),         allocator);

	if (m_timeResolution > 1)
	    layerObject.AddMember("resolution", m_timeResolution, allocator);
	if (m_layerFlag.size() > 0)
	    layerObject.AddMember("layerFlag", m_layerFlag.c_str(), allocator);
	
        // add the layer object to the layers array
        layersArray->PushBack(layerObject, allocator);
    }
    
    template <typename TDevice>
    void Layer<TDevice>::setCurrTrainingEpoch(const int curTrainingEpoch)
    {
	m_currTrainingEpoch = curTrainingEpoch;
    }
    
    template <typename TDevice>
    int& Layer<TDevice>::getCurrTrainingEpoch()
    {
	return m_currTrainingEpoch;
    }

    template <typename TDevice>
    void Layer<TDevice>::setCurrTrainingFrac(const int curTrainingFrac)
    {
	m_currTrainingFrac = curTrainingFrac;
    }
    
    template <typename TDevice>
    int& Layer<TDevice>::getCurrTrainingFrac()
    {
	return m_currTrainingFrac;
    }

    template <typename TDevice>
    void Layer<TDevice>::linkTargetLayer(Layer<TDevice> &targetLayer)
    {
	// do nothing
    }

    template <typename TDevice>
    const std::string& Layer<TDevice>::layerAddInfor(const int opt) const
    {
	// used together with feedbackOutputs
        static std::string s;
        if (s == "" && opt==1){
	    std::ostringstream Convert;
	    Convert << this->size() << "_";  
	    s = Convert.str();
	}else{
	    s = "";
	}
	
        return s;
    }
    
    template <typename TDevice>
    void Layer<TDevice>::prepareStepGeneration(const int timeStep)
    {
	// do nothing
    }

    template <typename TDevice>
    typename Layer<TDevice>::real_vector& Layer<TDevice>::feedbackOutputs(const bool flagTrain)
    {
        return m_outputs;
    }

    template <typename TDevice>
    void Layer<TDevice>::cleanGradidents()
    {
	//thrust::fill(m_outputErrors.begin(), m_outputErrors.end(), 0.0);
    }

    template <typename TDevice>
    int Layer<TDevice>::hiddenStateSize()
    {
	return 0;
    }

    template <typename TDevice>
    void Layer<TDevice>::retrieveHiddenState(const int timeStep, real_vector& readBuffer)
    {	
    }
    
    template <typename TDevice>
    void Layer<TDevice>::setHiddenState(const int timeStep, real_vector& writeBuffer)
    {	
    }
    
    template <typename TDevice>
    bool Layer<TDevice>::flagTrainingMode() const
    {
	return m_flagTrainingMode;
    }

    template <typename TDevice>
    void Layer<TDevice>::clearOutputBuffer()
    {
	m_outputs.clear();
	m_outputs.shrink_to_fit();
    }
    
    template <typename TDevice>
    void Layer<TDevice>::resizeOutputBuffer(const int bufferSize)
    {
	this->clearOutputBuffer();
	m_outputs = Cpu::real_vector(bufferSize);
	// Can't use m_outputs.resize() because:
	// Not really. The extent of Thrust's CUDA support for pure C++ code is
	// the bare minimum to allow device_vector to be constructed and
	// destroyed for POD types. That's why thrust::fill has a pure C++ 
	// implementation even for the CUDA backend -- so it can be called by 
	// device_vector's constructor.
	// https://groups.google.com/forum/#!topic/thrust-users/abVI3htMrkw
    }

    template <typename TDevice>
    void Layer<TDevice>::reduceOutputBuffer()
    {
	// default: do nothing
    }

    
    template <typename TDevice>
    int Layer<TDevice>::outputBufPtrBias(const int timeStepTimesParallel, const int nnState)
    {
	// don't shift
	return 0;
    }
    
    template <typename TDevice>
    void Layer<TDevice>::setSaveMemoryFlag(const bool newFlag)
    {
	m_flagSaveOutputMemory = newFlag;
    }

    template <typename TDevice>
    bool Layer<TDevice>::getSaveMemoryFlag() const
    {
	return m_flagSaveOutputMemory;
    }
    
    // explicit template instantiations
    template class Layer<Cpu>;
    template class Layer<Gpu>;

} // namespace layers
