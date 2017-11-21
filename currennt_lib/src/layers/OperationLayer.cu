/******************************************************************************
 * This file is an addtional component of CURRENNT. 
 * Xin WANG
 * National Institute of Informatics, Japan
 * 2016
 *
 * This file is part of CURRENNT. 
 * Copyright (c) 2013 Johannes Bergmann, Felix Weninger, Bjoern Schuller
 * Institute for Human-Machine Communication
 * Technische Universitaet Muenchen (TUM)
 * D-80290 Munich, Germany
 *
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

#include "OperationLayer.hpp"

#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../helpers/JsonClasses.hpp"
#include "../helpers/misFuncs.hpp"
#include "../activation_functions/Logistic.cuh"
#include "../activation_functions/Tanh.cuh"
#include "../MacroDefine.hpp"

#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/fill.h>
#include <thrust/random.h>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <vector>
#include <stdexcept>

#include "../Configuration.hpp"

#define NN_OPE_LAST_SHOT_MODE1 1  // use the last shot of sentence end
#define NN_OPE_LAST_SHOT_MODE2 2  // use the last shot of sentence end, repeat across frames
#define NN_OPE_LAST_SHOT_MODE3 3  // use the last shot of segments
#define NN_OPE_LAST_SHOT_MODE4 4  // use the last shot of segments, repeat across frames

namespace internal{
    
    struct genNoise
    {
	float a, b;
	int   seed;
	
	__host__ __device__
	genNoise(float _a=-1.f, float _b=1.f, int _seed=123) : a(_a), b(_b), seed(_seed) {};

	__host__ __device__
	float operator()(const unsigned int n) const
	{
	    thrust::default_random_engine rng(seed);
	    thrust::uniform_real_distribution<float> dist(a, b);
	    rng.discard(n);
	    return dist(rng);
	}
    };

    
    struct fillOutputVec
    {
	int curLayerSize;
	int preLayerSize;
	int noiseDim;
	int noiseRepeat;
	
	real_t *preOutput;
	real_t *noiseData;

	real_t *weights;
	
	const char *patTypes;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int outputIdx = t.get<1>();
	    int dimIdx    = outputIdx % curLayerSize;
	    int timeIdx   = outputIdx / curLayerSize;

	    if (dimIdx < preLayerSize){
		if ((patTypes != NULL && patTypes[timeIdx] == PATTYPE_NONE)){
		    t.get<0>() = 0.0;
		}else{
		    t.get<0>() = preOutput[timeIdx * preLayerSize + dimIdx] * weights[dimIdx];
		}
	    }else{
		// repeat the noise across time
		if (noiseRepeat == NN_OPERATOR_LAYER_NOISE_TIMEREPEAT) 
		    t.get<0>() = noiseData[(dimIdx - preLayerSize)];
		// repeat the noise across dimension
		else if (noiseRepeat == NN_OPERATOR_LAYER_NOISE_DIMREPEAT) 
		    t.get<0>() = noiseData[timeIdx * noiseDim];
		// normal case
		else             
		    t.get<0>() = noiseData[timeIdx * noiseDim + (dimIdx - preLayerSize)];
	    }	    
	}
    };


    struct timeResolutionChange
    {
	int         inputRes;   // time resolution of previous layer
	int         outputRes;  // time resolution of this layer
	int         layerSize;  // layer size
	int         parallel;   // parallel number
	
	real_t     *sourceData;
	const char *patTypes;

	// From 1 : T
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int outputIdx = t.get<1>();
	    int dimIdx    = outputIdx % layerSize;  // dimension index
	    int timeIdx   = outputIdx / layerSize;  // time index (regardless of parallel)
	    int BlockIdx  = timeIdx / parallel;     // time index (considering parallel mode)
	    int BlockInIdx= timeIdx % parallel;     // index within a parallel block

	    int fraction  = 1;  // change ratio

	    if (patTypes[timeIdx] == PATTYPE_NONE){
		t.get<0>() = 0;
		return;
	    }		
	    if (outputRes >= inputRes){
		// down sampling
		fraction = outputRes / inputRes;
		t.get<0>() = sourceData[((BlockIdx * fraction) * parallel + BlockInIdx) *
					layerSize + dimIdx];
	    }else{
		// up sampling
		fraction = inputRes / outputRes;
		t.get<0>() = sourceData[((BlockIdx / fraction) * parallel + BlockInIdx) *
					layerSize + dimIdx];
	    }
	    
	}
    };

    struct timeResolutionChangeGrad
    {
	int         inputRes;   // time resolution of previous layer
	int         outputRes;  // time resolution of this layer
	int         layerSize;  // layer size
	int         parallel;   // parallel number
	
	real_t     *sourceData; // source Data is the gradients of this layer
	const char *patTypes;   // previous layer's patTypes

	// From 1 : T (of the previous layer)
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int outputIdx = t.get<1>();
	    int dimIdx    = outputIdx % layerSize;  // dimension index
	    int timeIdx   = outputIdx / layerSize;  // time index (regardless of parallel)
	    int BlockIdx  = timeIdx / parallel;     // time index (considering parallel mode)
	    int BlockInIdx= timeIdx % parallel;     // index within a parallel block

	    int fraction  = 1;  // change ratio

	    if (patTypes[timeIdx] == PATTYPE_NONE){
		t.get<0>() = 0;
		return;
	    }		
	    
	    if (outputRes >= inputRes){
		// down sampling
		fraction = outputRes / inputRes;
		if (BlockIdx % fraction == 0){
		    t.get<0>() = sourceData[((BlockIdx/fraction) * parallel + BlockInIdx) *
					    layerSize + dimIdx];
		}else{
		    t.get<0>() = 0;
		}
	    }else{
		// up sampling
		fraction = inputRes / outputRes;
		t.get<0>() = 0;
		for (int i = 0; i<fraction; i++)
		    t.get<0>() += sourceData[((BlockIdx * fraction+i) * parallel + BlockInIdx) *
					     layerSize + dimIdx];
	    }
	    
	}
    };
    
    // #1
    
    struct outDuplicationOperation
    {
	int featureDim;
	int resolution;
	int maxTimeLength;
	int parall;
	
	real_t *dataMatrix;
	const char   *patTypes;
	
	__host__ __device__ void operator() (const thrust::tuple<const real_t&, int> &t) const
	{
	    int timeIdx  = t.get<1>() / featureDim;
	    int dimIdx   = t.get<1>() % featureDim;
	    int timeRel  = timeIdx    / parall;
	    int paraPos  = timeIdx    % parall;
	    
	    int blockIdx = timeRel    / resolution;

	    int copyIdx  = (blockIdx + 1) * resolution * parall - parall + paraPos;
	    if (((timeRel % resolution) == (resolution -1)) || patTypes[timeIdx] == PATTYPE_LAST)
		// either this is the last point of one block, or the end point of sentence
		return;
	    else{
		// if copyIdx is larger than sentence length, move back to end point of sentence
		while(patTypes[copyIdx]==PATTYPE_NONE){
		    copyIdx -= parall;
		}
		dataMatrix[t.get<1>()] = dataMatrix[copyIdx * featureDim + dimIdx];
	    }
	    //if (copyIdx < maxTimeLength)
	    //dataMatrix[t.get<1>()] = dataMatrix[copyIdx * featureDim + dimIdx];
	    //else
	    //dataMatrix[t.get<1>()] = dataMatrix[(maxTimeLength - parall + paraPos) * featureDim
	    //+ dimIdx];
	    
	}
    };


    struct outDuplicationGradOperation
    {
	int featureDim;
	int resolution;
	int maxTimeLength;
	int parall;
	
	real_t *dataMatrix;
	const char   *patTypes;
	
	__host__ __device__ void operator() (const thrust::tuple<const real_t&, int> &t) const
	{
	    int timeIdx  = t.get<1>() / featureDim;
	    int dimIdx   = t.get<1>() % featureDim;
	    int timeRel  = timeIdx    / parall;	    

	    if (patTypes[timeIdx] == PATTYPE_NONE){
		return;
	    }else if ((timeRel % resolution) == (resolution -1)){
		// accumulate the gradients
		int idx;
		for (int i = 1; i < resolution; i++){
		    idx = (timeIdx - i * parall) * featureDim + dimIdx;
		    dataMatrix[t.get<1>()] += dataMatrix[idx];
		    dataMatrix[idx] = 0;
		}
	    }else if (patTypes[timeIdx] == PATTYPE_LAST){
		int idx;
		for (int i = 1; i <= (timeRel % resolution); i++){
		    idx = (timeIdx - i * parall) * featureDim + dimIdx;
		    dataMatrix[t.get<1>()] += dataMatrix[idx];
		    dataMatrix[idx] = 0;
		}
	    }else{
		return;
	    }
	}
    };


    struct lastShotForward
    {
	int     featureDim;
	int     paralSeqNm;
	int     lastShotOp;
	
	int    *seqLengthD;
	real_t *sourceData;
	
	const char *patTypes;
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int outputIdx = t.get<1>();
	    int dimIdx    = outputIdx % featureDim;
	    int timeIdx   = outputIdx / featureDim;

	    if (patTypes[timeIdx] == PATTYPE_NONE){
		t.get<0>() = 0.0;
		return;
	    }
	    
	    int paralBlk  = timeIdx / paralSeqNm;
	    int sentIdx   = timeIdx % paralSeqNm;
	    int seqLength = seqLengthD[sentIdx];

	    if (lastShotOp == 1){
		// only copy the last timestep to the first timestep
		if (paralBlk == 0)
		    t.get<0>() = sourceData[((seqLength-1) * paralSeqNm + sentIdx) * featureDim
					    + dimIdx];
		else
		    t.get<0>() = 0.0;
	    }else{
		// copy the last timestep to all timesteps
		t.get<0>() = sourceData[((seqLength-1) * paralSeqNm + sentIdx) * featureDim
					+ dimIdx];
	    }
	}
	
    };


    struct lastShotForwardSegBoundary
    {
	int     featureDim;
	int     paralSeqNm;
	int     lastShotOp;
	
	int    *segBoundary;
	real_t *sourceData;
	
	const char *patTypes;
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int outputIdx = t.get<1>();
	    int dimIdx    = outputIdx % featureDim;
	    int timeIdx   = outputIdx / featureDim;

	    if (patTypes[timeIdx] == PATTYPE_NONE){
		t.get<0>() = 0.0;
		return;
	    }
	    
	    //int paralBlk  = timeIdx / paralSeqNm;
	    int sentIdx   = timeIdx % paralSeqNm;
	    int boundary  = segBoundary[timeIdx];

	    if (boundary < 0){
		t.get<0>() = 0.0;
	    }else{
		t.get<0>() = sourceData[(boundary * paralSeqNm + sentIdx) * featureDim + dimIdx];
	    }
	}
    };

    struct lastShotForwardSegBoundaryGrad
    {
	int     featureDim;
	int     paralSeqNm;
	int     lastShotOp;
	
	int    *segBoundary;
	real_t *targetData;
	
	const char *patTypes;
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int outputIdx = t.get<1>();
	    int dimIdx    = outputIdx % featureDim;
	    int timeIdx   = outputIdx / featureDim;

	    if (patTypes[timeIdx] == PATTYPE_NONE){
		t.get<0>() = 0.0;
		return;
	    }
	    
	    //int paralBlk  = timeIdx / paralSeqNm;
	    int sentIdx   = timeIdx % paralSeqNm;
	    int boundary  = segBoundary[timeIdx];

	    if (boundary < 0){
		// not boundary
	    }else{
		targetData[(boundary * paralSeqNm + sentIdx) * featureDim + dimIdx] = t.get<0>();
	    }
	}
    };

}


namespace layers{

    template <typename TDevice>
    OperationLayer<TDevice>::OperationLayer(const helpers::JsonValue &layerChild,
					    const helpers::JsonValue &weightsSection,
					    Layer<TDevice>           &precedingLayer,
					    int                       maxSeqLength)
	: TrainableLayer<TDevice>(layerChild, weightsSection, 0, 0, precedingLayer, maxSeqLength)
	, m_noiseMag    (1.0)
	, m_noiseSize   (0)
	, m_noiseRepeat (0)
	, m_outDupRate  (0)
	, m_lastShot    (0)
	, m_segLevel    (-1)
	, m_changeTimeRes (0)
    {

	/* ------ Configuration for noise generation ------ */
	// Note: in operation layer, noise is concatenated with input features
	//       for noises added to the input features, use skipIni layer
	m_noiseMag    = (layerChild->HasMember("noiseRatio") ? 
			 static_cast<real_t>((*layerChild)["noiseRatio"].GetDouble()) : 1.0);
	m_noiseSize   = (layerChild->HasMember("noiseDim") ? 
			 static_cast<real_t>((*layerChild)["noiseDim"].GetInt()) : 0);
	m_noiseRepeat = (layerChild->HasMember("noiseRepeat") ? 
			 static_cast<real_t>((*layerChild)["noiseRepeat"].GetInt()) : 0);
	// check the layer size
	if (this->size() != (this->precedingLayer().size() + m_noiseSize))
	    throw std::runtime_error("Error operator layer, noiseDim + preLayerSize = layerSize");
	// 
	m_noiseInput.resize(m_noiseSize * (this->precedingLayer().outputs().size() /
					   this->precedingLayer().size()), 0.0);

	/* ------ Configuration for weighting the input features ------ */
	// Configuration for the weight of input vector
	m_setZeroStr  = ((layerChild->HasMember("setZero")) ? 
			 ((*layerChild)["setZero"].GetString()) : (""));
	if (m_setZeroStr.size()){
	    m_setZeroVec_H.clear();
	    misFuncs::ParseFloatOpt(m_setZeroStr, m_setZeroVec_H);
	    m_setZeroVec_D = m_setZeroVec_H;
	}else{
	    m_setZeroVec_D.resize(this->precedingLayer().size(), 1.0);
	}
	
	if (this->precedingLayer().size() != m_setZeroVec_D.size())
	    throw std::runtime_error("Error operator setZero, unequal to previous layer size");

	/* ------ Configuration of the output duplication ------ */
	if (layerChild->HasMember("outputDownSampling")){
	    printf("\toutputDownSampling flag has been changed to outputDuplicating\n");
	    throw std::runtime_error("Error: old configuration name in OperationLayer");
	}
	m_outDupRate   = (layerChild->HasMember("outputDuplicating") ? 
			   static_cast<int>((*layerChild)["outputDuplicating"].GetInt()) : 0);

	/* ------ Configuration of last shot mode ------ */
	//
	// m_lastShot = 1: use last frame of precedingLayer as the first frame of this layer
	// m_lastShot = 2: use last frame of precedingLayer as the all frames of this layer
	// m_lastShot = 3: do m_lastShot=1 for every segment defined by the boundary
	// m_lastShot = 4: not implemented now
	m_lastShot      = (layerChild->HasMember("lastShot")?
			   static_cast<int>((*layerChild)["lastShot"].GetInt()) : 0);
	// Configuration of the extraction of the last time steps
	m_segLevel      = (layerChild->HasMember("segLevel")?
			   static_cast<int>((*layerChild)["segLevel"].GetInt()) : -1);
	if (m_lastShot > 0){
	    // only use the utterance end boundary	    
	    if (m_lastShot == NN_OPE_LAST_SHOT_MODE1 || m_lastShot == NN_OPE_LAST_SHOT_MODE2){
		m_seqLengthBuffH.resize(this->parallelSequences(), 0);
		m_seqLengthBuffD = m_seqLengthBuffH;
		cpu_real_vector tmp(this->parallelSequences() * this->maxSeqLength() +
				    this->parallelSequences() - 1, 0.0);
		if (m_lastShot == NN_OPE_LAST_SHOT_MODE2){
		    for (int i = this->parallelSequences(); i<tmp.size();
			 i+=this->parallelSequences())
			tmp[i-1] = 1.0;
		}else{
		    tmp[this->parallelSequences()-1] = 1.0;
		}
		m_oneVec = tmp;
		
	    // use the segmental boundary
	    }else if (m_lastShot == NN_OPE_LAST_SHOT_MODE3 || NN_OPE_LAST_SHOT_MODE4){
		if (m_segLevel < 0)
		    throw std::runtime_error("segLevel is not configured for Operationlayer");
		m_segBoundaryH.resize(this->maxSeqLength() * this->parallelSequences(), 0);
		m_segBoundaryD = m_segBoundaryH;
		
	    }else{
		throw std::runtime_error("Unknown lastShot option number");
	    }
	}

	/* ------- Configuration of the time resolution change */
	m_changeTimeRes   = (layerChild->HasMember("changeResolution") ? 
			     static_cast<int>((*layerChild)["changeResolution"].GetInt()) : 0);
	if (m_changeTimeRes && this->size() != this->precedingLayer().size())
	    throw std::runtime_error("Layer size unequal for time resolution change");
	if (this->getResolution() > this->precedingLayer().getResolution()){
	    // down sampling
	    if ((this->getResolution() % this->precedingLayer().getResolution()) != 0)
		throw std::runtime_error("Fractional resolution change is not supported");
	}else{
	    // up sampling
	    if ((this->precedingLayer().getResolution() % this->getResolution())!= 0)
		throw std::runtime_error("Fractional resolution change is not supported");
	}
	
	/* ------ print the information ------ */
	printf("\tOperator layer: \n");
	if (m_noiseSize > 0)
	    printf("\tinject noise: dim %d, u[-%f, %f]\n", m_noiseSize, m_noiseMag, m_noiseMag);
	
	if (m_setZeroStr.size())
	    printf("\tinput/output configuration: %s\n", m_setZeroStr.c_str());
	
	if (m_noiseRepeat){
	    if (m_noiseRepeat == NN_OPERATOR_LAYER_NOISE_TIMEREPEAT)
		printf("\trepeat the same noise across frames\n");
	    else if (m_noiseRepeat == NN_OPERATOR_LAYER_NOISE_DIMREPEAT)
		printf("\trepeat the same noise across dimension\n");
	    else
		printf("\tunknown noise repeat option\n");
	}
	
	if (m_outDupRate > 1)
	    printf("\toutput duplication at the rate of %d\n", m_outDupRate);

	if (m_changeTimeRes){
	    printf("\tTurn of time resolution change across layers: from %d to %d",
		   this->precedingLayer().getResolution(), this->getResolution());
	}
	
	if (m_lastShot > 0){
	    printf("\tlast shot is used [%d]\n", m_lastShot);
	    if (m_noiseSize > 0 || m_outDupRate > 1 || m_setZeroStr.size() || m_changeTimeRes > 0)
		throw std::runtime_error("lastShot mode can't be used with nother operation");
	    if (this->size() != this->precedingLayer().size())
		throw std::runtime_error("Layer size is unequal to previous one");
	}

	if (this->precedingLayer().getSaveMemoryFlag())
	    throw std::runtime_error("layer before operator is reduced in mem");  
	
    }

    template <typename TDevice>
    OperationLayer<TDevice>::~OperationLayer()
    {
    }

    template <typename TDevice>
    void OperationLayer<TDevice>::exportLayer(const helpers::JsonValue     &layersArray, 
					      const helpers::JsonAllocator &allocator) const
    {
        TrainableLayer<TDevice>::exportLayer(layersArray, allocator);
	if (m_setZeroStr.size())
	    (*layersArray)[layersArray->Size() - 1].AddMember("setZero",     m_setZeroStr.c_str(),
							      allocator);
	if (m_noiseSize > 0){
	    (*layersArray)[layersArray->Size() - 1].AddMember("noiseRatio",  m_noiseMag,
							      allocator);
	    (*layersArray)[layersArray->Size() - 1].AddMember("noiseDim",    m_noiseSize,
							      allocator);
	    (*layersArray)[layersArray->Size() - 1].AddMember("noiseRepeat", m_noiseRepeat,
							      allocator);
	}
	
	if (m_outDupRate > 1)
	    (*layersArray)[layersArray->Size() - 1].AddMember("outputDuplicating", m_outDupRate,
							      allocator);
	if (m_lastShot > 0){
	    (*layersArray)[layersArray->Size() - 1].AddMember("lastShot", m_lastShot,
							      allocator);
	    (*layersArray)[layersArray->Size() - 1].AddMember("segLevel", m_segLevel,
							      allocator);
	}

	if (m_changeTimeRes > 0)
	    (*layersArray)[layersArray->Size() - 1].AddMember("changeResolution", m_changeTimeRes,
							      allocator);
    }

    template <typename TDevice>
    void OperationLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction,
						const int nnState)
    {
	TrainableLayer<TDevice>::loadSequences(fraction, nnState);

	if (m_lastShot  == NN_OPE_LAST_SHOT_MODE1 || m_lastShot == NN_OPE_LAST_SHOT_MODE2){
	    
	    // load the sequence length 
	    for (int i = 0; i<fraction.numSequences(); i++)
		m_seqLengthBuffH[i] = fraction.seqInfo(i).length;
	    m_seqLengthBuffD = m_seqLengthBuffH;

	}else if (m_lastShot == NN_OPE_LAST_SHOT_MODE3 || m_lastShot == NN_OPE_LAST_SHOT_MODE4){
	    
	    // load the segments length
	    if (m_segLevel > CHAR_BIT)
		throw std::runtime_error("Operationlayer: segLevel is larger than expected");
	    if (fraction.auxPattypeData().size() == 0)
		throw std::runtime_error("Operationlayer: Last-shot requires boundary (auxData)");
	    
	    int pos;
	    int boundary;
	    char bitOp = (0b01 << m_segLevel);	
	    for (int i = 0; i < fraction.numSequences(); i++){
		// the last segment boundry is the end of utterance
		boundary = (fraction.seqInfo(i).length - 1);
		for (int time = fraction.seqInfo(i).length - 1; time>=0; time--){
		    pos = time * this->parallelSequences() + i;  // abosolute position
		    if (m_lastShot == NN_OPE_LAST_SHOT_MODE3){
			if (fraction.auxPattypeData()[pos] & bitOp)
			    m_segBoundaryH[pos] = boundary; // this is the start of segment
			else
			    m_segBoundaryH[pos] = -1;       // other frames
		    }else{
			m_segBoundaryH[pos] = boundary;     
		    }
			
		    if (fraction.auxPattypeData()[pos] & bitOp){
			// update the boundary
			boundary = time - 1;
		    }
		}
	    }
	    m_segBoundaryD = m_segBoundaryH;
	    
	// nothing 
	}else{

	}
    }
    
    template <typename TDevice>
    const std::string& OperationLayer<TDevice>::type() const
    {
        static std::string s;
        if (s.empty()) s = "operator";
        return s;
    }

    template <typename TDevice>
    void OperationLayer<TDevice>::computeForwardPass(const int nnState)
    {
	int timeLength = this->curMaxSeqLength() * this->parallelSequences();

	if (m_lastShot == NN_OPE_LAST_SHOT_MODE1 || m_lastShot == NN_OPE_LAST_SHOT_MODE2){
	    // use last shot mode
	    internal::lastShotForward fn1;
	    fn1.featureDim = this->size();
	    fn1.paralSeqNm = this->parallelSequences();
	    fn1.lastShotOp = this->m_lastShot;
	    fn1.seqLengthD = helpers::getRawPointer(m_seqLengthBuffD);
	    fn1.sourceData = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn1.patTypes   = helpers::getRawPointer(this->patTypes());
	    
	    int n = timeLength * this->size();
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + n,
				     thrust::counting_iterator<int>(0) + n)),
	       fn1);
	    
	}else if (m_lastShot == NN_OPE_LAST_SHOT_MODE3 || m_lastShot == NN_OPE_LAST_SHOT_MODE4){
	    // use last shot mode based on segmental boundary
	    internal::lastShotForwardSegBoundary fn1;
	    fn1.featureDim  = this->size();
	    fn1.paralSeqNm  = this->parallelSequences();
	    fn1.lastShotOp  = this->m_lastShot;
	    fn1.segBoundary = helpers::getRawPointer(m_segBoundaryD);
	    fn1.sourceData  = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn1.patTypes    = helpers::getRawPointer(this->patTypes());
	    
	    int n = timeLength * this->size();
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + n,
				     thrust::counting_iterator<int>(0) + n)),
	       fn1);
	    	    
	}else if (m_changeTimeRes){
	    
	    internal::timeResolutionChange fn1;
	    fn1.inputRes  = this->precedingLayer().getResolution();
	    fn1.outputRes = this->getResolution();
	    fn1.layerSize = this->size();
	    fn1.parallel  = this->parallelSequences();
	    fn1.sourceData  = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn1.patTypes    = helpers::getRawPointer(this->patTypes());
	    
	    int n = timeLength * this->size();
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + n,
				     thrust::counting_iterator<int>(0) + n)),
	       fn1);
	    
	}else{
	    // normal mode
	    if (m_noiseSize > 0){
		// generate the noise for all frames
		thrust::counting_iterator<unsigned int> index_sequence_begin(0);
		thrust::transform(
				  index_sequence_begin,
				  index_sequence_begin + timeLength * m_noiseSize,
				  m_noiseInput.begin(),
				  internal::genNoise(-1.0 * m_noiseMag, m_noiseMag,
						     (int)(misFuncs::GetRandomNumber()*10000.0)));

	    }
	
	    {
	    internal::fillOutputVec fn;
	    fn.curLayerSize = this->size();
	    fn.preLayerSize = this->precedingLayer().size();
	    fn.noiseDim     = m_noiseSize;
	    fn.noiseRepeat  = m_noiseRepeat;
	    
	    fn.weights   = helpers::getRawPointer(m_setZeroVec_D);
	    fn.preOutput = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn.noiseData = helpers::getRawPointer(this->m_noiseInput);
	    fn.patTypes  = helpers::getRawPointer(this->patTypes());

	    int n = timeLength * this->size();
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + n,
				     thrust::counting_iterator<int>(0) + n)),
	       fn);
	    
	    }
	    
	    if (m_outDupRate > 1){
		internal::outDuplicationOperation fn1;
		fn1.featureDim = this->size();
		fn1.resolution = m_outDupRate;
		fn1.maxTimeLength = timeLength;
		fn1.dataMatrix = helpers::getRawPointer(this->outputs());
		fn1.parall     = this->parallelSequences();
		fn1.patTypes  = helpers::getRawPointer(this->patTypes());
		
		int n = timeLength * this->size();
		thrust::for_each(
		    thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin(),
					   thrust::counting_iterator<int>(0))),
		    thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin()           + n,
					   thrust::counting_iterator<int>(0) + n)),
		    fn1);
	    }
	}

    }

    template <typename TDevice>
    void OperationLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
	int timeLength = this->curMaxSeqLength() * this->parallelSequences();

	if (m_lastShot == NN_OPE_LAST_SHOT_MODE1 || m_lastShot == NN_OPE_LAST_SHOT_MODE2){
	    // Tricky code
	    
	    /*
	      This part should be handled. LastShot mode is now only used before the VAE
	      layer. In the generation time, VAE with m_vaeUseageOpt==2 will not transform
	      the output from the lastShot layers. 

	      The code just set up the boundary of the segments
	     */
	    
	    // Although operator with last shot should not be used after a feedback layer
	    // (because it generates the output at the end of segment and uses it at the begining
	    //  of a segment), the boundary information can be generated 
	    if (timeStep == 0){
		thrust::fill(this->precedingLayer().outputs().begin(),
			     this->precedingLayer().outputs().begin()+timeLength * this->size(),
			     1.0);
		internal::lastShotForward fn1;
		fn1.featureDim = this->size();
		fn1.paralSeqNm = this->parallelSequences();
		fn1.lastShotOp = this->m_lastShot;
		fn1.seqLengthD = helpers::getRawPointer(m_seqLengthBuffD);
		fn1.sourceData = helpers::getRawPointer(this->precedingLayer().outputs());
		fn1.patTypes   = helpers::getRawPointer(this->patTypes());
	    
		int n = timeLength * this->size();
		thrust::for_each(
		 thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin(),
				     thrust::counting_iterator<int>(0))),
		 thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + n,
				     thrust::counting_iterator<int>(0) + n)),
		 fn1);
	    }
	}else if (m_lastShot == NN_OPE_LAST_SHOT_MODE3 || m_lastShot == NN_OPE_LAST_SHOT_MODE4){
	    /*
	      This part should be handled. LastShot mode is now only used before the VAE
	      layer. In the generation time, VAE with m_vaeUseageOpt==2 will not transform
	      the output from the lastShot layers. 

	      The code just set up the boundary of the segments
	     */
	    
	    // Last shot mode can not be used here
	    // 
	    if (timeStep == 0){
		thrust::fill(this->precedingLayer().outputs().begin(),
			     this->precedingLayer().outputs().begin()+timeLength * this->size(),
			     1.0);
		internal::lastShotForwardSegBoundary fn1;
		fn1.featureDim  = this->size();
		fn1.paralSeqNm  = this->parallelSequences();
		fn1.lastShotOp  = this->m_lastShot;
		fn1.segBoundary = helpers::getRawPointer(m_segBoundaryD);
		fn1.sourceData  = helpers::getRawPointer(this->precedingLayer().outputs());
		fn1.patTypes    = helpers::getRawPointer(this->patTypes());
	    
		int n = timeLength * this->size();
		thrust::for_each(
                 thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin(),
				     thrust::counting_iterator<int>(0))),
		 thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + n,
				     thrust::counting_iterator<int>(0) + n)),
		 fn1);
	    }
	}else if (m_changeTimeRes){

	    /* Input timeStep is the default time resolution of the network	      
	       Assume that timeResolution >= 1
	     */
	    
	    // time resolution has been considered in NeuralNetwork.cpp
	    //int st = (timeStep / this->getResolution()) * this->size();
	    //int et = (timeStep / this->getResolution()) * this->size() + this->size();
	    int st = timeStep * this->size();
	    int et = timeStep * this->size() + this->size();
	    
	    internal::timeResolutionChange fn1;
	    fn1.inputRes    = this->precedingLayer().getResolution();
	    fn1.outputRes   = this->getResolution();
	    fn1.layerSize   = this->size();
	    fn1.parallel    = this->parallelSequences();
	    fn1.sourceData  = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn1.patTypes    = helpers::getRawPointer(this->patTypes());
	    
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + st,
				     thrust::counting_iterator<int>(0) + st)),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + et,
				     thrust::counting_iterator<int>(0) + et)),
	       fn1);
	
	}else{
	
	    if (m_noiseSize > 0 && timeStep == 0){
		// generate the noise for all frames at the 1st timeStep
		thrust::counting_iterator<unsigned int> index_sequence_begin(0);
		thrust::transform(
				  index_sequence_begin,
				  index_sequence_begin + timeLength * m_noiseSize,
				  m_noiseInput.begin(),
				  internal::genNoise(-1.0 * m_noiseMag, m_noiseMag,
						     (int)(misFuncs::GetRandomNumber()*10000.0)));

	    }
	    {
	    internal::fillOutputVec fn;
	    fn.curLayerSize = this->size();
	    fn.preLayerSize = this->precedingLayer().size();
	    fn.noiseDim     = m_noiseSize;
	    fn.noiseRepeat  = m_noiseRepeat;
	    
	    fn.weights   = helpers::getRawPointer(m_setZeroVec_D);
	    fn.preOutput = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn.noiseData = helpers::getRawPointer(this->m_noiseInput);
	    fn.patTypes  = helpers::getRawPointer(this->patTypes());
	    
	    int st = timeStep * this->size();
	    int et = timeStep * this->size() + this->size();
	    
	    if (timeStep == 0 && this->precedingLayer().type()=="vae"){
		// for VAE layer, we need to load the noise for all frames
		st = 0;
		et = timeLength * this->size();
	    }
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + st,
				     thrust::counting_iterator<int>(0) + st)),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + et,
				     thrust::counting_iterator<int>(0) + et)),
	       fn);
	    }

	    if (m_outDupRate > 1){
		internal::outDuplicationOperation fn1;
		fn1.featureDim = this->size();
		fn1.resolution = m_outDupRate;
		fn1.maxTimeLength = timeLength;
		fn1.dataMatrix = helpers::getRawPointer(this->outputs());
		fn1.parall     = this->parallelSequences();
		fn1.patTypes  = helpers::getRawPointer(this->patTypes());
	    
		int st = timeStep * this->size();
		int et = timeStep * this->size() + this->size();
		thrust::for_each(
		 thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + st,
				     thrust::counting_iterator<int>(0) + st)),
		 thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + et,
				     thrust::counting_iterator<int>(0) + et)),
		 fn1);
	    }
	}

    }

    template <typename TDevice>
    void OperationLayer<TDevice>::computeBackwardPass(const int nnState)
    {	
	int timeLength = this->curMaxSeqLength() * this->parallelSequences();

	if (m_lastShot == NN_OPE_LAST_SHOT_MODE1 || m_lastShot == NN_OPE_LAST_SHOT_MODE2){
	    thrust::fill(this->precedingLayer().outputErrors().begin(),
			 this->precedingLayer().outputErrors().end(), 0.0);
	    for (int i = 0; i<this->parallelSequences(); i++){
		timeLength = this->parallelSequences() * this->m_seqLengthBuffH[i];
		// A trick for parallel training mode: circular move over one vector
		//       A B C D A B C D
		// 0 0 0 1 0 0 0 1 0 0 0          sum As, shift 3
		//   0 0 0 1 0 0 0 1 0 0 0        sum Bs, shift 2
		//     0 0 0 1 0 0 0 1 0 0 0      sum Cs, shift 1
		//       0 0 0 1 0 0 0 1 0 0 0    sum Ds, shift 0
		helpers::Matrix<TDevice> onevec  (&this->m_oneVec, timeLength, 1,
						  (this->parallelSequences() - 1 - i));
		helpers::Matrix<TDevice> source  (&this->outputErrors(), this->size(), timeLength);
		helpers::Matrix<TDevice> output  (&this->precedingLayer().outputErrors(), 
						  this->size(), 1,
						  (timeLength - this->parallelSequences() + i) *
						  this->size());
		// sum the gradients for a_k
		output.assignProduct(source, false, onevec, false);		
	    }
	    
	}else if (m_lastShot == NN_OPE_LAST_SHOT_MODE3 || m_lastShot == NN_OPE_LAST_SHOT_MODE4){
	    thrust::fill(this->precedingLayer().outputErrors().begin(),
			 this->precedingLayer().outputErrors().end(), 0.0);
	    if (m_lastShot == NN_OPE_LAST_SHOT_MODE3){
		// use last shot mode based on segmental boundary
		internal::lastShotForwardSegBoundaryGrad fn1;
		fn1.featureDim  = this->size();
		fn1.paralSeqNm  = this->parallelSequences();
		fn1.lastShotOp  = this->m_lastShot;
		fn1.segBoundary = helpers::getRawPointer(m_segBoundaryD);
		fn1.targetData  = helpers::getRawPointer(this->precedingLayer().outputErrors());
		fn1.patTypes    = helpers::getRawPointer(this->patTypes());
	    
		int n = timeLength * this->size();
		thrust::for_each(
                 thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputErrors().begin(),
				     thrust::counting_iterator<int>(0))),
		 thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputErrors().begin() + n,
				     thrust::counting_iterator<int>(0) + n)),
		 fn1);

	    }else{
		// not implemented
	    }
	}else if (m_changeTimeRes){

	    internal::timeResolutionChangeGrad fn1;
	    fn1.inputRes  = this->precedingLayer().getResolution();
	    fn1.outputRes = this->getResolution();
	    fn1.layerSize = this->size();
	    fn1.parallel  = this->parallelSequences();
	    fn1.sourceData  = helpers::getRawPointer(this->outputErrors());
	    fn1.patTypes    = helpers::getRawPointer(this->precedingLayer().patTypes());
	    
	    int n = (this->precedingLayer().curMaxSeqLength()    *
		     this->precedingLayer().parallelSequences()  * this->precedingLayer().size());
	    
	    thrust::for_each(
              thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputErrors().begin(),
				   thrust::counting_iterator<int>(0))),
	      thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputErrors().begin() + n,
				     thrust::counting_iterator<int>(0) + n)),
	       fn1);
	    
	    
	}else{
	    
	    if (m_outDupRate > 1){
		internal::outDuplicationGradOperation fn1;
		fn1.featureDim = this->size();
		fn1.resolution = m_outDupRate;
		fn1.maxTimeLength = timeLength;
		fn1.dataMatrix = helpers::getRawPointer(this->outputErrors());	
		fn1.parall     = this->parallelSequences();
		fn1.patTypes   = helpers::getRawPointer(this->patTypes());
		
		int n = timeLength * this->size();
		thrust::for_each(
                 thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputErrors().begin(),
				     thrust::counting_iterator<int>(0))),
		 thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputErrors().begin()      + n,
				     thrust::counting_iterator<int>(0) + n)),
		 fn1);
	    }
	
	    {
		// Although it is tricky to use the same function as in computeForwardPass,
		// it works by constraining the curLayerSize and preLayerSize
	    internal::fillOutputVec fn;
	    fn.curLayerSize = this->precedingLayer().size();
	    fn.preLayerSize = this->size();
	    fn.noiseDim     = m_noiseSize;
	    
	    fn.weights   = helpers::getRawPointer(m_setZeroVec_D);
	    fn.preOutput = helpers::getRawPointer(this->outputErrors());
	    fn.noiseData = helpers::getRawPointer(this->m_noiseInput);
	    fn.patTypes  = helpers::getRawPointer(this->patTypes());
	    
	    int n = timeLength * this->precedingLayer().size();
	    thrust::for_each(
               thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputErrors().begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		thrust::make_tuple(this->precedingLayer().outputErrors().begin() + n,
				     thrust::counting_iterator<int>(0)           + n)),
	       fn);
	    }
	}

    }

    template class OperationLayer<Cpu>;
    template class OperationLayer<Gpu>;
    
}



/*
    #1
    struct outDuplicationMatrix
    {
	int size;
	int resolution;
	int parall;
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int rowidx   = t.get<1>() % size;
	    int colidx   = t.get<1>() / size;

	    int rowidx2  = rowidx  / parall;
	    int colidx2  = colidx  / parall;
	    
	    //int paralPos = rowidx  % parall;
	    int blockidx = rowidx2 / resolution;
	    
	    // the matrix is column major
	    // [ 0 0 0 0 0 0 0 0
	    //   ...
	    //   1 1 ... 1 0 0 0  -> resolution-th row
	    //   |--------|
	    //   resolution columns
	    // move right and repeat this pattern

	    // each block
	    if (((rowidx2 % resolution) == (resolution - 1)) &&
		(colidx2 >= (blockidx * resolution)) &&
		(colidx2 <  ((blockidx + 1) * resolution)) &&
		(rowidx % parall) == (colidx % parall))
		t.get<0>() = 1.0/((real_t)resolution);

	    // last row
	    else if (rowidx2 == (size / parall - 1) &&
		     (colidx2 >= (blockidx * resolution)) &&
		     (rowidx % parall) == (colidx % parall))
		t.get<0>() = 1.0/((real_t)resolution);
	    
	}
    };
    
*/
