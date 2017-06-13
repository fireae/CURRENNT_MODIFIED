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




namespace internal{
namespace {
    
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

	real_t *zeroFlag;
	
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
		    t.get<0>() = preOutput[timeIdx * preLayerSize + dimIdx] * zeroFlag[dimIdx];
		}
	    }else{
		if (noiseRepeat == NN_OPERATOR_LAYER_NOISE_TIMEREPEAT)// use the first noise vector
		    t.get<0>() = noiseData[(dimIdx - preLayerSize)];
		else if (noiseRepeat == NN_OPERATOR_LAYER_NOISE_DIMREPEAT) 
		    t.get<0>() = noiseData[timeIdx * noiseDim];
		else             // use the noise of each frame
		    t.get<0>() = noiseData[timeIdx * noiseDim + (dimIdx - preLayerSize)];
	    }	    
	}
    };
    
}
}


namespace layers{

    template <typename TDevice>
    OperationLayer<TDevice>::OperationLayer(const helpers::JsonValue &layerChild,
					    const helpers::JsonValue &weightsSection,
					    Layer<TDevice>           &precedingLayer)
	: TrainableLayer<TDevice>(layerChild, weightsSection, 0, 0, precedingLayer)
	, m_noiseMag    (1.0)
	, m_noiseSize   (0)
	, m_noiseRepeat (0)
    {
	
	m_noiseMag    = (layerChild->HasMember("noiseRatio") ? 
			 static_cast<real_t>((*layerChild)["noiseRatio"].GetDouble()) : 1.0);
	m_noiseSize   = (layerChild->HasMember("noiseDim") ? 
			 static_cast<real_t>((*layerChild)["noiseDim"].GetInt()) : 0);
	m_noiseRepeat = (layerChild->HasMember("noiseRepeat") ? 
			 static_cast<real_t>((*layerChild)["noiseRepeat"].GetInt()) : 0);
	
	
	// check the layer size)
	if (this->size() != (this->precedingLayer().size() + m_noiseSize))
	    throw std::runtime_error("Error operator layer, noiseDim + preLayerSize = layerSize");
	
	m_setZeroStr  = ((layerChild->HasMember("setZero")) ? 
			 ((*layerChild)["setZero"].GetString()) : (""));
	if (m_setZeroStr.size()){
	    m_setZeroVec_H.clear();
	    ParseFloatOpt(m_setZeroStr, m_setZeroVec_H);
	    m_setZeroVec_D = m_setZeroVec_H;
	}else{
	    m_setZeroVec_D.resize(this->precedingLayer().size(), 1.0);
	}
	if (this->precedingLayer().size() != m_setZeroVec_D.size())
	    throw std::runtime_error("Error operator setZero, unequal to previous layer size");

	m_noiseInput.resize(m_noiseSize * (this->precedingLayer().outputs().size() /
					   this->precedingLayer().size()), 0.0);

	// print the information
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
        (*layersArray)[layersArray->Size() - 1].AddMember("setZero",    m_setZeroStr.c_str(),
							  allocator);
	(*layersArray)[layersArray->Size() - 1].AddMember("noiseRatio", m_noiseMag,
							  allocator);
	(*layersArray)[layersArray->Size() - 1].AddMember("noiseDim",   m_noiseSize,
							  allocator);
	(*layersArray)[layersArray->Size() - 1].AddMember("noiseRepeat",m_noiseRepeat,
							  allocator);
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
	
	if (m_noiseSize > 0){
	    // generate the noise for all frames
	    thrust::counting_iterator<unsigned int> index_sequence_begin(0);
	    thrust::transform(
			      index_sequence_begin,
			      index_sequence_begin + timeLength * m_noiseSize,
			      m_noiseInput.begin(),
			      internal::genNoise(-1.0 * m_noiseMag, m_noiseMag,
						 (int)(GetRandomNumber()*10000.0)));

	}
	
	{
	    internal::fillOutputVec fn;
	    fn.curLayerSize = this->size();
	    fn.preLayerSize = this->precedingLayer().size();
	    fn.noiseDim     = m_noiseSize;
	    fn.noiseRepeat  = m_noiseRepeat;
	    
	    fn.zeroFlag  = helpers::getRawPointer(m_setZeroVec_D);
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
    }

    template <typename TDevice>
    void OperationLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
	throw std::runtime_error("Not implmented OperationLayer forward(timestep)");
    }

    template <typename TDevice>
    void OperationLayer<TDevice>::computeBackwardPass(const int nnState)
    {	
	int timeLength = this->curMaxSeqLength() * this->parallelSequences();
	
	{
	    internal::fillOutputVec fn;
	    fn.curLayerSize = this->precedingLayer().size();
	    fn.preLayerSize = this->size();
	    fn.noiseDim     = m_noiseSize;
	    
	    fn.zeroFlag  = helpers::getRawPointer(m_setZeroVec_D);
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

    template class OperationLayer<Cpu>;
    template class OperationLayer<Gpu>;
    
}
