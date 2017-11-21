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

#include "../Configuration.hpp"
#include "vqLayer.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../helpers/JsonClasses.hpp"
#include "../helpers/misFuncs.hpp"

#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/fill.h>
#include <thrust/random.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string.hpp>
#include <vector>


namespace internal{

    struct computeDisMatrix
    {
	int featureDim;
	int codeBookSize;
	real_t *inputData;
	real_t *codeData;
	
	const char *patTypes;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int index = t.get<1>();
	    int timeIdx = index / codeBookSize;
	    int codeIdx = index % codeBookSize;
	    
	    real_t distance = 0.0;

	    if (patTypes[timeIdx] == PATTYPE_NONE)
		t.get<0>() = distance;
	    else{
		for (int i = 0; i < featureDim; i++){
		    distance += ((codeData[codeIdx * featureDim + i] -
				  inputData[timeIdx * featureDim + i]) *
				 (codeData[codeIdx * featureDim + i] -
				  inputData[timeIdx * featureDim + i]));
		}
		t.get<0>() = distance;
	    }   
	}
    };

    struct getBestIndex
    {
	int codeBookSize;
	real_t *disMatrix;
	
	const char *patTypes;

	// for 0 : T 
	__host__ __device__ void operator() (const thrust::tuple<int&, int> &t) const
	{
	    int timeIdx = t.get<1>();
	    if (patTypes[timeIdx] == PATTYPE_NONE)
		t.get<0>() = -1;
	    else{
		if (codeBookSize == 1)
		    t.get<0>() = -1;
		else{
		    real_t tempMin = disMatrix[timeIdx * codeBookSize];
		    real_t tempId  = 0; 
		    for (int i = 1; i < codeBookSize; i++){
			if (disMatrix[timeIdx * codeBookSize + i] < tempMin){
			    tempMin = disMatrix[timeIdx * codeBookSize + i];
			    tempId  = i;
			}
		    }
		    t.get<0>() = tempId;
		}
	    }
	}
    };


    struct LoadVq
    {
	int featureDim;
	int codeBookSize;
	
	real_t *codeData;
	int *index;
	
	const char *patTypes;

	// for 0 : T * featureDim
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int timeIdx = t.get<1>() / featureDim;
	    int featIdx = t.get<1>() % featureDim;
	    if (patTypes[timeIdx] == PATTYPE_NONE)
		t.get<0>() = 0.0;
	    else if (index[timeIdx] >= 0)
		t.get<0>() = codeData[index[timeIdx] * featureDim + featIdx];
	    else
		t.get<0>() = 0.0;
	}
    };


    struct CodeDiff
    {
	int featureDim;
	int codeBookSize;
	
	real_t *codeData;
	int *index;
	
	const char *patTypes;

	// for 0 : T * featureDim
	__host__ __device__ real_t operator() (const thrust::tuple<const real_t&, int> &t) const
	{
	    int timeIdx = t.get<1>() / featureDim;
	    int featIdx = t.get<1>() % featureDim;
	    if (patTypes[timeIdx] == PATTYPE_NONE)
		return 0.0;
	    else if (index[timeIdx] >= 0)
		return ((t.get<0>() - codeData[index[timeIdx] * featureDim + featIdx]) *
			(t.get<0>() - codeData[index[timeIdx] * featureDim + featIdx]));
	    else
		return 0.0;
	}
    };

    struct GradientForCodeBook
    {
	int featureDim;
	int codeBookSize;
	real_t  beta;
	
	real_t *inputData;
	real_t *codeData;
	real_t *preLayerGrad;
	int *index;

	int timeLength;
	const char *patTypes;

	// for codeBookSize * featureDim
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{

	    int codeIdx = t.get<1>() / featureDim;
	    int featIdx = t.get<1>() % featureDim;

	    real_t sumInput = 0.0;
	    real_t cnt = 0.0;
	    for (int i = 0; i<timeLength; i++){
		if (patTypes[i] == PATTYPE_NONE)
		    break;
		if (index[i] == codeIdx){
		    cnt += 1.0;
		    // Methods1/2: Moving average of the input latent codes
		    sumInput += (inputData[i * featureDim + featIdx] - sumInput)/cnt;
		    
		    // Propagate to the previous layer
		    preLayerGrad[i * featureDim + featIdx] +=
			beta * (inputData[i * featureDim + featIdx] - codeData[t.get<1>()]);
		}
	    }
	    // Method1: average the gradients over time
	    // t.get<0>() = codeData[t.get<1>()] - sumInput;
	    
	    // Method2: sum of the gradients over time 
	    t.get<0>() = (codeData[t.get<1>()] - sumInput) * cnt;
	}
    };

}


namespace layers{

    // Construct the layer
    template <typename TDevice>
    vqLayer<TDevice>::vqLayer(const helpers::JsonValue &layerChild,
			      const helpers::JsonValue &weightsSection,
			      Layer<TDevice> &precedingLayer,
			      int maxSeqLength)
	: TrainableLayer<TDevice>(layerChild, weightsSection, 0,
				  (layerChild->HasMember("vqCodeBookSize") ? 
				   ((*layerChild)["vqCodeBookSize"].GetInt()) : 0),
				  precedingLayer, maxSeqLength)
	, m_vqCodeBookSize ((layerChild->HasMember("vqCodeBookSize") ? 
			     ((*layerChild)["vqCodeBookSize"].GetInt()) : 0))
    {

	// Initial check
	if (precedingLayer.size() != this->size())
	    throw std::runtime_error("vqLayer layer size is different from previous layer");
	if (m_vqCodeBookSize < 1)
	    throw std::runtime_error("vqLayer vqCodeBookSize is not an positive integer");

	// Initialize the distance matrix
	cpu_real_vector temp(this->parallelSequences() * maxSeqLength * m_vqCodeBookSize, 0.0);
	m_disMatrix = temp;

	cpu_int_vector temp2(this->parallelSequences() * maxSeqLength, 0);
	m_selectedIdx = temp2;

	m_betaPara    = (layerChild->HasMember("beta") ? 
			 ((*layerChild)["beta"].GetDouble()) : 0.25);
	
    }	

    // Destructor
    template <typename TDevice>
    vqLayer<TDevice>::~vqLayer()
    {
    }
    
    // Load sequences
    template <typename TDevice>
    void vqLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction,
					 const int nnState)
    {
	TrainableLayer<TDevice>::loadSequences(fraction, nnState);
	if (Configuration::instance().vaeCodeInputDir().size()){
	    if (fraction.numSequences() > 1)
		throw std::runtime_error("Please turn off parallel mode");
	    std::string fileName = Configuration::instance().vaeCodeInputDir() + "/" +
		fraction.seqInfo(0).seqTag + ".bin";
	    cpu_real_vector codeData;
	    int numEle = misFuncs::ReadRealData(fileName, codeData);
	    if (numEle % this->size() != 0)
		throw std::runtime_error("Number of code data is inconsistent with code dim");
	    if (numEle / this->size() != this->curMaxSeqLength())
		throw std::runtime_error("Length of code data is inconsistent with utt length");
	    thrust::copy(codeData.begin(), codeData.begin() + numEle,
			 this->outputs().begin());
	}
    }

    // NN forward
    template <typename TDevice>
    void vqLayer<TDevice>::computeForwardPass(const int nnState)
    {
	
	{{
	    // step1. compute the distance matrix
	    internal::computeDisMatrix fn1;
	    fn1.featureDim   = this->size();
	    fn1.codeBookSize = this->m_vqCodeBookSize;
	    fn1.inputData    = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn1.codeData     = helpers::getRawPointer(this->weights());
	    fn1.patTypes     = helpers::getRawPointer(this->patTypes());

	    int n = this->curMaxSeqLength() * this->parallelSequences() * this->m_vqCodeBookSize;
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->m_disMatrix.begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->m_disMatrix.begin()         + n,
				     thrust::counting_iterator<int>(0) + n)),
	       fn1);
	    
	    // step2. search for the best index
	    internal::getBestIndex fn2;
	    fn2.codeBookSize = this->m_vqCodeBookSize;
	    fn2.disMatrix    = helpers::getRawPointer(this->m_disMatrix);
	    fn2.patTypes     = helpers::getRawPointer(this->patTypes());

	    int m = this->curMaxSeqLength() * this->parallelSequences();
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->m_selectedIdx.begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->m_selectedIdx.begin()       + m,
				     thrust::counting_iterator<int>(0) + m)),
	       fn2);	
        }}

	// step4. optional, calculate the error
	{{
	    internal::CodeDiff fn4;
	    fn4.featureDim   = this->size();
	    fn4.codeBookSize = this->m_vqCodeBookSize;
	    fn4.codeData     = helpers::getRawPointer(this->weights());
	    fn4.index        = helpers::getRawPointer(this->m_selectedIdx);
	    fn4.patTypes     = helpers::getRawPointer(this->patTypes());
	    
	    int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();
	    m_codeError = thrust::transform_reduce(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + n,
				     thrust::counting_iterator<int>(0) + n)),
	       fn4, (real_t)0.0, thrust::plus<real_t>());
	    m_codeError /= this->curMaxSeqLength() * this->parallelSequences();
	}}
	
	// step3. use the best vector as the output
	{{
	    internal::LoadVq fn3;
	    fn3.featureDim   = this->size();
	    fn3.codeBookSize = this->m_vqCodeBookSize;
	    fn3.codeData     = helpers::getRawPointer(this->weights());
	    fn3.index        = helpers::getRawPointer(this->m_selectedIdx);
	    fn3.patTypes     = helpers::getRawPointer(this->patTypes());
	    
	    int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputs().begin()           + n,
				     thrust::counting_iterator<int>(0) + n)),
	       fn3);
	    
	}}

    }

    // NN forward
    template <typename TDevice>
    void vqLayer<TDevice>::computeForwardPass(const int timeStep, const int nnState)
    {
	// To be implemented for inference stage
	if (this->precedingLayer().getSaveMemoryFlag())
	    throw std::runtime_error("The layer before vae is reduced in mem");
	

	if (Configuration::instance().vaeCodeInputDir().size()){
	    // the output (code) has been loaded from output. No need to compute again
	    return;
	}
	
	if (timeStep == 0){

	    static boost::mt19937 *gen = NULL;
            if (!gen) {
                gen = new boost::mt19937;
                gen->seed(Configuration::instance().randomSeed());
            }

	    // Generating standard noise ()
	    int timeLength = this->curMaxSeqLength() * this->parallelSequences();

	    boost::random::uniform_real_distribution<real_t> dist(0, this->m_vqCodeBookSize);
	    cpu_int_vector temp(timeLength);
	    for (int i = 0; i<timeLength; i++)
		temp[i] = (int)dist(*gen);
	    this->m_selectedIdx = temp;

	    // step3. use the best vector as the output
	    {{
		    internal::LoadVq fn3;
		    fn3.featureDim   = this->size();
		    fn3.codeBookSize = this->m_vqCodeBookSize;
		    fn3.codeData     = helpers::getRawPointer(this->weights());
		    fn3.index        = helpers::getRawPointer(this->m_selectedIdx);
		    fn3.patTypes     = helpers::getRawPointer(this->patTypes());
	    
		    int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();
		    thrust::for_each(
			thrust::make_zip_iterator(
		          thrust::make_tuple(this->outputs().begin(),
					     thrust::counting_iterator<int>(0))),
			thrust::make_zip_iterator(
		          thrust::make_tuple(this->outputs().begin()           + n,
					     thrust::counting_iterator<int>(0) + n)),
			fn3);
		}}
	}
    }


    // NN backward
    template <typename TDevice>
    void vqLayer<TDevice>::computeBackwardPass(const int nnState)
    {
	int timeLength = this->parallelSequences() * this->curMaxSeqLength();
	// Copy the gradients
	thrust::copy(this->outputErrors().begin(),
		     this->outputErrors().begin() + timeLength * this->size(),
		     this->precedingLayer().outputErrors().begin());
	
	// Update the codeBook
	{{
	    internal::GradientForCodeBook fn1;
	    fn1.featureDim   = this->size();
	    fn1.codeBookSize = this->m_vqCodeBookSize;
	    
	    fn1.codeData     = helpers::getRawPointer(this->weights());
	    fn1.inputData    = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn1.preLayerGrad = helpers::getRawPointer(this->precedingLayer().outputErrors());
	    fn1.beta         = m_betaPara;
	    fn1.index        = helpers::getRawPointer(this->m_selectedIdx);
	    
	    fn1.timeLength   = timeLength;
	    fn1.patTypes     = helpers::getRawPointer(this->patTypes());
	    
	    int n = this->size() * this->m_vqCodeBookSize;
	    thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->_weightUpdates().begin(),
				     thrust::counting_iterator<int>(0))),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->_weightUpdates().begin()    + n,
				     thrust::counting_iterator<int>(0) + n)),
	       fn1);

	}}
    }
	    
    template <typename TDevice>
    const std::string& vqLayer<TDevice>::type() const
    {
	static std::string s1("vqlayer");
        return s1;
    }

    template <typename TDevice>
    void vqLayer<TDevice>::exportLayer(const helpers::JsonValue &layersArray,
				       const helpers::JsonAllocator &allocator) const
    {
        TrainableLayer<TDevice>::exportLayer(layersArray, allocator);	
	(*layersArray)[layersArray->Size() - 1].AddMember("vqCodeBookSize",
							  m_vqCodeBookSize, allocator);
	(*layersArray)[layersArray->Size() - 1].AddMember("beta",
							  m_betaPara, allocator);
	
    }

    template <typename TDevice>
    real_t vqLayer<TDevice>::codeError() const
    {
	return m_codeError;
    }

    template class vqLayer<Cpu>;
    template class vqLayer<Gpu>;
    
}
