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


#ifdef _MSC_VER
#   pragma warning (disable: 4244) // thrust/iterator/iterator_adaptor.h(121): warning C4244: '+=' : conversion from '__int64' to 'int', possible loss of data
#endif


#include "CNNLayer.hpp"

#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../helpers/min.cuh"
#include "../helpers/max.cuh"
#include "../helpers/safeExp.cuh"
#include "../helpers/JsonClasses.hpp"
#include "../helpers/misFuncs.hpp"

#include "../activation_functions/Tanh.cuh"
#include "../activation_functions/Logistic.cuh"
#include "../activation_functions/Identity.cuh"
#include "../activation_functions/Relu.cuh"

#include "../Configuration.hpp"

#include <boost/foreach.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <fstream>
#include <cmath>
#include <vector>
#include <stdexcept>

#define DEBUG_LOCAL_CNN 1

namespace internal{
namespace{

    typedef activation_functions::Tanh     cell_act_fn_t;

    //
    // dubstin.txt 20170421x01
    
    struct ConvolutionCore
    {

	real_t *dataBuffer;
	real_t *targetBuff;
	real_t *biasWeight;
	
	int    *winSizeCum;
	int    *winHalfSize;
	int    *winTapInter;
	
	int     curLayerSize; 
	int     winTotalLength;
	
	const char *patTypes;
	int   paral;                
	int   maxSeqLength;         // max length of one utterance
	
        __host__ __device__ void operator() (const thrust::tuple<int&, int> &t) const
        {
	    
            // unpack the tuple
            int outputIdx = t.get<1>();

            // calculate the pattern index
            int timeIdx = outputIdx / curLayerSize;   //
	    int dimIdx  = outputIdx % curLayerSize;   // which filter

	    if (patTypes[timeIdx] == PATTYPE_NONE)
		return;

	    int dimS  = winSizeCum[dimIdx];     //
	    int dimE  = winSizeCum[dimIdx+1];   // 
	    int winHS = winHalfSize[dimIdx];    // half window size
	    int inter = winTapInter[dimIdx];    // tap interval
	    
	    // location of the element to be added;
	    int dTmp     = dimS + winHS;
	    int tTmp     = timeIdx;
	    int maxValue = 0;
	    
	    for (int shift = -1 * winHS; shift <= winHS; shift += 1){
		dTmp = (dimS + winHS) + shift;
		tTmp = timeIdx + shift * inter * paral;
		
		if (tTmp < 0                       || tTmp >= (maxSeqLength * paral) ||
		    patTypes[tTmp] == PATTYPE_NONE ||
		    dTmp < dimS                    || dTmp >= dimE)
		    continue;

		// accumulate the feature
		maxValue += dataBuffer[tTmp * winTotalLength + dTmp];
	    }

	    // add bias and pass through the activation function
	    targetBuff[outputIdx] = cell_act_fn_t::fn(maxValue + biasWeight[dimIdx]);
        }
    };


    struct ConvolutionCoreGra
    {

	real_t *dataBuffer;
	real_t *GradBuffer;

	int    *winSizeCum;
	int    *winHalfSize;
	int    *winTapInter;
	
	int     curLayerSize; 
	int     winTotalLength;
	
	const char *patTypes;
	int   paral;                
	int   maxSeqLength;         // max length of one utterance
	
        __host__ __device__ void operator() (const thrust::tuple<int&, int> &t) const
        {
	    
            // unpack the tuple
            int outputIdx = t.get<1>();

            // calculate the pattern index
            int timeIdx = outputIdx / curLayerSize;   //
	    int dimIdx  = outputIdx % curLayerSize;   // which filter

	    if (patTypes[timeIdx] == PATTYPE_NONE)
		return;

	    int dimS  = winSizeCum[dimIdx];     //
	    int dimE  = winSizeCum[dimIdx+1];   // 
	    int winHS = winHalfSize[dimIdx];    // half window size
	    int inter = winTapInter[dimIdx];    // tap interval
	    
	    // location of the element to be added;
	    int dTmp  = dimS + winHS;
	    int tTmp  = timeIdx;
	    
	    for (int shift = -1 * winHS; shift <= winHS; shift += 1){
		dTmp = (dimS + winHS) + shift;
		tTmp = timeIdx + shift * inter * paral;
		
		if (tTmp < 0                       || tTmp >= (maxSeqLength * paral) ||
		    patTypes[tTmp] == PATTYPE_NONE ||
		    dTmp < dimS                    || dTmp >= dimE)
		    continue;
		
		// copy the gradient
		dataBuffer[tTmp * winTotalLength + dTmp] = GradBuffer[outputIdx];
	    }
        }
    };
        
    struct ComputeDeltaFn
    {
        __host__ __device__ void operator() (const thrust::tuple<real_t&, const real_t&> &t) const
        {
            real_t delta = cell_act_fn_t::deriv(t.get<1>()) * t.get<0>();
            t.get<0>() = delta;
        }
    };

    
} // namespace 
} // namespace internal

namespace CNNTools{

    int winWidth(int opt){
	return (opt * 2 + 1);
    }
    
    int winTotalLength(Cpu::int_vector &opt){
	int cnt =0;
	for (int i=0; i < opt.size(); i++){
	    cnt += winWidth(opt[i]);
	}
	return cnt;
    }
    
    int getCNNWeight(const std::string winWidthOpt){
	Cpu::int_vector tmp;
	if (winWidthOpt.size() > 0){
	    ParseIntOpt(winWidthOpt, tmp);
	    return ((int)std::ceil(winTotalLength(tmp)/(float)tmp.size()));
	}else{
	    return 0;
	}
    }

}

namespace layers {
   
    /*****************************************************************************************
     * CNN layer 
     *****************************************************************************************/
    template <typename TDevice>
    CNNLayer<TDevice>::CNNLayer(
        const helpers::JsonValue &layerChild, 
        const helpers::JsonValue &weightsSection,
        Layer<TDevice> &precedingLayer)
	: m_winWidth_Opt    ((layerChild->HasMember("window_width")) ? 
			     ((*layerChild)["window_width"].GetString()) : (""))
	, m_winConRange_Opt ((layerChild->HasMember("window_convo_range")) ? 
			     ((*layerChild)["window_convo_range"].GetString()) : (""))
	, m_winInterval_Opt ((layerChild->HasMember("window_tap_interval")) ? 
			     ((*layerChild)["window_tap_interval"].GetString()) : (""))
	, TrainableLayer<TDevice>  (layerChild, weightsSection,
				    0,
				    (CNNTools::getCNNWeight(
					(layerChild->HasMember("window_width")) ? 
					((*layerChild)["window_width"].GetString()) : ("")) *
				     precedingLayer.size() + 1),
				    precedingLayer)
    {
	
	if (m_winWidth_Opt.size() < 1)
	    throw std::runtime_error("Fail to find window_width in network.jsn");

	// Parse the width of filter window
	m_winWidth_H.clear();
	ParseIntOpt(m_winWidth_Opt, m_winWidth_H);
	m_winWidth_D = m_winWidth_H;

	// total width of filter window
	m_winTotalL  = CNNTools::winTotalLength(m_winWidth_H);
	
	// number of weights (transformation matrices, not including the bias part)
	m_numMatrixW = m_winTotalL * precedingLayer.size(); 

	// parse the convolution range option
	m_winConRange_H.clear();
	if (m_winConRange_Opt.size())
	    ParseIntOpt(m_winConRange_Opt, m_winConRange_H);
	else{
	    m_winConRange_H = m_winWidth_H;
	    thrust::fill(m_winConRange_H.begin(), m_winConRange_H.end(), 1);
	}
	m_winConRange_D = m_winConRange_H;

	// parse the tap interval
	m_winInterval_H.clear();
	if (m_winInterval_Opt.size())
	    ParseIntOpt(m_winInterval_Opt, m_winInterval_H);
	else{
	    m_winInterval_H = m_winWidth_H;
	    thrust::fill(m_winInterval_H.begin(), m_winInterval_H.end(), 1);
	}
	m_winInterval_D = m_winInterval_H;
	
	
	if (m_winConRange_H.size() != m_winWidth_H.size() ||
	    m_winInterval_H.size() != m_winWidth_H.size() ||
	    m_winConRange_H.size() != this->size()        ||
	    m_winInterval_H.size() != this->size())
	    throw std::runtime_error("Incompatible layer size and window configuration in CNN");

	// Buffer to log down the max idx
	m_maxIdxBuffer.resize(this->precedingLayer().outputs().size(), 0);

	// Create index to the first weight cell of each window filter
	Cpu::int_vector tmp(m_winWidth_H.size() + 1, 0);
	Cpu::int_vector tmp2(m_winWidth_H.size() + 1, 0);
	for (int i = 1; i < (m_winWidth_H.size()+1); i++){
	    tmp[i] = tmp[i-1]  + CNNTools::winWidth(m_winWidth_H[i-1]) * precedingLayer.size();
	    tmp2[i]= tmp2[i-1] + CNNTools::winWidth(m_winWidth_H[i-1]);
	}
	m_weightIdx    = tmp;
	m_winWidth_Cum = tmp2;
	
	// allocate memory for convolution buffer (\sum_window_Length * Time)
	m_conBuffer.resize(this->precedingLayer().patTypes().size() * m_winTotalL, 0);

	// done
	printf("\n");
	printf("\t# CNN weights: %d \n", m_numMatrixW + this->size());
	printf("\t# CNN filter width %s, total width %d \n", m_winWidth_Opt.c_str(), m_winTotalL);
	printf("\t# Filter tap interval (1 default) %s \n", m_winInterval_Opt.c_str());
    }
    
    template <typename TDevice>
    CNNLayer<TDevice>::~CNNLayer()
    {
    }
    
    template <typename TDevice>
    void CNNLayer<TDevice>::computeForwardPass()
    {

	// Step1: prepare the data buffer by matrix transformation
	{{
	    helpers::Matrix<TDevice> weightMatrix   (&this->weights(),
						     this->precedingLayer().size(),
						     this->m_winTotalL);
	    
	    helpers::Matrix<TDevice> plOutputsMatrix(&this->precedingLayer().outputs(), 
						     this->precedingLayer().size(), 
						     this->curMaxSeqLength() * 
						     this->parallelSequences());

            helpers::Matrix<TDevice> outputsMatrix  (&this->m_conBuffer,                 
						     this->m_winTotalL,                  
						     this->curMaxSeqLength() * 
						     this->parallelSequences());

            outputsMatrix.assignProduct(weightMatrix, true, plOutputsMatrix, false);
	}}

	// Step2: sum the result
	{{
	    internal::ConvolutionCore fn;
	    	    
	    fn.dataBuffer       = helpers::getRawPointer(this->m_conBuffer);
	    fn.targetBuff       = helpers::getRawPointer(this->outputs());
	    fn.biasWeight       = helpers::getRawPointer(this->weights()) + m_numMatrixW;
	    
	    fn.winSizeCum       = helpers::getRawPointer(m_winWidth_Cum);
	    fn.winHalfSize      = helpers::getRawPointer(m_winWidth_D);
	    fn.winTapInter      = helpers::getRawPointer(m_winInterval_D);
		
	    fn.curLayerSize     = this->size();
	    fn.winTotalLength   = this->m_winTotalL;

	    fn.patTypes         = helpers::getRawPointer(this->patTypes());
	    fn.paral            = this->precedingLayer().parallelSequences();
	    fn.maxSeqLength     = this->curMaxSeqLength();
	    
	    int n =this->precedingLayer().curMaxSeqLength();
	    n = n*this->precedingLayer().parallelSequences();
	    n = n*this->size();

	    thrust::for_each(
	     thrust::make_zip_iterator(
			thrust::make_tuple(m_maxIdxBuffer.begin(),
					   thrust::counting_iterator<int>(0))),
	     thrust::make_zip_iterator(
			thrust::make_tuple(m_maxIdxBuffer.begin()+n, 
					   thrust::counting_iterator<int>(0)+n)),
	     fn);

	}}
	
	// dustbin.txt 20170421x02
    }
    
    template <typename TDevice>
    void CNNLayer<TDevice>::computeForwardPass(const int timeStep)
    {
	// Not implemented
	throw std::runtime_error("Not implemented yet");
    }
    
    template <typename TDevice>
    void CNNLayer<TDevice>::computeBackwardPass()
    {
	// Step1: Pass throught the nonlinear function
	{{
            internal::ComputeDeltaFn fn;
            int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();
            thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputErrors().begin(),   this->outputs().begin())),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputErrors().begin()+n, this->outputs().begin()+n)),
                fn);
	}}

	thrust::fill(this->precedingLayer().outputErrors().begin(),
		     this->precedingLayer().outputErrors().end(), 0.0);	

	// Step2: propagate the gradient
	{{
	    internal::ConvolutionCoreGra fn;
	    	    
	    fn.dataBuffer       = helpers::getRawPointer(this->m_conBuffer);
	    fn.GradBuffer       = helpers::getRawPointer(this->outputErrors());

	    fn.winSizeCum       = helpers::getRawPointer(m_winWidth_Cum);
	    fn.winHalfSize      = helpers::getRawPointer(m_winWidth_D);
	    fn.winTapInter      = helpers::getRawPointer(m_winInterval_D);
	    
	    fn.curLayerSize     = this->size();
	    fn.winTotalLength   = this->m_winTotalL;

	    fn.patTypes         = helpers::getRawPointer(this->patTypes());
	    fn.paral            = this->precedingLayer().parallelSequences();
	    fn.maxSeqLength     = this->curMaxSeqLength();
	    
	    int n =this->precedingLayer().curMaxSeqLength();
	    n = n*this->precedingLayer().parallelSequences();
	    n = n*this->size();

	    thrust::for_each(
	     thrust::make_zip_iterator(
			thrust::make_tuple(m_maxIdxBuffer.begin(),
					   thrust::counting_iterator<int>(0))),
	     thrust::make_zip_iterator(
			thrust::make_tuple(m_maxIdxBuffer.begin()+n, 
					   thrust::counting_iterator<int>(0)+n)),
	     fn);

	}}

	// Step3: gradient to previous layer
	{{
	    helpers::Matrix<TDevice> weightMatrix   (&this->weights(),
						     this->precedingLayer().size(),
						     this->m_winTotalL);

	    helpers::Matrix<TDevice> curErrorMatrix (&this->m_conBuffer,                 
						     this->m_winTotalL,                  
						     this->curMaxSeqLength() * 
						     this->parallelSequences());

            helpers::Matrix<TDevice> preErrorMatrix (&this->precedingLayer().outputErrors(),
						     this->precedingLayer().size(),
						     this->curMaxSeqLength() * 
						     this->parallelSequences());

            preErrorMatrix.assignProduct(weightMatrix, false, curErrorMatrix, false);
	}}

	// Step4: gradient to the weight
	{{
	    helpers::Matrix<TDevice> weightError   (&this->_weightUpdates(),
						     this->precedingLayer().size(),
						     this->m_winTotalL);

	    helpers::Matrix<TDevice> curErrorMatrix (&this->m_conBuffer,                 
						     this->m_winTotalL,                  
						     this->curMaxSeqLength() * 
						     this->parallelSequences());

            helpers::Matrix<TDevice> preOutputMatrix (&this->precedingLayer().outputs(),
						      this->precedingLayer().size(),
						      this->curMaxSeqLength() * 
						      this->parallelSequences());

            weightError.assignProduct(preOutputMatrix, false, curErrorMatrix, true);
	}}

	// Step5: gradient to the bias part
	{{
	    // Borrow the m_conBuffer as one vector [1, 1, 1, 1, 1]
	    thrust::fill(m_conBuffer.begin(),
			 m_conBuffer.begin() + this->curMaxSeqLength() * this->parallelSequences(),
			 1.0);
	    
	    helpers::Matrix<TDevice> biasError   (&this->_weightUpdates(), 1, this->size(),
						  m_numMatrixW);

	    helpers::Matrix<TDevice> curErrorMatrix (&this->outputErrors(),                 
						     this->size(),                  
						     this->curMaxSeqLength() * 
						     this->parallelSequences());

            helpers::Matrix<TDevice> onesVec (&this->m_conBuffer, 1,
					      this->curMaxSeqLength() * this->parallelSequences());

            biasError.assignProduct(onesVec, false, curErrorMatrix, true);
	    
	}}
	
	// dustbin.txt 20170421x03
    }

    template <typename TDevice>
    void CNNLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction)
    {
	// load the sequences for TrainableLayers
	TrainableLayer<TDevice>::loadSequences(fraction);
	
	// 
    }
    
    template <typename TDevice>
    const std::string& CNNLayer<TDevice>::type() const
    {
	static const std::string m("cnn");
	return m;
    }

    template <typename TDevice>
    void CNNLayer<TDevice>::exportLayer(
	const helpers::JsonValue     &layersArray, 
	const helpers::JsonAllocator &allocator) const
    {
        TrainableLayer<TDevice>::exportLayer(layersArray, allocator);
	(*layersArray)[layersArray->Size() - 1].AddMember("window_width",
							  m_winWidth_Opt.c_str(),
							  allocator);
        (*layersArray)[layersArray->Size() - 1].AddMember("window_convo_range",
							  m_winConRange_Opt.c_str(),
							  allocator);
        (*layersArray)[layersArray->Size() - 1].AddMember("window_tap_interval",
							  m_winInterval_Opt.c_str(),
							  allocator);
    }

    template class CNNLayer<Gpu>;
    template class CNNLayer<Cpu>;
}
