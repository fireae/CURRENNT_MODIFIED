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
#   pragma warning (disable: 4244) // thrust/iterator/iterator_adaptor.h(121): warning C4244: '+=' : conversion from '__int64' to 'int', possible loss of data
#endif

#include "FeedForwardLayer.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../activation_functions/Tanh.cuh"
#include "../activation_functions/Logistic.cuh"
#include "../activation_functions/Identity.cuh"
#include "../activation_functions/Relu.cuh"

#include "../helpers/JsonClasses.hpp"
#include "../Configuration.hpp"

#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <typeinfo>

#define BATCHNORM_GAMMA_INITIAL 0.01

namespace internal {
namespace {

    template <typename TActFn>
    struct ComputeOutputFn
    {
        int    layerSize;
        real_t bias;

        const real_t *biasWeights;

        __host__ __device__ real_t operator() (real_t a, const int &outputIdx) const
        {
            // calculate indices
            int blockIdx = outputIdx % layerSize; 

            // add the bias
            a += bias * biasWeights[blockIdx];

            // apply the activation function
            real_t b = TActFn::fn(a);

            // store the activation
            return b;
        }
    };

    template <typename TActFn>
    struct ComputeDeltaFn
    {
        // since calculating the derivatives is very cheap for our activation functions, 
        // we simple calculate the deltas of all timesteps, including dummies
        
        __host__ __device__ void operator() (const thrust::tuple<real_t&, const real_t&> &t) const
        {
            real_t delta = TActFn::deriv(t.get<1>()) * t.get<0>();
            t.get<0>() = delta;
        }
    };
    
    struct ComputeBiasWeightUpdateFn
    {
        int    layerSize;
        int    patternsCount;
        real_t bias;

        const real_t *deltas;
        
        __host__ __device__ real_t operator() (const int &biasWeightIdx) const
        {
            const real_t *offDeltas = deltas + biasWeightIdx;

            real_t wu = 0;
            for (int i = 0; i < patternsCount; ++i) {
                wu += bias * *offDeltas;
                offDeltas += layerSize;
            }

            return wu;
        }
    };

    /*struct GradientAverage
    {
	real_t timeStep;
	real_t *gradients;
        __host__ __device__ void operator() (const int &index) const
        {
	    *(gradients + index) = *(gradients + index)/timeStep;
        }
	};*/

    // 
    struct BatchSize
    {
	// over time t * parallel sentence
	const char *patTypes;
	
	__host__ __device__ real_t operator() (const thrust::tuple<const real_t&, int> &t) const
	{
	    int timeIdx = t.get<1>();
	    if (patTypes[timeIdx] == PATTYPE_NONE)
		return 0.0;// skip dummy node
	    else
		return 1.0;
	}
    };
    
    struct PrepareForMeanStd
    {
	int layerSize;
	bool   meanNotVar;

	
	const char *patTypes;   
	real_t     *data;
	real_t     *outdata;
	real_t     *mean;
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int dataIdx = t.get<1>();
	    int timeIdx = dataIdx / layerSize;
	    int dimIdx  = dataIdx % layerSize;
	    if (patTypes[timeIdx] == PATTYPE_NONE){
		// skip dummy node
		outdata[dataIdx] = 0.0; //
	    }else{
		if (meanNotVar)
		    outdata[dataIdx] = data[dataIdx]; //
		else
		    outdata[dataIdx] = (data[dataIdx]-mean[dimIdx]) * (data[dataIdx]-mean[dimIdx]);
	    }
	}
    };
    struct PrepareGrad
    {
	int    layerSize;
	bool   alphaNotBeta;
	const char *patTypes;   
	real_t     *grad;
	real_t     *data;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int dataIdx = t.get<1>();
	    int timeIdx = dataIdx / layerSize;
	    if (patTypes[timeIdx] == PATTYPE_NONE){
		t.get<0>() = 0.0; // skip dummy node
	    }else{
		if (alphaNotBeta)
		    t.get<0>() = grad[dataIdx] * data[dataIdx];
		else
		    t.get<0>() = grad[dataIdx];
	    }
	}
    };

    struct GetStd
    {
	real_t  stdConst;
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int dimIdx = t.get<1>();
	    t.get<0>() = sqrt(t.get<0>() +stdConst);
	}
    };

    struct AveMeanStd
    {
	real_t *meanStdBuf;
	real_t  cnt;
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int dimIdx = t.get<1>();
	    meanStdBuf[dimIdx] += (t.get<0>() - meanStdBuf[dimIdx]) / cnt;
	}
    };
    
    

    template <typename TActFn>
    struct ComputeBatchNorm_Transform
    {
	int layerSize;

	const char *patTypes;   
	real_t *data;
	real_t *outdata;
	real_t *meanStd;
	real_t *meanStdBuf;
	real_t *scale;
	bool    trainFlag;
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int dataIdx = t.get<1>();
	    int dimIdx  = dataIdx % layerSize;
	    int timeIdx = dataIdx / layerSize;
	    int varIdx  = dimIdx  + layerSize;
	    if (patTypes[timeIdx] == PATTYPE_NONE){
		// skip dummy node
	    }else{
		// \hat{x} = (x - \mu) / \sigma
		if (trainFlag)
		    data[dataIdx] = (data[dataIdx]-meanStd[dimIdx])/meanStd[varIdx];
		else
		    data[dataIdx] = (data[dataIdx]-meanStdBuf[dimIdx])/meanStdBuf[varIdx];

		// y =f(\alpha \hat{x} + \beta)
		outdata[dataIdx]   = TActFn::fn(data[dataIdx] * scale[dimIdx] + scale[varIdx]);
	    }
	}
    };
    
    struct ComputeBatchGradient_output
    {
	
	int layerSize;

	const char *patTypes;   
	real_t *errors;
	real_t *outNormed;
	real_t *meanStd;
	real_t *scale;
	real_t *scaleGrad;	
	real_t  batchSize;
	
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int dataIdx      = t.get<1>();
	    int dimIdx       = dataIdx % layerSize;
	    int timeIdx      = dataIdx / layerSize;
	    
	    if (patTypes[timeIdx] == PATTYPE_NONE){
		errors[dataIdx] = 0.0;
	    }else{
		// gradient =
		// alpha / std * (\deltaE/\delta{y} - \deltaE/\deltaBeta / batchSize -
		//                \deltaE/\deltaAlpha * dataNormed / batchSize)
		errors[dataIdx] = ((errors[dataIdx] -
				    scaleGrad[dimIdx] * outNormed[dataIdx]/ batchSize -
				    scaleGrad[dimIdx + layerSize] / batchSize ) *
				   scale[dimIdx] / meanStd[dimIdx + layerSize]);
	    }
	}
    };
    
} // anonymous namespace
} // namespace internal


namespace layers {

    // Additional weight due to batch normalization
    int weightForBatchNorm(const helpers::JsonValue &layerChild){
	if (layerChild->HasMember("batchnorm") && ((*layerChild)["batchnorm"].GetInt()))
	    return 3; // alpha, mean, std
	else
	    return 0;
    }
    
    template <typename TDevice, typename TActFn>
    FeedForwardLayer<TDevice, TActFn>::FeedForwardLayer(const helpers::JsonValue &layerChild, 
							const helpers::JsonValue &weightsSection, 
							Layer<TDevice> &precedingLayer)
        : TrainableLayer<TDevice>(layerChild, weightsSection, 1, weightForBatchNorm(layerChild),
				  precedingLayer)
    {
	
	// Initialization for batch normalization
	m_batchNorm = (weightForBatchNorm(layerChild)>0)? true : false;
	
	if (m_batchNorm){
	    // initialization
	    m_stdConst  = 0.001; m_batchCnt  = 0.0; m_preEpoch  = 1;
	    
	    // mean, std
	    Cpu::real_vector tmp;
	    tmp.resize(this->size() * 2, 0.0); 
	    m_stats     = tmp;

	    // all-one vector for vector summation
	    tmp.resize(this->outputs().size()/this->size(), 1.0);
	    m_oneVector = tmp;

	    // a tempopary buff
	    m_buff      = this->outputs();
	    m_outNormed = this->outputs();
		    
	    if (weightsSection.isValid() && weightsSection->HasMember(this->name().c_str())) {
		// read 
	    }else{
		// initialize 
		int transMatrixWeightNum = this->size() * this->precedingLayer().size();
		
		// alpha = 1.0
		thrust::fill(this->weights().begin() + transMatrixWeightNum,
			     this->weights().begin() + transMatrixWeightNum + this->size(),
			     BATCHNORM_GAMMA_INITIAL);
		
		// beta, mean, std
		thrust::fill(this->weights().begin() + transMatrixWeightNum + this->size(),
			     this->weights().end(),
			     0.0);
	    }
	    //const Configuration &config = Configuration::instance();
	    //m_trainFlag = config.trainingMode();
	}
	
    }

    template <typename TDevice, typename TActFn>
    FeedForwardLayer<TDevice, TActFn>::~FeedForwardLayer()
    {
    }
    
    template <typename TDevice, typename TActFn>
    const std::string& FeedForwardLayer<TDevice, TActFn>::type() const
    {
        static std::string s;
        if (s.empty()) {
            if (typeid(TActFn) == typeid(activation_functions::Tanh))
                s = "feedforward_tanh";
            else if (typeid(TActFn) == typeid(activation_functions::Logistic))
                s = "feedforward_logistic";
            else if (typeid(TActFn) == typeid(activation_functions::Identity))
                s = "feedforward_identity";
	    else if (typeid(TActFn) == typeid(activation_functions::Relu))
		s = "feedforward_relu";
            else
                throw std::runtime_error("Unsupported activation function");
        }    
        return s;
    }

    template <typename TDevice, typename TActFn>
    void FeedForwardLayer<TDevice, TActFn>::computeForwardPass(const int nnState)
    {

	// Fine, I am lazy to merge the code
	if (!m_batchNorm){
	    
	    // The conventional feedforward part
	    // collect outputs from preceding layer
	    {{
            helpers::Matrix<TDevice> weightsMatrix  (&this->weights(),                  
						     this->precedingLayer().size(), this->size());
	    
	    
            helpers::Matrix<TDevice> plOutputsMatrix(&this->precedingLayer().outputs(), 
						     this->precedingLayer().size(), 
						     this->curMaxSeqLength() * 
						     this->parallelSequences());

            helpers::Matrix<TDevice> outputsMatrix  (&this->_outputs(),                 
						     this->size(),                  
						     this->curMaxSeqLength() * 
						     this->parallelSequences());

            outputsMatrix.assignProduct(weightsMatrix, true, plOutputsMatrix, false);
	    }}
	
	    // calculate the outputs of the layer
	    {{
            internal::ComputeOutputFn<TActFn> fn;
            fn.layerSize        = this->size();
            fn.bias             = this->bias();
            fn.biasWeights      = (helpers::getRawPointer(this->weights()) + 
				   this->size() * this->precedingLayer().size());

            thrust::transform(
                this->_outputs().begin(),
                (this->_outputs().begin() + 
		 this->curMaxSeqLength() * this->parallelSequences() * this->size()),
                thrust::counting_iterator<int>(0),
                this->_outputs().begin(),
                fn
                );
	   }}
	    
	}else{
	    // if batch normalization is used
	    int transMatrixWeightNum = this->size() * this->precedingLayer().size();
	    
	    // Re-initialize the batch mean and variance
	    if (this->flagTrainingMode() && m_preEpoch > 0 &&
		m_preEpoch != this->getCurrTrainingEpoch()){
		// always update the mean, std for each epoch
		m_batchCnt = 0;
		thrust::fill(this->weights().begin() + transMatrixWeightNum + 2 * this->size(),
			     this->weights().end(),  0.0);
		m_preEpoch = this->getCurrTrainingEpoch();
	    }

	    // Wx
	    {{
		helpers::Matrix<TDevice> weightsMatrix  (&this->weights(),                  
							 this->precedingLayer().size(),
							 this->size());
		helpers::Matrix<TDevice> plOutputsMatrix(&this->precedingLayer().outputs(), 
							 this->precedingLayer().size(), 
							 this->curMaxSeqLength() * 
							 this->parallelSequences());
		helpers::Matrix<TDevice> outputsMatrix  (&this->m_outNormed,                 
							 this->size(),                  
							 this->curMaxSeqLength() * 
							 this->parallelSequences());
		outputsMatrix.assignProduct(weightsMatrix, true, plOutputsMatrix, false);
	    }}	    

	    // normalize the data
	    m_batchCnt++;
	    {{
	       int maxFrameNum = this->curMaxSeqLength() * this->parallelSequences();
	       int maxDataNum  = maxFrameNum * this->size();
	       
	       // Step1. calculate the batch size
	       //        For parallel sentences, there is dummy node. BatchSize should not count it.
	       internal::BatchSize fn0;
	       fn0.patTypes = helpers::getRawPointer(this->patTypes());
	       m_batchSize  = thrust::transform_reduce(
				thrust::make_zip_iterator(
				    thrust::make_tuple(
					this->m_buff.begin(), 
					thrust::counting_iterator<int>(0))),
				thrust::make_zip_iterator(
				    thrust::make_tuple(
					this->m_buff.begin()              + maxFrameNum, 
					thrust::counting_iterator<int>(0) + maxFrameNum)),
				fn0, (real_t)0.0, thrust::plus<real_t>());
	       
	       thrust::fill(this->m_oneVector.begin(), this->m_oneVector.end(), 1.0/m_batchSize);
	       
	       // Step2. accumulate the mean
	       internal::PrepareForMeanStd fn1;
	       fn1.layerSize  = this->size();
	       fn1.meanNotVar = true;
	       fn1.mean       = NULL;
	       fn1.patTypes   = helpers::getRawPointer(this->patTypes());
	       fn1.data       = helpers::getRawPointer(this->m_outNormed);
	       fn1.outdata    = helpers::getRawPointer(this->m_buff);	   
	       thrust::for_each(
		 thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin(), 
					   thrust::counting_iterator<int>(0))),
		 thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin()           + maxDataNum, 
					   thrust::counting_iterator<int>(0) + maxDataNum)),
		 fn1);
	   
	       helpers::Matrix<TDevice> onevec  (&this->m_oneVector, maxFrameNum,  1);
	       helpers::Matrix<TDevice> data    (&this->m_buff,      this->size(), maxFrameNum);
	       helpers::Matrix<TDevice> meanVec (&this->m_stats,     this->size(), 1);
	       meanVec.assignProduct(data, false, onevec, false);

	       // Step3. accumulate the var
	       fn1.meanNotVar = false;
	       fn1.mean       = helpers::getRawPointer(this->m_stats);; 
	       thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin()           + maxDataNum, 
					   thrust::counting_iterator<int>(0) + maxDataNum)),
		fn1);
	       
	       helpers::Matrix<TDevice> data2   (&this->m_buff,  this->size(), maxFrameNum);
	       helpers::Matrix<TDevice> stdVec  (&this->m_stats, this->size(), 1, this->size());
	       stdVec.assignProduct(data2, false, onevec, false);
	       
	       internal::GetStd fn3;
	       fn3.stdConst = m_stdConst;
	       thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(m_stats.begin() + this->size(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(m_stats.begin() + this->size() * 2, 
					   thrust::counting_iterator<int>(0) + this->size())),
		fn3);

	       // Step4. accumulate the mean and std, for generation stage
	       if (this->flagTrainingMode()){
		   internal::AveMeanStd fn5;
		   fn5.meanStdBuf = (helpers::getRawPointer(this->weights()) +
				     transMatrixWeightNum + this->size() * 2);
		   fn5.cnt        = m_batchCnt;
		   thrust::for_each(
		     thrust::make_zip_iterator(
			thrust::make_tuple(m_stats.begin(), 
					   thrust::counting_iterator<int>(0))),
		     thrust::make_zip_iterator(
			thrust::make_tuple(m_stats.begin() + this->size() * 2, 
					   thrust::counting_iterator<int>(0) + this->size() * 2)),
		     fn5);
	       }
	   
	       // Step5: normalize and scale the data
	       internal::ComputeBatchNorm_Transform<TActFn> fn2;
	       fn2.layerSize = this->size();
	       fn2.patTypes  = helpers::getRawPointer(this->patTypes());
	       fn2.data      = helpers::getRawPointer(this->m_outNormed);
	       fn2.outdata   = helpers::getRawPointer(this->outputs());
	       fn2.scale     = helpers::getRawPointer(this->weights()) + transMatrixWeightNum;
	       fn2.meanStd   = helpers::getRawPointer(this->m_stats);
	       fn2.meanStdBuf= (helpers::getRawPointer(this->weights()) +
				transMatrixWeightNum + this->size() * 2);
	       fn2.trainFlag = this->flagTrainingMode();
	   
	       thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(this->outputs().begin()           + maxDataNum, 
					   thrust::counting_iterator<int>(0) + maxDataNum)),
		fn2);

	    }}
	}
	
	// done
    }


    template <typename TDevice, typename TActFn>
    void FeedForwardLayer<TDevice, TActFn>::computeForwardPass(const int timeStep,
							       const int nnState)
    {
	if (m_batchNorm){
	    throw std::runtime_error("Error: batchnorm not available for online processing");
	}

	int effTimeStart = timeStep * this->parallelSequences();
	int effTimeEnd   = (timeStep+1) * this->parallelSequences();
	
	// Pointer to the output of previous layer (input buffer)
	int shiftIn  = this->precedingLayer().outputBufPtrBias(timeStep * this->parallelSequences(),
							       nnState);
	// Pointer to the output of this layer
	int shiftOut = this->outputBufPtrBias(timeStep * this->parallelSequences(), nnState);
	
	// collect outputs from preceding layer
        {{
            helpers::Matrix<TDevice> weightsMatrix  (&this->weights(),                  
						     this->precedingLayer().size(), this->size());
	    
	    
            helpers::Matrix<TDevice> plOutputsMatrix(&this->precedingLayer().outputs(), 
						     this->precedingLayer().size(), 
						     this->parallelSequences(),
						     (effTimeStart * this->precedingLayer().size()
						      - shiftIn));

            helpers::Matrix<TDevice> outputsMatrix  (&this->_outputs(),                 
						     this->size(), 
						     this->parallelSequences(),
						     (effTimeStart * this->size()
						      - shiftOut));

            outputsMatrix.assignProduct(weightsMatrix, true, plOutputsMatrix, false);
        }}

        // calculate the outputs of the layer
        {{
            internal::ComputeOutputFn<TActFn> fn;
            fn.layerSize        = this->size();
            fn.bias             = this->bias();
            fn.biasWeights      = (helpers::getRawPointer(this->weights()) + 
				   this->size() * this->precedingLayer().size());

            thrust::transform(
		this->_outputs().begin() + effTimeStart * this->size() - shiftOut,
		this->_outputs().begin() + effTimeEnd   * this->size() - shiftOut,
		thrust::counting_iterator<int>(0),
		this->_outputs().begin() + effTimeStart * this->size() - shiftOut,
		fn);
        }}
    }

    template <typename TDevice, typename TActFn>
    void FeedForwardLayer<TDevice, TActFn>::computeBackwardPass(const int nnState)
    {
	// compute deltas
	{{
            internal::ComputeDeltaFn<TActFn> fn;

            int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

            thrust::for_each(
               thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputErrors().begin(),   this->outputs().begin())),
	       thrust::make_zip_iterator(
		  thrust::make_tuple(this->outputErrors().begin()+n, this->outputs().begin()+n)),
                fn);
	}}

	if (m_batchNorm) {
	    // for batch normalization
	    int maxFrameNum          = this->curMaxSeqLength() * this->parallelSequences();
	    int maxDataNum           = maxFrameNum * this->size();
	    int transMatrixWeightNum = this->size() * this->precedingLayer().size();

	    thrust::fill(m_oneVector.begin(),            m_oneVector.end(),            1.0);
	    thrust::fill(m_buff.begin(),                 m_buff.end(),                 0.0);
	    thrust::fill(this->_weightUpdates().begin(), this->_weightUpdates().end(), 0.0);
	    
	    
	    // Step1. Calculate \deltaE/\delta{\alpha}
	    internal::PrepareGrad fn1;
	    fn1.layerSize    = this->size();
	    fn1.alphaNotBeta = true;
	    fn1.patTypes     = helpers::getRawPointer(this->patTypes());
	    fn1.grad         = helpers::getRawPointer(this->outputErrors());
	    fn1.data         = helpers::getRawPointer(this->m_outNormed);
	    
	    thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(this->m_buff.begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(this->m_buff.begin() + maxDataNum, 
					   thrust::counting_iterator<int>(0) + maxDataNum)),
		fn1);
	   
	    helpers::Matrix<TDevice> onevec    (&this->m_oneVector, maxFrameNum, 1);
	    helpers::Matrix<TDevice> data      (&this->m_buff,      this->size(), maxFrameNum);
	    helpers::Matrix<TDevice> gradAlpha (&this->_weightUpdates(), this->size(), 1,
						transMatrixWeightNum);
	   gradAlpha.assignProduct(data, false, onevec, false);

	   // Step2. Calculate \deltaE/\delta{\beta}
	   fn1.alphaNotBeta = false;	   
	   thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(this->m_buff.begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(this->m_buff.begin() + maxDataNum, 
					   thrust::counting_iterator<int>(0) + maxDataNum)),
		fn1);
	   
	   helpers::Matrix<TDevice> gradBeta (&this->_weightUpdates(), this->size(),1,
					      transMatrixWeightNum + this->size());
	   gradBeta.assignProduct(data, false, onevec, false);
	   

	   // Step3. Calculate \deltaE/\delta{x}
	   internal::ComputeBatchGradient_output fn2;
	   fn2.layerSize = this->size();
	   fn2.patTypes  = helpers::getRawPointer(this->patTypes());
	   fn2.errors    = helpers::getRawPointer(this->outputErrors());
	   fn2.outNormed = helpers::getRawPointer(m_outNormed);
	   fn2.meanStd   = helpers::getRawPointer(m_stats);
	   fn2.scale     = helpers::getRawPointer(this->weights())        + transMatrixWeightNum;
	   fn2.scaleGrad = helpers::getRawPointer(this->_weightUpdates()) + transMatrixWeightNum;
	   fn2.batchSize = m_batchSize;
	   
	   thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(m_outNormed.begin(), 
					   thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(
			thrust::make_tuple(m_outNormed.begin() + maxDataNum, 
					   thrust::counting_iterator<int>(0) + maxDataNum)),
		fn2);

	}

	
	// back-propagate the error to the preceding layer
	{{

	    // why only to Trainablelayer?
            TrainableLayer<TDevice> *pl = 
		dynamic_cast<TrainableLayer<TDevice>*>(&this->precedingLayer());
	    
	    if (pl) {
                helpers::Matrix<TDevice> weightsMatrix (&this->weights(),      
							pl->size(),   
							this->size());
                helpers::Matrix<TDevice> plErrorsMatrix(&pl->outputErrors(),   
							pl->size(),   
							this->curMaxSeqLength() * 
							this->parallelSequences());
                helpers::Matrix<TDevice> deltasMatrix  (&this->outputErrors(), 
							this->size(), 
							this->curMaxSeqLength() * 
							this->parallelSequences());
		
                plErrorsMatrix.assignProduct(weightsMatrix, false, deltasMatrix, false);
            }else{
		/* Add 16-02-22 Wang: for WE updating */
		// If the input layer will udpate the word vectors
		// we need to propagate the errors back to the input layer
		/* Add 17-05-01 for MiddleoutputLayer*/
		Layer<TDevice> *pl2 = dynamic_cast<Layer<TDevice>*>(&this->precedingLayer());
		if (this->precedingLayer().inputWeUpdate() ||
		    this->precedingLayer().type() == "middleoutput" ||
		    this->precedingLayer().type() == "featmatch"){
		    helpers::Matrix<TDevice> weightsMatrix (&this->weights(),      
							    pl2->size(),  
							    this->size());

		    helpers::Matrix<TDevice> plErrorsMatrix(&pl2->outputErrors(),  
							    pl2->size(),  
							    this->curMaxSeqLength() * 
							    this->parallelSequences());

		    helpers::Matrix<TDevice> deltasMatrix  (&this->outputErrors(), 
							    this->size(), 
							    this->curMaxSeqLength() * 
							    this->parallelSequences());
		    plErrorsMatrix.assignProduct(weightsMatrix, false, deltasMatrix, false);
		}
	    }
	}}

	// compute the input weight updates
	{{
            helpers::Matrix<TDevice> weightUpdatesMatrix(&this->_weightUpdates(),           
							 this->precedingLayer().size(), 
							 this->size());

            helpers::Matrix<TDevice> plOutputsMatrix    (&this->precedingLayer().outputs(), 
							 this->precedingLayer().size(), 
							 this->curMaxSeqLength() * 
							 this->parallelSequences());
	    
            helpers::Matrix<TDevice> deltasMatrix       (&this->outputErrors(),             
							 this->size(),                  
							 this->curMaxSeqLength() * 
							 this->parallelSequences());

            weightUpdatesMatrix.assignProduct(plOutputsMatrix, false, deltasMatrix, true);
	}}

	if (!m_batchNorm){
	    // compute the bias weight updates
	    {{
            internal::ComputeBiasWeightUpdateFn fn;
            fn.layerSize     = this->size();
            fn.patternsCount = this->curMaxSeqLength() * this->parallelSequences();
            fn.bias          = this->bias();
            fn.deltas        = helpers::getRawPointer(this->outputErrors());

            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(0) + this->size(),
                this->_weightUpdates().begin() + this->precedingLayer().size() * this->size(),
                fn
                );
	    }}
	}
	/* Gradient averaging ?
	      if (this->_optOpt()){
	      {{
	      internal::GradientAverage fn;
	      fn.timeStep = (real_t)(this->curMaxSeqLength() * this->parallelSequences());
	      fn.gradients= helpers::getRawPointer(this->_weightUpdates());		
	      thrust::for_each(
	      thrust::counting_iterator<int>(0),
	      thrust::counting_iterator<int>(0) + this->_weightUpdates().size(),
	      fn);
	      }}
	      }*/
    }

    template <typename TDevice, typename TActFn>
    void FeedForwardLayer<TDevice, TActFn>::exportLayer(
	const helpers::JsonValue     &layersArray, 
	const helpers::JsonAllocator &allocator) const
    {
        TrainableLayer<TDevice>::exportLayer(layersArray, allocator);
        (*layersArray)[layersArray->Size() - 1].AddMember("batchnorm", (int)m_batchNorm, allocator);
    }


    template <typename TDevice, typename TActFn>
    void FeedForwardLayer<TDevice, TActFn>::reduceOutputBuffer()
    {
	// only for no-batch-normalized module
	if (m_batchNorm){
	    // do thing
	}else{
	    this->resizeOutputBuffer(this->parallelSequences() * this->size());
	    this->setSaveMemoryFlag(true);
	    printf("\t[mem saved]");
	}
    }
    
    template <typename TDevice, typename TActFn>
    int FeedForwardLayer<TDevice, TActFn>::outputBufPtrBias(const int timeStepTimesParallel,
							    const int nnState)
    {
	if (this->getSaveMemoryFlag()){
	    return timeStepTimesParallel * this->size();
	}else{
	    return 0;
	}
    }	

    // explicit template instantiations
    template class FeedForwardLayer<Cpu, activation_functions::Tanh>;
    template class FeedForwardLayer<Gpu, activation_functions::Tanh>;
    template class FeedForwardLayer<Cpu, activation_functions::Logistic>;
    template class FeedForwardLayer<Gpu, activation_functions::Logistic>;
    template class FeedForwardLayer<Cpu, activation_functions::Identity>;
    template class FeedForwardLayer<Gpu, activation_functions::Identity>;
    template class FeedForwardLayer<Cpu, activation_functions::Relu>;
    template class FeedForwardLayer<Gpu, activation_functions::Relu>;

} // namespace layers
