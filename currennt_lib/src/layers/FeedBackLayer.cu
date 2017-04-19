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

#include "FeedBackLayer.hpp"

#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../helpers/JsonClasses.hpp"
#include "../activation_functions/Logistic.cuh"
#include "../activation_functions/Tanh.cuh"

#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/fill.h>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <vector>
#include <stdexcept>

#include "../Configuration.hpp"

#define FEEDBACKLAYER_DEBUG 0

namespace internal{
namespace {

    typedef activation_functions::Tanh     cell_act_fn_t;
    
    // dustbin.txt/Block1226x02
    
    struct vectorFillForward
    {
	// Copy the output of preceding layer to the output of this layer
	// Copy the output of target layer to the output of this layer

	int dimInput1;      // dimension of output of preceding layer
	int dimInput2;      // dimension of output of target layer (to be fed back, in total dim)
	int dimInput2Start; // from which dimension of the target to load (may not be 0)
	
	int dimOutput;      // dimension of output of this layer
	int parallel;       // number of parallel sentences

	int dim1Step;
	
	real_t *input1;     // preceding layer
	real_t *input2;     // target layer
	real_t *output;     // this layer

	int    *lookBack;   // lookback step
	int     lookBackStepNM; // how many steps to look back ?
	int     crossBoundary;
	// dispatched over Dim * T * Parallel
	__host__ __device__ void operator() (const thrust::tuple<const real_t&, int> &t)
	{
	    int outputEffIdx = t.get<1>();
	    int timeStep     = outputEffIdx / dimOutput;
	    int dimIdx       = outputEffIdx % dimOutput;

	    // Idx in the output of this layer
	    int outputIdx    = timeStep * dimOutput + dimIdx;
	    int lookBackTime = 0;

	    if (dimIdx < (dimInput1 + lookBackStepNM * dim1Step)){
		if (dimIdx >= dimInput1){
		    // copy from the target layer (feedback part)
		    
		    // get the dimension index (across multiple time steps)
		    dimIdx       = dimIdx - dimInput1;
		    
		    // get the time shift to be looked backwards
		    if (lookBack != NULL)
			lookBackTime = lookBack[dimIdx / dim1Step] * parallel;
		    else
			lookBackTime = 1;
		    
		    // get the dimension index in each time step
		    dimIdx       = dimIdx % dim1Step;
		    
		    if (timeStep < lookBackTime)      // loopback one step
			output[outputIdx] = 0.0;
		    else{
			output[outputIdx] = input2[(timeStep - lookBackTime) * dimInput2 +
						   dimIdx + dimInput2Start];
			
			// crossBoundary should be deleted
			if (crossBoundary == 3 &&
			    input2[(timeStep - lookBackTime) * dimInput2 + dimInput2Start] > 0.98){
			    output[outputIdx] = 0.0;
			    // Set the feedback to zero if previous frame is silence
			}
			//
		    }
		    
		}else{
		    //output[outputIdx] = 0;
		    output[outputIdx] = input1[timeStep * dimInput1 + dimIdx];
		}
	    }else{
		// this is section for aggregating information
	    }
	}
    };
    

    struct vectorAggregateForward
    {
	// Copy the output of preceding layer to the output of this layer
	// Copy the output of target layer to the output of this layer

	int dimInput2;      // dimension of output of target layer (to be fed back, in total dim)
	int dimInput2Start; // from which dimension of the target to load (may not be 0)
	
	int dimOutput;      // dimension of output of this layer
	int dimOutputStart;
	int dim1Band;
	int bandNum;
	
	real_t *input2;     // target layer
	real_t *output;     // this layer

	char   *boundaryInfo;
	int     startTime;
	int     endTime;

	real_t *aggBuffer;
	
	int   crossBoundary; // deliver the aggregation across the boundary

	// dispatched over Dim * Band
	__host__ __device__ void operator() (const thrust::tuple<const real_t&, int> &t)
	{
	    int     dimIdxRel  = t.get<1>();                             // relative dimension
	    
	    int     outputIdx  = dimOutputStart + dimIdxRel;           
	    int     inputIdx   = dimInput2Start + dimIdxRel % dim1Band;
	    
	    int     bandIdx    = dimIdxRel / dim1Band;           // which band this dimension is in?
	    int     boundTime  = 0;
	    real_t  aggreInfo  = 0.0;
	    
	    
	    for (int time = startTime ; time < endTime; time++){
		
		if (crossBoundary){
		    // deliver the aggregation across boundary
		    output[outputIdx]= aggreInfo;
		    if (boundaryInfo[time * bandNum + bandIdx] < 1 || time < 1){
			aggreInfo    = 0.0;  
			boundTime    = time; 
		    }
		}else{
		    // not deliver across boundary
		    if (boundaryInfo[time * bandNum + bandIdx] < 1 || time < 1){
			aggreInfo    = 0.0;  
			boundTime    = time; 
		    }
		    output[outputIdx]= aggreInfo;
		}

		if (crossBoundary == 3 && (inputIdx - dimInput2)>0 &&
		    input2[inputIdx - dimInput2 - dimIdxRel % dim1Band] > 0.98){
		    output[outputIdx] = 0;
		    // set the previous frame to zero if it is silence
		}
		
		if (crossBoundary == 3 && input2[inputIdx - dimIdxRel % dim1Band] > 0.98){
		    // don't aggregate this frame
		}else{
		    // aggregating information using tanh and moving average
		    aggreInfo  = (((time - boundTime) / ((time - boundTime)+1.0)) * aggreInfo +
				  cell_act_fn_t::fn(input2[inputIdx]) / ((time-boundTime)+1.0));
		}
		outputIdx += dimOutput;
		inputIdx  += dimInput2;
	    }
	}
    };


    struct vectorAggregateForwardInfer
    {
	int dimInput2;      // dimension of output of target layer (to be fed back, in total dim)
	int dimInput2Start; // from which dimension of the target to load (may not be 0)
	
	int dimOutput;      // dimension of output of this layer
	int dimOutputStart;
	int dim1Band;
	int bandNum;
	
	real_t *input2;     // target layer
	real_t *output;     // this layer

	char   *boundaryInfo;
	int     startTime;
	int     endTime;

	real_t *aggBuffer;
	
	int   crossBoundary; // deliver the aggregation across the boundary

	// dispatched over Dim * Band
	__host__ __device__ void operator() (const thrust::tuple<const real_t&, int> &t)
	{
	    int     dimIdxRel  = t.get<1>();                             // relative dimension
	    
	    /*********************** FATAL ERROR ***********************************
	       Fatal Error: for inferrence stage, outputIdx and inputIdx should point
	       to current time step:
	       for (int time = 0; time < startTime; time++)
	           outputIdx += dimOutput
                   inputIdx  += dimInput2
	    ************************************************************************/
	    // Error: 
	    //int     outputIdx  = dimOutputStart + dimIdxRel;           
	    //int     inputIdx   = dimInput2Start + dimIdxRel % dim1Band;
	    // Modified
	    int     outputIdx = startTime * dimOutput + dimOutputStart + dimIdxRel;
	    int     inputIdx  = startTime * dimInput2 + dimInput2Start + dimIdxRel % dim1Band; 
	    
	    int     bandIdx   = dimIdxRel / dim1Band;      // which band this dimension is in?	    
	    int     preTime   = 0;

	    // after the first frame
	    // take the accumulation the frames before the previous frame
	    real_t  aggreInfo = aggBuffer[dimIdxRel];      
	    int     boundTime = aggBuffer[bandNum * dim1Band + dimIdxRel];
	    
	    
	    for (int time = startTime ; time < endTime; time++){
		
		preTime = time - 1;
		if (preTime < 0){
		    // the first frame
		    output[outputIdx]     = 0.0;
		    aggBuffer[dimIdxRel]  = 0.0;
		    aggBuffer[bandNum * dim1Band + dimIdxRel] = time;
		    
		}else{
		    
		    // aggregating the previous frame
		    if (crossBoundary == 3 &&
			input2[inputIdx - dimInput2 - dimIdxRel % dim1Band] > 0.98){
			
			
		    }else{
			aggreInfo  = ((preTime-boundTime) / (preTime - boundTime + 1.0)) *
			    aggreInfo +
			    cell_act_fn_t::fn(input2[inputIdx - dimInput2]) /
			    (preTime - boundTime+1.0);
		    }
		    
		    // propagate the info to the current frame
		    if (crossBoundary == 3 &&
			input2[inputIdx - dimInput2 - dimIdxRel % dim1Band] > 0.98){
			output[outputIdx]= 0;
			
		    }else if (crossBoundary == 1){
			// deliver the aggregation across boundary
			output[outputIdx]= aggreInfo;
			if (boundaryInfo[time * bandNum + bandIdx] < 1){
			    aggreInfo    = 0.0;  
			    boundTime    = time; 
			}
		    }else{
			// not deliver across boundary
			if (boundaryInfo[time * bandNum + bandIdx] < 1){
			    aggreInfo    = 0.0;  
			    boundTime    = time; 
			}
			output[outputIdx]= aggreInfo;
		    }
		    // save the aggregation information for next time (during generation)
		    aggBuffer[dimIdxRel] = aggreInfo;
		    aggBuffer[bandNum * dim1Band + dimIdxRel] = boundTime; 
		}
		outputIdx += dimOutput;
		inputIdx  += dimInput2;
	    }
	}
    };

    
    struct vectorFillBackward
    {
	int dimInput1;      // dimension of the preceding layer
	int dimOutput;      // dimension of this layer
	
	real_t *outputError;
	
	// dispatched over Dim * T * Parallel
	// Dim here is the dimension of the previous layer
	__host__ __device__ real_t operator() (const int &outputIdx) const
	{
	    int timeStep  = outputIdx / dimInput1;
	    int dimIdx    = outputIdx % dimInput1;
	    return outputError[timeStep * dimOutput + dimIdx];
	}
    };
    
}
}

namespace layers{

    // dustbin.txt/Block 1226x01
    int ParseLayerOpt(const std::string options){
	std::vector<std::string> tempArgs;
	boost::split(tempArgs, options, boost::is_any_of("_"));
	return boost::lexical_cast<int>(tempArgs[0]);
    }

    void ParseLookBackStep(const std::string options, Cpu::int_vector &optVec){
	std::vector<std::string> tempArgs;
	boost::split(tempArgs, options, boost::is_any_of("_"));
	optVec.resize(tempArgs.size(), 0);
	for (int i =0 ; i<tempArgs.size(); i++)
	    optVec[i] = boost::lexical_cast<int>(tempArgs[i]);
    }

    void ConvertBoundaryInfo(Cpu::pattype_vector &boundary, Cpu::pattype_vector &distance,
			     Cpu::int_vector & aggOpt, const int curMaxLength)
    {
	// The boundary information logs the distance of this frame to the previous boundary
	// ex. 0 1 2 3 4 .. 10 0 1 2 .. 32 0
	std::vector<int> outTemp(aggOpt.size(), 0);
	for (int time = 0; time < curMaxLength; time++){
	    for (int band = 0; band < aggOpt.size(); band++){
		if (boundary[time] & (0b01 << aggOpt[band]))
		    outTemp[band] = 0;
		else
		    outTemp[band] = outTemp[band] + 1;
		distance[time * aggOpt.size() + band] = outTemp[band];
	    }
	}
    }
    
    template <typename TDevice>
    FeedBackLayer<TDevice>::FeedBackLayer(const helpers::JsonValue &layerChild,
					  const helpers::JsonValue &weightsSection,
					  Layer<TDevice>           &precedingLayer
					  )
	: TrainableLayer<TDevice>(layerChild, weightsSection, 0, 0, precedingLayer)
	, m_targetDim   (-1)
	, m_targetLayer (NULL)
    {
	m_targetBuffer.clear();
	
	const Configuration &config = Configuration::instance();
	
	// get ClockRNN state
	m_lookBackStr = ((layerChild->HasMember("lookback")) ? 
			 ((*layerChild)["lookback"].GetString()) : (""));
	if (m_lookBackStr.size()){
	    if (m_lookBackStr.size()==1 && m_lookBackStr[0] == '0'){
		// special case where lookback is not used
		m_lookBack.clear();
	    }else{
		// when lookback is explicitly specified
		cpu_int_vector tempOpt;
		ParseLookBackStep(m_lookBackStr, tempOpt);
		m_lookBack = tempOpt;
	    }
	}else{
	    // default only look back 1 step
	    m_lookBack.resize(1,1); 
	}

	// get aggregation information
	m_aggStr         = ((layerChild->HasMember("aggregate")) ? 
			    ((*layerChild)["aggregate"].GetString()) : (""));
	m_crossBoundary  = (layerChild->HasMember("aggregate_cross_boundary") ? 
			    (*layerChild)["aggregate_cross_boundary"].GetInt() : 0);

	if (m_aggStr.size()){
	    // configuratio for F0 aggregation
	    cpu_int_vector tempOpt;
	    ParseLookBackStep(m_aggStr, tempOpt);
	    m_aggOpt = tempOpt;
	    m_boundaryInfo.resize(m_aggOpt.size() * precedingLayer.maxSeqLength(), 0);
	    m_aggOptSyn      = config.aggregateOpt();
	}else{
	    // default, don't use aggregate
	    m_aggOpt.clear(); 
	}
    }

    template <typename TDevice>
    FeedBackLayer<TDevice>::~FeedBackLayer()
    {
    }

    template <typename TDevice>
    void FeedBackLayer<TDevice>::exportLayer(const helpers::JsonValue     &layersArray, 
					     const helpers::JsonAllocator &allocator) const
    {
        TrainableLayer<TDevice>::exportLayer(layersArray, allocator);
        (*layersArray)[layersArray->Size() - 1].AddMember("lookback",  m_lookBackStr.c_str(),
							  allocator);
        (*layersArray)[layersArray->Size() - 1].AddMember("aggregate", m_aggStr.c_str(),
							  allocator);
        (*layersArray)[layersArray->Size() - 1].AddMember("aggregate_cross_boundary", 
							  m_crossBoundary,
							  allocator);
    }

    template <typename TDevice>
    void FeedBackLayer<TDevice>::linkTargetLayer(Layer<TDevice> &targetLayer)
    {
	m_targetDim      = ParseLayerOpt(targetLayer.layerAddInfor(1));
	m_targetLayer    = &targetLayer;

	// Now, use all target features for feedback
	// To be completed
	m_targetDimStart = 0;
	m_targetDimEnd   = m_targetDim;

	// dim * look_back + dim * aggregate + preceding_layer
	int dimExpected = ((m_targetDimEnd - m_targetDimStart) * m_lookBack.size() +
			   (m_targetDimEnd - m_targetDimStart) * m_aggOpt.size()   +
			   this->precedingLayer().size());
	
	if (dimExpected !=this->size()){
	    printf("Feedback dim + Feedforward dim = %d\n", dimExpected);
	    throw std::runtime_error("Error in network.jsn feedback layer size");
	}
	if (m_targetDimEnd > m_targetDim || m_targetDimStart > m_targetDim ||
	    m_targetDimEnd < m_targetDimStart){
	    throw std::runtime_error("Error in configuration of targetDimStart, targetDimEnd");
	}

	// initialize m_aggBuffer
	//     m_aggBuffer stores the intermediate state of aggregation and the previous boundary
	//     time
	if (m_aggOpt.size())
	    m_aggBuffer.resize((m_targetDimEnd - m_targetDimStart) * m_aggOpt.size() * 2, 0.0);
	
	// print information
	printf("\nCreating the feedback link:\n");
	printf("\tFrom %s [%d-%d]", targetLayer.type().c_str(), m_targetDimStart, m_targetDimEnd);
	printf("\tLook Back [%s]", m_lookBackStr.c_str());
	if (m_aggOpt.size()){
	    printf("\tAggregating [%s]", m_aggStr.c_str());
	    if (m_crossBoundary)
		printf(" cross boundary");
	}
	printf("\n");
    }

    template <typename TDevice>
    void FeedBackLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction)
    {
	TrainableLayer<TDevice>::loadSequences(fraction);

	// read in the boundary information
	if (m_aggStr.size()){
	    //
	    if (this->parallelSequences()>1){
		printf("Please use parallel_sequences = 1\n");
		throw std::runtime_error("Not implemented: F0 aggregation for parallel training");
	    }
	    
	    if (fraction.auxDataDim()>0){
		if (m_aggOpt.size() > CHAR_BIT)
		    throw std::runtime_error("Aggregate information is larger than CHAR_BIT");

		// Read in the aux label information
		Cpu::pattype_vector auxInfo = fraction.auxPattypeData();
		if (auxInfo.size() != this->curMaxSeqLength())
		    throw std::runtime_error("Error unequal length of clockTime size");
	    
		// Convert the boundary information into distance information
		Cpu::pattype_vector tempDistance(m_boundaryInfo.size(), 0);
		cpu_int_vector      tmpAggOpt = m_aggOpt;
		ConvertBoundaryInfo(auxInfo, tempDistance, tmpAggOpt, this->curMaxSeqLength());
		m_boundaryInfo = tempDistance;
		
		if (FEEDBACKLAYER_DEBUG){
		    for (int i = 0; i < this->curMaxSeqLength(); i++){
			printf("%d:%3d\t", i, auxInfo[i]);
			for (int j = 0; j<m_aggOpt.size(); j++)
			    printf("%3d ", tempDistance[i*m_aggOpt.size()+j]);
			printf("\n");
		    }
		}
		
		// prepare the aggregate buffer (which will be used in generation)
		m_aggBuffer.resize((m_targetDimEnd - m_targetDimStart) * m_aggOpt.size() * 2, 0.0);
	    }else {
		throw std::runtime_error("No boundary information is provided");
	    }
	}else{
	    // nothing if aggregation is not used
	}
    }
    
    template <typename TDevice>
    const std::string& FeedBackLayer<TDevice>::type() const
    {
        static std::string s;
        if (s.empty()) s = "feedback";
        return s;
    }

    // computeForward: 
    //  in training stage, target data are known
    template <typename TDevice>
    void FeedBackLayer<TDevice>::computeForwardPass()
    {
	if (m_targetLayer == NULL)
	    throw std::runtime_error("Target layer is not linked");
	
	thrust::fill(this->outputs().begin(), this->outputs().end(), 0.0);
	{{
	    // Concatenate the output of the preceding layer and the feedback layer
	    int previousSize  = this->precedingLayer().size();
	    
	    internal::vectorFillForward fn;
	    fn.dimInput1      = previousSize;     // the dimension from preceding layer
	    
	    fn.dimInput2      = m_targetDim;      // the dimension of the output of target layer
	    fn.dimInput2Start = m_targetDimStart; // from which dimension to load from target layer
	    fn.dim1Step       = m_targetDimEnd - m_targetDimStart; // dimension for 1 step
		
	    fn.dimOutput      = this->size();     
	    fn.parallel       = this->parallelSequences();

	    fn.input1         = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn.input2         = helpers::getRawPointer(m_targetLayer->secondOutputs(true));
	    fn.output         = helpers::getRawPointer(this->outputs());
	    fn.lookBack       = helpers::getRawPointer(this->m_lookBack);

	    fn.lookBackStepNM = this->m_lookBack.size();
	    fn.crossBoundary  = m_crossBoundary;
	    int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();
	    thrust::for_each(
		thrust::make_zip_iterator(thrust::make_tuple(this->outputs().begin(),
							     thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(thrust::make_tuple(this->outputs().begin()+n,
							     thrust::counting_iterator<int>(0)+n)),
		fn);
	    // dustbin.txt/Block1226x03
	}}
	
	{{
	    // aggregating
	    if (m_aggOpt.size()){
		internal::vectorAggregateForward fn;

	    
		fn.dimInput2      = m_targetDim;      // 
		fn.dimInput2Start = m_targetDimStart; //

		fn.dim1Band       = m_targetDimEnd - m_targetDimStart; // dimension for 1 band
		fn.dimOutput      = this->size();
		fn.dimOutputStart = (this->precedingLayer().size() +
				     this->m_lookBack.size() * (m_targetDimEnd - m_targetDimStart));
		
		fn.input2         = helpers::getRawPointer(m_targetLayer->secondOutputs(true));
		fn.output         = helpers::getRawPointer(this->outputs());
		fn.bandNum        = this->m_aggOpt.size();
		
		fn.boundaryInfo   = helpers::getRawPointer(this->m_boundaryInfo);
		fn.startTime      = 0;
		fn.endTime        = this->curMaxSeqLength();
		
		fn.aggBuffer      = NULL;
		fn.crossBoundary  = m_crossBoundary;

		int n = (m_targetDimEnd - m_targetDimStart) * m_aggOpt.size();
		thrust::for_each(
			thrust::make_zip_iterator(
				thrust::make_tuple(this->outputs().begin(),
						   thrust::counting_iterator<int>(0))),
			thrust::make_zip_iterator(
				thrust::make_tuple(this->outputs().begin()+n,
						   thrust::counting_iterator<int>(0)+n)),
		fn);
	    }
	}}
	
	/*
	Cpu::real_vector tmp = this->outputs();
	for (int i = 0; i<this->curMaxSeqLength(); i++){
	    printf("%3d\t", i);
	    for (int j = 0; j < this->size(); j++){
		if (j<2)
		    printf("%0.2f\t", tmp[i*this->size() + j]);
		else
		    if (tmp[i*this->size() + j] * tmp[i*this->size() + j] > 0.00001)
			printf("One-hot: %3d", j);
	    }
	    printf("\n");
	}
	printf("\n");*/
    }

    // computeForwardPass
    // in synthesis stage, when the target must be predicted frame by frame
    template <typename TDevice>
    void FeedBackLayer<TDevice>::computeForwardPass(const int timeStep)
    {
	if (m_targetLayer == NULL){
	    throw std::runtime_error("Target layer is not linked");
	}	
	
	int effTimeStepS = timeStep     * this->parallelSequences();
	int effTimeStepE = (timeStep+1) * this->parallelSequences();
	int dimension    = 0;
	thrust::fill(this->outputs().begin() + effTimeStepS * this->size(), 
		     this->outputs().begin() + effTimeStepE * this->size(), 0.0);
	
	{{
	    // The dimension of the concatenated feature (if no softmax exists)
	    int previousSize  = this->precedingLayer().size();
	    
	    // Concatenate the feature vector 
	    // (by treating the 1 dimensional softmax Index as a normal feature)
	    internal::vectorFillForward fn;
	    
	    fn.dimInput1      = previousSize;
	    fn.dimInput2      = m_targetDim;
	    
	    fn.dimOutput      = this->size();
	    fn.parallel       = this->parallelSequences();
	    fn.dimInput2Start = m_targetDimStart;


	    fn.input1         = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn.input2         = helpers::getRawPointer(m_targetLayer->secondOutputs(false));
	    fn.output         = helpers::getRawPointer(this->outputs());

	    fn.dim1Step       = m_targetDimEnd - m_targetDimStart; // dimension for 1 step
	    fn.lookBack       = helpers::getRawPointer(this->m_lookBack);

	    fn.lookBackStepNM = this->m_lookBack.size();
	    fn.crossBoundary  = m_crossBoundary;
	    thrust::for_each(
	       thrust::make_zip_iterator(
		 thrust::make_tuple(
			this->outputs().begin()+ effTimeStepS * this->size(),
			thrust::counting_iterator<int>(0)+ effTimeStepS * this->size())),
	       thrust::make_zip_iterator(
		 thrust::make_tuple(
			this->outputs().begin()+ effTimeStepE * this->size(),
			thrust::counting_iterator<int>(0)+ effTimeStepE * this->size())),
			fn);
	    // dustbin.txt/Block1226x04
	    
	}}

	{{
	    // aggregating
	    if (m_aggOptSyn==1 && m_aggOpt.size()){
		// strategy one:
		//   use the same aggration algorithm as in the training stage
		if (timeStep == 0)
		    thrust::fill(this->m_aggBuffer.begin(), this->m_aggBuffer.end(), 0.0);
		
		internal::vectorAggregateForwardInfer fn;
		fn.dimInput2      = m_targetDim;      // 
		fn.dimInput2Start = m_targetDimStart; //

		fn.dim1Band       = m_targetDimEnd - m_targetDimStart; // dimension for 1 band
		fn.dimOutput      = this->size();
		fn.dimOutputStart = (this->precedingLayer().size() +
				     this->m_lookBack.size() * (m_targetDimEnd - m_targetDimStart));
		fn.bandNum        = this->m_aggOpt.size();
				
		fn.input2         = helpers::getRawPointer(m_targetLayer->secondOutputs(false));
		fn.output         = helpers::getRawPointer(this->outputs());

		fn.boundaryInfo   = helpers::getRawPointer(this->m_boundaryInfo);
		fn.startTime      = timeStep;
		fn.endTime        = timeStep + 1;
		fn.aggBuffer      = helpers::getRawPointer(this->m_aggBuffer);;

		fn.crossBoundary  = m_crossBoundary;		
		dimension         = (m_targetDimEnd - m_targetDimStart) * m_aggOpt.size();

		
		thrust::for_each(
			thrust::make_zip_iterator(
				thrust::make_tuple(this->outputs().begin(),
						   thrust::counting_iterator<int>(0))),
			thrust::make_zip_iterator(
				thrust::make_tuple(this->outputs().begin()+dimension,
						   thrust::counting_iterator<int>(0)+dimension)),
						   fn);
	    }else if (m_aggOptSyn == 2){
		
		int previousSize  = this->precedingLayer().size();
		
		internal::vectorFillForward fn;
		
		fn.dimInput1      = previousSize;
		fn.dimInput2      = m_targetDim;
	    
		fn.dimOutput      = this->size();
		fn.parallel       = this->parallelSequences();
		fn.dimInput2Start = m_targetDimStart;
		

		fn.input1         = helpers::getRawPointer(this->precedingLayer().outputs());
		fn.input2         = helpers::getRawPointer(m_targetLayer->secondOutputs(false));
		fn.output         = helpers::getRawPointer(this->outputs());

		fn.dim1Step       = m_targetDimEnd - m_targetDimStart; // dimension for 1 step
		fn.crossBoundary  = m_crossBoundary;
		Cpu::int_vector  tmp(2,1);
		int_vector       tmpGPU = tmp;
		fn.lookBack       = helpers::getRawPointer(tmpGPU);

		fn.lookBackStepNM = m_aggOpt.size();
		
		thrust::for_each(
	         thrust::make_zip_iterator(
		  thrust::make_tuple(
			this->outputs().begin()+ effTimeStepS * this->size(),
			thrust::counting_iterator<int>(0)+ effTimeStepS * this->size())),
		 thrust::make_zip_iterator(
		  thrust::make_tuple(
			this->outputs().begin()+ effTimeStepE * this->size(),
			thrust::counting_iterator<int>(0)+ effTimeStepE * this->size())),
			fn);

	    }
	}}

    }

    // 
    template <typename TDevice>
    void FeedBackLayer<TDevice>::computeBackwardPass()
    {
	{{
	   // Copy the gradient for the preceding layer
	   internal::vectorFillBackward fn;
	   fn.dimInput1      = this->precedingLayer().size();
	   fn.dimOutput      = this->size();
	   fn.outputError    = helpers::getRawPointer(this->outputErrors());

	   int n = (this->curMaxSeqLength() * this->parallelSequences() *
		    this->precedingLayer().size());
	   
	   thrust::transform(thrust::counting_iterator<int>(0),
			     thrust::counting_iterator<int>(0)+n,
			     this->precedingLayer().outputErrors().begin(),
			     fn);	   
	}}
    }
    
    template class FeedBackLayer<Cpu>;
    template class FeedBackLayer<Gpu>;
    
}
