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


#include "SkipAddLayer.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"

#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/fill.h>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>


namespace layers{

    // Construct the layer
    template <typename TDevice>
    SkipAddLayer<TDevice>::SkipAddLayer(
					const helpers::JsonValue &layerChild,
					const helpers::JsonValue &weightsSection,
					std::vector<Layer<TDevice>*> precedingLayers
					)
	// use preLayers[0] as fake preceding layers
	: SkipLayer<TDevice>(layerChild, weightsSection, precedingLayers, false)
    {
	m_preLayers.assign(precedingLayers.begin(), precedingLayers.end());
	// m_outputErrorsFromSkipLayer = Cpu::real_vector(this->outputs().size(), (real_t)0.0);

	if (precedingLayers.size()<2){
	    m_flagSkipInit = true;
	}else{
	    m_flagSkipInit = false;
	}

    }	

    // Destructor
    template <typename TDevice>
    SkipAddLayer<TDevice>::~SkipAddLayer()
    {
    }
	
    // NN forward
    template <typename TDevice>
    void SkipAddLayer<TDevice>::computeForwardPass()
    {
	// initialization
	thrust::fill(this->outputs().begin(), 
		     (this->outputs().begin() + 
		      this->curMaxSeqLength() * this->parallelSequences() * this->size()),
		     0.0
		     );
	
	// initialization for backward pass
	thrust::fill(this->outputErrors().begin(), 
		     (this->outputErrors().begin() + 
		      this->curMaxSeqLength() * this->parallelSequences() * this->size()),
		     0.0
		     );

	thrust::fill(this->outputErrorsFromSkipLayer().begin(),
		     (this->outputErrorsFromSkipLayer().begin() + 
		      this->curMaxSeqLength() * this->parallelSequences() * this->size()),
		     0.0);

	// accumulating the outputs of previous layers
	BOOST_FOREACH (Layer<TDevice> *layer, m_preLayers) {
	    thrust::transform(layer->outputs().begin(),
			      (layer->outputs().begin() + 
			       this->curMaxSeqLength() * this->parallelSequences() * this->size()),
			      this->outputs().begin(),
			      this->outputs().begin(),
			      thrust::plus<real_t>()
			      );	    
	}	
    }

    // NN forward
    template <typename TDevice>
    void SkipAddLayer<TDevice>::computeForwardPass(const int timeStep)
    {
	int effTimeS = timeStep     * this->parallelSequences();
	int effTimeE = (timeStep+1) * this->parallelSequences();

	// initialization
	thrust::fill(this->outputs().begin() + effTimeS * this->size(), 
		     this->outputs().begin() + effTimeE * this->size(), 
		     0.0);
	
	// accumulating the outputs of previous layers
	BOOST_FOREACH (Layer<TDevice> *layer, m_preLayers) {
	    thrust::transform(layer->outputs().begin() + effTimeS * this->size(),
			      layer->outputs().begin() + effTimeE * this->size(),
			      this->outputs().begin()  + effTimeS * this->size(),
			      this->outputs().begin()  + effTimeS * this->size(),
			      thrust::plus<real_t>()
			      );	    
	}	
    }


    // NN backward
    template <typename TDevice>
    void SkipAddLayer<TDevice>::computeBackwardPass()
    {
	// 
	// at first, add the errors in both this->outputErrorsFromSkipLayer() and m_outputErrors
	thrust::transform(this->outputErrorsFromSkipLayer().begin(),
			  (this->outputErrorsFromSkipLayer().begin() + 
			   this->curMaxSeqLength() * this->parallelSequences() * this->size()),
			  this->outputErrors().begin(),
			  this->outputErrors().begin(),
			  thrust::plus<real_t>()
			  );

	// send erros to the all the previous layers
	BOOST_REVERSE_FOREACH (Layer<TDevice> *layer, m_preLayers) {
	    SkipLayer<TDevice>* tempLayer = dynamic_cast<SkipLayer<TDevice>*>(layer);
	    if(tempLayer){
		// this is an SkipAdd Layer, erros should be accumulated to 
		// this->outputErrorsFromSkipLayer()
		thrust::transform(this->outputErrors().begin(),
				  (this->outputErrors().begin() +
				   this->curMaxSeqLength()      * 
				   this->parallelSequences()    * this->size()),
				  tempLayer->outputErrorsFromSkipLayer().begin(),
				  tempLayer->outputErrorsFromSkipLayer().begin(),
				  thrust::plus<real_t>()
				  );
	    }else{
		// else, just copy the data to the outputErrors
		thrust::copy(this->outputErrors().begin(),
			     (this->outputErrors().begin() + 
			      this->curMaxSeqLength() * this->parallelSequences() * this->size()),
			     layer->outputErrors().begin()
			     );
	    }
	    
	}
    }
	
    // return all the preceding layers
    template <typename TDevice>
    std::vector<Layer<TDevice>*> SkipAddLayer<TDevice>::PreLayers()
    {
	return m_preLayers;
    }

    template <typename TDevice>
    typename SkipAddLayer<TDevice>::real_vector& SkipAddLayer<TDevice>::outputFromGate()
    {
	
	return this->outputFromGate();
    }
    
    template <typename TDevice>
    const std::string& SkipAddLayer<TDevice>::type() const
    {
	static std::string s;
	if (m_flagSkipInit){
	    s = "skipini";
	}
	else{
	    s = "skipadd";
	}
        return s;
    }
   
    /*
    template <typename TDevice>
    typename SkipAddLayer<TDevice>::real_vector& SkipAddLayer<TDevice>::outputErrorsFromSkipLayer()
    {
        return this->outputErrorsFromSkipLayer();
    }
    */

    template class SkipAddLayer<Cpu>;
    template class SkipAddLayer<Gpu>;
    
}
