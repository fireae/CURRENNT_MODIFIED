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
/*
 */

#ifndef LAYERS_OPERATIONLAYER_HPP
#define LAYERS_OPERATIONLAYER_HPP


#include "TrainableLayer.hpp"

namespace layers{

    template <typename TDevice>
    class OperationLayer : public TrainableLayer<TDevice>
    {
	typedef typename TDevice::real_vector     real_vector;
	typedef typename Cpu::real_vector         cpu_real_vector;
	typedef typename TDevice::int_vector      int_vector;
	typedef typename TDevice::pattype_vector  pattype_vector;
	typedef typename Cpu::int_vector          cpu_int_vector;

    public:
	real_vector     m_setZeroVec_D;
	cpu_real_vector m_setZeroVec_H;
	std::string     m_setZeroStr;

	int             m_noiseSize;
	real_t          m_noiseMag;
	real_vector     m_noiseInput;
	int             m_noiseRepeat;

	int             m_downSampRes;

	real_vector     m_oneVec;
	int             m_lastShot;
	
	cpu_int_vector  m_seqLengthBuffH;  // the length of each sequence
	int_vector      m_seqLengthBuffD;  // the length of each sequence
	
	cpu_int_vector  m_segBoundaryH;    // position of the end of segment (for each frame)
	int_vector      m_segBoundaryD;
	int             m_segLevel;        // which level to be used ?
	
	OperationLayer(
	    const helpers::JsonValue &layerChild,
	    const helpers::JsonValue &weightsSection,
            Layer<TDevice>           &precedingLayer,
	    int                       maxSeqLength
	);

	virtual ~OperationLayer();
	
	
	virtual const std::string& type() const;
	
	// NN forward
	virtual void computeForwardPass(const int nnState);
	
	// NN forward, per frame
	virtual void computeForwardPass(const int timeStep, const int nnState);
	
	// NN backward
	virtual void computeBackwardPass(const int nnState);

	// export
	virtual void exportLayer(const helpers::JsonValue &layersArray, 
				 const helpers::JsonAllocator &allocator) const;
	//
	virtual void loadSequences(const data_sets::DataSetFraction &fraction, const int nnState);
    };
    
}

#endif
