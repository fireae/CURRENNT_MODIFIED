// Obsolete
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

#ifndef LAYERS_CNNLAYER_HPP
#define LAYERS_CNNLAYER_HPP

#include "TrainableLayer.hpp"
#include <boost/shared_ptr.hpp>

namespace layers {

    /******************************************************************************************//**
     * CNN layer 
     *********************************************************************************************/
    template <typename TDevice>
    class CNNLayer : public TrainableLayer<TDevice>
    {
	typedef typename TDevice::real_vector real_vector;
	typedef typename Cpu::real_vector cpu_real_vector;
	typedef typename TDevice::int_vector int_vector;
	typedef typename Cpu::int_vector cpu_int_vector;

    protected:
	cpu_int_vector  m_winWidth_H;     // filter dimension (width of the filter window)
	int_vector      m_winWidth_D;     // 
	std::string     m_winWidth_Opt;   //

	int_vector      m_winWidth_Cum;   // cumsum of the filter dimension
	
	cpu_int_vector  m_winConRange_H;  // filter convolution range
	int_vector      m_winConRange_D;
	std::string     m_winConRange_Opt;

	cpu_int_vector  m_winInterval_H;  // interval between window 
	int_vector      m_winInterval_D;
	std::string     m_winInterval_Opt;

	int_vector      m_maxIdxBuffer;   //
	int_vector      m_weightIdx;      // idx to access the weight of each window filter

	real_vector     m_conBuffer;      // data buffer
	int             m_winTotalL;      // sum of the width of filter
	int             m_numMatrixW;     // total number of weights of filter
	
    public:
	// initializer and destructor
	CNNLayer(const helpers::JsonValue &layerChild,
		 const helpers::JsonValue &weightsSection,
		 Layer<TDevice> &precedingLayer);

	virtual ~CNNLayer();

	virtual const std::string& type() const;
	
	virtual void computeForwardPass();

	virtual void computeForwardPass(const int timeStep);
	
	virtual void computeBackwardPass();

        virtual void loadSequences(const data_sets::DataSetFraction &fraction);

	// export
	virtual void exportLayer(const helpers::JsonValue &layersArray, 
				 const helpers::JsonAllocator &allocator) const;
    };
    
    
}
#endif
