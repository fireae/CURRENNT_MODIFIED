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

#ifndef LAYERS_AMALGAMATE_HPP
#define LAYERS_AMALGAMATE_HPP

#include "Layer.hpp"
#include "TrainableLayer.hpp"
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>

namespace layers {
    template <typename TDevice>
    class AmalgamateLayer : public TrainableLayer<TDevice>
    {
	typedef typename TDevice::real_vector     real_vector;
	typedef typename TDevice::int_vector      int_vector;
	typedef typename TDevice::pattype_vector  pattype_vector;
	typedef typename Cpu::int_vector          cpu_int_vector;
	
    private:
	bool            m_biDirectional;  // 
	std::string     m_aggStr;         //
	int_vector      m_aggOpt;         //
	pattype_vector  m_boundaryInfo;   // buffer to store the boundary information
	real_vector     m_aggBuffer;
	
    public:
	
	AmalgamateLayer(
	    const helpers::JsonValue &layerChild,
	    const helpers::JsonValue &weightsSection,
            Layer<TDevice>           &precedingLayer
	);

	virtual ~AmalgamateLayer();
	
	
	virtual const std::string& type() const;
	
	// NN forward
	virtual void computeForwardPass();
	
	// NN forward, per frame
	virtual void computeForwardPass(const int timeStep);
	
	// NN backward
	virtual void computeBackwardPass();

	// export
	virtual void exportLayer(const helpers::JsonValue &layersArray, 
				 const helpers::JsonAllocator &allocator) const;
	
	virtual void setDirection(const bool direction);
	
	// load sequences
	virtual void loadSequences(const data_sets::DataSetFraction &fraction);
    };

}


#endif
