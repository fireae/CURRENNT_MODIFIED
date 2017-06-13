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

#ifndef LAYERS_POSTOUTPUTLAYER_HPP
#define LAYERS_POSTOUTPUTLAYER_HPP

#include "TrainableLayer.hpp"


namespace layers {

    /******************************************************************************************//**
     * This layer is used as the very last layer which store the target outputs and calculates the
     * error between the target outputs and the actual outputs
     *
     * @param TDevice The computation device (Cpu or Gpu)
     *********************************************************************************************/
    template <typename TDevice>
    class PostOutputLayer : public Layer<TDevice>
    {
        typedef typename TDevice::real_vector real_vector;
	typedef typename Cpu::real_vector cpu_real_vector;
    private:
        Layer<TDevice> &m_precedingLayer;

	PostOutputLayer<TDevice> *m_precedingMiddleOutLayer;
	
	/* Add 0401 for weighted MSE */
	real_vector     m_outputMseWeights;  // vector for MSE output weights
	cpu_real_vector m_outputMseWeightsCPU; //
	bool            m_flagMseWeight;     // whether to use m_flagMseWeight
	
	/* Add 1012 for mean and variance vector */
	real_vector     m_targetDataMV;      // [mean, variance]

	// Add 170411 for feedback
	real_vector     m_feedBackOutput;    // Buffer for the feedback data

	int             m_ganState;
	
    protected:
        real_vector&    _targets();
        real_vector&    _actualOutputs();
        real_vector&    _outputErrors();
	
	/* Add 0401 for weighted MSE */
	real_vector&     _mseWeight();
	cpu_real_vector& _mseWeightCPU();
	/* Add 1012 for */
	real_vector&     _mvVector();
	
    public:
        /**
         * Constructs the Layer
         *
         * @param layerChild     The layer child of the JSON configuration for this layer
         * @param precedingLayer The layer preceding this one
         * @param createOutputs  If false, then the outputs vector will be left empty
         */
        PostOutputLayer(
            const helpers::JsonValue &layerChild, 
            Layer<TDevice>  &precedingLayer,
            int requiredSize,
            bool createOutputs = true
            );
	
	/* Add 0401 for weighted MSE */
	bool readMseWeight(const std::string mseWeightPath);
	bool flagMseWeight();
	
	/* Add 1012 for read mean and variance vector */
	bool readMV(const cpu_real_vector &mVec, const cpu_real_vector &vVec);

        /**
         * Destructs the Layer
         */
        virtual ~PostOutputLayer();

	Layer<TDevice>& precedingLayer();

        /**
         * @see Layer::loadSequences()
         */
        virtual void loadSequences(const data_sets::DataSetFraction &fraction, const int nnState);

        /**
         * Computes the error with respect to the target outputs
         *
         * @return The error 
         */
        virtual real_t calculateError() =0;

	virtual void computeBackwardPass(const int nnState);
	
	/**
	 * Re-initialize the network
	   only defines for Trainable Layers, here do nothing
	 */
	virtual void reInitWeight();

	/* *
	 * Functions to enable the trainable features of PostOutput Layer
	 */
	// Whether this layer is trainable (default false)
	virtual bool flagTrainable() const;

	/**
	 * Functions to retrieve the feedback data
	 */
	virtual void retrieveFeedBackData();

	virtual void retrieveFeedBackData(real_vector& randNum, const int method);

	virtual void retrieveFeedBackData(const int timeStep, const int method=0);

	// Used by feedbackLayer
	virtual real_vector& feedbackOutputs(const bool flagTrain);

	// get from the middleoutput layer
	virtual real_vector& secondOutputs();

	virtual void linkMiddleOutptu(PostOutputLayer<TDevice> *precedingMiddleOutLayer);
	
	// export
	virtual void exportLayer(const helpers::JsonValue &layersArray, 
				 const helpers::JsonAllocator &allocator) const;

	virtual int ganState();

    };

} // namespace layers


#endif // LAYERS_POSTOUTPUTLAYER_HPP
