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

#include "./misFuncs.hpp"
#include <string>
#include <vector>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

/* ***** Functions for string process ***** */
void ParseStrOpt(const std::string stringOpt, std::vector<std::string> &optVec){
    std::vector<std::string> tempArgs;
    boost::split(tempArgs, stringOpt, boost::is_any_of("_"));
    for (int i =0 ; i<tempArgs.size(); i++)
	optVec.push_back(tempArgs[i]);
    return;
}

void ParseIntOpt(const std::string stringOpt, Cpu::int_vector &optVec){
    std::vector<std::string> tempArgs;
    boost::split(tempArgs, stringOpt, boost::is_any_of("_"));
    optVec.resize(tempArgs.size(), 0);
    for (int i =0 ; i<tempArgs.size(); i++)
	optVec[i] = boost::lexical_cast<int>(tempArgs[i]);
}

int SumCpuIntVec(Cpu::int_vector &temp){
    int result = 0;
    for (int i = 0; i<temp.size(); i++)
	result += temp[i];
    return result;
}
