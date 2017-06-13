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


#include "../Configuration.hpp"
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>

/* ***** Functions for string process ***** */
void ParseStrOpt(const std::string stringOpt, std::vector<std::string> &optVec,
		 const std::string para){
    std::vector<std::string> tempArgs;
    boost::split(tempArgs, stringOpt, boost::is_any_of(para));
    for (int i =0 ; i<tempArgs.size(); i++)
	optVec.push_back(tempArgs[i]);
    return;
}

void ParseIntOpt(const std::string stringOpt, Cpu::int_vector &optVec){
    std::vector<std::string> tempArgs;
    std::vector<std::string> tempArgs2;
    std::vector<int> tmpresult;
    
    boost::split(tempArgs, stringOpt, boost::is_any_of("_"));
    for (int i =0 ; i<tempArgs.size(); i++){
	boost::split(tempArgs2, tempArgs[i], boost::is_any_of("*"));
	if (tempArgs2.size() == 2){
	    int cnt = boost::lexical_cast<int>(tempArgs2[0]);
	    for (int j = 0; j < cnt; j++)
		tmpresult.push_back(boost::lexical_cast<int>(tempArgs2[1]));
	}else{
	    tmpresult.push_back(boost::lexical_cast<int>(tempArgs[i]));
	}
    }
    optVec.resize(tmpresult.size(), 0.0);
    for (int i=0;i<optVec.size();i++)
	optVec[i] = tmpresult[i];
}

void ParseFloatOpt(const std::string stringOpt, Cpu::real_vector &optVec){
    std::vector<std::string> tempArgs;
    std::vector<std::string> tempArgs2;
    std::vector<real_t> tmpresult;
    
    boost::split(tempArgs, stringOpt, boost::is_any_of("_"));
    for (int i =0 ; i<tempArgs.size(); i++){
	boost::split(tempArgs2, tempArgs[i], boost::is_any_of("*"));
	if (tempArgs2.size() == 2){
	    int cnt = boost::lexical_cast<int>(tempArgs2[0]);
	    for (int j = 0; j < cnt; j++)
		tmpresult.push_back(boost::lexical_cast<real_t>(tempArgs2[1]));
	}else{
	    tmpresult.push_back(boost::lexical_cast<real_t>(tempArgs[i]));
	}
    }
    optVec.resize(tmpresult.size(), 0.0);
    for (int i=0;i<optVec.size();i++)
	optVec[i] = tmpresult[i];
}

int SumCpuIntVec(Cpu::int_vector &temp){
    int result = 0;
    for (int i = 0; i<temp.size(); i++)
	result += temp[i];
    return result;
}

int MaxCpuIntVec(Cpu::int_vector &temp){
    if (temp.size()>0){
	int max = temp[0];
	for (int i = 1; i<temp.size(); i++)
	    if (temp[i] > max){
		max = temp[i];
	    }
	return max;
    }else{
	printf("Input vector is void");
	return 0;
    }
}

real_t GetRandomNumber(){
    static boost::mt19937 *gen = NULL;
    if (!gen) {
	gen = new boost::mt19937;
	gen->seed(Configuration::instance().randomSeed());
    }
    boost::random::uniform_real_distribution<real_t> dist(0, 1);
    return dist(*gen); 
}


int flagUpdateDiscriminator(const int epoch, const int frac){
    /*if (epoch % 2){
	return (frac % 2) == 0;
    }else{
	return (frac % 2) == 1;
	}*/
    return ((frac + 1) % 3);
}

bool closeToZero(const real_t t1, const real_t lowBound, const real_t upBound)
{
    return ((t1 > lowBound) && (t1 < upBound));
}
