/*
 * train.h
 *
 *  Created on: Jul 5, 2016
 *      Author: sebastian
 */

#ifndef TRAIN_H_
#define TRAIN_H_

#include <vector>
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
using namespace std;


void export_descriptors(vector<CompactPatchDescriptor> & descriptors, size_t order, string const & outname) {
	ofstream out_descriptors;
	string directory = "./3DZD_train";
	makeDirectory(directory.c_str());
	out_descriptors.open(directory + "/" + outname + "_train_descriptors_N" + to_string(order) + ".txt", ofstream::out | ofstream::trunc);
	for (auto & d : descriptors) {
		CompactPatchDescriptor::to_ostream(out_descriptors,  d, order);
			out_descriptors << endl;
	}
	out_descriptors.close();

}




#endif /* TRAIN_H_ */
