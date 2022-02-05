/*
 * test.h
 *
 *  Created on: Jul 5, 2016
 *      Author: sebastian
 */

#ifndef TEST_H_
#define TEST_H_

#include <vector>
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>

using namespace std;

void export_descriptors(vector<CompactPatchDescriptor> & descriptors, size_t order, string const & outname) {
	ofstream out_descriptors, out_centers, out_truth;
	string directory = "./3DZD_test";
	makeDirectory(directory.c_str());
	out_descriptors.open(directory + "/" + outname + "_test_descriptors_N" + to_string(order) + ".txt", ofstream::out | ofstream::trunc);
	out_centers.open(directory + "/" + outname + "_patch_centers.txt", ofstream::out | ofstream::trunc);
	out_truth.open(directory + "/" + outname + "_patch_truth.txt", ofstream::out | ofstream::trunc);

	for (auto & d : descriptors) {
		CompactPatchDescriptor::to_ostream(out_descriptors,  d, order);
			out_descriptors << endl;
		out_centers << d.center.x << "\t" << d.center.y << "\t" << d.center.z << "\t" << endl;
		if (d.isInterface)
			out_truth << "+1" << endl;
		else
			out_truth << "-1" << endl;
	}
	out_descriptors.close();
	out_centers.close();
	out_truth.close();

}


#endif /* TEST_H_ */
