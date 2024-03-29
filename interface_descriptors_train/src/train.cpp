/*
 * validation.cpp
 *
 *  Created on: 6/mag/2016
 *      Author: sebastian
 */

#include "CommandLineParser/CommandLineParser.h"
#include "DockingMethods/DockingMethods.h"
#include "exceptions/ParsingOpenDXException.h"
#include "exceptions/ParsingPQRException.h"
#include "hydrophobicity/hydrophobicity.h"
#include "MolecularComplex/MolecularComplex.h"
#include "Molecule/Molecule.h"
#include "utils/disclaimer.h"
#include "utils/elapsedTime.h"
#include "utils/makeDirectory.h"
#include "Zernike/BoundingSphere.h"
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <utility>

#include <functional>

#include "train.h"

using namespace std;

int main (int argc, char* argv[]) {
//---------------------------variables and parameters------------------------------------
	float patchRadius; // patch sphere radius
	float minCenterDist_noninterface; // minimum distance between patch centers
	float minCenterDist_interface; // minimum distance between patch centers

	float probeRadius; // probe sphere radius
	float resolution; // resolution^3 = #voxels/Å^3

	float interface_distance;

	string inname_receptor; // input filename
	string inname_ligand; // input filename

	string openDX_receptor; // input DX potentials file
	string openDX_ligand; // input DX potentials file

	string outname_receptor; // output filename
	string outname_ligand; // output filename

	string inname_radii;

	bool no_hydrogen;
	bool no_hetatm;

	bool help = false; // if true, print the help message
	bool version = false; // if true, print the program version
	bool surf_description = false; // if true, print the three surface descriptions
	bool license = false; // if true, print the license information

	float threshold;

	int maxOrder; // max Zernike Descriptor order - N in paper

    auto const t_start = chrono::high_resolution_clock::now();

	if (!CommandLineParser::parseCommandLine(argc, argv, patchRadius,
			minCenterDist_noninterface, minCenterDist_interface, probeRadius, resolution, inname_receptor,
			inname_ligand, outname_receptor,
			outname_ligand, inname_radii, no_hydrogen, no_hetatm, maxOrder,
			interface_distance, threshold, help, version, surf_description, license))
		return EXIT_FAILURE;
	if (help || version || surf_description || license)
		return EXIT_SUCCESS;
//-----------------------------print config---------------------------------------------
	PROGRAM_INFO
	/* summary of the parsed parameters */
	cout << "The specification is: \n" << "input filenames: " << inname_receptor << " and " << inname_ligand << "\n";
	cout << "SES computation algorithm:\tRegion Growing EDT with speed-optimized \n";
	cout << "\t\tdata structures, \n";
	cout << "probe radius:\t" << probeRadius << "Å, \n";
	cout << "resolution:\t" << pow((double) resolution, 3.0)
			<< " voxels per Å³, \n";
	cout << "interface distance threshold: " << interface_distance << "Å, \n";
	cout << "interface patch area threshold: " << threshold * 100 << "%\n";

	if (no_hetatm)
		cout << "include HETATM records: no\n";
	else
		cout << "include HETATM records: yes\n";
	if (no_hydrogen)
		cout << "include hydrogen atoms: no\n";
	else
		cout << "include hydrogen atoms: yes\n";

	string extension_receptor = inname_receptor.substr(inname_receptor.find_last_of(".") + 1);
	string extension_ligand = inname_ligand.substr(inname_ligand.find_last_of(".") + 1);
	if(extension_ligand == "pdb" || extension_ligand == "PDB"
			|| extension_receptor == "pdb" || extension_receptor == "PDB") {
		cout << "atomic radii: " << inname_radii << "\n";
	}
	cout << "patch radius:\t" << patchRadius << "Å, \n";
	cout << "minimum distance between non-interface patch centers:\t" << minCenterDist_noninterface
			<< "Å, \n";
	cout << "minimum distance between interface patch centers:\t" << minCenterDist_interface
			<< "Å, \n";
	cout << "maximum Zernike descriptor order:\t" << maxOrder << ". \n";
	cout << "**************************************************\n";
//-----------------------------computation--------------------------------------------
	try {
		Molecule *receptor, *ligand;
#pragma omp parallel sections
{
	#pragma omp section
			{
		receptor = new Molecule(patchRadius, minCenterDist_noninterface, probeRadius, resolution,
				inname_receptor, openDX_receptor, outname_receptor, inname_radii,
				maxOrder, no_hydrogen, no_hetatm);
			}
	#pragma omp section
			{
		ligand = new Molecule(patchRadius, minCenterDist_noninterface, probeRadius, resolution,
				inname_ligand, openDX_ligand, outname_ligand, inname_radii,
				maxOrder, no_hydrogen, no_hetatm);
			}
}
		MolecularComplex complex(*receptor, *ligand, interface_distance);

		vector<CompactPatchDescriptor> receptor_descriptors, ligand_descriptors;


		receptor->calculateDescriptors(inname_receptor, minCenterDist_noninterface, minCenterDist_interface, patchRadius, maxOrder, complex.receptor_interface, threshold, receptor_descriptors);
		ligand->calculateDescriptors(inname_ligand, minCenterDist_noninterface, minCenterDist_interface, patchRadius, maxOrder, complex.ligand_interface, threshold, ligand_descriptors);

		export_descriptors(receptor_descriptors, maxOrder,	inname_receptor.substr(0, inname_receptor.length() - 4));
		export_descriptors(ligand_descriptors, maxOrder, inname_ligand.substr(0, inname_ligand.length() - 4));

		cout << "**************************************************\n";
		cout << "Total calculation time:\t" << elapsedTime(t_start, chrono::high_resolution_clock::now()) << "\n";
		cout << "**************************************************\n";
		delete receptor;
		delete ligand;
	} catch (ParsingPQRException const & e) {
		cerr << "error: " << e.what() << "\n";
		return EXIT_FAILURE;
	} catch (ParsingOpenDXException const & e) {
		cerr << "error: " << e.what() << "\n";
		return EXIT_FAILURE;
	} catch (fstream::failure const & e) {
		cerr << "error: " << e.what() << "\n";
		return EXIT_FAILURE;
	} catch (out_of_range const & e) {
		cerr << "error: " << e.what() << "\n";
		return EXIT_FAILURE;
	} catch (invalid_argument const & e) {
		cerr << "error: " << e.what() << "\n";
		return EXIT_FAILURE;
	} catch (logic_error const & e) {
		cerr << "error: " << e.what() << "\n";
		return EXIT_FAILURE;
	} catch (runtime_error const & e) {
		cerr << "error: " << e.what() << "\n";
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
