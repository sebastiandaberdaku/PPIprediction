/*
 * disclaimer.h
 *
 *  Created on: 29/ago/2014
 *      Author: sebastian
 */

#ifndef DESCRIPTOREVALUATION_BACKUP_SRC_UTILS_DISCLAIMER_H_
#define DESCRIPTOREVALUATION_BACKUP_SRC_UTILS_DISCLAIMER_H_

#include <iostream>

/**
 * Author
 */
#define AUTHOR "Sebastian Daberdaku"
/**
 * email
 */
#define EMAIL "sebastian.daberdaku@dei.unipd.it"
/**
 * 3-clause BSD License.
 */
#define DISCLAIMER \
	cout <<\
"Copyright (c) 2017, Sebastian Daberdaku - \n\
DIPARTIMENTO DI INGEGNERIA DELL'INFORMAZIONE - \n\
Università degli Studi di Padova.\n\
All rights reserved.\n\n\
\
Redistribution and use in source and binary forms, with or without\n\
modification, are permitted provided that the following conditions are met:\n\
  * Redistributions of source code must retain the above copyright\n\
    notice, this list of conditions and the following disclaimer.\n\
  * Redistributions in binary form must reproduce the above copyright\n\
    notice, this list of conditions and the following disclaimer in the\n\
    documentation and/or other materials provided with the distribution.\n\
  * The name of the author may not be used to endorse or promote products\n\
    derived from this software without specific prior written permission.\n\n\
\
THIS SOFTWARE IS PROVIDED BY THE AUTHOR ''AS IS'' AND ANY EXPRESS OR\n\
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES\n\
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.\n\
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,\n\
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT\n\
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,\n\
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY\n\
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT\n\
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF\n\
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.\n\n\n";
/**
 * Name of the program
 */
#define PROGRAM_NAME "PPI prediction - Testing samples generator.\n"
/**
 * Program version
 */
#define PROGRAM_VERSION "1.0"
/**
 * Program information.
 */
#define PROGRAM_INFO \
	cout <<\
"\n"<<\
"**************************************************\n"<<\
" PPI prediction - Testing samples generator.\n"<<\
" Interface prediction program for protein-protein\n"<<\
" complexes based on surface-patch descriptors.\n"<<\
" Program: "<<argv[0]<<" in source "<<__FILE__<<",\n"<<\
" v"<<PROGRAM_VERSION<<", compiled on "<<__DATE__<<" at "<<__TIME__<<".\n"<<\
" Author: "<<AUTHOR<<"\n"<<\
" Contact: "<<EMAIL"\n"<<\
"**************************************************\n\n";
/**
 * Definition of VWS
 */
#define VWS "Van der Waals surface (or van der Waals envelope) \n\
Named after Johannes Diderik van der Waals, it is the imaginary surface \
of the union of spherical atom surfaces defined by the so-called van der Waals \
radius of each atom in the molecule representation. The van der Waals surface \
enclosed volume reference is \"molecular volume\". \n\n"
/**
 * Definition of SAS
 */
#define SAS "Solvent-accessible surface (or Lee-Richards surface) \n\
Described by the center of the probe as it rolls along the atoms' van der Waals \
spheres. \n\
Corresponds to using an extended sphere around each atom (at a distance from \
the atom centre equal to the sum of the atom and probe radii), and eliminating \
those points that lie within neighbour spheres.\
The accessible surface is therefore sort of an expanded van der Waals surface.\
It is larger (more external) than the \"molecular\" surface. \n\n"
/**
 * Definition of MS
 */
#define MS "Molecular surface (or solvent-excluded surface, or Connolly surface) \n\
\
Described by the closest point of the solvent \"probe\" as it rolls along the atoms' \
vdW spheres. In other words: the surface traced by the inward-facing surface of the \
probe. Still in other words: the evolute of the surface of the probe. Composed of \n\
- contact surface: is the part of the vdW surface that can be touched by the probe. \n\
- reentrant surface: formed by the inward-facing part of the probe when it is in \
contact with more than one atom. \n\
(Delimits the solvent-excluded volume in Connolly's terminology, equal to \
van der Waals volume plus interstitial volume). The surface is smaller (more internal) \
than the solvent-accesible surface. \n\n"

#endif /* DESCRIPTOREVALUATION_BACKUP_SRC_UTILS_DISCLAIMER_H_ */
