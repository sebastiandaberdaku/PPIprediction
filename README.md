# PPIprediction

./train -h

PPI prediction - Training samples generator. Usage:

  -h [ --help ]                         Display this brief help message.
  
  --receptor arg                        Name of the receptor's PDB file.
  
  --ligand arg                          Name of the ligand's PDB file.
  
  --atom_radii arg                      File containing the radius information 
                                        of each atom. If not specified, the 
                                        default CHARMM22 radius values will be 
                                        used for PDB files. It is ignored if 
                                        specified for PQR files.
                                        
  --no_hetatm                           Ignore HETATM records in the surface 
                                        computation.
                                        
  --no_hydrogen                         Ignore hydrogen atoms in the surface 
                                        computation.
                                      
                                        NOTE: X-ray crystallography cannot 
                                        resolve hydrogen atoms in most protein 
                                        crystals, so in most PDB files, 
                                        hydrogen atoms are absent. Sometimes 
                                        hydrogens are added by modeling. 
                                        Hydrogens are always present in PDB 
                                        files resulting from NMR analysis, and 
                                        are usually present in theoretical 
                                        models.
                                        
  --surf_description                    Prints a brief description of the 
                                        supported surface types.
                                        
  -p [ --probe_radius ] arg (=1.4)      Probe radius (in Å), floating point 
                                        number in (0, 2.0] (default is 1.4Å).
                                        
  -R [ --patch_radius ] arg (=6)        Patch radius (in Å), positive floating
                                        point (default is 6.0Å).
                                        
  -t [ --patch_threshold ] arg (=0.8)   Threshold value used for the interface 
                                        patch determination (between 0 and 1, 
                                        default is 0.8).
  -i [ --interface_distance ] arg (=4.5)
                                        Threshold distance for the interface 
                                        determination (in Å), positive 
                                        floating point (default is 4.5Å).
                                        
  -d [ --nint_patch_dist ] arg (=4.5)   Minimum distance between non-interface 
                                        patch centers (in Å), positive 
                                        floating point (default is 4.5Å).
                                        
  -D [ --int_patch_dist ] arg (=1)      Minimum distance between interface 
                                        patch centers (in Å), positive 
                                        floating point (default is 1.0Å).
                                        
  -r [ --resolution ] arg (=4)          Resolution factor, positive floating 
                                        point (default is 4.0). This value's 
                                        cube determines the number of voxels 
                                        per Å³.
                                        
  -N [ --max_order ] arg (=20)          Maximum Zernike descriptor order 
                                        (default is 20).
                                        
  -l [ --license ]                      View license information.
  
  -v [ --version ]                      Display the version number
  
############################################################################

./test -hPPI prediction - Testing samples generator. Usage:

  -h [ --help ]                         Display this brief help message.
  
  --receptor arg                        Name of the receptor's PDB file.
  
  --ligand arg                          Name of the ligand's PDB file.
  
  --atom_radii arg                      File containing the radius information 
                                        of each atom. If not specified, the 
                                        default CHARMM22 radius values will be 
                                        used for PDB files. It is ignored if 
                                        specified for PQR files.
                                        
  --no_hetatm                           Ignore HETATM records in the surface 
                                        computation.
                                        
  --no_hydrogen                         Ignore hydrogen atoms in the surface 
                                        computation.
                                        
                                        NOTE: X-ray crystallography cannot 
                                        resolve hydrogen atoms in most protein 
                                        crystals, so in most PDB files, 
                                        hydrogen atoms are absent. Sometimes 
                                        hydrogens are added by modeling. 
                                        Hydrogens are always present in PDB 
                                        files resulting from NMR analysis, and 
                                        are usually present in theoretical 
                                        models.
                                        
  --surf_description                    Prints a brief description of the 
                                        supported surface types.
                                        
  -p [ --probe_radius ] arg (=1.4)      Probe radius (in Å), floating point 
                                        number in (0, 2.0] (default is 1.4Å).
                                        
  -R [ --patch_radius ] arg (=6)        Patch radius (in Å), positive floating
                                        point (default is 6.0Å).
                                        
  -t [ --patch_threshold ] arg (=0.8)   Threshold value used for the interface 
                                        patch determination (between 0 and 1, 
                                        default is 0.8).
                                        
  -i [ --interface_distance ] arg (=4.5)
                                        Threshold distance for the interface 
                                        determination (in Å), positive 
                                        floating point (default is 4.5Å).
                                        
  -d [ --patch_dist ] arg (=1.8)        Minimum distance between patch centers 
                                        (in Å), positive floating point 
                                        (default is 1.8Å).
                                        
  -r [ --resolution ] arg (=4)          Resolution factor, positive floating 
                                        point (default is 4.0). This value's 
                                        cube determines the number of voxels 
                                        per Å³.
                                        
  -N [ --max_order ] arg (=20)          Maximum Zernike descriptor order 
                                        (default is 20).
                                        
  -l [ --license ]                      View license information.
  
  -v [ --version ]                      Display the version number

############################################################################

The "CHARMM22" file contains the atomic radii information used during the surface computation process. Users can append new entries at the end of this file, modify existing ones, or create their own file with the desired radius assignment for each atom type. It is required by both the "train" and "test" binaries to function correctly. The easiest way to correctly run the binaries would be to place all files in the same directory: train/test binary, CHARMM22 file and the two PDB files. The location of the CHARMM22 file can also be supplied to the binaries with the "--atom_radii" command line option followed with the full path to the file.
