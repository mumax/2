//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

// This files provides the package documentation
// Author: Arne Vansteenkiste

// 3D Array indexing
//
// Internal dimensions are labeled (0,1,2), 0 being the outermost dimension, 2 the innermost.
// A typical loop thus reads:
//	for i:=0; i<N0; i++{
//		for j:=0; j<N1; j++{
//			for k:=0; k<N2; k++{
//
//			}
//		}
//	}
//
// 0 may be a small dimension, but 2 must preferentially be large and alignable in CUDA memory.
//
// The underlying contiguous storage is indexed as:
// 	index := i*N1*N2 + j*N2 + k
//
// The "internal" (0,1,2) dimensions correspond to the "user" dimensions (Z,Y,X)!
// Z is typically the smallest dimension like the thickness.
package common
