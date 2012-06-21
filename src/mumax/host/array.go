//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package host

// This file implements 3-dimensional arrays of N-vectors on the host.
// Author: Arne Vansteenkiste

import (
	cu "cuda/driver"
	. "mumax/common"
	"sync"
	"unsafe"
	"math"
)

// A MuMax Array represents a 3-dimensional array of N-vectors.
type Array struct {
	List           []float32       // Underlying contiguous storage
	Array          [][][][]float32 // Array in the usual way
	Comp           [][]float32     // Components as contiguous lists
	Size           [4]int          // INTERNAL {components, size0, size1, size2}
	Size4D         []int           // {components, size0, size1, size2}
	Size3D         []int           // {size0, size1, size2}
	SizeInElements int64           // The total number of elements in the array
	SizeInBytes    int64           // The total size of the array in bytes
	isPinned       int             // Indicates that array is pinned to GPU
	sync.RWMutex                   // mutex for safe concurrent access to this array
}

// Initializes a pre-allocated Array struct
func (t *Array) Init(components int, size3D []int) {
	Assert(len(size3D) == 3)
	t.List, t.Array = Array4D(components, size3D[0], size3D[1], size3D[2])
	t.Comp = Slice2D(t.List, []int{components, Prod(size3D)})
	t.Size[0] = components
	t.Size[1] = size3D[0]
	t.Size[2] = size3D[1]
	t.Size[3] = size3D[2]
	t.Size4D = t.Size[:]
	t.Size3D = t.Size[1:]

	t.SizeInElements = int64(components) * int64(size3D[0]) * int64(size3D[1]) * int64(size3D[2])
	t.SizeInBytes = int64(SIZEOF_FLOAT) * t.SizeInElements
	t.isPinned = 0
}

// Allocates an returns a new Array
func NewArray(components int, size3D []int) *Array {
	t := new(Array)
	t.Init(components, size3D)
	t.isPinned = 0
	return t
}

func NewArrayPinned(components int, size3D []int) *Array {
	t := new(Array)
	t.Init(components, size3D)
	cu.MemHostRegister(cu.HostPtr(unsafe.Pointer(&t.List[0])), t.SizeInBytes, cu.MEMHOSTREGISTER_PORTABLE)
	Debug("Successfully pinned.")
	t.isPinned = 1
	return t
}

func (a *Array) Pin() {
	if a.isPinned == 0 {
		cu.MemHostRegister(cu.HostPtr(unsafe.Pointer(&a.List[0])), a.SizeInBytes, cu.MEMHOSTREGISTER_PORTABLE)
		Debug("Successfully pinned.")
		a.isPinned = 1
	}
}

func (a *Array) Rank() int {
	return len(a.Size)
}

func (a *Array) Len() int {
	return a.Size[0] * a.Size[1] * a.Size[2] * a.Size[3]
}

func (a *Array) NComp() int {
	return a.Size[0]
}

// Component array, shares storage with original
func (a *Array) Component(component int) *Array {
	comp := new(Array)
	copy(comp.Size[:], a.Size[:])
	comp.Size[0] = 1 // 1 component
	comp.Size4D = comp.Size[:]
	comp.Size3D = comp.Size[1:]
	comp.List = a.Comp[component]
	comp.Array = Slice4D(comp.List, comp.Size4D)
	comp.Comp = Slice2D(comp.List, []int{1, Prod(comp.Size3D)})
	comp.isPinned = 0
	return comp
}

// Get maximum absolute values per each channel of array
// The function returns non-absolute values
func (a *Array) GetMaxAbsValuesPerChannel() []float64 {
    ncomp := a.NComp()
    mul := make([]float64, ncomp)
    
    for c := 0; c < ncomp; c++ {
        mul[c] = 0.0
    }
    
    if ncomp != 0 {
                 
        sx := a.Size3D[0]
        sy := a.Size3D[1]
        sz := a.Size3D[2]
         
        val := 0.0
        aval := 0.0
        max_aval := math.Inf(-1)
        max_val := 0.0
        
        for c := 0; c < ncomp; c++ {
            val = 0.0
            aval = 0.0
            max_aval = math.Inf(-1)
            max_val = 0.0
	        for i := 0; i < sx; i++ {
		        for j := 0; j < sy; j++ {
			        for k := 0; k < sz; k++ {    
			            val = float64(a.Array[c][i][j][k])
			            aval = math.Abs(val)
				        if aval > max_aval {
				            max_aval = aval
				            max_val = val
				        }
			        }
		        }
	        }
	        mul[c] = max_val
	    }
    }  
    return mul
}

// Normalize each channel of the array by corresponding maximum absolute value
func (a *Array) NormalizeChannelsSeparately(mul []float64) {
    ncomp := a.NComp()
    i_mul := make([]float64, ncomp)
    for c := 0; c < ncomp; c++ {
        mul_val := mul[c]
        if (mul_val != 0.0) {
            i_mul[c] = 1.0 / mul_val
        }
    }
    
    if ncomp != 0 {
                 
        sx := a.Size3D[0]
        sy := a.Size3D[1]
        sz := a.Size3D[2]
                     
        for c := 0; c < ncomp; c++ {
	        for i := 0; i < sx; i++ {
		        for j := 0; j < sy; j++ {
			        for k := 0; k < sz; k++ {
			            a.Array[c][i][j][k] = float32(float64(a.Array[c][i][j][k]) * i_mul[c])
			        }
		        }
	        }
	    }
    }  
}

// Check whenever host array is spatially uniform or not
func (a *Array) IsArrayUniform() bool {

    isUniform := true
    
    ncomp := a.NComp()
        
    if ncomp != 0 {
                 
        sx := a.Size3D[0]
        sy := a.Size3D[1]
        sz := a.Size3D[2]
                     
        for c := 0; c < ncomp; c++ {
            init_val := float64(a.Array[c][0][0][0])
	        for i := 0; i < sx; i++ {
		        for j := 0; j < sy; j++ {
			        for k := 0; k < sz; k++ {
			            val := float64(a.Array[c][i][j][k])
			            if (val != init_val) {
			                isUniform = false
			                break;
			            }
			        }
		        }
	        }
	    }
    }
    return isUniform
}


