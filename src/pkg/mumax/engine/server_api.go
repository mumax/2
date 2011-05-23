//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// This file implements the publicly exported API of the mumax engine.
// These methods are accessible to the outside world via RPC.
// Author: Arne Vansteenkiste

// engine.API and engine.Client are the same data structure, 
// but API exports only the publicly available functions.
type API Client

//func (eng *API) SetGridSize(Nx, Ny, Nz int) {
//	(*Client)(eng).Init([]int{Nz, Ny, Nx}, []int{0, 0, 0}, true, false)
//}