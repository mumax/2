//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.



package omf

import (
	"mumax/host"
)


// Represents an omf file
type File struct {
	Header
	*host.Array
}


// Represents the header part of an omf file.
type Header struct {
	Desc            map[string]interface{}
	Size            [3]int
	ValueMultiplier float32
	ValueUnit       string
	Format          string // binary or text
	DataFormat      string // 4 or 8
	StepSize        [3]float32
	MeshUnit        string
}
