//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// This file implements a "Universe" in which time, space and physical quantities are defined.
//
// Author: Arne Vansteenkiste.

import (
	. "mumax/common"
	"fmt"
)

// A universe defines time, space (size of discretization grid)
// and a number of physical fields that are defined on the grid.
// 
// E.g.: a Universe may have size 1 x 64 x 64 and contain a magnetization,
// field and energy density, each of that size.
//
type Universe struct {
	// Space
	_size3D    [3]int // INTERNAL
	size3D     []int  // Discretization grid size
	_periodic  [3]int // INTERNAL
	periodic   []int  // Periodicity in each direction. 0 = no periodicity, >0 = repeat 2*N+1 times in that direction.
	hasVolumeNodes  bool
	hasSurfaceNodes bool

	// Time
	timeId int     // Integer representation of time ("number of time steps taken")
	time   float64 // Time in internal units

	// Fields
	fields []*Field

	*Logger
}


func (u *Universe) Init(size3D, periodic []int) {
	Assert(len(size3D) == 3)
	Assert(len(periodic) == 3)
	u.size3D = u._size3D[:]
	u.periodic = u._periodic[:]
	copy(u.size3D, size3D)
	copy(u.periodic, periodic)
	u.hasVolumeNodes = true
	u.hasSurfaceNodes = false
}


func (u *Universe) HasVolumeNodes() bool{
	return u.hasVolumeNodes
}

func (u *Universe) HasSurfaceNodes() bool{
	return u.hasSurfaceNodes
}

func (u *Universe) Size3D() []int{
	return u.size3D
}

func (u *Universe) AddField(name string, nComp int){
	u.addFieldOrValue(name, nComp, u.Size3D())
}

func (u *Universe) AddValue(name string, nComp int){
	u.addFieldOrValue(name, nComp, nil)
}

// INTERNAL
func (u *Universe) addFieldOrValue(name string, nComp int, size3D []int){
	field := newField(name, nComp, size3D)
	u.fields = append(u.fields, field)
}

func (u *Universe) String() string{
	str := "Universe " + fmt.Sprintf("%p\n", u) + 
		"\tsize:         " + fmt.Sprintln(u.size3D) + 
		"\tperiodic:     " + fmt.Sprintln(u.periodic) + 
		"\tvolume nodes :" + fmt.Sprintln(u.HasVolumeNodes()) + 
		"\tsurface nodes:" + fmt.Sprintln(u.HasSurfaceNodes())  +
		"\tfields:       \n"

	for i:= range u.fields{
		str += "\t             " + fmt.Sprintln(u.fields[i]) 
	}
	return str
}

