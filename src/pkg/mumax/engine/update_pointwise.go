//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

import (
	. "mumax/common"
)

type PointwiseUpdater struct {
	quant   *Quant
	lastIdx int         // Index of last time, for fast lookup of next
	points  [][]float64 // List of time+value lines: [time0, valx, valy, valz], [time1, ...
}

func (field *PointwiseUpdater) Update() {
	if len(field.points) < 2 {
		panic(InputErr("Pointwise definition needs at least two points"))
	}
	time := engine.time.Scalar()

	//find closest times

	// first search backwards in time, 
	// multi-stage solvers may have gone back in time.
	i := 0
	for i = field.lastIdx; i >= 0; i-- {
		if field.points[i][0] < time {
			break
		}
	}
	// then search forward
	for ; i < len(field.points); i++ {
		if field.points[i][0] >= time {
			break
		}
	}
	// i now points to a time >= engine.time
	field.lastIdx = i

	// out of range: value = unchanged
	if i-1 < 0 || i >= len(field.points) {
		// or should we zero it?
		return
	}

	t1 := field.points[i-1][0]
	t2 := field.points[i][0]
	v1 := field.points[i-1][1:]
	v2 := field.points[i][1:]
	dt := t2 - t1         //pt2[0] - pt1[0]
	t := (time - t1) / dt // 0..1
	value := field.quant.multiplier
	for i := range value {
		value[i] = v1[i] + t*(v2[i]-v1[i])
	}
}
