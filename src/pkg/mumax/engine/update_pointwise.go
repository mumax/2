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

// Updates a quantity according to a point-wise defined function of time.
type PointwiseUpdater struct {
	quant   *Quant
	lastIdx int         // Index of last time, for fast lookup of next
	points  [][]float64 // List of time+value lines: [time0, valx, valy, valz], [time1, ...
}

func newPointwiseUpdater(q *Quant) *PointwiseUpdater {
	u := new(PointwiseUpdater)
	u.quant = q
	u.points = make([][]float64, 0, 100)
	engine.Depends(q.Name(), "t") // declare time-dependence
	return u
}

func (field *PointwiseUpdater) Update() {

	//Debug(field)

	if len(field.points) < 2 {
		panic(InputErr("Pointwise definition needs at least two points"))
	}
	time := engine.time.Scalar()

	//find closest times

	// first search backwards in time, 
	// multi-stage solvers may have gone back in time.
	i := 0
	for i = field.lastIdx; i > 0; i-- {
		if field.points[i][0] < time {
			break
		}
	}
	// then search forward
	for ; i < len(field.points); i++ {
		//Debug("i", i)
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
	Assert(time >= 0 && time <= 1)
	value := field.quant.multiplier
	for i := range value {
		value[i] = v1[i] + t*(v2[i]-v1[i])
	}
	field.quant.Invalidate() //SetValue(value) //?
	//Debug("pointwise update", field.quant.Name(), "time=", time, "i=", i, "value=", value)
}

func (p *PointwiseUpdater) Append(time float64, value []float64) {
	nComp := p.quant.NComp()
	checkComp(p.quant, len(value))
	if len(p.points) > 0 {
		if p.points[len(p.points)-1][0] > time {
			panic(InputErrF("Pointwise definition should be in chronological order, but", p.points[len(p.points)-1][0], ">", time))
		}
	}

	entry := make([]float64, nComp+1)
	entry[0] = time
	copy(entry[1:], value)
	p.points = append(p.points, entry)

}
