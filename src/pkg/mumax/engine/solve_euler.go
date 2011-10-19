//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	"mumax/gpu"
	"fmt"
)

// Euler solver
type EulerSolver struct {
	y, ybuf, dy, t, dt *Quant
}

func NewEuler(e *Engine, y, dy *Quant) *EulerSolver {
	buffer := newQuant("eulerBuffer", y.NComp(), e.size3D, FIELD, y.Unit(), "hidden buffer to cache output quant")
	return &EulerSolver{y, buffer, dy, e.time, e.dt}
}

func (s *EulerSolver) AdvanceBuffer() {
	s.dy.Update()

	y := s.y.Array()
	dy := s.dy.Array()
	dyMul := s.dy.multiplier
	checkUniform(dyMul)
	dt := s.dt.Scalar()

	gpu.Madd(s.ybuf.Array(), y, dy, float32(dt*dyMul[0]))

	s.y.Invalidate()
}

func (s *EulerSolver) CopyBuffer() {
	s.y.Array().CopyFromDevice(s.ybuf.Array())
}

func (s *EulerSolver) ProposeDt() float64 {
	return 0 // this is not an adaptive step solver
}

func (e *EulerSolver) Deps() (in, out []*Quant) {
	in = []*Quant{e.dy}
	out = []*Quant{e.y}
	return
}

//DEBUG
func checkUniform(array []float64) {
	for _, v := range array {
		if v != array[0] {
			panic(Bug(fmt.Sprint("should be all equal:", array)))
		}
	}
}
