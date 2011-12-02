//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Author: Arne Vansteenkiste

import ()

type HeunSolver struct {
	y, dy *Quant // Input
	y1    *Quant // First stores euler solution, then heun solution.
	dy0   *Quant // saves initial dy estimate
	t, dt *Quant
}

func NewHeun(e *Engine, y, dy *Quant) *HeunSolver {
	y1 := newQuant("heun_y1", y.NComp(), e.size3D, FIELD, y.Unit(), false, "hidden buffer")
	dy0 := newQuant("heun_dy0", y.NComp(), e.size3D, FIELD, y.Unit(), false, "hidden buffer")
	return &HeunSolver{y, dy, y1, dy0, e.time, e.dt}
}

func (s *HeunSolver) AdvanceBuffer() {
	panic("todo")
	//	y := s.y
	//	dy := s.dy
	//	y1 := s.y1
	//	dy0 := s.dy0

	//dy.Update()
	//dy0.Array().CopyFromDevice(dy.Array()) // Save dy0 for later

	//dyMul := s.dy.multiplier
	//checkUniform(dyMul)
	//dt := s.dt.Scalar()

	//gpu.Madd(s.ybuf.Array(), y, dy, float32(dt*dyMul[0]))

	//s.y.Invalidate()
}

func (s *HeunSolver) CopyBuffer() {
	//s.y.Array().CopyFromDevice(s.ybuf.Array())
}

func (s *HeunSolver) ProposeDt() float64 {
	return 0 // this is not an adaptive step solver yet
}

func (e *HeunSolver) Deps() (in, out []*Quant) {
	in = []*Quant{e.dy}
	out = []*Quant{e.y}
	return
}
