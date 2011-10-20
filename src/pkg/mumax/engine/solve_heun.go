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

// Heun solver
// TODO: implement
type HeunSolver struct {
	y, dy, t, dt *Quant
}

func NewHeun(y, dy, t, dt *Quant) *HeunSolver {
	return &EulerSolver{y, dy, t, dt}
}

func (s *HeunSolver) Step() {
	y := s.y.Array()
	dy := s.dy.Array()
	dyMul := s.dy.multiplier
	checkUniform(dyMul)

	t := s.t
	dt := s.dt.Scalar()
	if dt <= 0 {
		panic(InputErr(fmt.Sprint("dt=", dt)))
	}

	gpu.Madd(y, y, dy, float32(dt*dyMul[0]))

	t.SetScalar(t.Scalar() + dt) // do not use quant dt.Scalar, which might get updated
}

//DEBUG
func checkUniform(array []float64) {
	for _, v := range array {
		if v != array[0] {
			panic(Bug(fmt.Sprint("should be all equal:", array)))
		}
	}
}
