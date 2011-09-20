//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

import (
	. "mumax/common"
	"mumax/gpu"
	"fmt"
)

type EulerSolver struct {
	y, dy, dt *Quant
}


func NewEuler(y, dy, dt *Quant)*EulerSolver{
	return &EulerSolver{y,dy,dt}
}

func (s *EulerSolver) Step() {
	y := s.y.Array()
	dy := s.dy.Array()
	dyMul := s.dy.multiplier
	checkUniform(dyMul)
	dt := s.dt.Scalar()

	Debug("dt intern: ", dt*dyMul[0])
	gpu.Madd(y, y, dy, float32(dt*dyMul[0]))

}

//DEBUG
func checkUniform(array []float64) {
	for _, v := range array {
		if v != array[0] {
			panic(Bug(fmt.Sprint("should be all equal:", array)))
		}
	}
}
