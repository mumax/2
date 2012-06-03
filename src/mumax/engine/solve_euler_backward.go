//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Author: Mykola Dvornik, Arne Vansteenkiste

import (
	"fmt"
	. "mumax/common"
	"mumax/gpu"
)

// Naive Backward Euler solver
type BDFEuler struct {
	y0buffer []*gpu.Array // initial value
	dybuffer []*gpu.Array // initial derivative
	err      []*Quant     // error estimates for each equation
	peakErr  []*Quant     // maximum error for each equation
	maxErr   []*Quant     // maximum error for each equation
	diff     []gpu.Reductor
	minDt    *Quant
	maxDt    *Quant
	iterations *Quant
}

func (s *BDFEuler) Step() {
	e := GetEngine()
	equation := e.equation
	// get dt here to avoid updates later on.
	dt := engine.dt.Scalar()
    // Advance time and update all inputs  
    e.time.SetScalar(e.time.Scalar() + dt)
	for i := range equation {
		Assert(equation[i].kind == EQN_PDE1)
		equation[i].input[0].Update()
	}

	

	// Then step all outputs (without intermediate updates!)
	// and invalidate them.
	// Do initial Euler step with 
	for i := range equation {
	    err := 1.0e38
	    for err > 1e-6 {
	        y := equation[i].output[0]
	        dy := equation[i].input[0]
	        dyMul := dy.multiplier
	        checkUniform(dyMul)
	        gpu.MAdd1Async(y.Array(), dy.Array(), float32(dt*dyMul[0]), y.Array().Stream) // TODO: faster MAdd
	        y.Array().Sync()
	        y.Invalidate()
        }
	}

	// Advance step	
	e.step.SetScalar(e.step.Scalar() + 1) // advance time step
}

func (s *BDFEuler) Dependencies() (children, parents []string) {
	children = []string{"t", "step"}
	parents = []string{"dt"}
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

// Register this module
func init() {
	RegisterModule("solver/euler_backward", "Fixed-step Backward Euler solver", LoadBDFEuler)
}

func LoadBDFEuler(e *Engine) {
    s := new(BDFSolver)
	s.iterations = e.AddNewQuant("iterations", SCALAR, VALUE, Unit(""), "Number of iterations per step")

	equation := e.equation
	s.dybuffer = make([]*gpu.Array, len(equation))
	s.y0buffer = make([]*gpu.Array, len(equation))
	s.err = make([]*Quant, len(equation))
	s.peakErr = make([]*Quant, len(equation))
	s.maxErr = make([]*Quant, len(equation))
	s.diff = make([]gpu.Reductor, len(equation))
	e.SetSolver(s)

	for i := range equation {

		eqn := &(equation[i])
		Assert(eqn.kind == EQN_PDE1)
		out := eqn.output[0]
		s.diff[i].Init(out.Array().NComp(), out.Array().Size3D())
		s.maxErr[i].SetVerifier(Positive)

		// TODO: recycle?
		y := equation[i].output[0]
		s.dybuffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.y0buffer[i] = Pool.Get(y.NComp(), y.Size3D())

	}
	e.SetSolver(s)
}
