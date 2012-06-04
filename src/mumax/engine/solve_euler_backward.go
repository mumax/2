//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Author: Mykola Dvornik, Arne Vansteenkiste

import (
	//"fmt"
	. "mumax/common"
	"mumax/gpu"
)

// Naive Backward Euler solver
type BDFEuler struct {
	ybuffer []*gpu.Array // initial derivative
	y0buffer []*gpu.Array // initial derivative
	err      []*Quant     // error estimates for each equation
	diff     []gpu.Reductor
	iterations *Quant
}

func (s *BDFEuler) Step() {
	e := GetEngine()
	equation := e.equation
	// get dt here to avoid updates later on.
	dt := engine.dt.Scalar()
    // Advance time and update all inputs  
    e.time.SetScalar(e.time.Scalar() + dt)
	
	// Then step all outputs (without intermediate updates!)
	// and invalidate them.
	// Do initial Euler step with 
	    
	for i := range equation {
	    err := 1.0e38
	    s.iterations.SetScalar(0)
	    
	    // Do zero order approximation
	    y := equation[i].output[0]
		dy := equation[i].input[0]
		dyMul := dy.multiplier
		h := float32(dt * dyMul[0])
		s.y0buffer[i].CopyFromDevice(y.Array()) // save for later
		
		gpu.Madd(y.Array(), s.y0buffer[i], dy.Array(), h)
		
		y.Invalidate()
		
	    for err > 0.1 {
	        equation[i].input[0].Update()
	        y = equation[i].output[0]
		    dy = equation[i].input[0]
	        gpu.Madd(s.ybuffer[i], s.y0buffer[i], dy.Array(), h)
	        s.ybuffer[i].Sync()
	        y.Array().Sync()
	        iterationDiff := s.diff[i].MaxDiff(y.Array(), s.ybuffer[i])
			 
	        err = float64(iterationDiff)
            //Debug("Iteration error:", err)
			s.err[i].SetScalar(err)

	        y.Array().CopyFromDevice(s.ybuffer[i])
	        y.Invalidate()
			s.iterations.SetScalar(s.iterations.Scalar() + 1)
        }
	}

	// Advance step	
	e.step.SetScalar(e.step.Scalar() + 1) // advance time step
}

func (s *BDFEuler) Dependencies() (children, parents []string) {
	children = []string{"t", "step", "iterations"}
	parents = []string{"dt"}
	return
}

// Register this module
func init() {
	RegisterModule("solver/bdf_euler", "Fixed-step Backward Euler solver", LoadBDFEuler)
}

func LoadBDFEuler(e *Engine) {
    s := new(BDFEuler)
	s.iterations = e.AddNewQuant("iterations", SCALAR, VALUE, Unit(""), "Number of iterations per step")

	equation := e.equation
	s.ybuffer = make([]*gpu.Array, len(equation))
	s.y0buffer = make([]*gpu.Array, len(equation))
	
	s.err = make([]*Quant, len(equation))
	s.diff = make([]gpu.Reductor, len(equation))

	for i := range equation {

		eqn := &(equation[i])
		Assert(eqn.kind == EQN_PDE1)
		out := eqn.output[0]
		unit := out.Unit()
		s.err[i] = e.AddNewQuant(out.Name()+"_error", SCALAR, VALUE, unit, "Error/step estimate for "+out.Name())
		s.diff[i].Init(out.Array().NComp(), out.Array().Size3D())

		// TODO: recycle?
		y := equation[i].output[0]
		s.ybuffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.y0buffer[i] = Pool.Get(y.NComp(), y.Size3D())

	}
	e.SetSolver(s)
}
