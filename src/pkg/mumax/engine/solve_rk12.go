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
)

type RK12Solver struct {
	buffer []*gpu.Array
	error  []*Quant // error estimates for each equation
	maxErr []*Quant // maximum error for each equation
	diff   []gpu.Reductor
}

func LoadRK12(e *Engine) {
	s := new(RK12Solver)
	equation := e.equation
	s.buffer = make([]*gpu.Array, len(equation))
	s.error = make([]*Quant, len(equation))
	s.maxErr = make([]*Quant, len(equation))
	s.diff = make([]gpu.Reductor, len(equation))
	e.SetSolver(s)

	for i := range equation {

		eqn := &(equation[i])
		Assert(eqn.kind == EQN_PDE1)
		out := eqn.output[0]
		unit := out.Unit()
		e.AddQuant(out.Name()+"_error", SCALAR, VALUE, unit, "Error/step estimate for "+out.Name())
		s.error[i] = e.Quant(out.Name() + "_error")
		e.AddQuant(out.Name()+"_maxError", SCALAR, VALUE, unit, "Maximum error/step for "+out.Name())
		s.maxErr[i] = e.Quant(out.Name() + "_maxError")
		s.diff[i].Init(out.Array().NComp(), out.Array().Size3D())
		s.maxErr[i].SetVerifier(Positive)
	}
}

func (s *RK12Solver) Dependencies() (children, parents []string) {
	children = []string{"dt", "step"}
	parents = []string{"dt"}
	for i := range s.error {
		parents = append(parents, s.maxErr[i].Name())
		children = append(children, s.error[i].Name())
	}
	return
}

// Register this module
func init() {
	RegisterModule("solver/rk12", "Adaptive Heun solver (Runge-Kutta 1+2)", LoadRK12)
}

func (s *RK12Solver) Step() {
	e := GetEngine()
	equation := e.equation

	// First update all inputs
	dt := engine.dt.Scalar()
	for i := range equation {
		Assert(equation[i].kind == EQN_PDE1)
		equation[i].input[0].Update()
	}

	// Then step all outputs
	// and invalidate them.

	// stage 0
	for i := range equation {
		y := equation[i].output[0]
		dy := equation[i].input[0]
		dyMul := dy.multiplier
		checkUniform(dyMul)
		s.buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.buffer[i].CopyFromDevice(dy.Array()) // save for later

		gpu.Madd(y.Array(), y.Array(), dy.Array(), float32(dt*dyMul[0])) // initial euler step

		y.Invalidate()
	}

	// Advance time
	e.time.SetScalar(e.time.Scalar() + dt)

	// update inputs again
	for i := range equation {
		Assert(equation[i].kind == EQN_PDE1)
		equation[i].input[0].Update()
	}

	// stage 1
	minFactor := 2.0
	for i := range equation {
		y := equation[i].output[0]
		dy := equation[i].input[0]
		dyMul := dy.multiplier

		h := float32(dt * dyMul[0])
		gpu.MAdd2Async(y.Array(), dy.Array(), 0.5*h, s.buffer[i], -0.5*h, y.Array().Stream) // corrected step
		y.Array().Sync()

		// error estimate
		stepDiff := s.diff[i].MaxDiff(dy.Array(), s.buffer[i]) * h
		error := float64(stepDiff)
		s.error[i].SetScalar(error)
		factor := s.maxErr[i].Scalar() / error

		// TODO: give user the control:
		if factor < 0.01 {
			factor = 0.01
		}
		if factor < minFactor {
			minFactor = factor
		} // take minimum time increase factor of all eqns.

		Pool.Recycle(&s.buffer[i])
		y.Invalidate()
	}

	e.dt.SetScalar(dt * minFactor)
	e.step.SetScalar(e.step.Scalar() + 1) // advance time step
}
