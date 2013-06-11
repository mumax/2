//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Author: Mykola Dvornik, Arne Vansteenkiste

import (
	"container/list"
	"fmt"
	"math"
	. "mumax/common"
	"mumax/gpu"
	"sort"
)

// The adaptive implicit method, predictor: Implicit Euler, corrector: Trapezoidal
type BDFEulerAuto struct {
	ybuffer []*gpu.Array // current value of the quantity

	y0buffer []*gpu.Array // the value of quantity at the begining of the step
	y1buffer []*gpu.Array // the value of quantity after pedictor step

	dy0buffer  []*gpu.Array // the value of quantity derivative at the begining of the step
	dybuffer   []*gpu.Array // the buffer for quantity derivative
	err        []*Quant     // error estimates for each equation
	maxAbsErr  []*Quant     // maximum absolute error per step for each equation
	maxRelErr  []*Quant     // maximum absolute error per step for each equation
	maxIterErr *Quant       // error iterator error estimates for each equation
	maxIter    *Quant       // maximum number of iterations per step
	newDt      []float64    //
	diff       []gpu.Reductor
	err_list   []*list.List
	iterations *Quant
	badSteps   *Quant
	minDt      *Quant
	maxDt      *Quant
}

func (s *BDFEulerAuto) Step() {
	e := GetEngine()
	t0 := e.time.Scalar()

	s.badSteps.SetScalar(0)
	s.iterations.SetScalar(0)

	equation := e.equation
	// make sure that errors history is wiped for t0 = 0s!
	if t0 == 0.0 {
		for i := range equation {
			s.err_list[i].Init()
		}
	}
	// save everything in the begining
	for i := range equation {
		equation[i].input[0].Update()
		y := equation[i].output[0]
		dy := equation[i].input[0]
		s.y0buffer[i].CopyFromDevice(y.Array())   // save for later
		s.dy0buffer[i].CopyFromDevice(dy.Array()) // save for later
	}

	const maxTry = 6 // undo at most this many bad steps
	const headRoom = 0.8

	// try to integrate maxTry times at most
	for try := 0; try < maxTry; try++ {
		dt := engine.dt.Scalar()
		badStep := false
		badIterator := false

		for i := range equation {
			// get dt here to avoid updates later on.
			y := equation[i].output[0]
			dy := equation[i].input[0]
			dyMul := dy.multiplier
			t_step := dt * dyMul[0]
			// Do zero order approximation with forward Euler method
			// The zero-order approximation is used as a starting point for fixed-point iteration
			gpu.Madd(y.Array(), s.y0buffer[i], s.dy0buffer[i], t_step)
			y.Invalidate()
		}

		s.iterations.SetScalar(s.iterations.Scalar() + 1)

		// Since implicit methods use derivative at right side

		e.time.SetScalar(t0 + dt)

		for i := range equation {
			equation[i].input[0].Update()
			s.dybuffer[i].CopyFromDevice(equation[i].input[0].Array())
		}

		// Do higher order approximation until converges
		maxIterErr := s.maxIterErr.Scalar()
		maxIter := int(s.maxIter.Scalar())
		er := make([]float64, len(equation))

		// Do predictor: BDF Euler (aka Adams-Moulton 0)
		iter := 0
		err := 1e10
		for err > maxIterErr {
			for i := range equation {
				y := equation[i].output[0]
				dy := equation[i].input[0]
				dyMul := dy.multiplier
				t_step := dt * dyMul[0]
				gpu.Madd(s.ybuffer[i], s.y0buffer[i], dy.Array(), t_step)
				tErr := float64(s.diff[i].MaxDiff(y.Array(), s.ybuffer[i]))
				//~ sum := float64(s.diff[i].MaxSum(y.Array(), s.ybuffer[i]))
				//~ if sum > 0.0 {
				//~ tErr = tErr / sum
				//~ }
				er[i] = tErr
				y.Array().CopyFromDevice(s.ybuffer[i])
				y.Invalidate()
			}
			for i := range equation {
				equation[i].input[0].Update()
			}
			// Get the largest error
			sort.Float64s(er)
			err = er[len(equation)-1]
			iter = iter + 1
			s.iterations.SetScalar(s.iterations.Scalar() + 1)
			if iter > maxIter {
				badIterator = true
				break
			}
		}
		// If fixed-point iterator cannot converge, then panic
		if badIterator && try == (maxTry-1) {
			panic(Bug(fmt.Sprintf("The BDF Euler iterator cannot converge! Please increase the maximum number of iterations and re-run!")))
		} else if badIterator {
			engine.dt.SetScalar(0.5 * dt)
			continue
		}

		// Save the derivative for the comparator
		// and restore dy as estimated by Forward Euler
		for i := range equation {
			s.y1buffer[i].CopyFromDevice(equation[i].output[0].Array())
			equation[i].input[0].Array().CopyFromDevice(s.dybuffer[i])
		}

		// Do corrector: BDF trapezoidal (aka Adams-Moulton 1)
		iter = 0
		err = 1e10
		// There is no such method in the literature
		// So lets do it like RK does, start from the initial guess, but not from the predicted one
		for err > maxIterErr {
			for i := range equation {
				y := equation[i].output[0]
				dy := equation[i].input[0]
				dyMul := dy.multiplier
				t_step := dt * dyMul[0]
				h := float32(t_step)
				gpu.AddMadd(s.ybuffer[i], s.y0buffer[i], dy.Array(), s.dy0buffer[i], 0.5*h)
				tErr := float64(s.diff[i].MaxDiff(y.Array(), s.ybuffer[i]))
				//~ sum := float64(s.diff[i].MaxSum(y.Array(), s.ybuffer[i]))
				//~ if sum > 0.0 {
				//~ tErr = tErr / sum
				//~ }
				er[i] = tErr
				y.Array().CopyFromDevice(s.ybuffer[i])
				y.Invalidate()
			}
			for i := range equation {
				equation[i].input[0].Update()
			}
			// Get the largest error
			sort.Float64s(er)
			err = er[len(equation)-1]
			iter = iter + 1
			s.iterations.SetScalar(s.iterations.Scalar() + 1)
			if iter > maxIter {
				badIterator = true
				break
			}
		}
		if badIterator && try == (maxTry-1) {
			// If fixed-point iterator cannot converge, then panic
			panic(Bug(fmt.Sprintf("The BDF Trapezoidal iterator cannot converge! Please decrease the error the maximum number of iterations and re-run!")))
		} else if badIterator {
			engine.dt.SetScalar(0.5 * dt)
			continue
		}

		for i := range equation {
			y := equation[i].output[0]
			// The error is estimated mainly by BDF Euler
			abs_dy := float64(s.diff[i].MaxDiff(y.Array(), s.y1buffer[i]))
			max_y := 0.5 * float64(s.diff[i].MaxSum(y.Array(), s.y1buffer[i]))

			s.err[i].SetScalar(abs_dy)

			maxAbsStepErr := s.maxAbsErr[i].Scalar()
			maxRelStepErr := s.maxRelErr[i].Scalar() * max_y

			StepErr := abs_dy
			maxStepErr := maxAbsStepErr

			// if step is large then threshold then badstep is reported
			if StepErr >= maxAbsStepErr || StepErr >= maxRelStepErr {
				s.badSteps.SetScalar(s.badSteps.Scalar() + 1)
				badStep = true
				// We don't know which particular condition has triggered the badstep, so let us pick the smallest error threshold
				maxStepErr = math.Min(maxAbsStepErr, maxRelStepErr)
			}

			if maxStepErr == 0.0 {
				maxStepErr = maxAbsStepErr
			}

			if abs_dy == 0.0 {
				// Highly unlikely, but possible situation
				StepErr = 0.001 * maxStepErr
			}

			// Let us compare the current error to the error from the previous steps
			errRatio := 1.0e10
			if !badStep {
				//do hardcore adjustment if there is not enough history and err < err0
				errRatio = math.Sqrt(maxStepErr / StepErr)
			} else {
				//do hardcore adjustment if there is not enough history and (or) err > err0
				errRatio = math.Pow(maxStepErr/StepErr, 1./3.)
			}

			step_corr := errRatio * headRoom
			// Keep the history of 'good' errors
			if !badStep || try == (maxTry-1) {
				s.err_list[i].PushFront(StepErr)
				if s.err_list[i].Len() == 10 {
					s.err_list[i].Remove(s.err_list[i].Back())
				}
			}

			if step_corr > 1.5 {
				step_corr = 1.5
			}
			if step_corr < 0.1 {
				step_corr = 0.1
			}

			new_dt := dt * step_corr

			if new_dt < s.minDt.Scalar() {
				new_dt = s.minDt.Scalar()
			}
			if new_dt > s.maxDt.Scalar() {
				new_dt = s.maxDt.Scalar()
			}
			s.newDt[i] = new_dt
		}
		// Get new timestep
		sort.Float64s(s.newDt)
		nDt := s.newDt[0]
		engine.dt.SetScalar(nDt)
		if !badStep || nDt == s.minDt.Scalar() {
			break
		}
	}
	// Advance step
	e.step.SetScalar(e.step.Scalar() + 1) // advance time step
}

func (s *BDFEulerAuto) Dependencies() (children, parents []string) {
	children = []string{"dt", "bdf_iterations", "t", "step", "badsteps", "maxIterError", "maxIterations"}
	parents = []string{"dt", "mindt", "maxdt"}
	for i := range s.err {
		parents = append(parents, s.maxAbsErr[i].Name())
		parents = append(parents, s.maxRelErr[i].Name())
	}
	return
}

// Register this module
func init() {
	RegisterModule("solver/am01", "Adaptive Adams-Moulton 0+1 solver", LoadBDFEulerAuto)
}

func LoadBDFEulerAuto(e *Engine) {
	s := new(BDFEulerAuto)

	// Minimum/maximum time step
	s.minDt = e.AddNewQuant("mindt", SCALAR, VALUE, Unit("s"), "Minimum time step")
	s.minDt.SetScalar(1e-38)
	s.minDt.SetVerifier(Positive)
	s.maxDt = e.AddNewQuant("maxdt", SCALAR, VALUE, Unit("s"), "Maximum time step")
	s.maxDt.SetVerifier(Positive)
	s.maxDt.SetScalar(1e38)

	s.iterations = e.AddNewQuant("bdf_iterations", SCALAR, VALUE, Unit(""), "Number of iterations per step")
	s.badSteps = e.AddNewQuant("badsteps", SCALAR, VALUE, Unit(""), "Number of time steps that had to be re-done")

	s.maxIterErr = e.AddNewQuant("maxIterError", SCALAR, VALUE, Unit(""), "The maximum error of iterator")
	s.maxIterErr.SetScalar(1e-6)
	s.maxIterErr.SetVerifier(Positive)
	s.maxIter = e.AddNewQuant("maxIterations", SCALAR, VALUE, Unit(""), "Maximum number of evaluations per step")
	s.maxIter.SetScalar(3)
	s.maxIter.SetVerifier(Positive)

	equation := e.equation
	s.ybuffer = make([]*gpu.Array, len(equation))
	s.y0buffer = make([]*gpu.Array, len(equation))
	s.y1buffer = make([]*gpu.Array, len(equation))
	s.dy0buffer = make([]*gpu.Array, len(equation))
	s.dybuffer = make([]*gpu.Array, len(equation))

	s.err = make([]*Quant, len(equation))
	s.err_list = make([]*list.List, len(equation))

	for i := range equation {
		s.err_list[i] = list.New()
	}

	s.maxAbsErr = make([]*Quant, len(equation))
	s.maxRelErr = make([]*Quant, len(equation))
	s.diff = make([]gpu.Reductor, len(equation))
	s.newDt = make([]float64, len(equation))

	for i := range equation {

		eqn := &(equation[i])
		Assert(eqn.kind == EQN_PDE1)
		out := eqn.output[0]
		unit := out.Unit()
		s.err[i] = e.AddNewQuant(out.Name()+"_error", SCALAR, VALUE, unit, "Error/step estimate for "+out.Name())
		s.maxAbsErr[i] = e.AddNewQuant(out.Name()+"_maxAbsError", SCALAR, VALUE, unit, "Maximum absolute error per step for "+out.Name())
		s.maxAbsErr[i].SetScalar(1e-4)
		s.maxRelErr[i] = e.AddNewQuant(out.Name()+"_maxRelError", SCALAR, VALUE, Unit(""), "Maximum relative error per step for "+out.Name())
		s.maxRelErr[i].SetScalar(1e-3)
		s.diff[i].Init(out.Array().NComp(), out.Array().Size3D())

		s.maxAbsErr[i].SetVerifier(Positive)
		s.maxRelErr[i].SetVerifier(Positive)

		// TODO: recycle?

		y := equation[i].output[0]
		s.ybuffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.y0buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.y1buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.dy0buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.dybuffer[i] = Pool.Get(y.NComp(), y.Size3D())

	}
	e.SetSolver(s)
}
