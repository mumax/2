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
	"math"
	. "mumax/common"
	"mumax/gpu"
	"sort"
	"container/list"
)

// Naive Backward Euler solver
type BDFEulerAuto struct {
	ybuffer []*gpu.Array // current value of the quantity

	y0buffer []*gpu.Array // the value of quantity at the begining of the step
	y1buffer []*gpu.Array // the value of quantity after pedictor step

	dy0buffer  []*gpu.Array // the value of quantity derivative at the begining of the step
	dybuffer   []*gpu.Array // the buffer for quantity derivative 
	err        []*Quant     // error estimates for each equation
	maxErr     []*Quant     // maximum error per step for each equation
	maxIterErr []*Quant     // error iterator error estimates for each equation
	maxIter    []*Quant     // maximum number of iterations per step
	newDt      []float64    // 
	diff       []gpu.Reductor
	err_list   *list.List
	iterations *Quant
	badSteps   *Quant
	minDt      *Quant
	maxDt      *Quant
}

func (s *BDFEulerAuto) Step() {
	//Debug("Enter solver")
	e := GetEngine()
	t0 := e.time.Scalar()

	s.badSteps.SetScalar(0)
	s.iterations.SetScalar(0)

	isBadStep := 1
	equation := e.equation
	// save everything in the begining
	topIdx := len(equation) - 1
	const headRoom = 0.8
	//Debug("Update quantities")
	for i := range equation {
		equation[i].input[0].Update()
		y := equation[i].output[0]
		dy := equation[i].input[0]
		s.y0buffer[i].CopyFromDevice(y.Array())   // save for later
		s.dy0buffer[i].CopyFromDevice(dy.Array()) // save for later
	}

	for isBadStep > 0 {
		isBadStep = 0
		// get dt here to avoid updates later on.
		dt := engine.dt.Scalar()
		// Advance time and update all inputs  
		e.time.SetScalar(t0 + dt)
		iter := 0

		for i := range equation {
			//Debug("Do forward Euler")	         
			// Do zero order approximation with Euler method
			y := equation[i].output[0]
			dy := equation[i].input[0]
			dyMul := dy.multiplier
			t_step := dt * dyMul[0]
			h := float32(t_step)
			gpu.Madd(y.Array(), s.y0buffer[i], s.dy0buffer[i], h)
			y.Invalidate()
			iter = iter + 1
			s.iterations.SetScalar(s.iterations.Scalar() + 1)
			// Do higher order approximation until converges    
			maxIterErr := t_step / s.maxIterErr[i].Scalar()
			maxIter := int(s.maxIter[i].Scalar())
			iter = 0
			err := 1.0e38
			// Do predictor: BDF Euler
			// Debug("Do backward Euler")
			equation[i].input[0].Update()
			for err > maxIterErr {
				gpu.Madd(s.ybuffer[i], s.y0buffer[i], dy.Array(), h)
				iterationDiff := s.diff[i].MaxDiff(y.Array(), s.ybuffer[i])
				err_y := float64(iterationDiff)
				s.err[i].SetScalar(err)
				y.Array().CopyFromDevice(s.ybuffer[i])
				s.dybuffer[i].CopyFromDevice(dy.Array()) // save for later 
				y.Invalidate()
				equation[i].input[0].Update()
				err_dy := float64(s.diff[i].MaxDiff(dy.Array(), s.dybuffer[i]))
				err = err_y / err_dy
				iter = iter + 1
				s.iterations.SetScalar(s.iterations.Scalar() + 1)
				if iter > maxIter {
					isBadStep = isBadStep + 1
					s.badSteps.SetScalar(s.badSteps.Scalar() + 1)
					//Debug("Badstep in Euler")
					break
				}
			}
			s.y1buffer[i].CopyFromDevice(y.Array()) // save for later                       

			iter = 0
			err = 1e38
			//Debug("Do trapezodila")
			// Do corrector: BDF trapezoidal
			for err > maxIterErr {
				gpu.Add(s.ybuffer[i], dy.Array(), s.dy0buffer[i])
				gpu.Madd(s.ybuffer[i], s.y0buffer[i], s.ybuffer[i], 0.5*h)
				iterationDiff := s.diff[i].MaxDiff(y.Array(), s.ybuffer[i])
				err_y := float64(iterationDiff)
				s.err[i].SetScalar(err)
				y.Array().CopyFromDevice(s.ybuffer[i])
				s.dybuffer[i].CopyFromDevice(dy.Array()) // save for later 
				y.Invalidate()
				equation[i].input[0].Update()
				err_dy := float64(s.diff[i].MaxDiff(dy.Array(), s.dybuffer[i]))
				err = err_y / err_dy
				iter = iter + 1
				s.iterations.SetScalar(s.iterations.Scalar() + 1)
				if iter > 2 {
					isBadStep = isBadStep + 1
					s.badSteps.SetScalar(s.badSteps.Scalar() + 1)
					//Debug("Badstep in Trapezoidal")
					break
				}
			}
           
			maxStepErr := s.maxErr[i].Scalar()
			StepErr := float64(s.diff[i].MaxDiff(y.Array(), s.y1buffer[i]))
			
			s.err_list.PushFront(StepErr)
			s.err_list.Remove(s.err_list.Back())
			
			if StepErr == 0.0 {
				StepErr = headRoom * headRoom * maxStepErr
			}
			// step is large then threshould then badstep is reported
			if StepErr > maxStepErr {
				isBadStep = isBadStep + 1
				s.badSteps.SetScalar(s.badSteps.Scalar() + 1)
				//Debug("General Badstep")
			}
			// Let us compare BDF Euler and BDF Trapezoidal
			// to guess next time step            
			errRatio := 0.0
			
			if e.step.Scalar() > 3 {
			    //do softcore adjustment
			    elem := s.err_list.Front()
			    e := elem.Value.(float64)
			    ep1 := elem.Next().Value.(float64)
			    ep2 := elem.Next().Value.(float64)
			    pre_f1 := math.Pow((ep1/e),0.075)
			    pre_f2 := math.Pow((ep1*ep1/(e*ep2)), 0.01)
			    pre := math.Pow((maxStepErr / e), 0.175)
			    errRatio = pre_f1 * pre * pre_f2
			    //Debug(e,ep1,ep2)
			} else {
			    //do hardcore adjustment
			    errRatio = headRoom * maxStepErr / StepErr
			}
			
			step_corr := math.Abs(errRatio)

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
		//Debug("Get new timestep")
		sort.Float64s(s.newDt)
		nDt := s.newDt[topIdx]
		engine.dt.SetScalar(nDt)
		if isBadStep >= 3 || nDt == s.minDt.Scalar() {
			break //give up
		}
	}

	// Advance step	
	e.step.SetScalar(e.step.Scalar() + 1) // advance time step
}

func (s *BDFEulerAuto) Dependencies() (children, parents []string) {
	children = []string{"dt", "bdf_iterations", "t", "step", "badsteps"}
	parents = []string{"dt", "mindt", "maxdt"}
	for i := range s.err {
		parents = append(parents, s.maxErr[i].Name())
		parents = append(parents, s.maxIter[i].Name())
		parents = append(parents, s.maxIterErr[i].Name())
	}
	return
}

// Register this module
func init() {
	RegisterModule("solver/bdf-euler-auto", "Multi-step Backward Euler solver", LoadBDFEulerAuto)
}

func LoadBDFEulerAuto(e *Engine) {
	s := new(BDFEulerAuto)
    s.err_list = list.New()
    
    s.err_list.PushFront(0.0)
    s.err_list.PushFront(0.0)
    s.err_list.PushFront(0.0)
    
	// Minimum/maximum time step
	s.minDt = e.AddNewQuant("mindt", SCALAR, VALUE, Unit("s"), "Minimum time step")
	s.minDt.SetScalar(1e-38)
	s.minDt.SetVerifier(Positive)
	s.maxDt = e.AddNewQuant("maxdt", SCALAR, VALUE, Unit("s"), "Maximum time step")
	s.maxDt.SetVerifier(Positive)
	s.maxDt.SetScalar(1e38)

	s.iterations = e.AddNewQuant("bdf_iterations", SCALAR, VALUE, Unit(""), "Number of iterations per step")
	s.badSteps = e.AddNewQuant("badsteps", SCALAR, VALUE, Unit(""), "Number of time steps that had to be re-done")

	equation := e.equation
	s.ybuffer = make([]*gpu.Array, len(equation))
	s.y0buffer = make([]*gpu.Array, len(equation))
	s.y1buffer = make([]*gpu.Array, len(equation))
	s.dy0buffer = make([]*gpu.Array, len(equation))
	s.dybuffer = make([]*gpu.Array, len(equation))

	s.err = make([]*Quant, len(equation))
	s.maxErr = make([]*Quant, len(equation))
	s.maxIterErr = make([]*Quant, len(equation))
	s.maxIter = make([]*Quant, len(equation))
	s.diff = make([]gpu.Reductor, len(equation))
	s.newDt = make([]float64, len(equation))

	for i := range equation {

		eqn := &(equation[i])
		Assert(eqn.kind == EQN_PDE1)
		out := eqn.output[0]
		unit := out.Unit()
		s.err[i] = e.AddNewQuant(out.Name()+"_error", SCALAR, VALUE, unit, "Error/step estimate for "+out.Name())
		s.maxErr[i] = e.AddNewQuant(out.Name()+"_maxError", SCALAR, VALUE, unit, "Maximum error/step for "+out.Name())
		s.maxIterErr[i] = e.AddNewQuant(out.Name()+"_maxIterError", SCALAR, VALUE, unit, "The maximum error of iterator"+out.Name())
		s.maxIter[i] = e.AddNewQuant(out.Name()+"_maxIterations", SCALAR, VALUE, unit, "Maximum number of iterations per step"+out.Name())
		s.diff[i].Init(out.Array().NComp(), out.Array().Size3D())

		s.maxErr[i].SetVerifier(Positive)
		s.maxIterErr[i].SetVerifier(Positive)
		s.maxIter[i].SetVerifier(Positive)

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
