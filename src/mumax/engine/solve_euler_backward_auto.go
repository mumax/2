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
	"math"
	"sort"
)

// Naive Backward Euler solver
type BDFEulerAuto struct {
	ybuffer []*gpu.Array  // current value of the quantity
	y0buffer []*gpu.Array // the value of quantity at the begining of the step
	dy0buffer []*gpu.Array // the value of quantity derivative at the begining of the step
	err      []*Quant     // error estimates for each equation
	maxErr   []*Quant     // maximum error per step for each equation
	maxIterErr []*Quant     // error iterator error estimates for each equation
	maxIter  []*Quant     // maximum number of iterations per step
	newDt    []float64   // 
	diff     []gpu.Reductor
	iterations *Quant
}

func (s *BDFEulerAuto) Step() {
    e := GetEngine()
    t0 := e.time.Scalar()

    isBadStep := 1
    
    equation := e.equation
    // save everything in the begining
    topIdx := len(equation) - 1
    //Debug("Upper bound is", topIdx)
    for i := range equation {
        equation[i].input[0].Update()
        y := equation[i].output[0]
		dy := equation[i].input[0]
        s.y0buffer[i].CopyFromDevice(y.Array()) // save for later
        s.dy0buffer[i].CopyFromDevice(dy.Array()) // save for later
    }

    for isBadStep > 0 { 
    
        isBadStep = 0 
	   	
	    // Then step all outputs (without intermediate updates!)
	    // and invalidate them.
	    // Do initial Euler step with 
        // get dt here to avoid updates later on.
        dt := engine.dt.Scalar()
        // Advance time and update all inputs  
        e.time.SetScalar(t0 + dt)
          
	    for i := range equation {
	        err := 1.0e38
	        iter := 0.0
	        
	        s.iterations.SetScalar(0)
	        // Do zero order approximation
	        y := equation[i].output[0]
		    dy := equation[i].input[0]
		    dyMul := dy.multiplier
		    h := float32(dt * dyMul[0])
		    //s.y0buffer[i].CopyFromDevice(y.Array()) // save for later
		    gpu.Madd(y.Array(), s.y0buffer[i], s.dy0buffer[i], h)
		    y.Invalidate()
		
		    iter = s.iterations.Scalar() + 1
		    s.iterations.SetScalar(iter)
            
		    // Do higher order approximation until converges
		    maxIterErr := s.maxIterErr[i].Scalar()
		    maxIter := s.maxIter[i].Scalar()
		     
	        for err > maxIterErr {
	            
	            equation[i].input[0].Update()
	            y = equation[i].output[0]
		        dy = equation[i].input[0]
	            gpu.Madd(s.ybuffer[i], s.y0buffer[i], dy.Array(), h)
	            s.ybuffer[i].Sync()
	            y.Array().Sync()
	            iterationDiff := s.diff[i].MaxDiff(y.Array(), s.ybuffer[i])
			     
	            err = float64(iterationDiff)
                
			    s.err[i].SetScalar(err)

	            y.Array().CopyFromDevice(s.ybuffer[i])
	            y.Invalidate()
	            iter := s.iterations.Scalar() + 1
	            if iter > maxIter {
	                isBadStep = isBadStep + 1
	                break
	            }
			    s.iterations.SetScalar(iter)
            }
            
            
            // Let us compare Euler and final BDF Euler
            StepErr := float64(s.diff[i].MaxDiff(dy.Array(), s.dy0buffer[i]) * h)
            maxStepErr := s.maxErr[i].Scalar() 
            errRatio := math.Abs(StepErr / maxStepErr)
            Debug("Step error:", errRatio)
            // If iterator reported bad step then correct timestep futher
            if isBadStep > 0 {
                errRatio = errRatio * math.Abs(err / maxIterErr)
            }        
            
            // If correction is to large then there is a badstep
            if (errRatio > 2) {
                isBadStep = isBadStep + 1
            }
            
            step_corr := math.Pow(float64(errRatio), -0.2)
            
            if step_corr > 1.2 {
                step_corr = 1.2
            }
            if step_corr < 0.1 {
                step_corr = 0.1
            }
            if step_corr == 0.0 {
                step_corr = 1.0
            }
            new_dt := dt * step_corr
               
            //Debug("New dt:", new_dt)
            s.newDt[i] = new_dt
	    }
	    
	    sort.Float64s(s.newDt)
	    
	    nDt := s.newDt[topIdx]
	    engine.dt.SetScalar(nDt) 
    }
	// Advance step	
	e.step.SetScalar(e.step.Scalar() + 1) // advance time step
}

func (s *BDFEulerAuto) Dependencies() (children, parents []string) {
	children = []string{"dt", "bdf_iterations", "t", "step"}
	parents = []string{"dt"}
	for i := range s.err {
		parents = append(parents, s.maxErr[i].Name())
		parents = append(parents, s.maxIter[i].Name())
		parents = append(parents, s.maxIterErr[i].Name())
	}
	return
}

// Register this module
func init() {
	RegisterModule("solver/bdf_euler_auto", "Multi-step Backward Euler solver", LoadBDFEulerAuto)
}

func LoadBDFEulerAuto(e *Engine) {
    s := new(BDFEulerAuto)
	s.iterations = e.AddNewQuant("bdf_iterations", SCALAR, VALUE, Unit(""), "Number of iterations per step")
    
	equation := e.equation
	s.ybuffer = make([]*gpu.Array, len(equation))
	s.y0buffer = make([]*gpu.Array, len(equation))
	s.dy0buffer = make([]*gpu.Array, len(equation))
	
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
		s.dy0buffer[i] = Pool.Get(y.NComp(), y.Size3D())

	}
	e.SetSolver(s)
}
