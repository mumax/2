//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// This file implements the adaptive Bogacki-Shampine scheme (RK32), the adaptive Cash-Karp {RK4(5)5CK}
// scheme, the adaptive Fehlberg {RK4(5)5F} scheme, the adaptive Dormand-Prince scheme {RK5(4)7FC},
// the adaptive Dormand-Prince scheme {RK5(4)7M}, and the adaptive Dormand-Prince scheme {RK5(4)7S}
//
// Details of RK4(5)5CK can be found in J. R. Cash, and A. H. Karp, "A variable order Runge-Kutta method
// for initial value problems with rapidly varying right-hand sides," ACM Transactions on Mathematical
// Software vol. 16, no. 3, p. 201-222, 1990
//
// Details of RK4(5)5F can be found in E. Fehlberg, "Low-order classical Runge-Kutta formulas with
// step size control and their application to some heat transfer problems," NASA Technical Report 315,
// 1969
//
// Details of RK5(4)7FC can be found in J. R. Domand, and P. J. Prince, "A reconsideration of some
// embedded Runge-Kutta formulae," J. of Computational and Applied Mathematics vol. 15, issue 2
// p. 203-211, 1986
//
// Details of RK5(4)7M and RK5(4)7S can be found in J. R. Domand, and P. J. Prince, "A family of
// embedded Runge-Kutta formulae," J. of Computational and Applied Mathematics vol. 6, no. 1
// p. 19-26, 1980
//

// Author: Xuanyao Fong (Kelvin) - Email: xfong@ecn.purdue.edu, xuanyao.fong@gmail.com

import (
	. "mumax/common"
	"mumax/gpu"
	"math"
)

type RK32Solver struct {
	y0buffer  []*gpu.Array // value of the quantity at the beginning of the step

	dy0buffer []*gpu.Array // value of the quantity derivative at the beginning of the step
	dy0Mul    []float64
	dy1buffer []*gpu.Array // value of quantity derivative at first intermediate step
	dy2buffer []*gpu.Array // value of quantity derivative at second intermediate step
	dy3buffer []*gpu.Array // value of quantity derivative at third intermediate step
	dybuffer  []*gpu.Array // the buffer for error estimate from quantity derivative
	dyhbuffer []*gpu.Array // the buffer for quantity derivative for high order corrector

	// Hard code tableau for RK methods
	tableauA [3][3]float32
//	tableauB [2][4]float32 // Not needed since the high order term is computed in the last iteration and we only need the error estimate
	tableauC [3]float64

	errTab   [4]float32   // Tableau for error estimate

	err      []*Quant     // error estimates for each equation
	peakErr  []*Quant     // maximum error for each equation
	maxErr   []*Quant     // maximum error for each equation
	relErr   []*Quant     // relative error tolerance for each equation
	diff     []gpu.Reductor
	minDt    *Quant
	maxDt    *Quant
	badSteps *Quant
	maxTry   int          // maximum number of iterations per step to achieve convergence

	order    float64
	headRoom float64
}

type RK45Solver struct {
	y0buffer  []*gpu.Array // value of the quantity at the beginning of the step

	dy0buffer []*gpu.Array // value of the quantity derivative at the beginning of the step
	dy0Mul    []float64
	dy1buffer []*gpu.Array // value of quantity derivative at first intermediate step
	dy2buffer []*gpu.Array // value of quantity derivative at second intermediate step
	dy3buffer []*gpu.Array // value of quantity derivative at third intermediate step
	dy4buffer []*gpu.Array // value of quantity derivative at fourth intermediate step
	dy5buffer []*gpu.Array // value of quantity derivative at fifth intermediate step
	dybuffer  []*gpu.Array // the buffer for error estimate from quantity derivative
	dyhbuffer []*gpu.Array // the buffer for quantity derivative

	// Hard code tableau for RK methods
	tableauA [5][5]float32
	tableauB [6]float32    // Need only for high order term since errTab is used for computing errors
//	tableauB [2][6]float32
	tableauC [5]float64

	errTab   [6]float32   // Tableau for error estimate

	err      []*Quant     // error estimates for each equation
	peakErr  []*Quant     // maximum error for each equation
	maxErr   []*Quant     // maximum error for each equation
	relErr   []*Quant     // relative error tolerance for each equation
	diff     []gpu.Reductor
	minDt    *Quant
	maxDt    *Quant
	badSteps *Quant
	maxTry   int

	order    float64
	headRoom float64
}

type RKF54Solver struct {
	y0buffer  []*gpu.Array // value of the quantity at the beginning of the step

	dy0buffer []*gpu.Array // value of the quantity derivative at the beginning of the step
	dy0Mul    []float64
	dy1buffer []*gpu.Array // value of quantity derivative at first intermediate step
	dy2buffer []*gpu.Array // value of quantity derivative at second intermediate step
	dy3buffer []*gpu.Array // value of quantity derivative at third intermediate step
	dy4buffer []*gpu.Array // value of quantity derivative at fourth intermediate step
	dy5buffer []*gpu.Array // value of quantity derivative at fifth intermediate step
	dy6buffer []*gpu.Array // value of quantity derivative at sixth intermediate step
	dybuffer []*gpu.Array // the buffer for error estimate from quantity derivative
	dyhbuffer []*gpu.Array // the buffer for quantity derivative

	// Hard code tableau for RK methods
	tableauA [6][6]float32
//	tableauB [2][7]float32 // Not needed since the high order term is computed in the last iteration and we only need the error estimate
	tableauC [6]float64

	errTab [7]float32     // Tableau for error estimate

	err      []*Quant     // error estimates for each equation
	peakErr  []*Quant     // maximum error for each equation
	maxErr   []*Quant     // maximum error for each equation
	relErr   []*Quant     // relative error tolerance for each equation
	diff     []gpu.Reductor
	minDt    *Quant
	maxDt    *Quant
	badSteps *Quant
	maxTry   int          // maximum number of iterations per step to achieve convergence

	order float64
	headRoom float64
}

// Load the solvers into the Engine
func LoadRK32(e *Engine) {
	s := new(RK32Solver)

	// Minimum/maximum time step
	s.minDt = e.AddNewQuant("mindt", SCALAR, VALUE, Unit("s"), "Minimum time step")
	s.minDt.SetScalar(1e-38)
	s.minDt.SetVerifier(Positive)
	s.maxDt = e.AddNewQuant("maxdt", SCALAR, VALUE, Unit("s"), "Maximum time step")
	s.maxDt.SetVerifier(Positive)
	s.maxDt.SetScalar(1e38)
	s.badSteps = e.AddNewQuant("badsteps", SCALAR, VALUE, Unit(""), "Number of time steps that had to be re-done")

	equation := e.equation
	s.y0buffer = make([]*gpu.Array, len(equation))

	s.dybuffer = make([]*gpu.Array, len(equation))
	s.dyhbuffer = make([]*gpu.Array, len(equation))
	s.dy0buffer = make([]*gpu.Array, len(equation))
	s.dy0Mul = make([]float64, len(equation))
	s.dy1buffer = make([]*gpu.Array, len(equation))
	s.dy2buffer = make([]*gpu.Array, len(equation))
	s.dy3buffer = make([]*gpu.Array, len(equation))

	s.err = make([]*Quant, len(equation))
	s.peakErr = make([]*Quant, len(equation))
	s.maxErr = make([]*Quant, len(equation))
	s.relErr = make([]*Quant, len(equation))
	s.diff = make([]gpu.Reductor, len(equation))

	// Hard code Butcher tableau for the RK scheme
	// Tableau A
	s.tableauA[0][0], s.tableauA[0][1], s.tableauA[0][2] = 1.0 / 2.0,       0.0,       0.0
	s.tableauA[1][0], s.tableauA[1][1], s.tableauA[1][2] =       0.0, 3.0 / 4.0,       0.0
	s.tableauA[2][0], s.tableauA[2][1], s.tableauA[2][2] = 2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0

	// Tableau B (Optimized for Bogacki-Shampine scheme)
//	s.tableauB[0][0], s.tableauB[0][1], s.tableauB[0][2], s.tableauB[0][3] = 7.0 / 24.0, 1.0 / 4.0, 1.0 / 3.0, 1.0 / 8.0
//	s.tableauB[1][0], s.tableauB[1][1], s.tableauB[1][2], s.tableauB[1][3] =  2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0,       0.0

	// Tableau C
	s.tableauC[0], s.tableauC[1], s.tableauC[2] = 1.0 / 2.0, 3.0 / 4.0, 1.0

	// Tableau for computing error estimate
	s.errTab[0], s.errTab[1], s.errTab[2], s.errTab[3] = 5.0 / 72.0, -1.0 / 12.0, -1.0 / 9.0, 1.0 / 8.0

	e.SetSolver(s)
	s.order = 2.0
	s.headRoom = 0.8
	s.maxTry = 3

	for i := range equation {

		eqn := &(equation[i])
		Assert(eqn.kind == EQN_PDE1)
		out := eqn.output[0]
		unit := out.Unit()
		s.err[i] = e.AddNewQuant(out.Name()+"_error", SCALAR, VALUE, unit, "Error/step estimate for "+out.Name())
		s.peakErr[i] = e.AddNewQuant(out.Name()+"_peakerror", SCALAR, VALUE, unit, "All-time maximum error/step for "+out.Name())
		s.maxErr[i] = e.AddNewQuant(out.Name()+"_maxError", SCALAR, VALUE, unit, "Maximum error/step for "+out.Name())
		s.relErr[i] = e.AddNewQuant(out.Name()+"_relError", SCALAR, VALUE, unit, "Relative error/step for "+out.Name())
		s.diff[i].Init(out.Array().NComp(), out.Array().Size3D())
		s.maxErr[i].SetVerifier(Positive)

		// TODO: recycle?
		y := equation[i].output[0]
		s.y0buffer[i] = Pool.Get(y.NComp(), y.Size3D())

		s.dy0Mul[i] = 0.0
		s.dybuffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.dyhbuffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.dy0buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.dy1buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.dy2buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.dy3buffer[i] = Pool.Get(y.NComp(), y.Size3D())

	}
}

func LoadRK45CK(e *Engine) {
	s0 := new(RK45Solver)

	// Minimum/maximum time step
	s0.minDt = e.AddNewQuant("mindt", SCALAR, VALUE, Unit("s"), "Minimum time step")
	s0.minDt.SetScalar(1e-38)
	s0.minDt.SetVerifier(Positive)
	s0.maxDt = e.AddNewQuant("maxdt", SCALAR, VALUE, Unit("s"), "Maximum time step")
	s0.maxDt.SetVerifier(Positive)
	s0.maxDt.SetScalar(1e38)
	s0.badSteps = e.AddNewQuant("badsteps", SCALAR, VALUE, Unit(""), "Number of time steps that had to be re-done")

	equation := e.equation
	s0.y0buffer = make([]*gpu.Array, len(equation))

	s0.dybuffer = make([]*gpu.Array, len(equation))
	s0.dyhbuffer = make([]*gpu.Array, len(equation))
	s0.dy0buffer = make([]*gpu.Array, len(equation))
	s0.dy0Mul = make([]float64, len(equation))
	s0.dy1buffer = make([]*gpu.Array, len(equation))
	s0.dy2buffer = make([]*gpu.Array, len(equation))
	s0.dy3buffer = make([]*gpu.Array, len(equation))
	s0.dy4buffer = make([]*gpu.Array, len(equation))
	s0.dy5buffer = make([]*gpu.Array, len(equation))

	s0.err = make([]*Quant, len(equation))
	s0.peakErr = make([]*Quant, len(equation))
	s0.maxErr = make([]*Quant, len(equation))
	s0.relErr = make([]*Quant, len(equation))
	s0.diff = make([]gpu.Reductor, len(equation))

	// Hard code Butcher tableau for the RK scheme
	// Tableau A
	s0.tableauA[0][0], s0.tableauA[0][1], s0.tableauA[0][2], s0.tableauA[0][3], s0.tableauA[0][4] =    1.0 /     5.0,           0.0,             0.0,                0.0,            0.0
	s0.tableauA[1][0], s0.tableauA[1][1], s0.tableauA[1][2], s0.tableauA[1][3], s0.tableauA[1][4] =    3.0 /    40.0,   9.0 /  40.0,             0.0,                0.0,            0.0
	s0.tableauA[2][0], s0.tableauA[2][1], s0.tableauA[2][2], s0.tableauA[2][3], s0.tableauA[2][4] =    3.0 /    10.0,  -9.0 /  10.0,   6.0 /     5.0,                0.0,            0.0
	s0.tableauA[3][0], s0.tableauA[3][1], s0.tableauA[3][2], s0.tableauA[3][3], s0.tableauA[3][4] =  -11.0 /    54.0,   5.0 /   2.0, -70.0 /    27.0,    35.0 /     27.0,            0.0
	s0.tableauA[4][0], s0.tableauA[4][1], s0.tableauA[4][2], s0.tableauA[4][3], s0.tableauA[4][4] = 1631.0 / 55296.0, 175.0 / 512.0, 575.0 / 13824.0, 44275.0 / 110592.0, 253.0 / 4096.0

	// Tableau B
	s0.tableauB[0], s0.tableauB[1], s0.tableauB[2], s0.tableauB[3], s0.tableauB[4], s0.tableauB[5] =   37.0 /   378.0, 0.0,   250.0 /   621.0,   125.0 /   594.0,               0.0, 512.0 / 1771.0
//	s0.tableauB[0][0], s0.tableauB[0][1], s0.tableauB[0][2], s0.tableauB[0][3], s0.tableauB[0][4], s0.tableauB[0][5] = 2825.0 / 27648.0, 0.0, 18575.0 / 48384.0, 13525.0 / 55296.0,   277.0 / 14336.0,   1.0 /    4.0
//	s0.tableauB[1][0], s0.tableauB[1][1], s0.tableauB[1][2], s0.tableauB[1][3], s0.tableauB[1][4], s0.tableauB[1][5] =   37.0 /   378.0, 0.0,   250.0 /   621.0,   125.0 /   594.0,               0.0, 512.0 / 1771.0

	// Tableau C
	s0.tableauC[0], s0.tableauC[1], s0.tableauC[2], s0.tableauC[3], s0.tableauC[4] = 1.0 / 5.0, 3.0 / 10.0,  3.0 / 5.0, 1.0, 7.0 / 8.0

	// Tableau for computing error estimate
	s0.errTab[0], s0.errTab[1], s0.errTab[2], s0.errTab[3], s0.errTab[4], s0.errTab[5] = -277.0 / 64512.0, 0.0, 6925.0 / 370944.0, -6925.0 / 202752.0, -277.0 / 14336.0, 277.0 / 7084.0

	e.SetSolver(s0)
	s0.order = 4.0
	s0.headRoom = 0.85
	s0.maxTry = 3

	for i := range equation {

		eqn := &(equation[i])
		Assert(eqn.kind == EQN_PDE1)
		out := eqn.output[0]
		unit := out.Unit()
		s0.err[i] = e.AddNewQuant(out.Name()+"_error", SCALAR, VALUE, unit, "Error/step estimate for "+out.Name())
		s0.peakErr[i] = e.AddNewQuant(out.Name()+"_peakerror", SCALAR, VALUE, unit, "All-time maximum error/step for "+out.Name())
		s0.maxErr[i] = e.AddNewQuant(out.Name()+"_maxError", SCALAR, VALUE, unit, "Maximum error/step for "+out.Name())
		s0.relErr[i] = e.AddNewQuant(out.Name()+"_relError", SCALAR, VALUE, unit, "Relative error/step for "+out.Name())
		s0.diff[i].Init(out.Array().NComp(), out.Array().Size3D())
		s0.maxErr[i].SetVerifier(Positive)

		// TODO: recycle?
		y := equation[i].output[0]
		s0.y0buffer[i] = Pool.Get(y.NComp(), y.Size3D())

		s0.dy0Mul[i] = 0.0
		s0.dybuffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s0.dyhbuffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s0.dy0buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s0.dy1buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s0.dy2buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s0.dy3buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s0.dy4buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s0.dy5buffer[i] = Pool.Get(y.NComp(), y.Size3D())

	}
}

func LoadRK45F(e *Engine) {
	s := new(RK45Solver)

	// Minimum/maximum time step
	s.minDt = e.AddNewQuant("mindt", SCALAR, VALUE, Unit("s"), "Minimum time step")
	s.minDt.SetScalar(1e-38)
	s.minDt.SetVerifier(Positive)
	s.maxDt = e.AddNewQuant("maxdt", SCALAR, VALUE, Unit("s"), "Maximum time step")
	s.maxDt.SetVerifier(Positive)
	s.maxDt.SetScalar(1e38)
	s.badSteps = e.AddNewQuant("badsteps", SCALAR, VALUE, Unit(""), "Number of time steps that had to be re-done")

	equation := e.equation
	s.y0buffer = make([]*gpu.Array, len(equation))

	s.dybuffer = make([]*gpu.Array, len(equation))
	s.dyhbuffer = make([]*gpu.Array, len(equation))
	s.dy0buffer = make([]*gpu.Array, len(equation))
	s.dy0Mul = make([]float64, len(equation))
	s.dy1buffer = make([]*gpu.Array, len(equation))
	s.dy2buffer = make([]*gpu.Array, len(equation))
	s.dy3buffer = make([]*gpu.Array, len(equation))
	s.dy4buffer = make([]*gpu.Array, len(equation))
	s.dy5buffer = make([]*gpu.Array, len(equation))

	s.err = make([]*Quant, len(equation))
	s.peakErr = make([]*Quant, len(equation))
	s.maxErr = make([]*Quant, len(equation))
	s.relErr = make([]*Quant, len(equation))
	s.diff = make([]gpu.Reductor, len(equation))

	// Hard code Butcher tableau for the RK scheme
	// Tableau A
	s.tableauA[0][0], s.tableauA[0][1], s.tableauA[0][2], s.tableauA[0][3], s.tableauA[0][4] =    1.0 /    4.0,              0.0,              0.0,             0.0,          0.0
	s.tableauA[1][0], s.tableauA[1][1], s.tableauA[1][2], s.tableauA[1][3], s.tableauA[1][4] =    3.0 /   32.0,     9.0 /   32.0,              0.0,             0.0,          0.0
	s.tableauA[2][0], s.tableauA[2][1], s.tableauA[2][2], s.tableauA[2][3], s.tableauA[2][4] = 1932.0 / 2197.0, -7200.0 / 2197.0,  7296.0 / 2197.0,             0.0,          0.0
	s.tableauA[3][0], s.tableauA[3][1], s.tableauA[3][2], s.tableauA[3][3], s.tableauA[3][4] =  439.0 /  216.0,             -8.0,  3680.0 /  513.0, -845.0 / 4104.0,          0.0
	s.tableauA[4][0], s.tableauA[4][1], s.tableauA[4][2], s.tableauA[4][3], s.tableauA[4][4] =   -8.0 /   27.0,              2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0

	// Tableau B
	s.tableauB[0], s.tableauB[1], s.tableauB[2], s.tableauB[3], s.tableauB[4], s.tableauB[5] = 25.0 / 216.0, 0.0, 1408.0 /  2565.0,  2197.0 /  4104.0, -1.0 /  5.0,         0.0
//	s.tableauB[0][0], s.tableauB[0][1], s.tableauB[0][2], s.tableauB[0][3], s.tableauB[0][4], s.tableauB[0][5] = 16.0 / 135.0, 0.0, 6656.0 / 12825.0, 28561.0 / 56430.0, -9.0 / 50.0,  2.0 / 55.0
//	s.tableauB[1][0], s.tableauB[1][1], s.tableauB[1][2], s.tableauB[1][3], s.tableauB[1][4], s.tableauB[1][5] = 25.0 / 216.0, 0.0, 1408.0 /  2565.0,  2197.0 /  4104.0, -1.0 /  5.0,         0.0

	// Tableau C
	s.tableauC[0], s.tableauC[1], s.tableauC[2], s.tableauC[3], s.tableauC[4] = 1.0 / 4.0, 3.0 / 8.0, 12.0 / 13.0, 1.0, 1.0 / 2.0

	// Tableau for computing error estimate
	s.errTab[0], s.errTab[1], s.errTab[2], s.errTab[3], s.errTab[4], s.errTab[5] = -1.0 / 360.0, 0.0, 2432.0 / 81225.0, 41743.0 / 1429560.0, -1.0 / 50.0, -2.0 / 55.0

	e.SetSolver(s)
	s.order = 4.0
	s.headRoom = 0.85
	s.maxTry = 3

	for i := range equation {

		eqn := &(equation[i])
		Assert(eqn.kind == EQN_PDE1)
		out := eqn.output[0]
		unit := out.Unit()
		s.err[i] = e.AddNewQuant(out.Name()+"_error", SCALAR, VALUE, unit, "Error/step estimate for "+out.Name())
		s.peakErr[i] = e.AddNewQuant(out.Name()+"_peakerror", SCALAR, VALUE, unit, "All-time maximum error/step for "+out.Name())
		s.maxErr[i] = e.AddNewQuant(out.Name()+"_maxError", SCALAR, VALUE, unit, "Maximum error/step for "+out.Name())
		s.relErr[i] = e.AddNewQuant(out.Name()+"_relError", SCALAR, VALUE, unit, "Relative error/step for "+out.Name())
		s.diff[i].Init(out.Array().NComp(), out.Array().Size3D())
		s.maxErr[i].SetVerifier(Positive)

		// TODO: recycle?
		y := equation[i].output[0]
		s.y0buffer[i] = Pool.Get(y.NComp(), y.Size3D())

		s.dy0Mul[i] = 0.0
		s.dybuffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.dyhbuffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.dy0buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.dy1buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.dy2buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.dy3buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.dy4buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.dy5buffer[i] = Pool.Get(y.NComp(), y.Size3D())

	}
}

func LoadRKF54(e *Engine) {
	s0 := new(RKF54Solver)

	// Minimum/maximum time step
	s0.minDt = e.AddNewQuant("mindt", SCALAR, VALUE, Unit("s"), "Minimum time step")
	s0.minDt.SetScalar(1e-38)
	s0.minDt.SetVerifier(Positive)
	s0.maxDt = e.AddNewQuant("maxdt", SCALAR, VALUE, Unit("s"), "Maximum time step")
	s0.maxDt.SetVerifier(Positive)
	s0.maxDt.SetScalar(1e38)
	s0.badSteps = e.AddNewQuant("badsteps", SCALAR, VALUE, Unit(""), "Number of time steps that had to be re-done")

	equation := e.equation
	s0.y0buffer = make([]*gpu.Array, len(equation))

	s0.dybuffer = make([]*gpu.Array, len(equation))
	s0.dyhbuffer = make([]*gpu.Array, len(equation))
	s0.dy0buffer = make([]*gpu.Array, len(equation))
	s0.dy0Mul = make([]float64, len(equation))
	s0.dy1buffer = make([]*gpu.Array, len(equation))
	s0.dy2buffer = make([]*gpu.Array, len(equation))
	s0.dy3buffer = make([]*gpu.Array, len(equation))
	s0.dy4buffer = make([]*gpu.Array, len(equation))
	s0.dy5buffer = make([]*gpu.Array, len(equation))
	s0.dy6buffer = make([]*gpu.Array, len(equation))

	s0.err = make([]*Quant, len(equation))
	s0.peakErr = make([]*Quant, len(equation))
	s0.maxErr = make([]*Quant, len(equation))
	s0.relErr = make([]*Quant, len(equation))
	s0.diff = make([]gpu.Reductor, len(equation))

	// Hard code Butcher tableau for the RK scheme
	// Tableau A
	s0.tableauA[0][0], s0.tableauA[0][1], s0.tableauA[0][2], s0.tableauA[0][3], s0.tableauA[0][4], s0.tableauA[0][5] =    1.0 /    5.0,            0.0,              0.0,                0.0,             0.0,           0.0
	s0.tableauA[1][0], s0.tableauA[1][1], s0.tableauA[1][2], s0.tableauA[1][3], s0.tableauA[1][4], s0.tableauA[1][5] =    3.0 /   40.0,   9.0 /   40.0,              0.0,                0.0,             0.0,           0.0
	s0.tableauA[2][0], s0.tableauA[2][1], s0.tableauA[2][2], s0.tableauA[2][3], s0.tableauA[2][4], s0.tableauA[2][5] =  264.0 / 2197.0, -90.0 / 2197.0,  840.0 /  2197.0,                0.0,             0.0,           0.0
	s0.tableauA[3][0], s0.tableauA[3][1], s0.tableauA[3][2], s0.tableauA[3][3], s0.tableauA[3][4], s0.tableauA[3][5] =  932.0 / 3645.0, -14.0 /   27.0, 3256.0 /  5103.0,   7436.0 / 25515.0,             0.0,           0.0
	s0.tableauA[4][0], s0.tableauA[4][1], s0.tableauA[4][2], s0.tableauA[4][3], s0.tableauA[4][4], s0.tableauA[4][5] = -367.0 /  513.0,  30.0 /   19.0, 9940.0 /  5643.0, -29575.0 /  8208.0, 6615.0 / 3344.0,           0.0
	s0.tableauA[5][0], s0.tableauA[5][1], s0.tableauA[5][2], s0.tableauA[5][3], s0.tableauA[5][4], s0.tableauA[5][5] =   35.0 /  432.0,            0.0, 8500.0 / 14553.0, -28561.0 / 84672.0,  405.0 /  704.0,  19.0 / 196.0

	// Tableau B
//	s0.tableauB[0][0], s0.tableauB[0][1], s0.tableauB[0][2], s0.tableauB[0][3], s0.tableauB[0][4], s0.tableauB[0][5], s0.tableauB[0][6] = 35.0 / 432.0, 0.0,  8500.0 / 14553.0, -28561.0 / 84672.0, 405.0 / 704.0,   19.0 /  196.0,         0.0
//	s0.tableauB[1][0], s0.tableauB[1][1], s0.tableauB[1][2], s0.tableauB[1][3], s0.tableauB[1][4], s0.tableauB[1][5], s0.tableauB[1][6] = 11.0 / 108.0, 0.0,  6250.0 / 14553.0,  -2197.0 / 21168.0,  81.0 / 176.0,  171.0 / 1960.0,  1.0 / 40.0

	// Tableau C
	s0.tableauC[0], s0.tableauC[1], s0.tableauC[2], s0.tableauC[3], s0.tableauC[4], s0.tableauC[5] = 1.0 / 5.0, 3.0 / 10.0, 6.0 / 13.0, 2.0 / 3.0, 1.0, 1.0

	// Tableau for computing error estimate
	s0.errTab[0], s0.errTab[1], s0.errTab[2], s0.errTab[3], s0.errTab[4], s0.errTab[5], s0.errTab[6] = -1.0 / 48.0, 0.0, 250.0 / 1617.0, -2197.0 / 9408.0, 81.0 / 704.0, 19.0 / 1960.0, -1.0 / 40.0

	e.SetSolver(s0)
	s0.order = 5.0
	s0.headRoom = 0.9
	s0.maxTry = 5

	for i := range equation {

		eqn := &(equation[i])
		Assert(eqn.kind == EQN_PDE1)
		out := eqn.output[0]
		unit := out.Unit()
		s0.err[i] = e.AddNewQuant(out.Name()+"_error", SCALAR, VALUE, unit, "Error/step estimate for "+out.Name())
		s0.peakErr[i] = e.AddNewQuant(out.Name()+"_peakerror", SCALAR, VALUE, unit, "All-time maximum error/step for "+out.Name())
		s0.maxErr[i] = e.AddNewQuant(out.Name()+"_maxError", SCALAR, VALUE, unit, "Maximum error/step for "+out.Name())
		s0.relErr[i] = e.AddNewQuant(out.Name()+"_relError", SCALAR, VALUE, unit, "Relative error/step for "+out.Name())
		s0.diff[i].Init(out.Array().NComp(), out.Array().Size3D())
		s0.maxErr[i].SetVerifier(Positive)

		// TODO: recycle?
		y := equation[i].output[0]
		s0.y0buffer[i] = Pool.Get(y.NComp(), y.Size3D())

		s0.dy0Mul[i] = 0.0
		s0.dybuffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s0.dyhbuffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s0.dy0buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s0.dy1buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s0.dy2buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s0.dy3buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s0.dy4buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s0.dy5buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s0.dy6buffer[i] = Pool.Get(y.NComp(), y.Size3D())

	}
}

func LoadRKF54M(e *Engine) {
	s1 := new(RKF54Solver)

	// Minimum/maximum time step
	s1.minDt = e.AddNewQuant("mindt", SCALAR, VALUE, Unit("s"), "Minimum time step")
	s1.minDt.SetScalar(1e-38)
	s1.minDt.SetVerifier(Positive)
	s1.maxDt = e.AddNewQuant("maxdt", SCALAR, VALUE, Unit("s"), "Maximum time step")
	s1.maxDt.SetVerifier(Positive)
	s1.maxDt.SetScalar(1e38)
	s1.badSteps = e.AddNewQuant("badsteps", SCALAR, VALUE, Unit(""), "Number of time steps that had to be re-done")

	equation := e.equation
	s1.y0buffer = make([]*gpu.Array, len(equation))

	s1.dybuffer = make([]*gpu.Array, len(equation))
	s1.dyhbuffer = make([]*gpu.Array, len(equation))
	s1.dy0buffer = make([]*gpu.Array, len(equation))
	s1.dy0Mul = make([]float64, len(equation))
	s1.dy1buffer = make([]*gpu.Array, len(equation))
	s1.dy2buffer = make([]*gpu.Array, len(equation))
	s1.dy3buffer = make([]*gpu.Array, len(equation))
	s1.dy4buffer = make([]*gpu.Array, len(equation))
	s1.dy5buffer = make([]*gpu.Array, len(equation))
	s1.dy6buffer = make([]*gpu.Array, len(equation))

	s1.err = make([]*Quant, len(equation))
	s1.peakErr = make([]*Quant, len(equation))
	s1.maxErr = make([]*Quant, len(equation))
	s1.relErr = make([]*Quant, len(equation))
	s1.diff = make([]gpu.Reductor, len(equation))

	// Hard code Butcher tableau for the RK scheme
	// Tableau A
	s1.tableauA[0][0], s1.tableauA[0][1], s1.tableauA[0][2], s1.tableauA[0][3], s1.tableauA[0][4], s1.tableauA[0][5] =     1.0 /    5.0,               0.0,              0.0,            0.0,               0.0,         0.0
	s1.tableauA[1][0], s1.tableauA[1][1], s1.tableauA[1][2], s1.tableauA[1][3], s1.tableauA[1][4], s1.tableauA[1][5] =     3.0 /   40.0,      9.0 /   40.0,              0.0,            0.0,               0.0,         0.0
	s1.tableauA[2][0], s1.tableauA[2][1], s1.tableauA[2][2], s1.tableauA[2][3], s1.tableauA[2][4], s1.tableauA[2][5] =    44.0 /   45.0,    -56.0 /   15.0,    32.0 /    9.0,            0.0,               0.0,         0.0
	s1.tableauA[3][0], s1.tableauA[3][1], s1.tableauA[3][2], s1.tableauA[3][3], s1.tableauA[3][4], s1.tableauA[3][5] = 19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0,               0.0,         0.0
	s1.tableauA[4][0], s1.tableauA[4][1], s1.tableauA[4][2], s1.tableauA[4][3], s1.tableauA[4][4], s1.tableauA[4][5] =  9017.0 / 3168.0,   -355.0 /   33.0, 46732.0 / 5247.0,   49.0 / 176.0, -5103.0 / 18656.0,         0.0
	s1.tableauA[5][0], s1.tableauA[5][1], s1.tableauA[5][2], s1.tableauA[5][3], s1.tableauA[5][4], s1.tableauA[5][5] =    35.0 /  384.0,               0.0,   500.0 / 1113.0,  125.0 / 192.0, -2187.0 /  6784.0, 11.0 / 84.0

	// Tableau B
//	s1.tableauB[0][0], s1.tableauB[0][1], s1.tableauB[0][2], s1.tableauB[0][3], s1.tableauB[0][4], s1.tableauB[0][5], s1.tableauB[0][6] =   35.0 / 384.0, 0.0,  500.0 /  1113.0, -125.0 / 192.0,  -2187.0 /   6784.0,  11.0 /   84.0,        0.0
//	s1.tableauB[1][0], s1.tableauB[1][1], s1.tableauB[1][2], s1.tableauB[1][3], s1.tableauB[1][4], s1.tableauB[1][5], s1.tableauB[1][6] = 5179.0 / 57600, 0.0, 7571.0 / 16695.0,  393.0 / 640.0, -92097.0 / 339200.0, 187.0 / 2100.0, 1.0 / 40.0

	// Tableau C
	s1.tableauC[0], s1.tableauC[1], s1.tableauC[2], s1.tableauC[3], s1.tableauC[4], s1.tableauC[5] = 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0

	// Tableau for computing error estimate
	s1.errTab[0], s1.errTab[1], s1.errTab[2], s1.errTab[3], s1.errTab[4], s1.errTab[5], s1.errTab[6] = 71.0 / 57600.0, 0.0, -71.0 / 16695.0, 71.0 / 1920.0, -17253.0 / 339200.0, 22.0 / 525.0, -1.0 / 40.0

	e.SetSolver(s1)
	s1.order = 5.0
	s1.headRoom = 0.9
	s1.maxTry = 5

	for i := range equation {

		eqn := &(equation[i])
		Assert(eqn.kind == EQN_PDE1)
		out := eqn.output[0]
		unit := out.Unit()
		s1.err[i] = e.AddNewQuant(out.Name()+"_error", SCALAR, VALUE, unit, "Error/step estimate for "+out.Name())
		s1.peakErr[i] = e.AddNewQuant(out.Name()+"_peakerror", SCALAR, VALUE, unit, "All-time maximum error/step for "+out.Name())
		s1.maxErr[i] = e.AddNewQuant(out.Name()+"_maxError", SCALAR, VALUE, unit, "Maximum error/step for "+out.Name())
		s1.relErr[i] = e.AddNewQuant(out.Name()+"_relError", SCALAR, VALUE, unit, "Relative error/step for "+out.Name())
		s1.diff[i].Init(out.Array().NComp(), out.Array().Size3D())
		s1.maxErr[i].SetVerifier(Positive)

		// TODO: recycle?
		y := equation[i].output[0]
		s1.y0buffer[i] = Pool.Get(y.NComp(), y.Size3D())

		s1.dy0Mul[i] = 0.0
		s1.dybuffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s1.dyhbuffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s1.dy0buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s1.dy1buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s1.dy2buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s1.dy3buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s1.dy4buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s1.dy5buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s1.dy6buffer[i] = Pool.Get(y.NComp(), y.Size3D())

	}
}

func LoadRKF54S(e *Engine) {
	s := new(RKF54Solver)

	// Minimum/maximum time step
	s.minDt = e.AddNewQuant("mindt", SCALAR, VALUE, Unit("s"), "Minimum time step")
	s.minDt.SetScalar(1e-38)
	s.minDt.SetVerifier(Positive)
	s.maxDt = e.AddNewQuant("maxdt", SCALAR, VALUE, Unit("s"), "Maximum time step")
	s.maxDt.SetVerifier(Positive)
	s.maxDt.SetScalar(1e38)
	s.badSteps = e.AddNewQuant("badsteps", SCALAR, VALUE, Unit(""), "Number of time steps that had to be re-done")

	equation := e.equation
	s.y0buffer = make([]*gpu.Array, len(equation))

	s.dybuffer = make([]*gpu.Array, len(equation))
	s.dyhbuffer = make([]*gpu.Array, len(equation))
	s.dy0buffer = make([]*gpu.Array, len(equation))
	s.dy0Mul = make([]float64, len(equation))
	s.dy1buffer = make([]*gpu.Array, len(equation))
	s.dy2buffer = make([]*gpu.Array, len(equation))
	s.dy3buffer = make([]*gpu.Array, len(equation))
	s.dy4buffer = make([]*gpu.Array, len(equation))
	s.dy5buffer = make([]*gpu.Array, len(equation))
	s.dy6buffer = make([]*gpu.Array, len(equation))

	s.err = make([]*Quant, len(equation))
	s.peakErr = make([]*Quant, len(equation))
	s.maxErr = make([]*Quant, len(equation))
	s.relErr = make([]*Quant, len(equation))
	s.diff = make([]gpu.Reductor, len(equation))

	// Hard code Butcher tableau for the RK scheme
	// Tableau A
	s.tableauA[0][0], s.tableauA[0][1], s.tableauA[0][2], s.tableauA[0][3], s.tableauA[0][4], s.tableauA[0][5] =  2.0 /    9.0,           0.0,         0.0,            0.0,         0.0,        0.0
	s.tableauA[1][0], s.tableauA[1][1], s.tableauA[1][2], s.tableauA[1][3], s.tableauA[1][4], s.tableauA[1][5] =  1.0 /   12.0,   1.0 /   4.0,         0.0,            0.0,         0.0,        0.0
	s.tableauA[2][0], s.tableauA[2][1], s.tableauA[2][2], s.tableauA[2][3], s.tableauA[2][4], s.tableauA[2][5] =  55.0 / 324.0, -25.0 / 108.0, 50.0 / 81.0,            0.0,         0.0,        0.0
	s.tableauA[3][0], s.tableauA[3][1], s.tableauA[3][2], s.tableauA[3][3], s.tableauA[3][4], s.tableauA[3][5] =  83.0 / 330.0, -13.0 /  22.0, 61.0 / 66.0,    9.0 / 110.0,         0.0,        0.0
	s.tableauA[4][0], s.tableauA[4][1], s.tableauA[4][2], s.tableauA[4][3], s.tableauA[4][4], s.tableauA[4][5] = -19.0 /  28.0,   9.0 /   4.0,  1.0 /  7.0,  -27.0 /   7.0, 22.0 /  7.0,        0.0
	s.tableauA[5][0], s.tableauA[5][1], s.tableauA[5][2], s.tableauA[5][3], s.tableauA[5][4], s.tableauA[5][5] =  19.0 / 200.0,           0.0,  3.0 /  5.0, -243.0 / 400.0, 33.0 / 40.0, 7.0 / 80.0

	// Tableau B
//	s.tableauB[0][0], s.tableauB[0][1], s.tableauB[0][2], s.tableauB[0][3], s.tableauB[0][4], s.tableauB[0][5], s.tableauB[0][6] =   19.0 / 200.0, 0.0,   3.0 /   5.0,  -243.0 /   400.0,  33.0 /   40.0,   7.0 /   80.0,         0.0
//	s.tableauB[1][0], s.tableauB[1][1], s.tableauB[1][2], s.tableauB[1][3], s.tableauB[1][4], s.tableauB[1][5], s.tableauB[1][6] = 431.0 / 5000.0, 0.0, 333.0 / 500.0, -7857.0 / 10000.0, 957.0 / 1000.0, 193.0 / 2000.0, -1.0 / 50.0

	// Tableau C
	s.tableauC[0], s.tableauC[1], s.tableauC[2], s.tableauC[3], s.tableauC[4], s.tableauC[5] = 2.0 / 9.0, 1.0 / 3.0, 5.0 / 9.0, 2.0 / 3.0, 1.0, 1.0

	// Tableau for computing error estimate
	s.errTab[0], s.errTab[1], s.errTab[2], s.errTab[3], s.errTab[4], s.errTab[5], s.errTab[6] = 11.0 / 1250.0, 0.0, -33.0 / 500.0, 891.0 / 5000.0, -33.0 / 250.0, -9.0 / 1000.0, 1.0 / 50.0

	e.SetSolver(s)
	s.order = 5.0
	s.headRoom = 0.9
	s.maxTry = 5

	for i := range equation {

		eqn := &(equation[i])
		Assert(eqn.kind == EQN_PDE1)
		out := eqn.output[0]
		unit := out.Unit()
		s.err[i] = e.AddNewQuant(out.Name()+"_error", SCALAR, VALUE, unit, "Error/step estimate for "+out.Name())
		s.peakErr[i] = e.AddNewQuant(out.Name()+"_peakerror", SCALAR, VALUE, unit, "All-time maximum error/step for "+out.Name())
		s.maxErr[i] = e.AddNewQuant(out.Name()+"_maxError", SCALAR, VALUE, unit, "Maximum error/step for "+out.Name())
		s.relErr[i] = e.AddNewQuant(out.Name()+"_relError", SCALAR, VALUE, unit, "Relative error/step for "+out.Name())
		s.diff[i].Init(out.Array().NComp(), out.Array().Size3D())
		s.maxErr[i].SetVerifier(Positive)

		// TODO: recycle?
		y := equation[i].output[0]
		s.y0buffer[i] = Pool.Get(y.NComp(), y.Size3D())

		s.dy0Mul[i] = 0.0
		s.dybuffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.dyhbuffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.dy0buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.dy1buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.dy2buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.dy3buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.dy4buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.dy5buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.dy6buffer[i] = Pool.Get(y.NComp(), y.Size3D())

	}
}

// Declares this solvers' special dependencies
func (s *RK32Solver) Dependencies() (children, parents []string) {
	children = []string{"dt", "step", "t", "badsteps"}
	parents = []string{"dt", "mindt", "maxdt"}
	for i := range s.err {
		parents = append(parents, s.maxErr[i].Name(), s.relErr[i].Name())
		children = append(children, s.peakErr[i].Name(), s.err[i].Name())
	}
	return
}

func (s *RK45Solver) Dependencies() (children, parents []string) {
	children = []string{"dt", "step", "t", "badsteps"}
	parents = []string{"dt", "mindt", "maxdt"}
	for i := range s.err {
		parents = append(parents, s.maxErr[i].Name(), s.relErr[i].Name())
		children = append(children, s.peakErr[i].Name(), s.err[i].Name())
	}
	return
}

func (s *RKF54Solver) Dependencies() (children, parents []string) {
	children = []string{"dt", "step", "t", "badsteps"}
	parents = []string{"dt", "mindt", "maxdt"}
	for i := range s.err {
		parents = append(parents, s.maxErr[i].Name(), s.relErr[i].Name())
		children = append(children, s.peakErr[i].Name(), s.err[i].Name())
	}
	return
}

// Register this module
func init() {
	RegisterModule("solver/rk32", "Adaptive Bogacki-Shampine solver (Runge-Kutta 2+3)", LoadRK32)
	RegisterModule("solver/rk45ck", "Adaptive Cash-Karp solver (Runge-Kutta 4+5) RK4(5)5CK", LoadRK45CK)
	RegisterModule("solver/rk45f", "Adaptive Fehlberg solver (Runge-Kutta 4+5) RK4(5)5F", LoadRK45F)
	RegisterModule("solver/rkf54", "Adaptive Dormand-Prince solver (Runge-Kutta 4+5) RK5(4)7FC", LoadRKF54)
	RegisterModule("solver/rkf54m", "Adaptive Dormand-Prince solver (Runge-Kutta 4+5) RK5(4)7M", LoadRKF54M)
	RegisterModule("solver/rkf54s", "Adaptive Dormand-Prince solver (Runge-Kutta 4+5) RK5(4)7S", LoadRKF54S)
}

// Take one time step
func (s *RK32Solver) Step() {
	e := GetEngine()
	equation := e.equation

	// stage 0
	t0 := e.time.Scalar()

	// First update all inputs
	dt := engine.dt.Scalar()
	for i := range equation {
		Assert(equation[i].kind == EQN_PDE1)
		equation[i].input[0].Update()
		y := equation[i].output[0]
		dy := equation[i].input[0]
		dyMul := dy.multiplier
		s.y0buffer[i].CopyFromDevice(y.Array()) // save for later
		s.dy0buffer[i].CopyFromDevice(dy.Array()) // save for later
		s.dy0Mul[i] = dyMul[0]
//		s.dybuffer[i].Zero()
//		s.dyhbuffer[i].Zero()
//		s.dy1buffer[i].Zero()
//		s.dy2buffer[i].Zero()
//		s.dy3buffer[i].Zero()
	}

	maxTry := s.maxTry // undo at most this many bad steps
	headRoom := s.headRoom
	for try := 0; try < maxTry; try++ {

		// Get updated time step to try...
		dt = engine.dt.Scalar()

		// Fill arrays of slopes at all intermediate points...
		// Initial slope and initial point are already saved in s.y0buffer and s.dybuffer
		for i0 := 0; i0 < 3; i0++ {

			// Update time so that equation[i].input[0].Update() will pick up the right inputs
			stageTime := t0 + s.tableauC[i0] * dt
			e.time.SetScalar(stageTime)

			// 1. Calculate next point for ALL equations to get actual intermediate point
			// 2. After 1),  get the slope at intermediate point
		       	for i := range equation {

			      	// This is intermediate point computed from the previous i0
				y := equation[i].output[0]
				h := float32(s.dy0Mul[i] * dt)
//			      	if ((i0 == 0) && (try > 0)) { // Reset buffers whenever we have a bad step
//					s.dybuffer[i].Zero()
//					s.dyhbuffer[i].Zero()
//					s.dy1buffer[i].Zero()
//					s.dy2buffer[i].Zero()
//					s.dy3buffer[i].Zero()
//				}

			       	//  Update intermediate point
				if i0 == 0 {

					// The first intermediate point involve only the initial slope
					gpu.Madd(y.Array(), s.y0buffer[i], s.dy0buffer[i], h * s.tableauA[0][0])
				} else {
					if i0 == 1 {
						gpu.SMul(s.dybuffer[i], s.dy1buffer[i], s.tableauA[1][1])
					} else if i0 == 2 {
						gpu.LinearCombination3(s.dybuffer[i], s.dy0buffer[i], s.tableauA[2][0], s.dy1buffer[i], s.tableauA[2][1], s.dy2buffer[i], s.tableauA[2][2])

						// In Bogacki-Shampine scheme, higher order corrector is used to get
						// an intermediate point used for computing lower order predictor
						s.dyhbuffer[i].CopyFromDevice(s.dybuffer[i])
					}
					gpu.Madd(y.Array(), s.y0buffer[i], s.dybuffer[i], h)
				}
				y.Invalidate()
			}

			// Only get the slope after intermediate points for ALL equations have been calculated
			for i := range equation {
				// Now Update() sees the fully updated equation[i].output[0]
				equation[i].input[0].Update()
			}

			// Slope at intermediate point is computed by now
			for i := range equation {
				dy := equation[i].input[0]

				// Remember the slopes in buffers
				if i0 == 0 {
			     		s.dy1buffer[i].CopyFromDevice(dy.Array())
				} else if i0 == 1 {
			       		s.dy2buffer[i].CopyFromDevice(dy.Array())
				} else if i0 == 2 {
			       		s.dy3buffer[i].CopyFromDevice(dy.Array())
				}
			}
		}

		// All the intermediate slopes have been computed by now
		// Time to calculate predictor and corrector
		// Initialize the scaling factors to use for scaling time step
		badStep := false
		minFactor := 100.0
		maxFactor := 100.0

		// Advance time and update all inputs
		e.time.SetScalar(t0 + dt)

		// Calculate predictor and corrector and determine error
		for i := range equation {
			// y := equation[i].output[0]
			h := float32(s.dy0Mul[i] * dt)

			// The higher order dy/dt was already computed to obtain s.dy3buffer[i]
			// Compute the error estimate
			gpu.LinearCombination4(s.dybuffer[i], s.dy0buffer[i], s.errTab[0], s.dy1buffer[i], s.errTab[1], s.dy2buffer[i], s.errTab[2], s.dy3buffer[i], s.errTab[3])

			stepDiff := s.diff[i].MaxAbs(s.dybuffer[i]) * h
			err := float64(stepDiff)
			s.err[i].SetScalar(err)
			maxErr := s.maxErr[i].Scalar()
			stepRelDiff := s.diff[i].MaxAbs(s.dyhbuffer[i])
			maxDy := float64(stepRelDiff * h) * s.relErr[i].Scalar()

			// If error calculated is higher than tolerance...
			if ((err > maxErr) || (err > maxDy)) {
				if try == maxTry - 1.0 {
					s.badSteps.SetScalar(s.badSteps.Scalar() + 1)
				}
				badStep = true
			}
			if (!badStep || try == maxTry-1) && err > s.peakErr[i].Scalar() {

				// peak error should be that of good step, unless last trial which will not be undone
				s.peakErr[i].SetScalar(err)
			}
			factor := headRoom

			// Handle zero error to avoid divide by zero
			if err == 0 {
				factor = maxFactor
			} else {
				maxVal := maxErr
				if maxDy < maxVal {
					maxVal = maxDy
					factor = headRoom * math.Pow( maxVal / err, 1.0 / s.order)
				} else {
					factor = headRoom * math.Pow( maxVal / err, 1.0 / (s.order + 1.0))
				}
			}

			// Compare factor for each equation but only remember the smallest,
			// which we will use to change time step
			if factor < minFactor {
				minFactor = factor
			}

		}

		// Set new time step but do not go beyond min/max bounds
		newDt := dt * minFactor
		if newDt < s.minDt.Scalar() {
			newDt = s.minDt.Scalar()
		}
		if newDt > s.maxDt.Scalar() {
			newDt = s.maxDt.Scalar()
		}
		e.dt.SetScalar(newDt)
		if !badStep || newDt == s.minDt.Scalar() {
			for i := range equation {
				y := equation[i].output[0]

				// Only calculate the next point iff the step size is fine.
				// Helps to save on one full array MAdd operation per step
				// thus, improving run time
				if !badStep {
					h := float32(s.dy0Mul[i] * dt)
					gpu.Madd(y.Array(), s.y0buffer[i], s.dyhbuffer[i], h)
				}

				y.Invalidate()

			}

			break

		}

		for i := range equation {
			y := equation[i].output[0]
			y.Invalidate()
		}

	} // end try

	// advance time step
	e.step.SetScalar(e.step.Scalar() + 1)
}

func (s *RK45Solver) Step() {
	e := GetEngine()
	equation := e.equation

	// stage 0
	t0 := e.time.Scalar()

	// First update all inputs
	dt := engine.dt.Scalar()
	for i := range equation {
		Assert(equation[i].kind == EQN_PDE1)
		equation[i].input[0].Update()
		y := equation[i].output[0]
		dy := equation[i].input[0]
		dyMul := dy.multiplier
		s.y0buffer[i].CopyFromDevice(y.Array()) // save for later
		s.dy0buffer[i].CopyFromDevice(dy.Array()) // save for later
		s.dy0Mul[i] = dyMul[0]
//		s.dybuffer[i].Zero()
//		s.dy1buffer[i].Zero()
//		s.dy2buffer[i].Zero()
//		s.dy3buffer[i].Zero()
//		s.dy4buffer[i].Zero()
//		s.dy5buffer[i].Zero()
	}


	maxTry := s.maxTry // undo at most this many bad steps
	headRoom := s.headRoom
	for try := 0; try < maxTry; try++ {

		// Get updated time step to try...
		dt = engine.dt.Scalar()

		// Fill arrays of slopes at all intermediate points...
		// Initial slope and initial point are already saved in s.y0buffer and s.dybuffer
		for i0 := 0; i0 < 5; i0++ {

			// Update time stamp so that equation[i].input[0].Update()
			// will pick up the right inputs
			stageTime := t0 + s.tableauC[i0] * dt
			e.time.SetScalar(stageTime)
		       	for i := range equation {

			      	// This is intermediate point computed from the previous iteration
				y := equation[i].output[0]
				h := float32(s.dy0Mul[i] * dt)
//			      	if ((i0 == 0) && (try > 0)) { // Reset buffers whenever we have a bad step
//					s.dybuffer[i].Zero()
//					s.dy1buffer[i].Zero()
//					s.dy2buffer[i].Zero()
//					s.dy3buffer[i].Zero()
//					s.dy4buffer[i].Zero()
//					s.dy5buffer[i].Zero()
//				}

			       	//  Update intermediate point
				if i0 == 0 {

					// The first intermediate point involve only the initial slope
					gpu.Madd(y.Array(), s.y0buffer[i], s.dy0buffer[i], h * s.tableauA[0][0])
				} else {
					if i0 == 1 {
						gpu.LinearCombination2(s.dybuffer[i], s.dy0buffer[i], s.tableauA[1][0], s.dy1buffer[i], s.tableauA[1][1])
					} else if i0 == 2 {
						gpu.LinearCombination3(s.dybuffer[i], s.dy0buffer[i], s.tableauA[2][0], s.dy1buffer[i], s.tableauA[2][1], s.dy2buffer[i], s.tableauA[2][2])
					} else if i0 == 3 {
						gpu.LinearCombination4(s.dybuffer[i], s.dy0buffer[i], s.tableauA[3][0], s.dy1buffer[i], s.tableauA[3][1], s.dy2buffer[i], s.tableauA[3][2], s.dy3buffer[i], s.tableauA[3][3])
					} else if i0 == 4 {
						gpu.LinearCombination5(s.dybuffer[i], s.dy0buffer[i], s.tableauA[4][0], s.dy1buffer[i], s.tableauA[4][1], s.dy2buffer[i], s.tableauA[4][2], s.dy3buffer[i], s.tableauA[4][3], s.dy4buffer[i], s.tableauA[4][4])
					}
					gpu.Madd(y.Array(), s.y0buffer[i], s.dybuffer[i], h)
				}
				y.Invalidate()
			}

			// Only get the slope after intermediate points for ALL equations have been calculated
			for i:= range equation {

				// Now Update() sees the fully updated equation[i].output[0]
				equation[i].input[0].Update()
			}

			for i:= range equation {

				// Slope at intermediate point is computed by now
				dy := equation[i].input[0]

				// Remember the slopes in buffers
				if i0 == 0 {
			     		s.dy1buffer[i].CopyFromDevice(dy.Array())
				} else if i0 == 1 {
			       		s.dy2buffer[i].CopyFromDevice(dy.Array())
				} else if i0 == 2 {
			       		s.dy3buffer[i].CopyFromDevice(dy.Array())
				} else if i0 == 3 {
			       		s.dy4buffer[i].CopyFromDevice(dy.Array())
				} else if i0 == 4 {
			       		s.dy5buffer[i].CopyFromDevice(dy.Array())
				}
			}
		}

		// All the intermediate slopes have been computed by now
		// Time to calculate predictor and corrector
		// Initialize the scaling factors to use for scaling time step
		badStep := false
		minFactor := 100.0
		maxFactor := 100.0

		// Advance time and update all inputs
		e.time.SetScalar(t0 + dt)

		// Calculate predictor and corrector and determine error
		for i := range equation {
			// y := equation[i].output[0]
			h := float32(s.dy0Mul[i] * dt)

			// Calculate dy for corrector
			gpu.LinearCombination5(s.dyhbuffer[i], s.dy0buffer[i], s.tableauB[0], s.dy2buffer[i], s.tableauB[2], s.dy3buffer[i], s.tableauB[3], s.dy4buffer[i], s.tableauB[4], s.dy5buffer[i], s.tableauB[5])

			// Compute the error estimate
			gpu.LinearCombination5(s.dybuffer[i], s.dy0buffer[i], s.errTab[0], s.dy2buffer[i], s.errTab[2], s.dy3buffer[i], s.errTab[3], s.dy4buffer[i], s.errTab[4], s.dy5buffer[i], s.errTab[5])

			stepDiff := s.diff[i].MaxAbs(s.dybuffer[i]) * h
			err := float64(stepDiff)
			s.err[i].SetScalar(err)
			maxErr := s.maxErr[i].Scalar()
			stepRelDiff := s.diff[i].MaxAbs(s.dyhbuffer[i])
			maxDy := float64(stepRelDiff * h) * s.relErr[i].Scalar()

			// If error calculated is higher than tolerance...
			if ((err > maxErr) || (err > maxDy)) {
				if try == maxTry - 1.0 {
					s.badSteps.SetScalar(s.badSteps.Scalar() + 1)
				}
				badStep = true
			}
			if (!badStep || try == maxTry-1) && err > s.peakErr[i].Scalar() {

				// peak error should be that of good step, unless last trial which will not be undone
				s.peakErr[i].SetScalar(err)
			}
			factor := headRoom

			// Handle zero error to avoid divide by zero
			if err == 0 {
				factor = maxFactor
			} else {
				maxVal := maxErr
				if maxDy < maxVal {
					maxVal = maxDy
					factor = headRoom * math.Pow( maxVal / err, 1.0 / s.order)
				} else {
					factor = headRoom * math.Pow( maxVal / err, 1.0 / (s.order + 1.0))
				}
			}

			// Compare factor for each equation but only remember the smallest,
			// which we will use to change time step
			if factor < minFactor {
				minFactor = factor
			}

		}

		// Set new time step but do not go beyond min/max bounds
		newDt := dt * minFactor
		if newDt < s.minDt.Scalar() {
			newDt = s.minDt.Scalar()
		}
		if newDt > s.maxDt.Scalar() {
			newDt = s.maxDt.Scalar()
		}
		e.dt.SetScalar(newDt)
		if !badStep || newDt == s.minDt.Scalar() {
			for i := range equation {
				y := equation[i].output[0]

				// Only calculate the next point iff the step size is fine.
				// Helps to save on one full array MAdd operation per step
				// thus, improving run time
				if !badStep {
					h := float32(s.dy0Mul[i] * dt)
					gpu.Madd(y.Array(), s.y0buffer[i], s.dyhbuffer[i], h)
				}

				y.Invalidate()

			}

			break

		}

		for i := range equation {
			y := equation[i].output[0]
			y.Invalidate()
		}

	} // end try

	// advance time step
	e.step.SetScalar(e.step.Scalar() + 1)

}

func (s *RKF54Solver) Step() {
	e := GetEngine()
	equation := e.equation

	// stage 0
	t0 := e.time.Scalar()

	// First update all inputs
	dt := engine.dt.Scalar()
	for i := range equation {
		Assert(equation[i].kind == EQN_PDE1)
		equation[i].input[0].Update()
		y := equation[i].output[0]
		dy := equation[i].input[0]
		dyMul := dy.multiplier
		s.y0buffer[i].CopyFromDevice(y.Array()) // save for later
		s.dy0buffer[i].CopyFromDevice(dy.Array()) // save for later
		s.dy0Mul[i] = dyMul[0]
//		s.dybuffer[i].Zero()
//		s.dy1buffer[i].Zero()
//		s.dy2buffer[i].Zero()
//		s.dy3buffer[i].Zero()
//		s.dy4buffer[i].Zero()
//		s.dy5buffer[i].Zero()
//		s.dy6buffer[i].Zero()
	}


	maxTry := s.maxTry // undo at most this many bad steps
	headRoom := s.headRoom
	for try := 0; try < maxTry; try++ {

		// Get updated time step to try...
		dt = engine.dt.Scalar()

		// Fill arrays of slopes at all intermediate points...
		// Initial slope and initial point are already saved in s.y0buffer and s.dybuffer
		for i0 := 0; i0 < 6; i0++ {

			// Update time stamp so that equation[i].input[0].Update()
			// will pick up the right inputs
			stageTime := t0 + s.tableauC[i0] * dt
			e.time.SetScalar(stageTime)
		       	for i := range equation {

			      	// This is intermediate point computed from the previous iteration
				y := equation[i].output[0]
				h := float32(s.dy0Mul[i] * dt)
//			      	if ((i0 == 0) && (try > 0)) { // Reset buffers whenever we have a bad step
//					s.dybuffer[i].Zero()
//					s.dy1buffer[i].Zero()
//					s.dy2buffer[i].Zero()
//					s.dy3buffer[i].Zero()
//					s.dy4buffer[i].Zero()
//					s.dy5buffer[i].Zero()
//					s.dy6buffer[i].Zero()
//				}

			       	//  Update intermediate point
				if i0 == 0 {

					// The first intermediate point involve only the initial slope
					gpu.Madd(y.Array(), s.y0buffer[i], s.dy0buffer[i], h * s.tableauA[0][0])
				} else {
					if i0 == 1 {
						gpu.LinearCombination2(s.dybuffer[i], s.dy0buffer[i], s.tableauA[1][0], s.dy1buffer[i], s.tableauA[1][1])
					} else if i0 == 2 {
						gpu.LinearCombination3(s.dybuffer[i], s.dy0buffer[i], s.tableauA[2][0], s.dy1buffer[i], s.tableauA[2][1], s.dy2buffer[i], s.tableauA[2][2])
					} else if i0 == 3 {
						gpu.LinearCombination4(s.dybuffer[i], s.dy0buffer[i], s.tableauA[3][0], s.dy1buffer[i], s.tableauA[3][1], s.dy2buffer[i], s.tableauA[3][2], s.dy3buffer[i], s.tableauA[3][3])
					} else if i0 == 4 {
						gpu.LinearCombination5(s.dybuffer[i], s.dy0buffer[i], s.tableauA[4][0], s.dy1buffer[i], s.tableauA[4][1], s.dy2buffer[i], s.tableauA[4][2], s.dy3buffer[i], s.tableauA[4][3], s.dy4buffer[i], s.tableauA[4][4])
					} else if i0 == 5 {
						gpu.LinearCombination5(s.dybuffer[i], s.dy0buffer[i], s.tableauA[5][0], s.dy2buffer[i], s.tableauA[5][2], s.dy3buffer[i], s.tableauA[5][3], s.dy4buffer[i], s.tableauA[5][4], s.dy5buffer[i], s.tableauA[5][5])

						// Since this is also the high order corrector, we will remember this in s.dyhbuffer[i]
						s.dyhbuffer[i].CopyFromDevice(s.dybuffer[i])
					}
					gpu.Madd(y.Array(), s.y0buffer[i], s.dybuffer[i], h)
				}
				y.Invalidate()
			}

			// Only get the slope after intermediate points for ALL equations have been calculated
			for i:= range equation {

				// Now Update() sees the fully updated equation[i].output[0]
				equation[i].input[0].Update()
			}

			for i:= range equation {

				// Slope at intermediate point is computed by now
				dy := equation[i].input[0]

				// Remember the slopes in buffers
				if i0 == 0 {
			     		s.dy1buffer[i].CopyFromDevice(dy.Array())
				} else if i0 == 1 {
			       		s.dy2buffer[i].CopyFromDevice(dy.Array())
				} else if i0 == 2 {
			       		s.dy3buffer[i].CopyFromDevice(dy.Array())
				} else if i0 == 3 {
			       		s.dy4buffer[i].CopyFromDevice(dy.Array())
				} else if i0 == 4 {
			       		s.dy5buffer[i].CopyFromDevice(dy.Array())
				} else if i0 == 5 {
			       		s.dy6buffer[i].CopyFromDevice(dy.Array())
				}
			}
		}

		// All the intermediate slopes have been computed by now
		// Time to calculate predictor and corrector
		// Initialize the scaling factors to use for scaling time step
		badStep := false
		minFactor := 100.0
		maxFactor := 100.0

		// Advance time and update all inputs
		e.time.SetScalar(t0 + dt)

		// Calculate predictor and corrector and determine error
		for i := range equation {
			// y := equation[i].output[0]
			h := float32(s.dy0Mul[i] * dt)

			// Calculate error estimate
			gpu.LinearCombination6(s.dybuffer[i], s.dy0buffer[i], s.errTab[0], s.dy2buffer[i], s.errTab[2], s.dy3buffer[i], s.errTab[3], s.dy4buffer[i], s.errTab[4], s.dy5buffer[i], s.errTab[5], s.dy6buffer[i], s.errTab[6])

			// error estimate using difference of high and low order dy
			// stepDiff := s.diff[i].MaxDiff(s.dyhbuffer[i], s.dybuffer[i]) * h
			stepDiff := s.diff[i].MaxAbs(s.dybuffer[i]) * h
			err := float64(stepDiff)
			s.err[i].SetScalar(err)
			maxErr := s.maxErr[i].Scalar()
			stepRelDiff := s.diff[i].MaxAbs(s.dyhbuffer[i])
			maxDy := float64(stepRelDiff * h) * s.relErr[i].Scalar()

			if ((err > maxErr) || (err > maxDy)) {
				if try == maxTry - 1.0 {
					s.badSteps.SetScalar(s.badSteps.Scalar() + 1)
				}
				badStep = true
			}
			if (!badStep || try == maxTry-1) && err > s.peakErr[i].Scalar() {

				// peak error should be that of good step, unless last trial which will not be undone
				s.peakErr[i].SetScalar(err)
			}
			factor := headRoom

			// Handle zero error to avoid divide by zero
			if err == 0 {
				factor = maxFactor
			} else {
				maxVal := maxErr
				if maxDy < maxVal {
					maxVal = maxDy
					factor = headRoom * math.Pow( maxVal / err, 1.0 / s.order)
				} else {
					factor = headRoom * math.Pow( maxVal / err, 1.0 / (s.order + 1.0))
				}
			}

			// Compare factor for each equation but only remember the smallest,
			// which we will use to change time step
			if factor < minFactor {
				minFactor = factor
			}

		}

		// Set new time step but do not go beyond min/max bounds
		newDt := dt * minFactor
		if newDt < s.minDt.Scalar() {
			newDt = s.minDt.Scalar()
		}
		if newDt > s.maxDt.Scalar() {
			newDt = s.maxDt.Scalar()
		}
		e.dt.SetScalar(newDt)
		if !badStep || newDt == s.minDt.Scalar() {
			for i := range equation {
				y := equation[i].output[0]

				// Only calculate the next point iff the step size is fine.
				// Helps to save on one full array MAdd operation per step
				// thus, improving run time
				if !badStep {
					h := float32(s.dy0Mul[i] * dt)
					gpu.Madd(y.Array(), s.y0buffer[i], s.dyhbuffer[i], h)
				}

				y.Invalidate()

			}

			break

		}

		for i := range equation {
			y := equation[i].output[0]
			y.Invalidate()
		}

	} // end try

	// advance time step
	e.step.SetScalar(e.step.Scalar() + 1)
}
