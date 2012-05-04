//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Module implementing Slonczewski spin transfer torque.
// Authors: Mykol Dvornik, Graham Rowlands, Arne Vansteenkiste

import (
	. "mumax/common"
	. "mumax/engine"
	"mumax/gpu"
	//"math"
)

// Register this module
func init() {
	RegisterModule("slonczewski", "Slonczewski spin transfer torque.", LoadSlonczewskiTorque)
}

func LoadSlonczewskiTorque(e *Engine) {
	e.LoadModule("llg") // needed for alpha, hfield, ...

	// ============ New Quantities =============
	e.AddNewQuant("lambda", SCALAR, VALUE, Unit(""), "In-Plane term")
	e.AddNewQuant("p", VECTOR, MASK , Unit(""), "Polarization Vector")
	e.AddNewQuant("pol", SCALAR, VALUE, Unit(""), "Polarization efficiency")
	e.AddNewQuant("epsilon_prime", SCALAR, VALUE, Unit(""), "Field-like term")
	LoadUserDefinedCurrentDensity(e)
	stt := e.AddNewQuant("stt", VECTOR, FIELD, Unit("/s"), "Slonczewski Spin Transfer Torque")

	// ============ Dependencies =============
	e.Depends("stt", "lambda", "p", "pol", "epsilon_prime", "j", "m", "gamma", "Msat", "gamma")

	// ============ Updating the torque =============
	stt.SetUpdater(&slonczewskiUpdater{stt: stt})

	// Add spin-torque to LLG torque
	AddTermToQuant(e.Quant("torque"), stt)
}

type slonczewskiUpdater struct {
	stt *Quant
}

func (u *slonczewskiUpdater) Update() {
	e := GetEngine()

	worldSize := e.WorldSize()
	
	stt := u.stt
	m := e.Quant("m")
	msat := e.Quant("msat")
	pol := e.Quant("pol").Scalar()
	lambda := e.Quant("lambda").Scalar()
	epsilon_prime := e.Quant("epsilon_prime").Scalar()
	p := e.Quant("p")
	curr := e.Quant("j")
	gamma := e.Quant("gamma").Scalar()
	
	
    //njn := math.Sqrt(float64(curr.Multiplier()[0] * curr.Multiplier()[0]) + float64(curr.Multiplier()[1] * curr.Multiplier()[1]) + float64(curr.Multiplier()[2] * curr.Multiplier()[2]))
    
	nmsatn := msat.Multiplier()[0]
	//Debug("Reduced Planck's constant:", H_bar)
	
    beta := H_bar * gamma / (Mu0 * E * nmsatn) // njn is missing
    beta_prime := pol * beta  //beta_prime does not contain 
    pre_fld := beta * epsilon_prime
	lambda2 := lambda * lambda
	//Debug("beta_prime:",beta_prime," pre_fld:",pre_fld," lambda:",lambda2,"P:", p.Multiplier(), "J:", curr.Multiplier(), "wSize:", worldSize)

	gpu.LLSlon(stt.Array(), m.Array(), msat.Array(), p.Array(), curr.Array(), p.Multiplier(), curr.Multiplier(), float32(lambda2), float32(beta_prime), float32(pre_fld), worldSize)
}
