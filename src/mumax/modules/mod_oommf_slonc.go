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
	RegisterModule("oommf/slonczewski/int0", "OOMMF implementation of Slonczewski spin transfer torque.", LoadOommfSloncTorque0)
	RegisterModule("oommf/slonczewski/int1", "OOMMF implementation of Slonczewski spin transfer torque.", LoadOommfSloncTorque1)
	RegisterModule("oommf/slonczewski/int2", "OOMMF implementation of Slonczewski spin transfer torque.", LoadOommfSloncTorque2)
}

func LoadOommfSloncTorque0(e *Engine) {
	e.LoadModule("llg") // needed for alpha, hfield, ...

	// ============ New Quantities =============
	e.AddNewQuant("t_fl0", SCALAR, MASK, Unit(""), "Free layer thickness")

	labmda_fre := e.AddNewQuant("lambda_free0", SCALAR, MASK, Unit(""), "Scattering control parameter (free side)")
	labmda_fre.SetValue([]float64{1.0})
	labmda_fix := e.AddNewQuant("lambda_fixed0", SCALAR, MASK, Unit(""), "Scattering control parameter (fixed side)")
	labmda_fix.SetValue([]float64{1.0})
	e.AddNewQuant("p0", VECTOR, MASK, Unit(""), "Polarization Vector")

	pol_fre := e.AddNewQuant("pol_free0", SCALAR, MASK, Unit(""), "Polarization efficiency (free side)")
	pol_fre.SetValue([]float64{1.0})
	pol_fix := e.AddNewQuant("pol_fixed0", SCALAR, MASK, Unit(""), "Polarization efficiency (fixed side)")
	pol_fix.SetValue([]float64{1.0})
	epsilon_prime := e.AddNewQuant("epsilon_prime0", SCALAR, MASK, Unit(""), "Field-like term")
	epsilon_prime.SetValue([]float64{0.0})
	LoadUserDefinedCurrentDensity0(e)
	oommf_stt0 := e.AddNewQuant("oommf_stt0", VECTOR, FIELD, Unit("/s"), "OOMMF Slonczewski Spin Transfer Torque")

	// ============ Dependencies =============
	e.Depends("oommf_stt0", "lambda_free0", "lambda_fixed0", "p0", "pol_free0", "pol_fixed0", "epsilon_prime0", "j0", "m", "gamma", "msat", "gamma", "alpha", "t_fl0")

	// ============ Updating the torque =============
	oommf_stt0.SetUpdater(&sloncOommfUpdater0{oommf_stt0: oommf_stt0})

	// Add spin-torque to LLG torque
	AddTermToQuant(e.Quant("torque"), oommf_stt0)
}

type sloncOommfUpdater0 struct {
	oommf_stt0 *Quant
}

func (u *sloncOommfUpdater0) Update() {
	e := GetEngine()

	worldSize := e.WorldSize()

	stt := u.oommf_stt0
	m := e.Quant("m")
	msat := e.Quant("msat")
	pol_fre := e.Quant("pol_free0")
	pol_fix := e.Quant("pol_fixed0")
	lambda_fre := e.Quant("lambda_free0")
	lambda_fix := e.Quant("lambda_fixed0")
	epsilon_prime := e.Quant("epsilon_prime0")
	p := e.Quant("p0")
	curr := e.Quant("j0")
	alpha := e.Quant("alpha")
	gamma := e.Quant("gamma").Scalar()
	t_fl := e.Quant("t_fl0")

	//njn := math.Sqrt(float64(curr.Multiplier()[0] * curr.Multiplier()[0]) + float64(curr.Multiplier()[1] * curr.Multiplier()[1]) + float64(curr.Multiplier()[2] * curr.Multiplier()[2]))

	nmsatn := msat.Multiplier()[0]

	beta := H_bar * gamma / (Mu0 * E * nmsatn) // njn is missing
	pre_fld := beta * epsilon_prime.Multiplier()[0] // epsilon_primeMsk is missing
		
	gpu.LLSlonOOMMF(stt.Array(),
		m.Array(),
		msat.Array(),
		p.Array(),
		curr.Array(),
		alpha.Array(),
		t_fl.Array(),
		pol_fre.Array(),
		pol_fix.Array(),
		lambda_fre.Array(),
		lambda_fix.Array(),
		epsilon_prime.Array(),
		p.Multiplier(),
		curr.Multiplier(),
		float32(beta),
		float32(pre_fld),
		worldSize,
		alpha.Multiplier(),
		t_fl.Multiplier(),
		pol_fre.Multiplier(),
		pol_fix.Multiplier(),
		lambda_fre.Multiplier(),
		lambda_fix.Multiplier())
}

func LoadOommfSloncTorque1(e *Engine) {
	e.LoadModule("llg") // needed for alpha, hfield, ...

	// ============ New Quantities =============
	e.AddNewQuant("t_fl1", SCALAR, MASK, Unit(""), "Free layer thickness")

	labmda_fre := e.AddNewQuant("lambda_free1", SCALAR, MASK, Unit(""), "Scattering control parameter (free side)")
	labmda_fre.SetValue([]float64{1.0})
	labmda_fix := e.AddNewQuant("lambda_fixed1", SCALAR, MASK, Unit(""), "Scattering control parameter (fixed side)")
	labmda_fix.SetValue([]float64{1.0})
	e.AddNewQuant("p1", VECTOR, MASK, Unit(""), "Polarization Vector")

	pol_fre := e.AddNewQuant("pol_free1", SCALAR, MASK, Unit(""), "Polarization efficiency (free side)")
	pol_fre.SetValue([]float64{1.0})
	pol_fix := e.AddNewQuant("pol_fixed1", SCALAR, MASK, Unit(""), "Polarization efficiency (fixed side)")
	pol_fix.SetValue([]float64{1.0})
	epsilon_prime := e.AddNewQuant("epsilon_prime1", SCALAR, MASK, Unit(""), "Field-like term")
	epsilon_prime.SetValue([]float64{0.0})
	LoadUserDefinedCurrentDensity1(e)
	oommf_stt1 := e.AddNewQuant("oommf_stt1", VECTOR, FIELD, Unit("/s"), "OOMMF Slonczewski Spin Transfer Torque")

	// ============ Dependencies =============
	e.Depends("oommf_stt1", "lambda_free1", "lambda_fixed1", "p1", "pol_free1", "pol_fixed1", "epsilon_prime1", "j1", "m", "gamma", "msat", "gamma", "alpha", "t_fl1")

	// ============ Updating the torque =============
	oommf_stt1.SetUpdater(&sloncOommfUpdater1{oommf_stt1: oommf_stt1})

	// Add spin-torque to LLG torque
	AddTermToQuant(e.Quant("torque"), oommf_stt1)
}

type sloncOommfUpdater1 struct {
	oommf_stt1 *Quant
}

func (u *sloncOommfUpdater1) Update() {
	e := GetEngine()

	worldSize := e.WorldSize()

	stt := u.oommf_stt1
	m := e.Quant("m")
	msat := e.Quant("msat")
	pol_fre := e.Quant("pol_free1")
	pol_fix := e.Quant("pol_fixed1")
	lambda_fre := e.Quant("lambda_free1")
	lambda_fix := e.Quant("lambda_fixed1")
	epsilon_prime := e.Quant("epsilon_prime1")
	p := e.Quant("p1")
	curr := e.Quant("j1")
	alpha := e.Quant("alpha")
	gamma := e.Quant("gamma").Scalar()
	t_fl := e.Quant("t_fl1")

	//njn := math.Sqrt(float64(curr.Multiplier()[0] * curr.Multiplier()[0]) + float64(curr.Multiplier()[1] * curr.Multiplier()[1]) + float64(curr.Multiplier()[2] * curr.Multiplier()[2]))

	nmsatn := msat.Multiplier()[0]

	beta := H_bar * gamma / (Mu0 * E * nmsatn) // njn is missing
	pre_fld := beta * epsilon_prime.Multiplier()[0] // epsilon_primeMsk is missing
		
	gpu.LLSlonOOMMF(stt.Array(),
		m.Array(),
		msat.Array(),
		p.Array(),
		curr.Array(),
		alpha.Array(),
		t_fl.Array(),
		pol_fre.Array(),
		pol_fix.Array(),
		lambda_fre.Array(),
		lambda_fix.Array(),
		epsilon_prime.Array(),
		p.Multiplier(),
		curr.Multiplier(),
		float32(beta),
		float32(pre_fld),
		worldSize,
		alpha.Multiplier(),
		t_fl.Multiplier(),
		pol_fre.Multiplier(),
		pol_fix.Multiplier(),
		lambda_fre.Multiplier(),
		lambda_fix.Multiplier())
}

func LoadOommfSloncTorque2(e *Engine) {
	e.LoadModule("llg") // needed for alpha, hfield, ...

	// ============ New Quantities =============
	e.AddNewQuant("t_fl2", SCALAR, MASK, Unit(""), "Free layer thickness")

	labmda_fre := e.AddNewQuant("lambda_free2", SCALAR, MASK, Unit(""), "Scattering control parameter (free side)")
	labmda_fre.SetValue([]float64{1.0})
	labmda_fix := e.AddNewQuant("lambda_fixed2", SCALAR, MASK, Unit(""), "Scattering control parameter (fixed side)")
	labmda_fix.SetValue([]float64{1.0})
	e.AddNewQuant("p2", VECTOR, MASK, Unit(""), "Polarization Vector")

	pol_fre := e.AddNewQuant("pol_free2", SCALAR, MASK, Unit(""), "Polarization efficiency (free side)")
	pol_fre.SetValue([]float64{1.0})
	pol_fix := e.AddNewQuant("pol_fixed2", SCALAR, MASK, Unit(""), "Polarization efficiency (fixed side)")
	pol_fix.SetValue([]float64{1.0})
	epsilon_prime := e.AddNewQuant("epsilon_prime2", SCALAR, MASK, Unit(""), "Field-like term")
	epsilon_prime.SetValue([]float64{0.0})
	LoadUserDefinedCurrentDensity2(e)
	oommf_stt2 := e.AddNewQuant("oommf_stt2", VECTOR, FIELD, Unit("/s"), "OOMMF Slonczewski Spin Transfer Torque")

	// ============ Dependencies =============
	e.Depends("oommf_stt2", "lambda_free2", "lambda_fixed2", "p2", "pol_free2", "pol_fixed2", "epsilon_prime2", "j2", "m", "gamma", "msat", "gamma", "alpha", "t_fl2")

	// ============ Updating the torque =============
	oommf_stt2.SetUpdater(&sloncOommfUpdater2{oommf_stt2: oommf_stt2})

	// Add spin-torque to LLG torque
	AddTermToQuant(e.Quant("torque"), oommf_stt2)
}

type sloncOommfUpdater2 struct {
	oommf_stt2 *Quant
}

func (u *sloncOommfUpdater2) Update() {
	e := GetEngine()

	worldSize := e.WorldSize()

	stt := u.oommf_stt2
	m := e.Quant("m")
	msat := e.Quant("msat")
	pol_fre := e.Quant("pol_free2")
	pol_fix := e.Quant("pol_fixed2")
	lambda_fre := e.Quant("lambda_free2")
	lambda_fix := e.Quant("lambda_fixed2")
	epsilon_prime := e.Quant("epsilon_prime2")
	p := e.Quant("p2")
	curr := e.Quant("j2")
	alpha := e.Quant("alpha")
	gamma := e.Quant("gamma").Scalar()
	t_fl := e.Quant("t_fl2")

	//njn := math.Sqrt(float64(curr.Multiplier()[0] * curr.Multiplier()[0]) + float64(curr.Multiplier()[1] * curr.Multiplier()[1]) + float64(curr.Multiplier()[2] * curr.Multiplier()[2]))

	nmsatn := msat.Multiplier()[0]

	beta := H_bar * gamma / (Mu0 * E * nmsatn) // njn is missing
	pre_fld := beta * epsilon_prime.Multiplier()[0] // epsilon_primeMsk is missing
		
	gpu.LLSlonOOMMF(stt.Array(),
		m.Array(),
		msat.Array(),
		p.Array(),
		curr.Array(),
		alpha.Array(),
		t_fl.Array(),
		pol_fre.Array(),
		pol_fix.Array(),
		lambda_fre.Array(),
		lambda_fix.Array(),
		epsilon_prime.Array(),
		p.Multiplier(),
		curr.Multiplier(),
		float32(beta),
		float32(pre_fld),
		worldSize,
		alpha.Multiplier(),
		t_fl.Multiplier(),
		pol_fre.Multiplier(),
		pol_fix.Multiplier(),
		lambda_fre.Multiplier(),
		lambda_fix.Multiplier())
}
