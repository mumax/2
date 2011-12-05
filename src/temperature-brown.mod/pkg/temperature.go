//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package temperature_brown

// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	. "mumax/engine"
	"math"
	"mumax/gpu"
	"cuda/curand"
)

// Register this module
func init() {
	RegisterModule(ModTempBrown(0))
}

type ModTempBrown int

func (x ModTempBrown) Description() string {
	return "Thermal fluctuating field according to Brown."
}

func (x ModTempBrown) Name() string {
	return "temperature/brown"
}

func (x ModTempBrown) Load(e *Engine) {

	// TODO: make it a mask so we can have temperature gradients
	temp := 	e.AddQuant("Temp", SCALAR, VALUE, Unit("K"), "Temperature")
	temp.SetVerifier(NonNegative)
	Htherm := e.AddQuant("H_therm", VECTOR, FIELD, Unit("A/m"), "Thermal fluctuating field")
	//e.AddQuant("Therm_seed", SCALAR, VALUE, Unit(""), "Random seed for H_therm")
	e.Depends("H_therm", "Temp", "Step", "dt", "alpha", "gamma")//, "Therm_seed")
	Htherm.SetUpdater(NewTempBrownUpdater(Htherm))

	e.LoadModule("hfield")
	hfield := e.Quant("H")
	sum := hfield.GetUpdater().(*SumUpdater)
	sum.AddParent("H_therm")
}

type TempBrownUpdater struct {
	rng []curand.Generator
	htherm *Quant
}

func NewTempBrownUpdater(htherm *Quant) Updater {
	u := new(TempBrownUpdater)
	u.htherm = htherm
	u.rng = make([]curand.Generator, gpu.NDevice())
	for dev := range u.rng {
		gpu.SetDeviceForIndex(dev)
		u.rng[dev] = curand.CreateGenerator(curand.PSEUDO_DEFAULT)
		u.rng[dev].SetSeed(int64(dev)) // TODO: something else
	}
	return u
}

func (u *TempBrownUpdater) Update() {
	e := GetEngine()
	
	temp := e.Quant("temp").Scalar()
	if temp == 0{return}

	dt := e.Quant("dt").Scalar()
	cellSize := e.CellSize()
	V := cellSize[X] * cellSize[Y] * cellSize[Z]
	alpha := e.Quant("alpha").Scalar()
	gamma := e.Quant("gamma").Scalar()
	mSat := e.Quant("Msat").Scalar()

	stddev := float32(math.Sqrt((2*alpha*Kb*temp)/(gamma*Mu0*mSat*V*dt)))

	array := u.htherm.Array()
	devPointers := array.Pointers()
	N := int64(array.PartLen4D())

	for dev:=range u.rng{
		gpu.SetDeviceForIndex(dev)
		u.rng[dev].GenerateNormal(uintptr(devPointers[dev]), N, 0, stddev)
	}
}

