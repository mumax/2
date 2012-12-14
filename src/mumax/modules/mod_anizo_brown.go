//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Simple module for thermal fluctuations according to Brown.
// Author: Arne Vansteenkiste

import (
	"cuda/curand"
	. "mumax/common"
	. "mumax/engine"
	"mumax/gpu"
)

// Register this module
func init() {
	RegisterModule("temperature/anizotropic-brown", "Anizotropic thermal fluctuating field according to Brown.", LoadAnizBrown)
}

func LoadAnizBrown(e *Engine) {
	//e.LoadModule("llb")
	LoadTemp(e, LtempName) // load temperature

	Therm_seed := e.AddNewQuant("Therm_seed", SCALAR, VALUE, Unit(""), `Random seed for H\_therm`)
	Therm_seed.SetVerifier(Int)

	Htherm := e.AddNewQuant("H_therm", VECTOR, FIELD, Unit("A/m"), "Thermal fluctuating field")
	e.AddNewQuant("cutoff_dt", SCALAR, VALUE, "s", `Update thermal field at most once per cutoff\_dt. Works best with fixed time step equal to N*cutoff\_dt.`)

	// By declaring that H_therm depends on Step,
	// It will be automatically updated at each new time step
	// and remain constant during the stages of the step.

	e.AddNewQuant("lambda_lattice", VECTOR, MASK, Unit(""), "Anizotropic lattice coupling constant")
	e.Depends("H_therm", LtempName, "Step", "dt", "lambda_lattice", "gamma_LL", "Msat", "Therm_seed")
	Htherm.SetUpdater(NewAnizBrownUpdater(Htherm, Therm_seed))

	// Add thermal field to total field
	hfield := e.Quant("H_eff")
	sum := hfield.GetUpdater().(*SumUpdater)
	sum.AddParent("H_therm")
}

// Updates the thermal field
type AnizBrownUpdater struct {
	rng              []curand.Generator // Random number generator for each GPU
	htherm           *Quant             // The quantity I will update
	therm_seed       *Quant
	therm_seed_cache int64
	last_time        float64 // time of last htherm update
}

func NewAnizBrownUpdater(htherm *Quant, therm_seed *Quant) Updater {
	u := new(AnizBrownUpdater)
	u.therm_seed = therm_seed
	u.therm_seed_cache = -1e10
	u.htherm = htherm
	u.rng = make([]curand.Generator, gpu.NDevice())
	for dev := range u.rng {
		gpu.SetDeviceForIndex(dev)
		u.rng[dev] = curand.CreateGenerator(curand.PSEUDO_DEFAULT)
		//u.rng[dev].SetSeed(int64(dev)) // TODO: use proper seed
	}
	return u
}

// Updates H_therm
func (u *AnizBrownUpdater) Update() {
	e := GetEngine()

	therm_seed := int64(u.therm_seed.Scalar())

	if therm_seed != u.therm_seed_cache {
		for dev := range u.rng {
			seed := therm_seed + int64(dev)
			u.rng[dev].SetSeed(seed)
		}
	}

	u.therm_seed_cache = therm_seed

	// Nothing to do for zero temperature
	temp := e.Quant(LtempName)
	tempMul := temp.Multiplier()[0]
	if tempMul == 0 {
		u.htherm.Array().Zero()
		return
	}

	// Update only if we went past the dt cutoff
	t := e.Quant("t").Scalar()
	dt := e.Quant("dt").Scalar()
	cutoff_dt := e.Quant("cutoff_dt").Scalar()
	if dt < cutoff_dt {
		dt = cutoff_dt
		if u.last_time != 0 && t < u.last_time+dt {
			return
		}
	}

	// Make standard normal noise
	noise := u.htherm.Array()
	devPointers := noise.Pointers()
	N := int64(noise.PartLen4D())
	// Fills H_therm with gaussian noise.
	// CURAND does not provide an out-of-the-box way to do this in parallel over the GPUs
	for dev := range u.rng {
		gpu.SetDeviceForIndex(dev)
		u.rng[dev].GenerateNormal(uintptr(devPointers[dev]), N, 0, 1)
	}

	// Scale the noise according to local parameters
	cellSize := e.CellSize()
	V := cellSize[X] * cellSize[Y] * cellSize[Z]
	ll := e.Quant("lambda_lattice")

	gamma := e.Quant("gamma_LL").Scalar()
	mSat := e.Quant("Msat")
	mSatMask := mSat.Array()
	mSatMul := mSat.Multiplier()[0]
	tempMask := temp.Array()
	KB2tempMul := Kb * 2 * tempMul
	mu0VgammaDtMsatMul := float32(Mu0 * V * gamma * dt * mSatMul)

	for c := 0; c < 3; c++ {
		llMask := ll.Array().Component(X)
		llKB2tempMul := float32(ll.Multiplier()[c] * KB2tempMul)
		gpu.ScaleNoise(&(noise.Comp[c]), llMask, tempMask, llKB2tempMul, mSatMask, mu0VgammaDtMsatMul)
	}

	u.last_time = t
}
