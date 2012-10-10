//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// 6-neighbor exchange interaction
// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	. "mumax/engine"
	"mumax/gpu"
)

// Register this module
func init() {
	RegisterModule("langevin", "Temperature dependance of equilibrium value of saturation magnetization for J → ∞", LoadLangevin)
}

func LoadLangevin(e *Engine) {
    LoadFullMagnetization(e)
	LoadTemp(e, "Te")
	J0 := e.AddNewQuant("J0", SCALAR, MASK, Unit(""), "zero Fourier component of exchange integral")
	e.Depends("msat0", "msat0T0", "Te", "J0")
	msat0 := e.Quant("msat0")
	msat0.SetUpdater(&LangevinUpdater{msat0: msat0, msat0T0: e.Quant("msat0T0"), T: e.Quant("Te"), J0: J0})

}

type LangevinUpdater struct {
	msat0, msat0T0, T, J0 *Quant
}

func (u *LangevinUpdater) Update() {
	msat0 := u.msat0
	msat0T0 := u.msat0T0
	T := u.T
	J0 := u.J0
	stream := msat0.Array().Stream
	gpu.LangevinAsync(msat0.Array(), msat0T0.Array(), T.Array(), J0.Array(), msat0.Multiplier()[0], msat0T0.Multiplier()[0], J0.Multiplier()[0], stream)
	stream.Sync()
}
