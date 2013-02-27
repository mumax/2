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
	//~ . "mumax/common"
	. "mumax/engine"
	"mumax/gpu"
)

// Register this module
func init() {
	RegisterModule("mfa-brillouin/msat0", "Temperature dependance of equilibrium value of saturation magnetization for any finite J", LoadBrillouin)
}

func LoadBrillouin(e *Engine) {
	LoadFullMagnetization(e)
	LoadTemp(e, "Te")
	LoadMFAParams(e)

	S := e.Quant("J")
	Tc := e.Quant("Tc")

	e.Depends("msat0", "msat0T0", "Te", "J", "Tc")
	msat0 := e.Quant("msat0")
	msat0.SetUpdater(&BrillouinUpdater{msat0: msat0, msat0T0: e.Quant("msat0T0"), T: e.Quant("Te"), Tc: Tc, S: S})

}

type BrillouinUpdater struct {
	msat0, msat0T0, T, Tc, S *Quant
}

func (u *BrillouinUpdater) Update() {
	msat0 := u.msat0
	msat0T0 := u.msat0T0
	T := u.T
	S := u.S
	Tc := u.Tc
	stream := msat0.Array().Stream
	gpu.BrillouinAsync(msat0.Array(), msat0T0.Array(), T.Array(), Tc.Array(), S.Array(), msat0.Multiplier()[0], msat0T0.Multiplier()[0], Tc.Multiplier()[0], S.Multiplier()[0], stream)
	stream.Sync()
}
