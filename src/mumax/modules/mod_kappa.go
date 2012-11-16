//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// temperature dependence of longitudinal susceptibility as follows from mean-field approximation
// Author: Mykola Dvornik

import (
	. "mumax/common"
	. "mumax/engine"
	"mumax/gpu"
)

// Register this module
func init() {
	RegisterModule("mfa-brillouin/kappa", "Temperature dependence of longitudinal susceptibility for finite J", LoadBrillouinKappa)
}

func LoadBrillouinKappa(e *Engine) {

    LoadFullMagnetization(e)
	LoadTemp(e, "Te")
	LoadKappa(e)
	LoadMFAParams(e)
	
	e.Depends("kappa", "Te", "Tc", "msat0", "J", "msat0T0", "n")
	msat0 := e.Quant("msat0")
	msat0T0 := e.Quant("msat0T0")
	kappa := e.Quant("kappa")
	n := e.Quant("n")
	kappa.SetUpdater(&kappaUpdater{kappa: kappa, msat0: msat0, msat0T0: msat0T0, T: e.Quant("Te"), Tc: e.Quant("Tc"), S: e.Quant("J"), n: n})

}

type kappaUpdater struct {
	kappa, msat0, msat0T0, T, Tc, S, n, gamma *Quant
}

func (u *kappaUpdater) Update() {
    kappa := u.kappa
	msat0 := u.msat0
	msat0T0 := u.msat0T0
	T := u.T
	Tc := u.Tc
	S := u.S
	n := u.n
	stream := kappa.Array().Stream
	kappa.Multiplier()[0] = Mu0 / (n.Multiplier()[0] * Kb)
	
	gpu.KappaAsync(kappa.Array(), msat0.Array(), msat0T0.Array(), T.Array(), Tc.Array(), S.Array(), n.Array(), msat0.Multiplier()[0], msat0T0.Multiplier()[0], Tc.Multiplier()[0], S.Multiplier()[0], stream)
	stream.Sync()
}
