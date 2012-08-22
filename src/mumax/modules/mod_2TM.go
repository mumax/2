//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

import (
	. "mumax/common"
	. "mumax/engine"
	"mumax/gpu"
)

// Register this module
func init() {
	RegisterModule("2TM", "Two-temperature model module", Load2TM)
}

func Load2TM(e *Engine) {
    //Load Temperatures
    LoadT(e)
    //LoadGes(e)
    
    Te := e.Quant("Te")
    Tl := e.Quant("Tl")
    
    Q := e.AddNewQuant("Q", SCALAR, MASK, Unit("J/s"), "The power of external heater")
    gamma_e := e.AddNewQuant("gamma_e", SCALAR, MASK, Unit("J/(K^2*m^3)"), "The heat capacity of electrons")
    Cl := e.AddNewQuant("Cl", SCALAR, MASK, Unit("J/(K*m^3)"), "The heat capacity of phonons")
    
    Gel := e.AddNewQuant("Gel", SCALAR, MASK, Unit("W/(K*m^3)"), "The electron-phonon coupling coefficient")
    
    ke := e.AddNewQuant("k_e", SCALAR, MASK, Unit("W/(K*m)"), "Heat conductance of electrons")
    kl := e.AddNewQuant("k_l", SCALAR, MASK, Unit("W/(K*m)"), "Heat conductance of phonons")
    
    Qe := e.AddNewQuant("Qe", SCALAR, FIELD, Unit("W/(m^3)"), "The heat flux density of electrons")
    Ql := e.AddNewQuant("Ql", SCALAR, FIELD, Unit("W/(m^3)"), "The heat flux density of phonons")
    
    e.Depends("Qe", "Te", "Tl", "gamma_e", "Gel", "Q", "k_e")
    e.Depends("Ql", "Te", "Tl", "Cl",      "Gel", "k_l")
    
    Qe.SetUpdater(&Qe2TMUpdater{Qe: Qe, Te: Te, Tl: Tl, Q: Q, gamma_e: gamma_e, Gel: Gel, ke: ke })
    Ql.SetUpdater(&Ql2TMUpdater{Ql: Ql, Te: Te, Tl: Tl, Cl: Cl, Gel: Gel, kl: kl })
    
    e.AddPDE1("Te", "Qe")
    e.AddPDE1("Tl", "Ql")
    
}

type Qe2TMUpdater struct {
	Qe, Te, Tl, Q, gamma_e, Gel, ke *Quant
}

type Ql2TMUpdater struct {
	Ql, Te, Tl, Cl, Gel, kl *Quant
}

func (u *Qe2TMUpdater) Update() {
    e := GetEngine()
    pbc := e.Periodic()
    cellSize := e.CellSize()
    
    gpu.Q2TM_async(
        u.Qe.Array(),
        u.Te.Array(),
        u.Tl.Array(),
        u.Q.Array(),
        u.gamma_e.Array(),
        u.Gel.Array(),
        u.ke.Array(),
        u.Q.Multiplier(),
        u.gamma_e.Multiplier(),
        u.Gel.Multiplier(),
        u.ke.Multiplier(),
        int(1),
        cellSize,
        pbc)
    u.Qe.Array().Sync()
}

func (u *Ql2TMUpdater) Update() {
    e := GetEngine()
    pbc := e.Periodic()
    cellSize := e.CellSize()
    QMul := []float64{0.0, 0.0, 0.0}
    Q := gpu.NilArray(u.Ql.Array().NComp(), u.Ql.Array().Size3D())
    gpu.Q2TM_async(
        u.Ql.Array(),
        u.Tl.Array(),
        u.Te.Array(),
        Q,
        u.Cl.Array(),
        u.Gel.Array(),
        u.kl.Array(),
        QMul,
        u.Cl.Multiplier(),
        u.Gel.Multiplier(),
        u.kl.Multiplier(),
        int(0),
        cellSize,
        pbc)
    u.Ql.Array().Sync()
    Q.Free()
}


