//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// This file implements internal units.
// 
// Internal units are used to avoid that the numerical value of any
// quantity in the simulation becomes too large or too small.
// 
// In a previous implementation, the internal units were defined so that:
// Msat == Aexch == mu0 == gamma0 == 1
// However, this required Msat, Aexch to be defined before any other quantity
// could be converted to internal units.
// Therefore, we now choose fixed internal units so that these values are
// of the order of 1, but not necessarily exactly 1.

// Author: Arne Vansteenkiste.

import ()


func init() {
	UpdateUnits()
}


// Physical constants in SI units
// Our primal SI units are:
//	m A s J
// All other units are expressed in terms of these, if possible:
//	T = J/Am2
// Otherwise, the SI unit is used, like Kelvin for temperature.
const (
	Mu0SI    = 4 * PI * 1e-7    // Permeability of vacuum in J/Am2
	Gamma0SI = 2.211E5          // Gyromagnetic ratio in m/As
	KbSI     = 1.380650424E-23  // Boltzmann's constant in J/K
	MuBSI    = 9.2740091523E-24 // Bohr magneton in Am^2
	ESI      = 1.60217646E-19   // Electron charge in As
	PI       = 3.14159265358979323846264338327950288
)


// Primary internal units. 
// Should only be changed once at the beginning of a simulation.
// When changed, UpdateUnits() should be called to update the derived units.
var (
	UnitLength float64 = 1e-9  // m
	UnitField  float64 = 1e6   // A/m
	UnitTime   float64 = 1e-15 // s
	UnitEnergy float64 = 1e-18 // J
)
// Unit temperature = 1K


// Derived internal units.
// Should NOT be set directly. Instead,
// set the primary internal units and call UpdateUnits().
var (
	UnitInduction      float64 // Internal unit of magnetic induction ("B", not "H"), expressed in Tesla.
	UnitSurface        float64 // Internal unit of surface, expressed in m2
	UnitVolume         float64 // Internal unit of volume, expressed in m3
	UnitCurrent        float64 // Internal unit of current, expressed in A
	UnitCurrentDensity float64 // Internal unit of current density, expressed in A/m2
	UnitCharge         float64 // Internal unit of electrical charge, expressed in As
	UnitEnergyDensity  float64 // Internal unit of energy density, expressed in J/m3
	UnitMoment         float64 // Internal unit of magnetic moment, expressed in Am2
)


// Physical constants in internal units.
// Do not change.
var (
	Mu0    float64 // Permeability of vacuum in internal units
	Gamma0 float64 // Gyromagnetic ratio in internal units 
	Kb     float64 // Boltzmann's constant in  internal units
	MuB    float64 // Bohr magneton in  internal units
	E      float64 // Electron charge in  internal units
)


// Updates the derived internal units and physical constants
// after the primary internal units have changed.
func UpdateUnits() {
	UnitSurface = UnitLength * UnitLength
	UnitVolume = UnitLength * UnitLength * UnitLength

	UnitCurrent = UnitField * UnitLength
	UnitCurrentDensity = UnitCurrent / UnitVolume
	UnitCharge = UnitCurrent * UnitTime

	Mu0 = Mu0SI / (UnitEnergy / (UnitCurrent * UnitSurface))
	UnitInduction = UnitField * Mu0

	UnitEnergyDensity = UnitEnergy / UnitVolume

	UnitMoment = UnitCurrent * UnitSurface

	Gamma0 = Gamma0SI / (UnitLength / (UnitCurrent * UnitTime))
	Kb = KbSI / (UnitEnergy / 1)
	MuB = MuBSI / (UnitMoment)
	E = ESI / (UnitCharge)
}
