//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

//
//	Here the user input (X,Y,Z) is changed to internal input (Z,Y,X)
//

import (
	. "mumax/common"
	"mumax/host"
	"os"
)


// The API methods are accessible to the end-user through scripting languages.
type API struct {
	Engine *Engine
}


//________________________________________________________________________________ init

// Set the grid size 
func (a API) SetGridSize(x, y, z int) {
	a.Engine.SetSize([]int{z, y, x}) // convert to internal axes
	a.Engine.InitMicromagnetism()
}


//________________________________________________________________________________ quant


// Set the value of a scalar, space-independent quantity
func (a API) SetScalar(name string, value float32) {
	e := a.Engine
	q := e.GetQuant(name)
	q.SetScalar(value)
}


// Get the value of a scalar, space-independent quantity
func (a API) GetScalar(name string) float32 {
	return a.Engine.GetQuant(name).ScalarValue()
}


func (a API) LoadVectorField(quant, filename string) {
	panic("unimplemented")
}

func (a API) SetVectorField(quant string, field *host.Array) {
	panic("unimplemented")
}


func (a API) GetField(quant string) *host.Array {
	return a.Engine.GetQuant(quant).FieldValue()
}

// Get the value of a general quantity
func (a API) Get(name string) interface{} {
	e := a.Engine
	q := e.GetQuant(name)
	switch {
	case q.IsScalar():
		return q.ScalarValue()
	}
	panic(Bug("unimplemented case"))
}


//________________________________________________________________________________ misc

func (a API) SaveGraph(file string) {
	f, err := os.Create(file)
	defer f.Close()
	CheckIO(err)
	a.Engine.WriteDot(f)
	Debug("Wrote", file)
}
