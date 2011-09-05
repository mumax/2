//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine


import (
	. "mumax/common"
	"os"
)


type API struct {
	Engine *Engine
}


func (a API) SetSize(x, y, z int) {
	a.Engine.SetSize([]int{z, y, x}) // convert to internal axes
	a.Engine.InitMicromagnetism()
}

func (a API) SetConst(name string, value float32) {
	e := a.Engine
	q := e.GetQuant("name")
	q.SetScalar(value)
}

func (a API) Get(name string) interface{} {
	return 42
}


func (a API) SaveGraph(file string) {
	f, err := os.Create(file)
	defer f.Close()
	CheckIO(err)
	a.Engine.WriteDot(f)
	Debug("Wrote", file)
}
