//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// This file implements the simulation engine. The engine stores the
// entire simulation state and provides methods to run the simulation.
// An engine is typically steered by a driver, like a python program or
// command line interface.

type Engine struct {

}


func NewEngine() *Engine {
	return new(Engine)
}

// For testing purposes.
func (e *Engine) Version() int {
	return 2
}

// For testing purposes.
func (e *Engine) Echo(i int) int {
	return i
}

// For testing purposes.
func (e *Engine) Sink(b bool, i int, f float32, d float64, s string) {
	return
}

// For testing purposes.
func (e *Engine) GetFloat() float32 {
	return 42.
}

// For testing purposes.
func (e *Engine) GetDouble() float64 {
	return 42.
}

// For testing purposes.
func (e *Engine) GetString() string {
	return "hello"
}


// For testing purposes.
func (e *Engine) GetBool() bool {
	return true
}


// For testing purposes.
func (e *Engine) Sum(i, j int) int {
	return i + j
}
