//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package refsh

// Caller unifies anything that can be called: a Method or a FuncValue.

// Author: Arne Vansteenkiste

import (
	. "reflect"
)


// Caller unifies anything that can be called:
// a Method or a FuncValue
type Caller interface {
	Call(args []Value) []Value // Call it
	In(i int) Type             // Types of the input parameters
	NumIn() int                // Number of input parameters
}


// Wraps a method in the Caller interface
type MethodWrapper struct {
	reciever Value
	function *FuncValue
}

// Implements Caller
func (m *MethodWrapper) Call(args []Value) []Value {
	methargs := make([]Value, len(args)+1) // todo: buffer in method struct
	methargs[0] = m.reciever
	for i, arg := range args {
		methargs[i+1] = arg
	}
	return m.function.Call(methargs)
}

// Implements Caller
func (m *MethodWrapper) In(i int) Type {
	return (m.function.Type().(*FuncType)).In(i + 1) // do not treat the reciever (1st argument) as an actual argument
}

// Implements Caller
func (m *MethodWrapper) NumIn() int {
	return (m.function.Type().(*FuncType)).NumIn() - 1 // do not treat the reciever (1st argument) as an actual argument
}


// Wraps a function in the Caller interface
type FuncWrapper FuncValue

// Implements Caller
func (f *FuncWrapper) In(i int) Type {
	return (*FuncValue)(f).Type().(*FuncType).In(i)
}

// Implements Caller
func (f *FuncWrapper) NumIn() int {
	return (*FuncValue)(f).Type().(*FuncType).NumIn()
}

// Implements Caller
func (f *FuncWrapper) Call(args []Value) []Value {
	return (*FuncValue)(f).Call(args)
}
