//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine


import (
	. "mumax/common"
	"fmt"
	"io"
)


// Engine is the heart of a multiphysics simulation.
// The engine stores named quantities like "m", "B", "alpha", ...
// A data structure consisting of interconnected quantities
// determines what should be updated and when.
type Engine struct {
	size3D   []int             // size of the FD grid
	quantity map[string]*Quant // maps quantity names onto their data structures
}


// Make new engine.
func NewEngine() *Engine {
	e := new(Engine)
	e.init()
	return e
}


// initialize
func (e *Engine) init() {
	e.quantity = make(map[string]*Quant)
}


//__________________________________________________________________ set/get

// Sets the FD grid size
func (e *Engine) SetSize(size3D []int) {
	Debug("engine:SetSize", size3D)
	Assert(len(size3D) == 3)
	if e.size3D == nil {
		e.size3D = make([]int, 3)
	} else {
		panic(InputErr("Size already set"))
	}
	copy(e.size3D, size3D)
}


// Gets the FD grid size
func (e *Engine) Size() []int {
	if e.size3D == nil {
		panic(InputErr("Size should be set first"))
	}
	return e.size3D
}


// retrieve a quantity by its name
func (e *Engine) GetQuant(name string) *Quant {
	if q, ok := e.quantity[name]; ok {
		return q
	} else {
		panic(InputErr("engine: undefined: " + name))
	}
	return nil //silence gc
}

//__________________________________________________________________ add

// Add a scalar quantity
func (e *Engine) AddScalar(name string) {
	e.AddQuant(name, 1, nil)
}


// Adds a vector field
func (e *Engine) AddVectorField(name string) {
	e.AddQuant(name, 3, e.Size())
}


// INTERNAL: add an arbitrary quantity
func (e *Engine) AddQuant(name string, nComp int, size3D []int) {
	Debug("engine.Add", name, nComp, size3D)
	// quantity should not yet be defined
	if _, ok := e.quantity[name]; ok {
		panic(Bug("engine: Already defined: " + name))
	}
	e.quantity[name] = newQuant(name, nComp, size3D)
}


// Mark childQuantity to depend on parentQuantity
func (e *Engine) Depends(childQuantity, parentQuantity string) {
	child := e.GetQuant(childQuantity)
	parent := e.GetQuant(parentQuantity)

	for _, p := range child.parents {
		if p.name == parentQuantity {
			panic(Bug("engine:addDependency(" + childQuantity + ", " + parentQuantity + "): already present"))
		}
	}

	child.parents = append(child.parents, parent)
	parent.children = append(parent.children, child)
}


//__________________________________________________________________ output

// String representation
func (e *Engine) String() string {
	str := "engine\n"
	quants := e.quantity
	for k, v := range quants {
		str += "\t" + k + "("
		for _, p := range v.parents {
			str += p.name + " "
		}
		str += ")\n"
	}
	return str
}


// Write .dot file for graphviz, 
// representing the physics graph.
func (e *Engine) WriteDot(out io.Writer) {
	fmt.Fprintln(out, "digraph Physics{")
	fmt.Fprintln(out, "rankdir=LR")
	quants := e.quantity
	for k, v := range quants {
		fmt.Fprintln(out, k, " [shape=box];")
		for _, c := range v.children {
			fmt.Fprintln(out, k, "->", c.name, ";")
		}
	}
	fmt.Fprintln(out, "}")
}
