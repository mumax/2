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
	outputDir string
	quantity  map[string]*Quant
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


// Add a scalar quantity
func (e *Engine) AddScalar(name string) {
	e.addQuant(name, 1, nil)
}


// INTERNAL: add an arbitrary quantity
func (e *Engine) addQuant(name string, nComp int, size3D []int) {
	Debug("engine.Add", name, nComp, size3D)
	// quantity should not yet be defined
	if _, ok := e.quantity[name]; ok {
		panic(Bug("engine: Already defined: " + name))
	}
	e.quantity[name] = newQuant(name, nComp, size3D)
}


// Mark childQuantity to depend on parentQuantity
func (e *Engine) AddDependency(childQuantity, parentQuantity string) {
	child := e.getQuant(childQuantity)
	parent := e.getQuant(parentQuantity)

	for _, p := range child.parents {
		if p.name == parentQuantity {
			panic(Bug("engine:addDependency(" + childQuantity + ", " + parentQuantity + "): already present"))
		}
	}

	child.parents = append(child.parents, parent)
	parent.children = append(parent.children, child)
}


// retrieve a quantity by its name
func (e *Engine) getQuant(name string) *Quant {
		if q,ok :=  e.quantity[name]; ok{
			return q
		}else{
				panic(Bug("engine: undefined: " + name))
		}
		return nil//silence gc
}


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
	quants := e.quantity
	for k, v := range quants {
		fmt.Fprintln(out, k, " [shape=box];")
		for _, c := range v.children {
			fmt.Fprintln(out, k, "->", c.name, ";")
		}
	}
	fmt.Fprintln(out, "}")
}
