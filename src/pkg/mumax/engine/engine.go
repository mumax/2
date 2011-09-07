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
// An acyclic graph structure consisting of interconnected quantities
// determines what should be calculated and when.
type Engine struct {
	size3D_   [3]int            // INTENRAL
	size3D    []int             // size of the FD grid, nil means not yet set
	cellSize_ [3]float64        // INTENRAL
	cellSize  []float64         // size of the FD cells, nil means not yet set
	quantity  map[string]*Quant // maps quantity names onto their data structures
	ode       [][2]*Quant       // quantities coupled by differential equations: d ode[i][0] / d t = ode[i][1]
	time      *Quant            // time quantity is always present
	dt        *Quant            // time step quantity is always present
}


// Left-hand side and right-hand side indices for Engine.ode[i]
const (
	LHS = 0
	RHS = 1
)


// Make new engine.
func NewEngine() *Engine {
	e := new(Engine)
	e.init()
	return e
}


// initialize
func (e *Engine) init() {
	e.quantity = make(map[string]*Quant)
	// special quantities time and dt are always present
	e.AddQuant("t", SCALAR, VALUE)
	e.AddQuant("dt", SCALAR, VALUE)
	e.time = e.GetQuant("t")
	e.dt = e.GetQuant("dt")
}


//__________________________________________________________________ set/get


// Sets the FD grid size
func (e *Engine) SetGridSize(size3D []int) {
	Debug("Engine.SetGridSize", size3D)
	Assert(len(size3D) == 3)
	if e.size3D == nil {
		e.size3D = e.size3D_[:]
		copy(e.size3D, size3D)
	} else {
		panic(InputErr("Grid size already set"))
	}
}


// Gets the FD grid size
func (e *Engine) GridSize() []int {
	if e.size3D == nil {
		panic(InputErr("Grid size should be set first"))
	}
	return e.size3D
}


// Sets the FD cell size
func (e *Engine) SetCellSize(size []float64) {
	Debug("Engine.SetCellSize", size)
	Assert(len(size) == 3)
	if e.cellSize == nil {
		e.cellSize = e.cellSize_[:]
		copy(e.cellSize, size)
	} else {
		panic(InputErr("Cell size already set"))
	}
}


// Gets the FD cell size
func (e *Engine) CellSize() []float64 {
	if e.cellSize == nil {
		panic(InputErr("Cell size should be set first"))
	}
	return e.cellSize
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


// Add an arbitrary quantity
func (e *Engine) AddQuant(name string, nComp int, kind QuantKind) {
	Debug("engine.Add", name, nComp, size3D, kind)

	// quantity should not yet be defined
	if _, ok := e.quantity[name]; ok {
		panic(Bug("engine: Already defined: " + name))
	}

	e.quantity[name] = newQuant(name, nComp, size3D, kind)
}


	// AddQuant(name, nComp, VALUE)
func(e*Engine) AddValue(name string, nComp int){
	e.AddQuant(name, nComp, VALUE)
}

	// AddQuant(name, nComp, FIELD)
func(e*Engine) AddField(name string, nComp int){
	e.AddQuant(name, nComp, FIELD)
}

	// AddQuant(name, nComp, MAP)
func(e*Engine) AddMap(name string, nComp int){
	e.AddQuant(name, nComp, MAP)
}


// Mark childQuantity to depend on parentQuantity.
// Multiply adding the same dependency has no effect.
func (e *Engine) Depends(childQuantity, parentQuantity string) {
	child := e.GetQuant(childQuantity)
	parent := e.GetQuant(parentQuantity)

	for _, p := range child.parents {
		if p.name == parentQuantity {
			return // Dependency is already defined, do not add it twice
			//panic(Bug("Engine.addDependency(" + childQuantity + ", " + parentQuantity + "): already present"))
		}
	}

	child.parents = append(child.parents, parent)
	parent.children = append(parent.children, child)
}


// Add a 1st order differential equation:
//	d y / d t = diff
// E.g.: ODE1("m", "torque")
// No direct dependency should be declared between the arguments.
func (e *Engine) ODE1(y, diff string) {
	yQ := e.GetQuant(y)
	dQ := e.GetQuant(diff)
	if e.ode != nil {
		for _, ode := range e.ode {
			for _, q := range ode {
				if q.Name() == y || q.Name() == diff {
					panic(Bug("Already in ODE: " + y + ", " + diff))
				}
			}
		}
	}
	e.ode = append(e.ode, [2]*Quant{yQ, dQ})
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
	str += "ODEs:\n"
	for _, ode := range e.ode {
		str += "d " + ode[0].Name() + " / d t = " + ode[1].Name() + "\n"
	}
	return str
}


// Write .dot file for graphviz, 
// representing the physics graph.
func (e *Engine) WriteDot(out io.Writer) {
	fmt.Fprintln(out, "digraph Physics{")
	fmt.Fprintln(out, "rankdir=LR")

	// Add quantities
	quants := e.quantity
	for k, v := range quants {
		fmt.Fprintln(out, k, " [shape=box, group=", k[0:1], "];") // use first letter as group name.
		// Add dependencies
		for _, c := range v.children {
			fmt.Fprintln(out, k, "->", c.name, ";")
		}
	}

	fmt.Fprintln(out, "subgraph solver{")
	fmt.Fprintln(out, "rank=sink;")
	// Add ODE node
	for i, _ := range e.ode {
		ODE := "solver" + fmt.Sprint(i)
		fmt.Fprintln(out, ODE+" [style=filled, shape=box];")
	}
	fmt.Fprintln(out, "}")

	// Add ODE node
	for i, ode := range e.ode {
		ODE := "ODE" + fmt.Sprint(i)
		fmt.Fprintln(out, ODE+" [style=filled, shape=box];")
		fmt.Fprintln(out, ODE, "->", ode[0].Name(), ";")
		fmt.Fprintln(out, ode[1].Name(), "->", ODE, ";")
		fmt.Fprintln(out, "{rank=source;", ode[LHS].Name(), "};")
		fmt.Fprintln(out, "{rank=sink;", ode[RHS].Name(), "};")
	}

	// align similar nodes
	i := 0
	for _, a := range quants {
		j := 0
		for _, b := range quants {
			if i < j {
				if similar(a.Name(), b.Name()) {
					fmt.Fprintln(out, "{rank=same;", a.Name(), ";", b.Name(), "};")
				}
			}
			j++
		}
		i++
	}

	fmt.Fprintln(out, "}")
}


// true if a and b are similar names, to be equally ranked in the dot graph.
func similar(a, b string) (similar bool) {
	defer func() {
		if recover() != nil {
			return
		}
	}()
	similar = a[0] == b[0] && a[1] == '_' && b[1] == '_'
	return
}
