//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// This file implements generation of a graphviz .dot file
// representing the physics graph.
// Author: Arne Vansteenkiste

import (
	"fmt"
	"io"
)


// Write .dot file for graphviz, 
// representing the physics graph.
func (e *Engine) WriteDot(out io.Writer) {
	fmt.Fprintln(out, "digraph Physics{")
	fmt.Fprintln(out, "rankdir=LR")

	// Add quantities
	quants := e.quantity
	for k, v := range quants {
		label := ""
		if v.desc != "" {
			label = "label=" + `"` + k + "\\n(" + v.desc + `)"`
		}
		fmt.Fprintln(out, k, " [shape=box, group=", k[0:1], label, "];") // use first letter as group name.
		// Add dependencies
		for _, c := range v.children {
			fmt.Fprintln(out, k, "->", c.name, ";")
		}
	}

	// Add ODE node
	fmt.Fprintln(out, "subgraph cluster0{")
	fmt.Fprintln(out, "rank=sink;")
	for i, _ := range e.ode {
		ODE := "solver" + fmt.Sprint(i)
		fmt.Fprintln(out, ODE+" [style=filled, shape=box];")
	}
	fmt.Fprintln(out, "}")

	// Add ODE node
	for i, ode := range e.ode {
		ODE := "solver" + fmt.Sprint(i)
		//fmt.Fprintln(out, ODE+" [style=filled, shape=box];")
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
