//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	"io"
	"fmt"
)

// Table refers to an open data table 
// to which space-independent output is appended during the simulation.
type Table struct {
	out io.Writer
}

// New table that will write in the file.
func NewTable(fname string) *Table {
	t := new(Table)
	t.out = OpenWRONLY(fname)
	return t
}

// Append the quantities value to the table.
func (t *Table) Tabulate(quants []string) {
	e := GetEngine()
	for _, q := range quants {
		quant := e.Quant(q)
		checkKinds(quant, VALUE, MASK)
		v := quant.multiplier
		for _, num := range v {
			fmt.Fprint(t.out, num, "\t")
		}
	}
	fmt.Fprintln(t.out)
}
