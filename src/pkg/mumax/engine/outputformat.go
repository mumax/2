//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Auhtor: Arne Vansteenkiste

import(
		. "mumax/common"
		"io"
)

func init() {
	outputformats = make(map[string]OutputFormat)
}

// global map of registered output formats
var outputformats map[string]OutputFormat

type OutputFormat interface {
	Name() string // Name to register the format under. E.g. "ascii"
	Write(out io.Writer, q*Quant)
}

// registers an output format
func RegisterOutputFormat(format OutputFormat) {
	outputformats[format.Name()] = format
}

// Retrieves an output format from its name. E.g. "ascii"
func GetOutputFormat(name string)OutputFormat{
	f,ok:=outputformats[name]
	if !ok{
		options := ""
		for k,_ := range outputformats{
			options += k + " "
		}
		panic(IOErr("Unknown output format: " + name + ". Options are: " + options))
	}
	return f
}
