//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

// This package implements automated mumax API generation.
// Based on the exported methods of a desired type, an API
// library in any of the supported programming languages is
// automatically generated.

package client

import (
	. "mumax/common"
	"reflect"
	"fmt"
)

// the package name
const pkg = "mumax2"

// Auto-generate API libraries.
func APIGen() {
	// interpreter can extract the methods
	var ipc interpreter
	ipc.init(new(Client))

	// target languages
	langs := []lang{&python{}, &java{}, &lua{}}

	// output api files for each language.
	for _, lang := range langs {
		file := lang.filename()
		Log("Generating", file)
		out := FOpen(file)

		fmt.Fprintln(out, lang.comment(), "This file is automatically generated by mumax2 -apigen. DO NOT EDIT.\n")

		lang.writeHeader(out)

		for name, meth := range ipc.method {
			var returnType reflect.Type
			switch meth.Type().NumOut() {
			default:
				panic(Bug(""))
			case 0:
			case 1:
				returnType = meth.Type().Out(0)
			}
			lang.writeFunc(out, name, ArgTypes(meth), returnType)
		}

		lang.writeFooter(out)

		out.Close()
	}
}

func ArgTypes(c reflect.Value) []reflect.Type {
	types := make([]reflect.Type, c.Type().NumIn())
	for i := range types {
		types[i] = c.Type().In(i)
	}
	return types
}
