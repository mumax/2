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
		)

// the package name
const pkg = "mumax2"

// Auto-generate API libraries.
func APIGen() {
	//// make a refsh shell that will tell us the methods of a mumax client
	//c := new(client.Client)
	//r := refsh.New()
	//r.AddAllMethods(c)

	//// target languages
	//langs := []Lang{&Python{}, &Java{}}

	//// output api files for each language.
	//for _, lang := range langs {
	//	file := pkg + "." + lang.FileExt()
	//	out := FOpen(file)

	//	fmt.Fprintln(out, lang.Comment(), "This file is automatically generated. DO NOT EDIT.\n")

	//	lang.WriteHeader(out)

	//	for i := range r.Funcnames {
	//		var returnType reflect.Type
	//		switch r.Funcs[i].NumOut() {
	//		default:
	//			panic(Bug(""))
	//		case 0:
	//		case 1:
	//			returnType = r.Funcs[i].Out(0)
	//		}
	//		lang.WriteFunc(out, r.Funcnames[i], ArgTypes(r.Funcs[i]), returnType)
	//	}

	//	lang.WriteFooter(out)

	//	out.Close()
	//}
}

// Fetches argument types from refsh.
//func ArgTypes(c refsh.Caller) []reflect.Type {
//	types := make([]reflect.Type, c.NumIn())
//	for i := range types {
//		types[i] = c.In(i)
//	}
//	return types
//}
