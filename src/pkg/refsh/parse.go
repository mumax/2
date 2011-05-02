//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package refsh


// This file implements parsing of function arguments, represented by strings, to values.
// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	"reflect"
	"fmt"
)


// INTERNAL
// Parses the argument list "argv" to values suited for the function named by "fname".
func (refsh *Refsh) parseArgs(fname string, argv []string) []reflect.Value {
	function := refsh.resolve(fname)
	nargs := function.NumIn()

	if nargs != len(argv) {
		panic(InputErr(fmt.Sprintf(MSG_ARG_MISMATCH, fname, nargs, len(argv))))
	}

	args := make([]reflect.Value, nargs)
	for i := range args {
		args[i] = parseArg(argv[i], function.In(i))
	}
	return args
}


// INTERNAL
// Parses a string representation of a given type to a value
// TODO: we need to return Value, err
func parseArg(arg string, argtype reflect.Type) reflect.Value {
	switch argtype.Name() {
	default:
		panic(Bug(fmt.Sprint(MSG_CANT_PARSE, argtype)))
	case "int":
		return reflect.ValueOf(Atoi(arg))
	case "int64":
		return reflect.ValueOf(Atoi64(arg))
	case "float32":
		return reflect.ValueOf(Atof32(arg))
	case "float64":
		return reflect.ValueOf(Atof64(arg))
	case "bool":
		return reflect.ValueOf(Atob(arg))
	case "string":
		return reflect.ValueOf(arg)
	}
	panic("Bug")
	return reflect.ValueOf(666) // is never reached.
}
