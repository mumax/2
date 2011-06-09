//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package client

// This file implements Inter-Process-Communication
// between mumax and a scripting language.

import (
	. "mumax/common"
	"reflect"
	"unicode"
	"fmt"
)

type ipc struct {
	method map[string]reflect.Value
}


// add all exported methods of receiver to the ipc's map
func (c *ipc) init(receiver_ interface{}) {
	c.method = make(map[string]reflect.Value)
	receiver := reflect.ValueOf(receiver_)
	typ := reflect.TypeOf(receiver_)
	for i := 0; i < typ.NumMethod(); i++ {
		name := typ.Method(i).Name
		if unicode.IsUpper(int(name[0])) {
			c.method[name] = receiver.Method(i)
		}
	}
}

// calls the method determined by the funcName with given arguments and returns the return value
func (c *ipc) call(funcName string, args []string) (returnValues []reflect.Value) {
	f, ok := c.method[funcName]
	if !ok {
		panic(IOErr(fmt.Sprintf(msg_no_such_method, funcName)))
	}
	return f.Call(parseArgs(f, args))
}


// parses the argument list "argv" to values suited for the function named by "fname"
func parseArgs(function reflect.Value, argv []string) []reflect.Value {
	nargs := function.Type().NumIn()

	if nargs != len(argv) {
		panic(InputErr(fmt.Sprintf(msg_arg_mismatch, function.String(), nargs, len(argv))))
	}

	args := make([]reflect.Value, nargs)
	for i := range args {
		args[i] = parseArg(argv[i], function.Type().In(i))
	}
	return args
}


// parses a string representation of a given type to a value
func parseArg(arg string, argtype reflect.Type) reflect.Value {
	switch argtype.Name() {
	default:
		panic(Bug(fmt.Sprint(msg_cant_parse, argtype)))
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
	panic(Bug("Bug"))
	//return reflect.ValueOf(666) // is never reached.
}


// error message
const (
	msg_already_defined = "client ipc: %s already defined"
	msg_no_such_method  = "client ipc: no such method: %s"
	msg_no_such_command = "client ipc: no such command: %s. options: %v"
	msg_cant_parse      = "client ipc: do not know how to parse %s"
	msg_arg_mismatch    = "client ipc: %v needs %v arguments, but %v provided"
)
