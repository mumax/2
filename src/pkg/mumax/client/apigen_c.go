//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.


package client

import (
	"io"
	"reflect"
	"fmt"
)


type c struct{}

func (x *c) filename() string {
	return "mumax2.h"
}


func (x *c) comment() string {
	return "//"
}

func (x *c) writeHeader(out io.Writer) {
	fmt.Fprintln(out, `
#ifndef MUMAX2_H
#define MUMAX2_H

#ifdef __cplusplus
extern "C" {
#endif

`)
}


func (x *c) writeFooter(out io.Writer) {
	fmt.Fprint(out, `
#ifdef __cplusplus
}
#endif
#endif
`)
}


func (x *c) writeFunc(out io.Writer, funcName string, argTypes []reflect.Type, returnType reflect.Type) {

	// convert the return type to C
	ret := ""
	if returnType != nil {
		ret = returnType.String()
	}
	ret = c_type[ret]

	// make list of args
	args := ""
	for i := range argTypes {
		if i != 0 {
			args += ", "
		}
		args += c_type[argTypes[i].String()] + " "
		args += "arg" + fmt.Sprint(i+1)
	}

	code := fmt.Sprintf(`
%s %s(%s){
}
`,ret, funcName, args)

	fmt.Fprintln(out, code)

	//	fmt.Fprintln(out)
	//
	//	ret := ""
	//	if returnType != nil {
	//		ret = returnType.String()
	//	}
	//
	//	fmt.Fprintf(out, `
	//	public static %s %s(`,java_type[ret], funcName)
	//
	//	args := ""
	//	for i := range argTypes {
	//		if i != 0 {
	//			args += ", "
	//		}
	//		args += java_type[argTypes[i].String()] + " "
	//		args += "arg" + fmt.Sprint(i+1)
	//	}
	//	fmt.Fprintln(out, args, "){")
	//
	//	fmt.Fprintf(out, `		String returned = call("%s", new String[]{`, funcName)
	//
	//	for i := range argTypes {
	//		if i != 0 {
	//			fmt.Fprintf(out, ", ")
	//		}
	//		fmt.Fprintf(out, `"" + arg%v`, i+1)
	//	}
	//	fmt.Fprintln(out, "});")
	//	if returnType != nil {
	//		fmt.Fprintf(out, `		return %s(returned);`, java_parse[ret])
	//		fmt.Fprintln(out)
	//	}
	//	fmt.Fprintln(out, `	}`)
}


var (
	// functions for parsing go types from string
	c_parse map[string]string = map[string]string{"int": "Integer.parseInt",
		"float32": "Float.parseFloat",
		"float64": "Double.parseDouble",
		"bool":    "Boolean.parseBoolean"}
	// maps go types onto java types
	c_type map[string]string = map[string]string{"int": "int",
		"float32": "float",
		"float64": "double",
		"string":  "char*",
		"bool":    "bool",
		"":        "void"}
)
