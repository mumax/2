//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package apigen

import (
	"fmt"
	"io"
	"reflect"
)

type Lua struct{}

func (l *Lua) Filename() string {
	return "mumax2.lua"
}

func (l *Lua) Comment() string {
	return " -- "
}

func (p *Lua) WriteHeader(out io.Writer) {
	fmt.Fprintln(out, `
`)
}

func (l *Lua) WriteFooter(out io.Writer) {
}

func (l *Lua) WriteFunc(out io.Writer, name string, argTypes []reflect.Type, returnType reflect.Type) {

	// setup args
	args := ""
	for i := range argTypes {
		if i != 0 {
			args += ", "
		}
		args += "arg" + fmt.Sprint(i+1)
	}

	code := fmt.Sprintf(`
function %s (%s)
	
end
`,
		name, args)

	fmt.Fprintln(out, code)
	//fmt.Fprintln(out, args, "):")

	//var retType string
	//if returnType != nil {
	//	retType = returnType.String()
	//}
	//fmt.Fprintln(out, fmt.Sprintf(`	return %s(call("%s", [%s]))`, lua_convert[retType], name, args))
}

var (
	// maps go types to lua types	
	lua_convert map[string]string = map[string]string{"int": "int",
		"float32": "float",
		"float64": "float",
		"string":  "str",
		"bool":    "boolean",
		"":        ""}
)
