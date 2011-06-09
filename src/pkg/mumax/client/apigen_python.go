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


type python struct{}

func (p *python) fileExt() string {
	return "py"
}

func (p *python) comment() string {
	return "#"
}

func (p *python) writeHeader(out io.Writer) {

}


func (p *python) writeFooter(out io.Writer) {

}

func (p *python) writeFunc(out io.Writer, name string, argTypes []reflect.Type, returnType reflect.Type) {
	fmt.Fprintln(out)
	fmt.Fprint(out, "def ", name, "(")

	args := ""
	for i := range argTypes {
		if i != 0 {
			args += ", "
		}
		args += "arg" + fmt.Sprint(i+1)
	}
	fmt.Fprintln(out, args, "):")
	if len(args) != 0 {
		args = args + ","
	}
	fmt.Fprintln(out, "\tprint", name, ",", args, "\"\\n\"")
	fmt.Fprintln(out, "\tstdout.flush()")
}
