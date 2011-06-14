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


type java struct{}

func (p *java) filename() string {
	return "Mumax2.java"
}


func (j *java) comment() string {
	return "//"
}

func (j *java) writeHeader(out io.Writer) {
	fmt.Fprintln(out, `
public class Mumax2{
	private void recv(){
					
	}
`)
}


func (j *java) writeFooter(out io.Writer) {
	fmt.Fprint(out, `}`)
}


func (j *java) writeFunc(out io.Writer, funcName string, argTypes []reflect.Type, returnType reflect.Type) {
	fmt.Fprintln(out)
	ret := "void"
	if returnType != nil {
		ret = returnType.String()
	}
	fmt.Fprintf(out, `	public static %s %s(`, ret, funcName)

	args := ""
	for i := range argTypes {
		if i != 0 {
			args += ", "
		}
		args += returnType.String() + " "
		args += "arg" + fmt.Sprint(i+1)
	}
	fmt.Fprintln(out, args, "){")

	fmt.Fprintf(out, `		System.out.Print("%s");`, funcName)
	fmt.Fprintln(out)

	for i := range argTypes {
		fmt.Fprintln(out, "\t", "System.out.Print(\"\"+arg", i+1, ");")
	}
	fmt.Fprintln(out, "\t", "System.out.Println();")
	fmt.Fprintln(out, "\t", "System.out.Flush();")
	fmt.Fprintln(out, "}")
}
