//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

// refsh is a "reflective shell", an interpreter that parses
// commands, executing them via run-time reflection.
// Usage: first set up a new interpreter:
// sh := refsh.New()
// sh.AddFunc("shortname", Func)
// sh.AddMethod("shortname", &Reciever{...}, "MethodName")
// Then execute some commands:
// sh.Exec(reader)
//
package refsh

import (
	. "mumax/common"
	"reflect"
	"os"
	"io"
	"strings"
	"unicode"
	"fmt"
)


// Makes a new Refsh
func New() *Refsh {
	return NewRefsh()
}

// Error message
const (
	MSG_ALREADY_DEFINED = "refsh: %s already defined"
	MSG_NO_SUCH_METHOD  = "refsh: no such method: %s"
	MSG_NO_SUCH_COMMAND = "refsh: no such command: %s. options: %v"
	MSG_CANT_PARSE = "refsh: do not know how to parse %s"
	MSG_ARG_MISMATCH = "refsh: %v needs %v arguments, but %v provided"
)

// Adds a function to the list of known commands.
// example: refsh.Add("exit", Exit)
func (r *Refsh) AddFunc(funcname string, function interface{}) {
	f := reflect.NewValue(function)

	if r.resolve(funcname) != nil {
		panic(Bug(fmt.Sprintf(MSG_ALREADY_DEFINED, funcname)))
	}
	r.funcnames = append(r.funcnames, funcname)
	r.funcs = append(r.funcs, (*FuncWrapper)(f.(*reflect.FuncValue)))
}


// Adds a method to the list of known commands
// example: refsh.Add("field", reciever, "GetField")
// (command field ... will call reciever.GetField(...))
func (r *Refsh) AddMethod(funcname string, reciever interface{}, methodname string) {
	if r.resolve(funcname) != nil {
		panic(Bug(fmt.Sprintf(MSG_ALREADY_DEFINED, funcname)))
	}

	typ := reflect.Typeof(reciever)
	var f *reflect.FuncValue
	for i := 0; i < typ.NumMethod(); i++ {
		if typ.Method(i).Name == methodname {
			f = typ.Method(i).Func
		}
	}
	if f == nil {
		panic(Bug(fmt.Sprintf(MSG_NO_SUCH_METHOD, methodname)))
	}
	r.funcnames = append(r.funcnames, funcname)
	r.funcs = append(r.funcs, &MethodWrapper{reflect.NewValue(reciever), f})
}

// Adds all the public Methods of the reciever,
// giving them a lower-case command name
func (r *Refsh) AddAllMethods(reciever interface{}) {
	typ := reflect.Typeof(reciever)
	for i := 0; i < typ.NumMethod(); i++ {
		name := typ.Method(i).Name
		if unicode.IsUpper(int(name[0])) {
			r.AddMethod(strings.ToLower(name), reciever, name)
		}
	}
}


// parses and executes the commands read from in
// bash-like syntax:
// command arg1 arg2
// command arg1
// #comment
func (refsh *Refsh) Exec(in io.Reader) {
	for line, eof := ReadNonemptyLine(in); !eof; line, eof = ReadNonemptyLine(in) {
		cmd := line[0]
		args := line[1:]
		refsh.Call(cmd, args)
	}
}


const prompt = ">> "

// starts an interactive command line
// When an error is encountered, the program will not abort
// but print a message and continue
func (refsh *Refsh) Interactive() {
	in := os.Stdin
	fmt.Print(prompt)
	line, eof := ReadNonemptyLine(in)
	for !eof {
		cmd := line[0]
		args := line[1:]
		refsh.Call(cmd, args)
		fmt.Print(prompt)
		line, eof = ReadNonemptyLine(in)
	}
}


func exit() {
	os.Exit(0)
}


// Calls a function. Function name and arguments are passed as strings.
// The function name should first have been added by refsh.Add();
func (refsh *Refsh) Call(fname string, argv []string) (returnvalue []interface{}) {
	refsh.CallCount++

	function := refsh.resolve(fname)
	if function == nil {
		err:= InputErr(fmt.Sprintf(MSG_NO_SUCH_COMMAND, fname, refsh.funcnames))
		panic(err)
	} else {
		args := refsh.parseArgs(fname, argv)
		retval := function.Call(args)
		ret := make([]interface{}, len(retval))
		for i := range retval {
			ret[i] = retval[i].Interface()
		}
		returnvalue = ret
	}
	panic(Bug("BUG"))
	return
}

type Refsh struct {
	funcnames    []string          // known function or method names (we do not use a map to not exclude the possibility of overloading)
	funcs        []Caller          // functions/methods corresponding to funcnames
	help         map[string]string // help strings corresponding to funcnames
	CrashOnError bool              // crash the program on a syntax error or just report it (e.g. for interactive mode)
	CallCount    int               //counts number of commands executed
}


func NewRefsh() *Refsh {
	refsh := new(Refsh)
	CAPACITY := 10 // Initial function name capacity, but can grow
	refsh.funcnames = make([]string, CAPACITY)[0:0]
	refsh.funcs = make([]Caller, CAPACITY)[0:0]
	refsh.CrashOnError = true
	// built-in functions
	//refsh.AddMethod("include", refsh, "Include")
	return refsh
}


// executes the file
//func (refsh *Refsh) Include(file string) {
//	in, err := os.Open(file, os.O_RDONLY, 0666)
//	if err != nil {
//		panic(err)
//	}
//	refsh.Exec(in)
//}
