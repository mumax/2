//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package frontend


import (
	. "mumax/common"
	"io"
	"os"
	"json"
	"fmt"
	"reflect"
)


type jsonIPC struct {
	in  io.Reader
	out io.Writer
	*json.Decoder
	*json.Encoder
	receiver interface{}
	method   map[string]reflect.Value // list of methods that can be called
}


func (j *jsonIPC) Init(in io.Reader, out io.Writer, receiver interface{}) {
	j.in = in
	j.out = out
	j.Decoder = json.NewDecoder(in)
	j.Encoder = json.NewEncoder(out)
	j.receiver = receiver
	j.method = make(map[string]reflect.Value)
	AddMethods(j.method, receiver)
}


func (j *jsonIPC) Run() {
	for {
		v := new(interface{})
		err := j.Decode(v)
		if err == os.EOF {
			break
		}
		CheckErr(err, ERR_IO)

		if array, ok := (*v).([]interface{}); ok {
			Debug("call:", array)
			Assert(len(array) == 2)
			ret := j.Call(array[0].(string), array[1].([]interface{}))
			j.Encode(ret)
		} else {
			panic(IOErr(fmt.Sprint("json: ", *v)))
		}
	}
}

func (j *jsonIPC) Call(funcName string, args []interface{}) []interface{} {
	f, ok := j.method[funcName]
	if !ok {
		panic(fmt.Sprintf(msg_no_such_method, funcName))
	}

	// call
	// convert []interface{} to []reflect.Value  
	argvals := make([]reflect.Value, len(args))
	for i := range argvals {
		argvals[i] = reflect.ValueOf(args[i])
	}
	retVals := f.Call(argvals)

	// convert []reflect.Value to []interface{}
	ret := make([]interface{}, len(retVals))
	for i := range retVals {
		ret[i] = retVals[i].Interface()
	}
	return ret
}


// error message
const (
	//msg_already_defined = "interpreter: %s already defined"
	msg_no_such_method = "interpreter: no such method: %s"
	//msg_no_such_command = "interpreter: no such command: %s. options: %v"
	//msg_cant_parse      = "interpreter: do not know how to parse %s"
	//msg_arg_mismatch    = "interpreter: %v needs %v arguments, but %v provided"
)
