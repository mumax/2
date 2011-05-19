//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// This file implements loading/executing CUDA modules in a multi-GPU context.
// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	cu "cuda/driver"
	"strings"
	"fmt"
)


// INTERNAL
var _modules map[string]cu.Module

// INTERNAL
var _functions map[string]cu.Function

func init(){
	_modules = make(map[string]cu.Module)
	_functions = make(map[string]cu.Function)
}

// Loads a .ptx module for all GPUs.
func ModuleLoad(fname string) {
	Debug("Loading module: ", fname)
	Assert(strings.HasSuffix(fname, ".ptx"))
	
	// load the module into _modules
	name := fname[:len(fname)-len(".ptx")] // module name without .ptx
	_, ok := _modules[name]
	if ok {
		panic(Bug(fmt.Sprintf(ERR_MODULE_LOADED, fname)))
	}
	module := cu.ModuleLoad(fname)
	_modules[name] = module

	// load all functions into _functions
	funcArgs := parsePTXArgTypes(fname)
	for funcName := range funcArgs {
		_, ok := _functions[funcName]
		if ok {
			panic(Bug(fmt.Sprintf(ERR_FUNC_REDEFINED, funcName, fname)))
		}
		_functions[funcName] = module.GetFunction(funcName)
	}
}

// Error message
const(
	 ERR_MODULE_LOADED = "module already loaded: %s"
	 ERR_FUNC_REDEFINED = "function already defined: %s (%s)"
)

// Makes a new closure (function+arguments) from code in the module.
//func (m *Module) MakeClosure(funcName string, argCount int) Closure {
//	var c Closure
//	c.DeviceClosure = make([]cu.Closure, DeviceCount())
//	for i := range c.DeviceClosure {
//		c.DeviceClosure[i] = cu.Close(m.DevMod[i].GetFunction(funcName), (argCount))
//	}
//	c.ArgCount = argCount
//	return c
//}


// Multi-GPU analog of cuda/driver/Closure.
type Closure struct {
	DeviceClosure []cu.Closure // INTERNAL: separate closures for each GPU
	ArgCount      int          // INTERNAL: number of function arguments
}


// Sets the same argument for all GPUs.
func (c *Closure) SetArg(argIdx int, arg interface{}) {
	Assert(argIdx < c.ArgCount)
	for _, dc := range c.DeviceClosure {
		dc.SetArg(argIdx, arg)
	}
}

// Sets the same arguments for all GPUs.
func (c *Closure) SetArgs(args ...interface{}) {
	Assert(len(args) <= c.ArgCount)
	for i, arg := range args {
		for _, dc := range c.DeviceClosure {
			dc.SetArg(i, arg)
		}
	}
}

// Sets an argument for a specific GPU.
func (c *Closure) SetDeviceArg(deviceId, argIdx int, arg interface{}) {
	Assert(argIdx < c.ArgCount)
	Assert(deviceId < DeviceCount())
	c.DeviceClosure[deviceId].SetArg(argIdx, arg)
}

func (c *Closure) Go() {
	for _, dc := range c.DeviceClosure {
		dc.Go()
	}
}

func (c *Closure) Synchronize() {
	for _, dc := range c.DeviceClosure {
		dc.Synchronize()
	}
}

func (c *Closure) Call() {
	c.Go()
	c.Synchronize()
}

func (c *Closure) Configure1D(Nidx, N int) {
	// fixes argument Nidx, distributing N over the GPUs
}
