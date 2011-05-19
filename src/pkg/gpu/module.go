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
)


// Loads a __global__ CUDA function from a .ptx file.
// E.g.:
//	function := Global("testmodule", "testMemset")
// loads the function testMemset() form testmodule.ptx, where
// testmodule.ptx is searched for automatically.
func Global(modname, funcname string) Closure {
	AssureModule(modname)
	module := _modules[modname]
	function := module.GetFunction(funcname) // take from module to prevent accidental wrong module name
	Assert(function == _functions[funcname])
	argTypes := _funcArgs[funcname]
	var c Closure
	c.DevClosure = make([]cu.Closure, DeviceCount())
	for i := range c.DevClosure {
		c.DevClosure[i] = cu.Close(function, len(argTypes))
	}
	c.ArgCount = len(argTypes)
	return c
}


// Multi-GPU analog of cuda/driver/Closure.
type Closure struct {
	DevClosure []cu.Closure // INTERNAL: separate closures for each GPU
	ArgCount      int          // INTERNAL: number of function arguments
}


// Sets the same argument for all GPUs.
func (c *Closure) SetArg(argIdx int, arg interface{}) {
	Assert(argIdx < c.ArgCount)
	for _, dc := range c.DevClosure {
		dc.SetArg(argIdx, arg)
	}
}

// Sets the same arguments for all GPUs.
func (c *Closure) SetArgs(args ...interface{}) {
	Assert(len(args) <= c.ArgCount)
	for i, arg := range args {
		for _, dc := range c.DevClosure {
			dc.SetArg(i, arg)
		}
	}
}

// Sets an argument for a specific GPU.
func (c *Closure) SetDeviceArg(deviceId, argIdx int, arg interface{}) {
	Assert(argIdx < c.ArgCount)
	Assert(deviceId < DeviceCount())
	c.DevClosure[deviceId].SetArg(argIdx, arg)
}


// Asynchronous call. 
// Executes the closure with its currently set arguments
// and does not wait for the result.
func (c *Closure) Go() {
	for _, dc := range c.DevClosure {
		dc.Go()
	}
}


// Blocks until the previous Go() call has been completed.
func (c *Closure) Synchronize() {
	for _, dc := range c.DevClosure {
		dc.Synchronize()
	}
}


// Synchronous call. 
// Executes the closure with its currently set arguments
// and waits for the result.
func (c *Closure) Call() {
	c.Go()
	c.Synchronize()
}


func (c *Closure) Configure1D(argName string, N int) {

}
