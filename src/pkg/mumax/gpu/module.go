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
	"fmt"
)


// Loads a __global__ CUDA function from a .ptx file.
// E.g.:
//	function := Global("testmodule", "testMemset")
// loads the function testMemset() form testmodule.ptx, where
// testmodule.ptx is searched for automatically.
func Global(modname, funcname string) Closure {
	assureModule(modname)
	module := _modules[modname]
	function := module.GetFunction(funcname) // take from module to prevent accidental wrong module name
	Assert(function == _functions[funcname])
	argInfo := _funcArgs[funcname]
	var c Closure
	c.DevClosure = make([]cu.Closure, DeviceCount())
	for i := range c.DevClosure {
		c.DevClosure[i] = cu.Close(function, len(argInfo))
	}
	// Extract argument types and names from the PTX file
	c.ArgType = make([]int, len(argInfo))
	c.ArgPART = -1
	c.ArgN = -1
	for i := range argInfo {
		c.ArgType[i] = argInfo[i].Type
		if argInfo[i].Name == "PART" {
			c.ArgPART = i
		}
		if argInfo[i].Name == "N" {
			c.ArgN = i
		}
		if argInfo[i].Name == "N0" {
			c.ArgN = i
		}
		if argInfo[i].Name == "N1" {
			Assert(i == c.ArgN+1)
		} // N1 must follow N0
		if argInfo[i].Name == "N2" {
			Assert(i == c.ArgN+2)
		} // N2 must follow N1
	}
	return c
}


// Multi-GPU analog of cuda/driver/Closure.
type Closure struct {
	DevClosure []cu.Closure // INTERNAL: separate closures for each GPU
	ArgType    []int        // INTERNAL: types of the arguments (see ptxparse)
	ArgPART    int          // INTERNAL: index of automatically set argument "PART"
	ArgN       int          // INTERNAL: index of automatically set argument "N" (for 1D) or "N0" (for 3D, then "N1", "N2" should immediately follow)
}


// Sets the same argument for all GPUs.
func (c *Closure) SetArg(argIdx int, arg interface{}) {
	Assert(argIdx < len(c.ArgType))

	if arr, ok := arg.(Array); ok { // handle gpu.Array as a special case
		Assert(c.ArgType[argIdx] == u64)
		for i, dc := range c.DevClosure {
			dc.SetDevicePtr(argIdx, arr.DevicePtr(i))
		}
	} else {
		argType := c.ArgType[argIdx]
		for _, dc := range c.DevClosure {
			switch argType {
			default:
				panic(Bug(fmt.Sprintf("can not handle argument type: %v", argType)))
			case s32:
				dc.Seti(argIdx, arg.(int))
			}
		}
	}
}

// Sets the same arguments for all GPUs.
func (c *Closure) SetArgs(args ...interface{}) {
	Assert(len(args) <= len(c.ArgType))
	for i, arg := range args {
		for _, dc := range c.DevClosure {
			dc.SetArg(i, arg)
		}
	}
}

// Sets an argument for a specific GPU.
func (c *Closure) SetDeviceArg(deviceId, argIdx int, arg interface{}) {
	Assert(argIdx < len(c.ArgType))
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


func (c *Closure) Configure1D(N int) {

}


func (c *Closure) Configure3D(fullSize []int) {

}
