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


// Multi-GPU analog of cuda/driver/Module.
type Module struct {
	DeviceModule []cu.Module // separate modules for each GPU
}

// Loads a .ptx module for all GPUs.
func ModuleLoad(fname string) Module {
	var mod Module
	mod.DeviceModule = make([]cu.Module, DeviceCount())
	for i := range mod.DeviceModule {
		mod.DeviceModule[i] = cu.ModuleLoad(fname)
	}
	Debug("Loaded module: ", fname)
	return mod
}


// Makes a new closure (function+arguments) from code in the module.
func (m *Module) MakeClosure(funcName string, argCount int) Closure {
	var c Closure
	c.DeviceClosure = make([]cu.Closure, DeviceCount())
	for i := range c.DeviceClosure {
		c.DeviceClosure[i] = cu.Close(m.DeviceModule[i].GetFunction(funcName), (argCount))
	}
	c.ArgCount = argCount
	return c
}


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
