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
	DevClosure []cu.Closure // INTERNAL: separate closures for each GPU. WARNING: loop over i, not values!
	ArgType    []int        // INTERNAL: types of the arguments (see ptxparse)
	ArgPART    int          // INTERNAL: index of automatically set argument "PART"
	ArgN       int          // INTERNAL: index of automatically set argument "N" (for 1D) or "N0" (for 3D, then "N1", "N2" should immediately follow)
	Configured bool         // INTERNAL: false if no Configure*() has yet been called.
}


// Sets the same argument for all GPUs.
func (c *Closure) SetArg(argIdx int, arg interface{}) {
	Assert(argIdx < len(c.ArgType))

	//Debug("SetArg", argIdx, arg)
	if arr, ok := arg.(*Array); ok { // handle gpu.Array as a special case
		Assert(c.ArgType[argIdx] == u64)
		for i, dc := range c.DevClosure {
			dc.SetDevicePtr(argIdx, arr.DevicePtr(i))
		}
	} else {
		argType := c.ArgType[argIdx]
		for _, dc := range c.DevClosure {
			switch argType {
			default:
				panic(Bug(fmt.Sprintf("* Can not handle argument type: %v", argType)))
			case s32:
				dc.Seti(argIdx, arg.(int))
				//case u64:
				//dc.SetDevicePtr(argIdx, (arg.(*Array)).DevicePtr(i))
			}
		}
	}
}

// Sets the same arguments for all GPUs.
func (c *Closure) SetArgs(args ...interface{}) {
	//Assert(len(args) <= len(c.ArgType))
	for i, arg := range args {
		c.SetArg(i, arg)
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
	if !c.Configured {
		panic(Bug("mumax/gpu: kernel not configured"))
	}
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

func (c *Closure) Configure(gridDim, blockDim []int) {
	// check validity of blockDim, gridDim
	good := true
	// > check against device restrictions
	threads := 1
	for i := range maxBlockDim {
		if blockDim[i] > maxBlockDim[i] || blockDim[i] < 1 {
			good = false
		}
		threads *= blockDim[i]
		if gridDim[i] > maxGridDim[i] || gridDim[i] < 1 {
			good = false
		}
	}
	if threads > maxThreadsPerBlock {
		good = false
	}
	if !good {
		panic(Bug("Invalid launch configuration: " + fmt.Sprint(gridDim, blockDim)))
	}

	// > check against function-specific restrictions.
	funcMaxTPB := c.MaxThreadsPerBlock()
	if threads > funcMaxTPB {
		panic(Bug(fmt.Sprint("Too many threads per block for function: ", threads, ">", funcMaxTPB)))
	}

	for i := range c.DevClosure {
		Debug("setconfig", gridDim, blockDim)
		c.DevClosure[i].SetConfig(gridDim, blockDim)
	}
	c.Configured = true
}

// The maximum number of threads per block (per device) for this specific function.
// Depending on the resources used by the function, it may be less than the maximum
// imposed by the device.
func (c *Closure) MaxThreadsPerBlock() int {
	return c.DevClosure[0].Func.GetAttribute(cu.FUNC_A_MAX_THREADS_PER_BLOCK)
}

func (c *Closure) Configure1D(N int) {
	// configure the kernel launch for N elements
	// with the largest possible number of threads per block
	Assert(N%DeviceCount() == 0)
	Ndev := N / DeviceCount() // number of elements per device

	threads := c.MaxThreadsPerBlock()
	grid := DivUp(Ndev, threads)
	c.Configure([]int{grid, 1, 1}, []int{threads, 1, 1})
	//TODO: allow 2D config for very large N

	// set the special variables N and PART for each device
	for i := range c.DevClosure {
		c.DevClosure[i].Seti(c.ArgN, Ndev) // always set N, it must be present
		if c.ArgPART >= 0 {
			c.DevClosure[i].Seti(c.ArgPART, i)
		} // only set PART if present in source file
	}
}


// Default matrix tile in floats (16x16)
const DEFAULT_TILE = 16

func (c *Closure) Configure2D(size3D []int) {
	Assert(size3D[1]%DeviceCount() == 0)
	N0 := size3D[0]
	N1 := size3D[1] / DeviceCount()
	N2 := size3D[2]

	threads := []int{DEFAULT_TILE, DEFAULT_TILE, 1}
	blocks := []int{DivUp(N1, DEFAULT_TILE), DivUp(N2, DEFAULT_TILE), 1}
	c.Configure(blocks, threads)

	// set the special variables N and PART for each device
	for i := range c.DevClosure {
		c.DevClosure[i].Seti(c.ArgPART, i)
		c.DevClosure[i].Seti(c.ArgN, N0)
		c.DevClosure[i].Seti(c.ArgN+1, N1)
		c.DevClosure[i].Seti(c.ArgN+2, N2)
	}
}
