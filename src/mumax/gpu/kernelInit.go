//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// Magnetostatic kernel
// Author: Ben Van de Wiele

import (
	cu "cuda/driver"
	. "mumax/common"
	"mumax/host"
	"unsafe"
	//   "fmt"
)

func InitDipoleKernel6(size []int, cellsize []float64, periodic []int, accuracy int, kern *host.Array) {
	Debug("Calculating demag kernel", "size:", size, "cellsize:", cellsize, "accuracy:", accuracy, "periodic:", periodic)
	Start("kern_d")

	Assert(len(kern.Array) == 9) // TODO: should be able to change to 6
	CheckSize(kern.Size3D, size)

	// initialization Gauss quadrature points for integrations + copy to gpu ________________________
	dev_qd_W_10 := make([]cu.DevicePtr, NDevice())
	dev_qd_P_10 := make([]cu.DevicePtr, NDevice())
	Initialize_Gauss_quadrature(dev_qd_W_10, dev_qd_P_10, cellsize)
	// ______________________________________________________________________________________________

	// allocate array to store one component on the devices _________________________________________
	gpuBuffer := NewArray(1, size)
	defer gpuBuffer.Free()
	// ______________________________________________________________________________________________

	// initialize kernel elements and copy to host __________________________________________________
	for comp := 0; comp < 9; comp++ {
		gpuBuffer.Zero()
		InitDipoleKernel6Element(gpuBuffer, comp, periodic, cellsize, dev_qd_P_10, dev_qd_W_10)
		gpuBuffer.CopyToHost(kern.Component(comp))
	}
	// ______________________________________________________________________________________________

	// free everything ______________________________________________________________________________
	devices := getDevices()
	for i := range devices {
		setDevice(devices[i])
		dev_qd_W_10[i].Free()
		dev_qd_P_10[i].Free()
	}
	// ______________________________________________________________________________________________

	Stop("kern_d")
}

func InitRotorKernel(size []int, cellsize []float64, periodic []int, accuracy int, kern *host.Array) {
	Debug("Calculating demag kernel", "size:", size, "cellsize:", cellsize, "accuracy:", accuracy, "periodic:", periodic)
	Start("kern_r")

	Assert(len(kern.Array) == 9) // TODO: should be able to change to 6
	CheckSize(kern.Size3D, size)

	// initialization Gauss quadrature points for integrations + copy to gpu ________________________
	dev_qd_W_10 := make([]cu.DevicePtr, NDevice())
	dev_qd_P_10 := make([]cu.DevicePtr, NDevice())
	Initialize_Gauss_quadrature(dev_qd_W_10, dev_qd_P_10, cellsize)
	// ______________________________________________________________________________________________

	// allocate array to store one component on the devices _________________________________________
	gpuBuffer := NewArray(1, size)
	defer gpuBuffer.Free()
	// ______________________________________________________________________________________________

	// initialize kernel elements and copy to host __________________________________________________
	for comp := 0; comp < 9; comp++ {
		gpuBuffer.Zero()
		InitRotorKernelElement(gpuBuffer, comp, periodic, cellsize, dev_qd_P_10, dev_qd_W_10)
		gpuBuffer.CopyToHost(kern.Component(comp))
	}
	// ______________________________________________________________________________________________

	// free everything ______________________________________________________________________________
	devices := getDevices()
	for i := range devices {
		setDevice(devices[i])
		dev_qd_W_10[i].Free()
		dev_qd_P_10[i].Free()
	}
	// ______________________________________________________________________________________________

	Stop("kern_r")
}

func InitPointKernel(size []int, cellsize []float64, periodic []int, kern *host.Array) {
	Debug("Calculating demag kernel", "size:", size, "cellsize:", cellsize, "periodic:", periodic)
	Start("kern_p")

	Assert(len(kern.Array) == 3) // TODO: should be able to change to 6
	CheckSize(kern.Size3D, size)

	// initialization Gauss quadrature points for integrations + copy to gpu ________________________
	dev_qd_W_10 := make([]cu.DevicePtr, NDevice())
	dev_qd_P_10 := make([]cu.DevicePtr, NDevice())
	Initialize_Gauss_quadrature(dev_qd_W_10, dev_qd_P_10, cellsize)
	// ______________________________________________________________________________________________

	// allocate array to store one component on the devices _________________________________________
	gpuBuffer := NewArray(1, size)
	defer gpuBuffer.Free()
	// ______________________________________________________________________________________________

	// initialize kernel elements and copy to host __________________________________________________
	for comp := 0; comp < 3; comp++ {
		gpuBuffer.Zero()
		InitPointKernelElement(gpuBuffer, comp, periodic, cellsize, dev_qd_P_10, dev_qd_W_10)
		gpuBuffer.CopyToHost(kern.Component(comp))
	}
	// ______________________________________________________________________________________________

	// free everything ______________________________________________________________________________
	devices := getDevices()
	for i := range devices {
		setDevice(devices[i])
		dev_qd_W_10[i].Free()
		dev_qd_P_10[i].Free()
	}
	// ______________________________________________________________________________________________

	Stop("kern_p")
}

func Initialize_Gauss_quadrature(dev_qd_W_10, dev_qd_P_10 []cu.DevicePtr, cellSize []float64) {

	// initialize standard order 10 Gauss quadrature points and weights _____________________________
	std_qd_P_10 := make([]float64, 10)
	std_qd_P_10[0] = -0.97390652851717197
	std_qd_P_10[1] = -0.86506336668898498
	std_qd_P_10[2] = -0.67940956829902399
	std_qd_P_10[3] = -0.43339539412924699
	std_qd_P_10[4] = -0.14887433898163099
	std_qd_P_10[5] = -std_qd_P_10[4]
	std_qd_P_10[6] = -std_qd_P_10[3]
	std_qd_P_10[7] = -std_qd_P_10[2]
	std_qd_P_10[8] = -std_qd_P_10[1]
	std_qd_P_10[9] = -std_qd_P_10[0]
	host_qd_W_10 := make([]float32, 10)
	host_qd_W_10[0] = 0.066671344308687999
	host_qd_W_10[9] = 0.066671344308687999
	host_qd_W_10[1] = 0.149451349150581
	host_qd_W_10[8] = 0.149451349150581
	host_qd_W_10[2] = 0.21908636251598201
	host_qd_W_10[7] = 0.21908636251598201
	host_qd_W_10[3] = 0.26926671930999602
	host_qd_W_10[6] = 0.26926671930999602
	host_qd_W_10[4] = 0.29552422471475298
	host_qd_W_10[5] = 0.29552422471475298
	// ______________________________________________________________________________________________

	// Map the standard Gauss quadrature points to the used integration boundaries __________________
	host_qd_P_10 := make([]float32, 30)
	get_Quad_Points(host_qd_P_10, std_qd_P_10, 10, -0.5*cellSize[0], 0.5*cellSize[0], 0)
	get_Quad_Points(host_qd_P_10, std_qd_P_10, 10, -0.5*cellSize[1], 0.5*cellSize[1], 1)
	get_Quad_Points(host_qd_P_10, std_qd_P_10, 10, -0.5*cellSize[2], 0.5*cellSize[2], 2)
	// ______________________________________________________________________________________________

	// copy to the quadrature points and weights to the devices _____________________________________
	devices := getDevices()
	for i := range devices {
		setDevice(devices[i])
		dev_qd_W_10[i] = cu.MemAlloc(10 * SIZEOF_FLOAT)
		dev_qd_P_10[i] = cu.MemAlloc(30 * SIZEOF_FLOAT)
		cu.MemcpyHtoD(cu.DevicePtr(dev_qd_W_10[i]), cu.HostPtr(unsafe.Pointer(&host_qd_W_10[0])), 10*SIZEOF_FLOAT)
		cu.MemcpyHtoD(cu.DevicePtr(dev_qd_P_10[i]), cu.HostPtr(unsafe.Pointer(&host_qd_P_10[0])), 30*SIZEOF_FLOAT)
	}
	// ______________________________________________________________________________________________

	return
}

func get_Quad_Points(gaussQP []float32, stdGaussQP []float64, qOrder int, a, b float64, cnt int) {

	A := (b - a) / 2.0 // coefficients for transformation x'= Ax+B
	B := (a + b) / 2.0 // where x' is the new integration parameter
	for i := 0; i < qOrder; i++ {
		gaussQP[cnt*10+i] = float32(A*stdGaussQP[i] + B)
	}

}
