//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Magnetostatic kernel
// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	"mumax/host"
	"math"
)

// Calculates the magnetostatic kernel
//
// size: size of the kernel, usually 2 x larger than the size of the magnetization due to zero padding
// accuracy: use 2^accuracy integration points
//
// return value: A symmetric rank 5 tensor K[sourcedir][destdir][x][y][z]
// (e.g. K[X][Y][1][2][3] gives H_y at position (1, 2, 3) due to a unit dipole m_x at the origin.
// Only the non-redundant elements of this symmetric tensor are returned: XX, YY, ZZ, YZ, XZ, XY
// You can use the function KernIdx to convert from source-dest pairs like XX to 1D indices:
// K[KernIdx[X][X]] returns K[XX]
func FaceKernel6(size []int, cellsize []float64, periodic []int, accuracy int, kern *host.Array) {
	Debug("Calculating demag kernel", "size:", size, "cellsize:", cellsize, "accuracy:", accuracy, "periodic:", periodic)
	Start("kern_d")
	k := kern.Array

	Assert(len(kern.Array) == 9) // TODO: should be able to change to 6
	CheckSize(kern.Size3D, size)

	B := [3]float64{0, 0, 0} //NewVector()
	R := [3]float64{0, 0, 0} //NewVector()

	x1 := -(size[X] - 1) / 2
	x2 := size[X]/2 - 1
	// support for 2D simulations (thickness 1)
	if size[X] == 1 && periodic[X] == 0 {
		x2 = 0
	}

	y1 := -(size[Y] - 1) / 2
	y2 := size[Y]/2 - 1

	z1 := -(size[Z] - 1) / 2
	z2 := size[Z]/2 - 1

	x1 *= (periodic[X] + 1)
	x2 *= (periodic[X] + 1)
	y1 *= (periodic[Y] + 1)
	y2 *= (periodic[Y] + 1)
	z1 *= (periodic[Z] + 1)
	z2 *= (periodic[Z] + 1)

	R2 := [3]float64{0, 0, 0}   //NewVector()
	pole := [3]float64{0, 0, 0} //NewVector() // position of point charge on the surface

	for s := 0; s < 3; s++ { // source index Ksdxyz
		for x := x1; x <= x2; x++ { // in each dimension, go from -(size-1)/2 to size/2 -1, wrapped. It's crucial that the unused rows remain zero, otherwise the FFT'ed kernel is not purely real anymore.
			xw := Wrap(x, size[X])
			for y := y1; y <= y2; y++ {
				yw := Wrap(y, size[Y])
				for z := z1; z <= z2; z++ {
					zw := Wrap(z, size[Z])
					//R.Set(float64(x)*cellsize[X], float64(y)*cellsize[Y], float64(z)*cellsize[Z])
					R[X] = float64(x) * cellsize[X]
					R[Y] = float64(y) * cellsize[Y]
					R[Z] = float64(z) * cellsize[Z]

					n := accuracy                  // number of integration points = n^2
					u, v, w := s, (s+1)%3, (s+2)%3 // u = direction of source (s), v & w are the orthogonal directions

					R2[X], R2[Y], R2[Z] = 0, 0, 0
					pole[X], pole[Y], pole[Z] = 0, 0, 0

					surface := cellsize[v] * cellsize[w] // the two directions perpendicular to direction s
					charge := surface

					pu1 := cellsize[u] / 2. // positive pole
					pu2 := -pu1             // negative pole

					B[X], B[Y], B[Z] = 0, 0, 0 // accumulates magnetic field
					for i := 0; i < n; i++ {
						pv := -(cellsize[v] / 2.) + cellsize[v]/float64(2*n) + float64(i)*(cellsize[v]/float64(n))
						for j := 0; j < n; j++ {
							pw := -(cellsize[w] / 2.) + cellsize[w]/float64(2*n) + float64(j)*(cellsize[w]/float64(n))

							pole[u] = pu1
							pole[v] = pv
							pole[w] = pw

							R2[X], R2[Y], R2[Z] = R[X]-pole[X], R[Y]-pole[Y], R[Z]-pole[Z]
							r := math.Sqrt(R2[X]*R2[X] + R2[Y]*R2[Y] + R2[Z]*R2[Z])
							B[X] += R2[X] * charge / (4 * PI * r * r * r)
							B[Y] += R2[Y] * charge / (4 * PI * r * r * r)
							B[Z] += R2[Z] * charge / (4 * PI * r * r * r)

							pole[u] = pu2
							R2[X], R2[Y], R2[Z] = R[X]-pole[X], R[Y]-pole[Y], R[Z]-pole[Z]
							r = math.Sqrt(R2[X]*R2[X] + R2[Y]*R2[Y] + R2[Z]*R2[Z])
							B[X] += R2[X] * -charge / (4 * PI * r * r * r)
							B[Y] += R2[Y] * -charge / (4 * PI * r * r * r)
							B[Z] += R2[Z] * -charge / (4 * PI * r * r * r)
						}
					}
					scale := 1 / float64(n*n)
					B[X] *= scale
					B[Y] *= scale
					B[Z] *= scale

					for d := s; d < 3; d++ { // destination index Ksdxyz
						i := FullTensorIdx[s][d]
						k[i][xw][yw][zw] += float32(B[d]) // We have to ADD because there are multiple contributions in case of periodicity
					}
				}
			}
		}
	}
	Stop("kern_d")
}






func FaceKernel6_gpu(size []int, cellsize []float64, periodic []int, accuracy int, kern *host.Array) {
  Debug("Calculating demag kernel", "size:", size, "cellsize:", cellsize, "accuracy:", accuracy, "periodic:", periodic)
  Start("kern_d")
  k := kern.Array

  Assert(len(kern.Array) == 9) // TODO: should be able to change to 6
  CheckSize(kern.Size3D, size)

  
  // on each gpu: initialization Gauss quadrature points for integrations + copy to gpu ___________
  dev_qd_W_10 := make([]cu.DevicePtr, NDevice())
  dev_qd_P_10 := make([]cu.DevicePtr, NDevice())
  
  devices := getDevices()
  for i := range devices {
    setDevice(devices[i])
    dev_qd_W_10[i] = cu.MemAlloc(10*SIZEOF_FLOAT)
    dev_qd_P_10[i] = cu.MemAlloc(30*SIZEOF_FLOAT)
    Initialize_Gauss_quadrature_on_gpu_FaceKernel6(dev_qd_W_10[i], dev_qd_P_10[i], cellSize);
  }
  // ______________________________________________________________________________________________
  
  // allocate array to store one component on the devices _________________________________________
  gpuBuffer := make([]Array, NDev)
  gpuBuffer.Init(9, size, DO_ALLOC)
  // ______________________________________________________________________________________________

  // initialize kernel elements and copy to host __________________________________________________
  comp:=0
  gpuBuffer.Zero()
  for co1:=0; co1<3; co1++{
    for co2:=0; co2<3; co2++{
      InitFaceKernel6Element(&gpuBuffer[comp], co1, co2, periodic, cellSize, dev_qd_P_10, dev_qd_W_10)
      comp++
    }
  }
  k.CopyFromDevice(gpuBuffer)
  // ______________________________________________________________________________________________
  
  // free everything ______________________________________________________________________________
  gpuBuffer.Free()
  for i := range devices {
    setDevice(devices[i])
    dev_qd_W_10[i].Free()
    dev_qd_P_10[i].Free()
  }
  dev_qd_W_10.Free()
  dev_qd_P_10.Free()
  // ______________________________________________________________________________________________

}


func FaceKernel6_gpu(size []int, cellsize []float64, periodic []int, accuracy int, kern *host.Array)

func Initialize_Gauss_quadrature_on_gpu_FaceKernel6(dev_qd_W_10, dev_qd_P_10, cellSize []float64){

  // initialize standard order 10 Gauss quadrature points and weights _____________________________
    std_qd_P_10 := make([]float64, 10)
    std_qd_P_10[0] = -0.97390652851717197f;
    std_qd_P_10[1] = -0.86506336668898498f;
    std_qd_P_10[2] = -0.67940956829902399f;
    std_qd_P_10[3] = -0.43339539412924699f;
    std_qd_P_10[4] = -0.14887433898163099f;
    std_qd_P_10[5] = -std_qd_P_10[4];
    std_qd_P_10[6] = -std_qd_P_10[3];
    std_qd_P_10[7] = -std_qd_P_10[2];
    std_qd_P_10[8] = -std_qd_P_10[1];
    std_qd_P_10[9] = -std_qd_P_10[0];
    host_qd_W_10 := make([]float64, 10)
    host_qd_W_10[0] = host_qd_W_10[9] = 0.066671344308687999f;
    host_qd_W_10[1] = host_qd_W_10[8] = 0.149451349150581f;
    host_qd_W_10[2] = host_qd_W_10[7] = 0.21908636251598201f;
    host_qd_W_10[3] = host_qd_W_10[6] = 0.26926671930999602f;
    host_qd_W_10[4] = host_qd_W_10[5] = 0.29552422471475298f;
  // ______________________________________________________________________________________________

  // Map the standard Gauss quadrature points to the used integration boundaries __________________
    host_qd_P_10 := make([]float64, 30)
    get_Quad_Points_FaceKernel6(&host_qd_P_10[X*10], std_qd_P_10, 10, -0.5f*cellSize[X], 0.5f*cellSize[X]);
    get_Quad_Points_FaceKernel6(&host_qd_P_10[Y*10], std_qd_P_10, 10, -0.5f*cellSize[Y], 0.5f*cellSize[Y]);
    get_Quad_Points_FaceKernel6(&host_qd_P_10[Z*10], std_qd_P_10, 10, -0.5f*cellSize[Z], 0.5f*cellSize[Z]);
  // ______________________________________________________________________________________________

  // copy to the quadrature points and weights to the device ______________________________________
    cu.MemcpyHtoD(cu.DevicePtr(host_qd_W_10), cu.HostPtr(&dev_qd_W_10), 10*SIZEOF_FLOAT)
    cu.MemcpyHtoD(cu.DevicePtr(host_qd_P_10), cu.HostPtr(&dev_qd_P_10), 30*SIZEOF_FLOAT)
    memcpy_to_gpu (host_qd_W_10, dev_qd_W_10, 10);
    memcpy_to_gpu (host_qd_P_10, dev_qd_P_10, 3*10);
  // ______________________________________________________________________________________________

  Free(std_qd_P_10);
  Free(host_qd_P_10);
  Free(host_qd_W_10);

  return;
}

func get_Quad_Points_FaceKernel6(float *gaussQP, float *stdGaussQP []float64, qOrder int, a, b float64){

  A := (b-a)/2.0f; // coefficients for transformation x'= Ax+B
  B := (a+b)/2.0f; // where x' is the new integration parameter
  for i:=0; i<qOrder; i++{
    gaussQP[i] = A*stdGaussQP[i]+B;
  }
  
}


// Magnetostatic field at position r (integer, number of cellsizes away from source) for a given source magnetization direction m (X, Y, or
// s = source direction (x, y, z)
//func faceIntegral(B, R *vector, cellsize []float64, s int, accuracy int) {
//	n := accuracy                  // number of integration points = n^2
//	u, v, w := s, (s+1)%3, (s+2)%3 // u = direction of source (s), v & w are the orthogonal directions
//	R2 := NewVector()
//	pole := NewVector() // position of point charge on the surface
//
//
//	surface := cellsize[v] * cellsize[w] // the two directions perpendicular to direction s
//	charge := surface
//
//	pu1 := cellsize[u] / 2. // positive pole
//	pu2 := -pu1             // negative pole
//
//	B.Set(0., 0., 0.) // accumulates magnetic field
//	for i := 0; i < n; i++ {
//		pv := -(cellsize[v] / 2.) + cellsize[v]/float64(2*n) + float64(i)*(cellsize[v]/float64(n))
//		for j := 0; j < n; j++ {
//			pw := -(cellsize[w] / 2.) + cellsize[w]/float64(2*n) + float64(j)*(cellsize[w]/float64(n))
//
//			pole[u] = pu1
//			pole[v] = pv
//			pole[w] = pw
//
//			R2.SetTo(R)
//			R2.Sub(pole)
//			r := R2.Norm()
//			R2.Normalize()
//			R2.Scale(charge / (4 * math.Pi * r * r))
//			B.Add(R2)
//
//			pole[u] = pu2
//
//			R2.SetTo(R)
//			R2.Sub(pole)
//			r = R2.Norm()
//			R2.Normalize()
//			R2.Scale(-charge / (4 * math.Pi * r * r))
//			B.Add(R2)
//		}
//	}
//	B.Scale(1. / (float64(n * n))) // n^2 integration points
//}
