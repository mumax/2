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
func FaceKernel6(size []int, cellsize []float64, accuracy int, periodic []int, kern *host.Array) {
	Debug("Calculating demag kernel", "size:", size, "cellsize:", cellsize, "accuracy:", accuracy, "periodic:", periodic)
	Start("kern_d")
	k := kern.Array

	Assert(len(kern.Array) == 6)
	CheckSize(kern.Size3D, size)

	B := NewVector()
	R := NewVector()

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

	R2 := NewVector()
	pole := NewVector() // position of point charge on the surface

	for s := 0; s < 3; s++ { // source index Ksdxyz
		for x := x1; x <= x2; x++ { // in each dimension, go from -(size-1)/2 to size/2 -1, wrapped. It's crucial that the unused rows remain zero, otherwise the FFT'ed kernel is not purely real anymore.
			xw := Wrap(x, size[X])
			for y := y1; y <= y2; y++ {
				yw := Wrap(y, size[Y])
				for z := z1; z <= z2; z++ {
					zw := Wrap(z, size[Z])
					R.Set(float64(x)*cellsize[X], float64(y)*cellsize[Y], float64(z)*cellsize[Z])

					//faceIntegral(B, R, cellsize, s, accuracy)
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

//							R2.SetTo(R)
//							R2.Sub(pole)
							R2[X], R2[Y], R2[Z] = R[X]-pole[X], R[Y]-pole[Y], R[Z]-pole[Z]

							r := (&R2).Norm()
							(&R2).Normalize()
							(&R2).Scale(charge / (4 * math.Pi * r * r))
							(&B).Add(&R2)

							pole[u] = pu2

							(&R2).SetTo(&R)
							(&R2).Sub(&pole)
							r = (&R2).Norm()
							(&R2).Normalize()
							(&R2).Scale(-charge / (4 * math.Pi * r * r))
							(&B).Add(&R2)
						}
					}
					(&B).Scale(1. / (float64(n * n))) // n^2 integration points

					for d := s; d < 3; d++ { // destination index Ksdxyz
						i := kernIdx[s][d]                // 3x3 symmetric index to 1x6 index
						k[i][xw][yw][zw] += float32(B[d]) // We have to ADD because there are multiple contributions in case of periodicity
					}
				}
			}
		}
	}
	Stop("kern_d")
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

// A 3-component vector
type vector [3]float64

func NewVector() vector {
	return vector([3]float64{0,0,0})
}

func UnitVector(direction int) vector {
	v := NewVector()
	v[direction] = 1.
	return v
}

func (v *vector) Set(x, y, z float64) {
	v[0] = x
	v[1] = y
	v[2] = z
}

func (v *vector) SetTo(other *vector) {
	v[0] = other[0]
	v[1] = other[1]
	v[2] = other[2]
}

func (a *vector) Cross(b *vector) vector {
	var cross vector
	cross[0] = a[1]*b[2] - a[2]*b[1]
	cross[1] = a[0]*b[2] - a[2]*b[0]
	cross[2] = a[0]*b[1] - a[1]*b[0]
	return cross
}

func (a *vector) Dot(b *vector) float64 {
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

func (v *vector) Norm() float64 {
	return float64(math.Sqrt(float64(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])))
}

func (v *vector) Normalize() {
	invnorm := 1. / v.Norm()
	v[0] *= invnorm
	v[1] *= invnorm
	v[2] *= invnorm
}

func (v *vector) Scale(r float64) {
	v[0] *= r
	v[1] *= r
	v[2] *= r
}

func (v *vector) Divide(r float64) {
	v[0] /= r
	v[1] /= r
	v[2] /= r
}

func (v *vector) Sub(other *vector) {
	v[0] -= other[0]
	v[1] -= other[1]
	v[2] -= other[2]
}

func (v *vector) Add(other *vector) {
	v[0] += other[0]
	v[1] += other[1]
	v[2] += other[2]
}
