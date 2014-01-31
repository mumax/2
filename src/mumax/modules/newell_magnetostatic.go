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
	"math"
	. "mumax/common"
	"mumax/host"
	"time"
)

////////////////// Subroutines ///////////////////////

// For accurate summing
func accSum(n int, arr *[]float64) float64 {
	// Quick sort the values
	tmp0 := *arr
	qsort(n,arr)
	sum,corr := tmp0[n-1], float64(0.0)
	for idx := 1; idx<n; idx++ {
		i := n-1-idx
		x:=tmp0[i]
		y:=corr+x
		tmp:=y-corr
		u:=x-tmp
		t:=y+sum
		tmp=t-sum
		v:=y-tmp
		z:=u+v
		sum=t+z
		tmp=sum-t
		corr=z-tmp
	}
	return sum
}

// Special qsort function to sort array by magnitude
func qsort(n int, arr *[]float64) {
	// n = number of elements to sort in arr (beginning from the front)
	// arr = pointer to array to be sorted

	// Do quick check for sorting small arrays
	if (n <= 1) { return } // Do nothing if n is <= 1

	tmp0 := *arr

	if (n == 2) { // If there are only 2 items..
		if (math.Abs(tmp0[1]) < math.Abs(tmp0[0])) { // Swap if magnitudes are not sorted
			tmp1 := tmp0[1]
			tmp0[1] = tmp0[0]
			tmp0[0] = tmp1
		} else {
			return
		}
	}

	// Set pivot
	pvt := tmp0[n-1]
	pvtAbs := math.Abs(pvt)

	// Set up two bins for sorting
	var (
		botArr, topArr []float64
		botCnt, topCnt int
	)

	botCnt = 0
	topCnt = 0

	// Go through the portion of array that needs to be sorted
	for idx:= 0; idx < (n-1); idx++ {
		if (pvtAbs > math.Abs(tmp0[idx])) {
			botCnt++
			botArr = append(botArr,tmp0[idx]);
		} else {
			topCnt++
			topArr = append(topArr,tmp0[idx]);
		}
	}

	// By now, the bottom bin contains elements smaller than pivot
	// and the top bin contains elements larger than or equal to pivot.
	// Recurse through the bins if they are not empty

	if (botCnt > 1) { qsort(botCnt,&botArr) }
	if (topCnt > 1) { qsort(topCnt,&topArr) }

	// All bins are sorted by now. All we have to do is go through the array
	// and place elements in there accordingly
	idx := 0 // Setup index for filling array with sorted values (treat as stack)

	// Fill array with elements from bottom bin if bottom bin is not empty
	if (botCnt != 0) {
		for idx0 := 0; idx0 < botCnt; idx0++ {
			tmp0[idx] = botArr[idx0]
			idx++
		}
	}

	// Place pivot in proper place in array and update pointer to top of stack
	tmp0[idx] = pvt
	idx++

	// Fill array with elements from top bin if top bin is not empty
	if (topCnt != 0) {
		for idx0 := 0; idx0 < topCnt; idx0++ {
			tmp0[idx] = topArr[idx0]
			idx++
		}
	}
	return
}

// This is an upoptimized implementation where we will go through each and every cell to
// calculate the tensor. Hence, we need some additional functions
func CalculateSDA00(x, y, z, dx, dy, dz float64) float64 {
	result := float64(0.0)
	if ( (x == 0.0) && (y == 0.0) && (z == 0.0) ) {
		result = SelfDemagNx(dx,dy,dz)*(4.0*math.Pi*dx*dy*dz)
	} else {
		var arr []float64

		arr = append(arr,-1.0*Newell_f(x+dx,y+dy,z+dz))
		arr = append(arr,-1.0*Newell_f(x+dx,y-dy,z+dz))
		arr = append(arr,-1.0*Newell_f(x+dx,y-dy,z-dz))
		arr = append(arr,-1.0*Newell_f(x+dx,y+dy,z-dz))
		arr = append(arr,-1.0*Newell_f(x-dx,y+dy,z-dz))
		arr = append(arr,-1.0*Newell_f(x-dx,y+dy,z+dz))
		arr = append(arr,-1.0*Newell_f(x-dx,y-dy,z+dz))
		arr = append(arr,-1.0*Newell_f(x-dx,y-dy,z-dz))

		arr = append(arr,2.0*Newell_f(x,y-dy,z-dz))
		arr = append(arr,2.0*Newell_f(x,y-dy,z+dz))
		arr = append(arr,2.0*Newell_f(x,y+dy,z+dz))
		arr = append(arr,2.0*Newell_f(x,y+dy,z-dz))
		arr = append(arr,2.0*Newell_f(x+dx,y+dy,z))
		arr = append(arr,2.0*Newell_f(x+dx,y,z+dz))
		arr = append(arr,2.0*Newell_f(x+dx,y,z-dz))
		arr = append(arr,2.0*Newell_f(x+dx,y-dy,z))
		arr = append(arr,2.0*Newell_f(x-dx,y-dy,z))
		arr = append(arr,2.0*Newell_f(x-dx,y,z+dz))
		arr = append(arr,2.0*Newell_f(x-dx,y,z-dz))
		arr = append(arr,2.0*Newell_f(x-dx,y+dy,z))

		arr = append(arr,-4.0*Newell_f(x,y-dy,z))
		arr = append(arr,-4.0*Newell_f(x,y+dy,z))
		arr = append(arr,-4.0*Newell_f(x,y,z-dz))
		arr = append(arr,-4.0*Newell_f(x,y,z+dz))
		arr = append(arr,-4.0*Newell_f(x+dx,y,z))
		arr = append(arr,-4.0*Newell_f(x-dx,y,z))

		arr = append(arr,8.0*Newell_f(x,y,z))
		result = accSum(27,&arr)
	}
	return result
}

func CalculateSDA11(x, y, z, dx, dy, dz float64) float64 {
	return CalculateSDA00(y,x,z,dy,dx,dz)
}

func CalculateSDA22(x, y, z, dx, dy, dz float64) float64 {
	return CalculateSDA00(z,y,x,dz,dy,dx)
}

func CalculateSDA01(x, y, z, l, h, e float64) float64 {
	if ((x == 0.0) || (y == 0.0)) { return float64(0.0) }

	var arr []float64

	arr = append(arr,-1.0*Newell_g(x-l,y-h,z-e))
	arr = append(arr,-1.0*Newell_g(x-l,y-h,z+e))
	arr = append(arr,-1.0*Newell_g(x+l,y-h,z+e))
	arr = append(arr,-1.0*Newell_g(x+l,y-h,z-e))
	arr = append(arr,-1.0*Newell_g(x+l,y+h,z-e))
	arr = append(arr,-1.0*Newell_g(x+l,y+h,z+e))
	arr = append(arr,-1.0*Newell_g(x-l,y+h,z+e))
	arr = append(arr,-1.0*Newell_g(x-l,y+h,z-e))

	arr = append(arr,2.0*Newell_g(x,y+h,z-e))
	arr = append(arr,2.0*Newell_g(x,y+h,z+e))
	arr = append(arr,2.0*Newell_g(x,y-h,z+e))
	arr = append(arr,2.0*Newell_g(x,y-h,z-e))
	arr = append(arr,2.0*Newell_g(x-l,y-h,z))
	arr = append(arr,2.0*Newell_g(x-l,y+h,z))
	arr = append(arr,2.0*Newell_g(x-l,y,z-e))
	arr = append(arr,2.0*Newell_g(x-l,y,z+e))
	arr = append(arr,2.0*Newell_g(x+l,y,z+e))
	arr = append(arr,2.0*Newell_g(x+l,y,z-e))
	arr = append(arr,2.0*Newell_g(x+l,y-h,z))
	arr = append(arr,2.0*Newell_g(x+l,y+h,z))

	arr = append(arr,-4.0*Newell_g(x-l,y,z))
	arr = append(arr,-4.0*Newell_g(x+l,y,z))
	arr = append(arr,-4.0*Newell_g(x,y,z+e))
	arr = append(arr,-4.0*Newell_g(x,y,z-e))
	arr = append(arr,-4.0*Newell_g(x,y-h,z))
	arr = append(arr,-4.0*Newell_g(x,y+h,z))

	arr = append(arr,8.0*Newell_g(x,y,z))

	return accSum(27,&arr)
}

func CalculateSDA02(x, y, z, dx, dy, dz float64) float64 {
	return CalculateSDA01(x,z,y,dx,dz,dy)
}

func CalculateSDA12(x, y, z, dx, dy, dz float64) float64 {
	return CalculateSDA01(y,z,x,dy,dz,dx)
}

func Newell_Nxx(x, y, z, dx, dy, dz float64) float64 {
	var tmp0 []float64

	tmp0 = append(tmp0,float64(2.0)*Newell_FuncF(x, y, z, dx, dy, dz))
	tmp0 = append(tmp0,float64(-1.0)*Newell_FuncF(x+dx, y, z, dx, dy, dz))
	tmp0 = append(tmp0,float64(-1.0)*Newell_FuncF(x-dx, y, z, dx, dy, dz))

	return accSum(3,&tmp0)/float64(-4.0*math.Pi*dx*dy*dz)
}

func Newell_Nyy(x, y, z, dx, dy, dz float64) float64 {
	return Newell_Nxx(y, x, z, dy, dx, dz)
}

func Newell_Nzz(x, y, z, dx, dy, dz float64) float64 {
	return Newell_Nxx(z, y, x, dz, dy, dx)
}

func Newell_Nxy(x, y, z, dx, dy, dz float64) float64 {
	var tmp0 []float64

	tmp0 = append(tmp0,Newell_FuncG(x, y, z, dx, dy, dz))
	tmp0 = append(tmp0,float64(-1.0)*Newell_FuncG(x-dx, y, z, dx, dy, dz))
	tmp0 = append(tmp0,float64(-1.0)*Newell_FuncG(x, y+dy, z, dx, dy, dz))
	tmp0 = append(tmp0,Newell_FuncG(x-dx, y+dy, z, dx, dy, dz))

	return accSum(4,&tmp0)/float64(-4.0*math.Pi*dx*dy*dz)
}

func Newell_Nxz(x, y, z, dx, dy, dz float64) float64 {
	return Newell_Nxy(x, z, y, dx, dz, dy)
}

func Newell_Nyz(x, y, z, dx, dy, dz float64) float64 {
	return Newell_Nxy(y, z, x, dy, dz, dx)
}

func Newell_FuncF(x, y, z, dx, dy, dz float64) float64 {
	var tmp0 []float64

	tmp0 = append(tmp0,Newell_F1(x, y+dy, z+dz, dx, dy, dz))
	tmp0 = append(tmp0,float64(-1.0)*Newell_F1(x, y, z+dz, dx, dy, dz))
	tmp0 = append(tmp0,float64(-1.0)*Newell_F1(x, y+dy, z, dx, dy, dz))
	tmp0 = append(tmp0,Newell_F1(x, y, z, dx, dy, dz))

	return accSum(4,&tmp0)
}

func Newell_F1(x, y, z, dx, dy, dz float64) float64 {
	var tmp0 []float64

	tmp0 = append(tmp0,Newell_F2(x, y, z))
	tmp0 = append(tmp0,float64(-1.0)*Newell_F2(x, y-dy, z))
	tmp0 = append(tmp0,float64(-1.0)*Newell_F2(x, y, z-dz))
	tmp0 = append(tmp0,Newell_F2(x, y-dy, z-dz))

	return accSum(4,&tmp0)
}

func Newell_F2(x, y, z float64) float64 {
	var tmp0 []float64

	tmp0 = append(tmp0,Newell_f(x, y, z))
	tmp0 = append(tmp0,float64(-1.0)*Newell_f(x, 0, z))
	tmp0 = append(tmp0,float64(-1.0)*Newell_f(x, y, 0))
	tmp0 = append(tmp0,Newell_f(x, 0, 0))

	return accSum(4,&tmp0)
}

func Newell_FuncG(x, y, z, dx, dy, dz float64) float64 {
	var tmp0 []float64

	tmp0 = append(tmp0,Newell_G1(x, y, z, dx, dy, dz))
	tmp0 = append(tmp0,float64(-1.0)*Newell_G1(x, y-dy, z, dx, dy, dz))
	tmp0 = append(tmp0,float64(-1.0)*Newell_G1(x, y, z-dz, dx, dy, dz))
	tmp0 = append(tmp0,Newell_G1(x, y-dy, z-dz, dx, dy, dz))

	return accSum(4,&tmp0)
}

func Newell_G1(x, y, z, dx, dy, dz float64) float64 {
	var tmp0 []float64

	tmp0 = append(tmp0,Newell_G2(x+dx, y, z+dz))
	tmp0 = append(tmp0,float64(-1.0)*Newell_G2(x+dx, y, z))
	tmp0 = append(tmp0,float64(-1.0)*Newell_G2(x, y, z+dz))
	tmp0 = append(tmp0,Newell_G2(x, y, z))

	return accSum(4,&tmp0)
}

func Newell_G2(x, y, z float64) float64 {
	tmp0 := Newell_g(x, y, z)
	tmp1 := float64(-1.0)*Newell_g(x, y, 0)

	return (tmp0 + tmp1)
}

// Newell's f(x, y, z) function
func Newell_f(x, y, z float64) float64 {
	x = math.Abs(x)
	xsq := x*x
	y = math.Abs(y)
	ysq := y*y
	z = math.Abs(z)
	zsq := z*z

	Rsq := xsq+ysq+zsq
	if (Rsq <= 0.0) { return float64(0.0) }
	R:= math.Sqrt(Rsq)

	var piece []float64
	piececount := int(0)
	if (z>0.0) {
	   var temp1 float64
	   var temp2 float64
	   var temp3 float64
	   piece = append(piece, float64(2.0*(2.0*xsq-ysq-zsq)*R))
	   piececount++
	   temp1 = x*y*z
	   if (temp1 > 0.0) {
	      piece = append(piece, float64(-12.0*temp1*math.Atan2(y*z,x*R)))
	      piececount++
	   }
	   temp2 = xsq+zsq
	   if ((y > 0.0) && (temp2>0.0)) {
	      dummy := math.Log(((y+R)*(y+R))/temp2)
	      piece = append(piece, float64(3.0*y*(zsq-xsq)*dummy))
	      piececount++
	   }
	   temp3 = xsq+ysq
	   if (temp3 > 0.0) {
	      dummy := math.Log(((z+R)*(z+R))/temp3)
	      piece = append(piece, float64(3.0*z*(ysq-xsq)*dummy))
	      piececount++
	   }
	} else {
	  if (x==y) {
	    K := 2.0*math.Sqrt(2.0)-6.0*math.Log(1.0+math.Sqrt(2.0))
	    piece = append(piece, float64(K*xsq*x))
	    piececount++
	  } else {
	    piece = append(piece, float64(2.0*(2.0*xsq-ysq)*R))
	    piececount++
	    if ( (y > 0.0) && (x > 0.0)) {
	       piece = append(piece, float64(-6.0*y*xsq*math.Log((y+R)/x)))
	       piececount++
	    }
	  }
	}

	return float64(accSum(piececount,&piece)/12.0)
}

// Newell's g(x, y, z) function
func Newell_g(x, y, z float64) float64 {
	result_sign := float64(1.0)
	if (x < 0.0) { result_sign *= -1.0 }
	if (y < 0.0) { result_sign *= -1.0 }
	x = math.Abs(x)
	xsq := x*x
	y = math.Abs(y)
	ysq := y*y
	z = math.Abs(z)
	zsq := z*z

	Rsq := xsq+ysq+zsq
	if (Rsq <= 0.0) { return float64(0.0) }
	R := math.Sqrt(Rsq)

	var piece []float64
	piececount := int(0)
	piece = append(piece, float64(-2.0*x*y*R))
	piececount++

	if (z > 0.0) {
	   piece = append(piece, float64(-z*zsq*math.Atan2(x*y,z*R)))
	   piececount++
	   piece = append(piece, float64(-3.0*z*ysq*math.Atan2(x*z,y*R)))
	   piececount++
	   piece = append(piece, float64(-3.0*z*xsq*math.Atan2(y*z,x*R)))
	   piececount++

	   var temp1 float64
	   var temp2 float64
	   var temp3 float64

	   temp1=xsq+ysq
	   if (temp1>0.0) {
	      piece = append(piece, float64(3.0*x*y*z*math.Log((z+R)*(z+R)/temp1)))
	      piececount++
	   }
	   temp2=ysq+zsq
	   if (temp2>0.0) {
	      piece = append(piece, float64(0.5*y*(3.0*zsq-ysq)*math.Log((x+R)*(x+R)/temp2)))
	      piececount++
	   }
	   temp3=xsq+zsq
	   if (temp3>0.0) {
	      piece = append(piece, float64(0.5*x*(3.0*zsq-xsq)*math.Log((y+R)*(y+R)/temp3)))
	      piececount++
	   }
	} else {
	  if(y>0.0) {
	  	piece = append(piece, float64(-y*ysq*math.Log((x+R)/y)))
		piececount++
	  }
	  if(x>0.0) {
	  	piece = append(piece, float64(-x*xsq*math.Log((y+R)/x)))
		piececount++
	  }
	}
	return float64(result_sign*accSum(piececount,&piece)/6.0)
}

// Some subroutines for accurate self-demag computations
func SelfDemagNx(x, y, z float64) float64 {
	if ((x<=0.0) || (y<=0.0) || (z<=0.0)) {
		return float64(0.0)
	}
	if ((x==y) && (y==z)) {
		return (float64(1.0)/float64(3.0))
	}
	xsq,ysq,zsq := x*x,y*y,z*z
	R := math.Sqrt(xsq+ysq+zsq)
	Rxy := math.Sqrt(xsq+ysq)
	Rxz := math.Sqrt(xsq+zsq)
	Ryz := math.Sqrt(ysq+zsq)

	var arr []float64

	tmp := float64(2.0*x*y*z * ( (x/(x+Rxy)+(2*xsq+ysq+zsq)/(R*Rxy+x*Rxz))/(x+Rxz) + (x/(x+Rxz)+(2*xsq+ysq+zsq)/(R*Rxz+x*Rxy))/(x+Rxy) ) / ((x+R)*(Rxy+Rxz+R)))
	arr = append(arr,tmp)

	tmp = float64(-1.0*x*y*z * ( (y/(y+Rxy)+(2*ysq+xsq+zsq)/(R*Rxy+y*Ryz))/(y+Ryz) + (y/(y+Ryz)+(2*ysq+xsq+zsq)/(R*Ryz+y*Rxy))/(y+Rxy)) / ((y+R)*(Rxy+Ryz+R)))
	arr = append(arr,tmp)

	tmp = float64(-1.0*x*y*z * ( (z/(z+Rxz)+(2*zsq+xsq+ysq)/(R*Rxz+z*Ryz))/(z+Ryz) + (z/(z+Ryz)+(2*zsq+xsq+ysq)/(R*Ryz+z*Rxz))/(z+Rxz)) / ((z+R)*(Rxz+Ryz+R)))
	arr = append(arr,tmp)

	tmp = float64(6.0*math.Atan(y*z/(x*R)))
	arr = append(arr,tmp)

	piece4 := float64(-1.0*y*z*z*(1.0/(x+Rxz)+y/(Rxy*Rxz+x*R))/(Rxz*(y+Rxy)))
	if (piece4 > -0.5) {
		arr = append(arr,float64(3.0 * x * math.Log1p(piece4) / z))
	} else {
		arr = append(arr,float64(3.0 * x * math.Log(x*(y+R)/(Rxz*(y+Rxy))) / z))
	}

	piece5 := float64(-1.0*y*y*z*(1.0/(x+Rxy)+z/(Rxy*Rxz+x*R))/(Rxy*(z+Rxz)))
	if (piece5 > -0.5) {
		arr = append(arr,float64(3.0 * x * math.Log1p(piece5) / y))
	} else {
		arr = append(arr,float64(3.0 * x * math.Log(x*(z+R)/(Rxy*(z+Rxz))) / y))
	}

	piece6 := float64(-1.0 * xsq * z *(1/(y+Rxy) + z/(Rxy*Ryz+y*R)) / (Rxy*(z+Ryz)))
	if (piece6 > -0.5) {
		arr = append(arr,float64(-3.0 * y *math.Log1p(piece6) / x))
	} else {
		arr = append(arr,float64(-3.0 * y * math.Log(y*(z+R)/(Rxy*(z+Ryz))) / x))
	}

	piece7 := float64(-1.0 * xsq * y * (1/(z+Rxz) + y/(Rxz*Ryz+z*R)) / (Rxz*(y+Ryz)))
	if (piece7 > -0.5) {
		arr = append(arr,float64(-3.0 * z * math.Log1p(piece7) / x))
	} else {
		arr = append(arr,float64(-3.0 * z * math.Log(z*(y+R)/(Rxz*(y+Ryz))) / x))
	}

	return accSum(8,&arr) / (3.0 * math.Pi)
}

func SelfDemagNy(xsize, ysize, zsize float64) float64 {
     return SelfDemagNx(ysize,zsize,xsize)
}

func SelfDemagNz(xsize, ysize, zsize float64) float64 {
     return SelfDemagNx(zsize,xsize,ysize)
}

//////////////////////////////////////////////////////////////////////////
// Greatest common divisor via Euclid's algorithm
func Gcd(m, n float64) float64 {
     Assert((m == math.Floor(m)) && (n == math.Floor(n)))
     m = math.Floor(math.Abs(m))
     n = math.Floor(math.Abs(n))
     if ((m == 0.0) || (n == 0.0)) { return 0.0 }
     temp := m - math.Floor(m/n)*n
     for (temp > 0.0) {
     		m = n
		n = temp
		temp = m - math.Floor(m/n)*n
     }
     return n
}

//////////////////////////////////////////////////////////////////////////
// Simple continued fraction like rational approximator.  Unlike the usual
// continued fraction expansion, here we allow +/- on each term (rounds to
// nearest integer rather than truncating).  This converges faster (by
// producing fewer terms).
//   A more useful rat approx routine would probably take an error
// tolerance rather than a step count, but we'll leave that until the time
// when we have an actual use for this routine.  (There is a 4 term
// recursion relation for continued fractions in "Numerical Recipes in C"
// that can probably be used, and also remove the need for the coef[]
// array.)  -mjd, 23-Jan-2000. Takes x and y as separate imports, and also
// takes error tolerance imports. Fills exports p and q with best result,
// and returns 1 or 0 according to whether or not p/q meets the specified
// relative error.
func FindRatApprox(x float64, y float64, relerr float64, maxq float64, p *float64, q *float64) bool {
     sign := 1
     if(x<0.0) {
     	     x *= -1.0
	     sign *= -1
     }
     if(y<0.0) {
     	     y *= -1.0
	     sign *= -1
     }
     swap := false
     if (x<y) {
     	t := x
	x = y
	y = t
	swap = true
     }

     x0 := x
     y0 := y

     p0, p1 := float64(0.0), float64(1.0)
     q0, q1 := float64(1.0), float64(0.0)

     m := math.Floor(x/y)
     r := x - m*y
     p2 := m*p1 + p0
     q2 := m*q1 + q0
     flag := (q2<maxq) && (math.Abs(x0*q2 - p2*y0) > relerr*x0*q2)
     for (flag) {
     	   x = y
	   y = r
	   m = math.Floor(x/y)
	   r = x - m*y
	   p0 = p1
	   p1 = p2
	   p2 = m*p1 + p0
	   q0 = q1
	   q1 = q2
	   q2 = m*q1 + q0
     	   flag = (q2<maxq) && (math.Abs(x0*q2 - p2*y0) > relerr*x0*q2)
     }

     if (!swap) {
     	*p = p2
	*q = q2
     } else {
        *p = q2
	*q = p2
     }

     flag = (math.Abs(x0*q2 - p2*y0) <= relerr*x0*q2)
     return flag
}

func GetNextPowerOfTwo(n int) int {
     m := 1
     logsize := 0
     for (m < n) {
     	 m *= 2
	logsize += 1
	 Assert(m>0)
     }
     return m
}

/////////////////// Main Code ////////////////////////

// Calculates the magnetostatic kernel by brute-force integration
// of magnetic charges over the faces and averages over cell volumes.
// 
// size: size of the kernel, usually 2 x larger than the size of the magnetization due to zero padding
// accuracy: use 2^accuracy integration points
//
// return value: A symmetric rank 5 tensor K[sourcedir][destdir][x][y][z]
// (e.g. K[X][Y][1][2][3] gives H_y at position (1, 2, 3) due to a unit dipole m_x at the origin.
// You can use the function KernIdx to convert from source-dest pairs like XX to 1D indices:
// K[KernIdx[X][X]] returns K[XX]
func Kernel_Newell(size []int, cellsize []float64, periodic []int, asymptotic_radius, zero_self_demag int, kern *host.Array) {
//func Kernel_Newell(size []int, cellsize []float64, periodic []int, accuracy_ int, kern *host.Array) {

	Debug("Calculating demag kernel:", "size", size)

	// Sanity check
	{
		Assert(size[0] > 0 && size[1] > 1 && size[2] > 1)
		Assert(cellsize[0] > 0 && cellsize[1] > 0 && cellsize[2] > 0)
		Assert(periodic[0] == 0 && periodic[1] == 0 && periodic[2] == 0)
		// TODO: in case of PBC, this will not be met?
		Assert(size[1]%2 == 0 && size[2]%2 == 0)
		if size[0] > 1 {
			Assert(size[0]%2 == 0)
		}
	}

	array := kern.Array

	// Field (destination) loop ranges
	x1, x2 := -(size[X]-1)/2, size[X]/2-1
	y1, y2 := -(size[Y]-1)/2, size[Y]/2-1
	z1, z2 := -(size[Z]-1)/2, size[Z]/2-1
	// support for 2D simulations (thickness 1)
	if size[X] == 1 && periodic[X] == 0 {
		x2 = 0
	}
	{ // Repeat for PBC:
		x1 *= (periodic[X] + 1)
		x2 *= (periodic[X] + 1)
		y1 *= (periodic[Y] + 1)
		y2 *= (periodic[Y] + 1)
		z1 *= (periodic[Z] + 1)
		z2 *= (periodic[Z] + 1)
	}

	dx, dy, dz := cellsize[X], cellsize[Y], cellsize[Z]
	// Determine relative sizes of dx, dy and dz, since that is all that demag
	// calculation cares about
	
	var (
		p1, p2, q1, q2 float64
	)

	if ( FindRatApprox(dx,dy,1e-12,1000,&p1,&q1) && FindRatApprox(dz,dy,1e-12,1000,&p2,&q2) ) {
	   gcd := Gcd(q1,q2)
	   dx = p1*q2/gcd
	   dy = q1*q2/gcd
	   dz = p2*q1/gcd
	} else {
	   maxedge := dx
	   if (dy>maxedge) { maxedge = dy }
	   if (dz>maxedge) { maxedge = dz }
	   dx /= maxedge
	   dy /= maxedge
	   dz /= maxedge
	}

	// Start brute integration
	// 9 nested loops, does that stress you out?
	// Fortunately, the 5 inner ones usually loop over just one element.
	// It might be nice to get rid of that branching though.
	var (
		R  [3]float64     // cell center positions
		points int        // counts used integration points
	)
	
	t0 := time.Now()
	
	for x := x1; x <= x2; x++ { // in each dimension, go from -(size-1)/2 to size/2 -1, wrapped.
		xw := Wrap(x, size[X])
		R[X] = float64(x) * dx

		for y := y1; y <= y2; y++ {
			yw := Wrap(y, size[Y])
			R[Y] = float64(y) * dy

			for z := z1; z <= z2; z++ {
				zw := Wrap(z, size[Z])
				R[Z] = float64(z) * dz

				var B float64

				I := FullTensorIdx[0][0]
				B = CalculateSDA00(R[X], R[Y], R[Z], dx, dy, dz)/(-4.0*math.Pi*dx*dy*dz)
				array[I][xw][yw][zw] += float32(B)

				I = FullTensorIdx[1][1]
				B = CalculateSDA11(R[X], R[Y], R[Z], dx, dy, dz)/(-4.0*math.Pi*dx*dy*dz)
				array[I][xw][yw][zw] += float32(B)

				I = FullTensorIdx[2][2]
				B = CalculateSDA22(R[X], R[Y], R[Z], dx, dy, dz)/(-4.0*math.Pi*dx*dy*dz)
				array[I][xw][yw][zw] += float32(B)

				I = FullTensorIdx[0][1]
				B = CalculateSDA01(R[X], R[Y], R[Z], dx, dy, dz)/(-4.0*math.Pi*dx*dy*dz)
				array[I][xw][yw][zw] += float32(B)
				I = FullTensorIdx[1][0]
				array[I][xw][yw][zw] += float32(B)

				I = FullTensorIdx[0][2]
				B = CalculateSDA02(R[X], R[Y], R[Z], dx, dy, dz)/(-4.0*math.Pi*dx*dy*dz)
				array[I][xw][yw][zw] += float32(B)
				I = FullTensorIdx[2][0]
				array[I][xw][yw][zw] += float32(B)

				I = FullTensorIdx[1][2]
				B = CalculateSDA12(R[X], R[Y], R[Z], dx, dy, dz)/(-4.0*math.Pi*dx*dy*dz)
				array[I][xw][yw][zw] += float32(B)
				I = FullTensorIdx[2][1]
				array[I][xw][yw][zw] += float32(B)

				Debug("kern at [X, Y, Z] = [", xw, ", ", yw, ", ", zw,"]")
				for s := 0; s < 3; s++ {
					var (
						outVar	[3]float32
					)
					for d := 0; d < 3; d++ {
						I := FullTensorIdx[s][d]
						outVar[d] = array[I][xw][yw][zw]
					}
					Debug("    [ ", outVar," ]")
				}
			}
		}
	}
	t1 := time.Now()
	Debug("kernel used", points, "integration points")
	Debug("kernel calculation took", t1.Sub(t0))
}
