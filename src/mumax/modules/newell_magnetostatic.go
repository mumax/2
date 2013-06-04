//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Magnetostatic kernel
// Author: Kelvin Fong

import (
	"math"
	"sort"
	. "mumax/common"
	"mumax/host"
)

// Define some required data types
type Newell1DFFT struct {
	fftsize int // 
}

type Newell1DFFTS struct {
	fftsize int // 
}

type Newell3DFFT struct {
	fftx Newell1DFFT // 
	ffty Newell1DFFTS // 
	fftz Newell1DFFTS // 
}

type DemagNabData struct {
	x, y, z float64
	tx2, ty2, tz2 float64
	R, iR float64
	R2, iR2 float64
}

type DemagAsymptoticRefineData struct {
	rdx, rdy, rdz float64
	result_scale float64
	xcount, ycount, zcount int
}

type DemagNxxAsymptoticBase struct {
	cubic_cell int
	self_demag, lead_weight float64
	a1, a2, a3, a4, a5, a6 float64
	b1, b2, b3, b4, b5, b6, b7, b8, b9, b10 float64
	c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15 float64
}

type DemagNxxAsymptotic struct {
	refine_data DemagAsymptoticRefineData
	Nxx DemagNxxAsymptoticBase
}

// Some subroutines required for other subroutines
func asCompare(px *float64, py *float64) int {
     x := *px
     y := *py
     if (x < y) { return 1 }
     if (x > y) { return -1}
     return 0
}

func accSum(n int, arr *[]float64) float64 {
	// Quick sort the values
	tmp0 := *arr
	sort.Float64s(tmp0)
	sum,corr := tmp0[n-1], float64(0.0)
	for i := n-1; i>=0; i-- {
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

// Some subroutines required for computations
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

	var arrPt *[]float64
	arr := *arrPt

	arr[0] = 2.0 *x*y*z* ( (x/(x+Rxy)+(2*xsq+ysq+zsq)/(R*Rxy+x*Rxz))/(x+Rxz) + (x/(x+Rxz)+(2*xsq+ysq+zsq)/(R*Rxz+x*Rxy))/(x+Rxy) ) / ((x+R)*(Rxy+Rxz+R))
	arr[1] = -1.0 *x*y*z* ( (y/(y+Rxy)+(2*ysq+xsq+zsq)/(R*Rxy+y*Ryz))/(y+Ryz) + (y/(y+Ryz)+(2*ysq+xsq+zsq)/(R*Ryz+y*Rxy))/(y+Rxy) ) / ((y+R)*(Rxy+Ryz+R))
	arr[2] = -1.0 *x*y*z* ( (z/(z+Rxz)+(2*zsq+xsq+ysq)/(R*Rxz+z*Ryz))/(z+Ryz) + (z/(z+Ryz)+(2*zsq+xsq+ysq)/(R*Ryz+z*Rxz))/(z+Rxz) ) / ((z+R)*(Rxz+Ryz+R))
	arr[3] = 6.0 * math.Atan(y*z/(x*R))

	piece4 := -y*zsq*(1/(x+Rxz)+y/(Rxy*Rxz+x*R))/(Rxz*(y+Rxy))
	if (piece4 > -0.5) {
		arr[4] = 3.0 * x * math.Log1p(piece4)/z
	} else {
		arr[4] = 3.0 * x * math.Log(x*(y+R)/(Rxz*(y+Rxy)))/z
	}

	piece5 := -z*ysq*(1/(x+Rxy)+z/(Rxy*Rxz+x*R))/(Rxy*(z+Rxz))
	if (piece5 > -0.5) {
		arr[5] = 3.0 * x * math.Log1p(piece5)/y
	} else {
		arr[5] = 3.0 * x * math.Log(x*(z+R)/(Rxy*(z+Rxz)))/y
	}

	piece6 := -z*xsq*(1/(y+Rxy)+z/(Rxy*Ryz+y*R))/(Rxy*(z+Ryz))
	if (piece6 > -0.5) {
		arr[6] = -3.0 * y * math.Log1p(piece6)/x
	} else {
		arr[6] = -3.0 * y * math.Log(x*(z+R)/(Rxy*(z+Rxz)))/y
	}

	piece7 := -y*xsq*(1/(z+Rxz)+y/(Rxz*Ryz+z*R))/(Rxz*(y+Ryz))
	if (piece7 > -0.5) {
		arr[7] = -3.0 * z * math.Log1p(piece7)/x
	} else {
		arr[7] = -3.0 * z * math.Log(z*(y+R)/(Rxz*(y+Ryz)))/x
	}

	Nxx := accSum(8,arrPt) / (3.0 * math.Pi)
	return Nxx
}

func SelfDemagNy(xsize, ysize, zsize float64) float64 {
     return SelfDemagNx(ysize,zsize,xsize)
}

func SelfDemagNz(xsize, ysize, zsize float64) float64 {
     return SelfDemagNx(zsize,xsize,ysize)
}

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
	var piecePt *[]float64
	piece := *piecePt
	piececount := int(0)
	if (z>0.0) {
	   var temp1 float64
	   var temp2 float64
	   var temp3 float64
	   piece[piececount] = 2.0*(2.0*xsq-ysq-zsq)*R
	   piececount++
	   temp1 = x*y*z
	   if (temp1 > 0.0) {
	      piece[piececount] = -12.0*temp1*math.Atan2(y*z,x*R)
	      piececount++
	   }
	   temp2 = xsq+zsq
	   if ((y > 0.0) && (temp2>0.0)) {
	      dummy := math.Log(((y+R)*(y+R))/temp2)
	      piece[piececount] = 3.0*y*(zsq-xsq)*dummy
	      piececount++
	   }
	   temp3 = xsq+ysq
	   if (temp3 > 0.0) {
	      dummy := math.Log(((z+R)*(z+R))/temp3)
	      piece[piececount] = 3.0*z*(ysq-xsq)*dummy
	      piececount++
	   }
	} else {
	  if (x==y) {
	    K := 2.0*math.Sqrt(2.0)-6.0*math.Log(1.0+math.Sqrt(2.0))
	    piece[piececount] = K*xsq*x
	    piececount++
	  } else {
	    piece[piececount] = 2.0*(2.0*xsq-ysq)*R
	    piececount++
	    if ( (y > 0.0) && (x > 0.0)) {
	       piece[piececount] = -6.0*y*xsq*math.Log((y+R)/x)
	       piececount++
	    }
	  }
	}

	return float64(accSum(piececount,piecePt)/12.0)
}

func CalculateSDA00(x, y, z, dx, dy, dz float64) float64 {
	result := float64(0.0)
	if ( (x == 0.0) && (y == 0.0) && (z == 0.0) ) {
	   result = SelfDemagNx(dx,dy,dz)*(4.0*math.Pi*dx*dy*dz)
	} else {
	   var arrPt *[]float64
	   arr := *arrPt
	   arr[0] = -1.0*Newell_f(x+dx,y+dy,z+dz)
	   arr[1] = -1.0*Newell_f(x+dx,y-dy,z+dz)
	   arr[2] = -1.0*Newell_f(x+dx,y-dy,z-dz)
	   arr[3] = -1.0*Newell_f(x+dx,y+dy,z-dz)
	   arr[4] = -1.0*Newell_f(x-dx,y+dy,z-dz)
	   arr[5] = -1.0*Newell_f(x-dx,y+dy,z+dz)
	   arr[6] = -1.0*Newell_f(x-dx,y-dy,z+dz)
	   arr[7] = -1.0*Newell_f(x-dx,y-dy,z-dz)

	   arr[8] = 2.0*Newell_f(x,y-dy,z-dz)
	   arr[9] = 2.0*Newell_f(x,y-dy,z+dz)
	   arr[10] = 2.0*Newell_f(x,y+dy,z+dz)
	   arr[11] = 2.0*Newell_f(x,y+dy,z-dz)
	   arr[12] = 2.0*Newell_f(x+dx,y+dy,z)
	   arr[13] = 2.0*Newell_f(x+dx,y,z+dz)
	   arr[14] = 2.0*Newell_f(x+dx,y,z-dz)
	   arr[15] = 2.0*Newell_f(x+dx,y-dy,z)
	   arr[16] = 2.0*Newell_f(x-dx,y-dy,z)
	   arr[17] = 2.0*Newell_f(x-dx,y,z+dz)
	   arr[18] = 2.0*Newell_f(x-dx,y,z-dz)
	   arr[19] = 2.0*Newell_f(x-dx,y+dy,z)

	   arr[20] = -4.0*Newell_f(x,y-dy,z)
	   arr[21] = -4.0*Newell_f(x,y+dy,z)
	   arr[22] = -4.0*Newell_f(x,y,z-dz)
	   arr[23] = -4.0*Newell_f(x,y,z+dz)
	   arr[24] = -4.0*Newell_f(x+dx,y,z)
	   arr[25] = -4.0*Newell_f(x-dx,y,z)

	   arr[26] = 8.0*Newell_f(x-dx,y+dy,z)

	   result = 8.0*accSum(27,arrPt)
	}
	return result
}

func CalculateSDA11(x, y, z, dx, dy, dz float64) float64 {
	return CalculateSDA00(y,x,z,dy,dx,dz)
}

func CalculateSDA22(x, y, z, dx, dy, dz float64) float64 {
	return CalculateSDA00(z,y,x,dz,dy,dx)
}

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

	var piecePt *[]float64
	piece := *piecePt
	piececount := int(0)
	piece[piececount] = -2.0*x*y*R
	piececount++

	if (z > 0.0) {
	   piece[piececount] = -z*zsq*math.Atan2(x*y,z*R)
	   piececount++
	   piece[piececount] = -3.0*z*ysq*math.Atan2(x*z,y*R)
	   piececount++
	   piece[piececount] = -3.0*z*xsq*math.Atan2(y*z,x*R)
	   piececount++

	   var temp1 float64
	   var temp2 float64
	   var temp3 float64

	   temp1=xsq+ysq
	   if (temp1>0.0) {
	      piece[piececount] = 3.0*x*y*z*math.Log((z+R)*(z+R)/temp1)
	      piececount++
	   }
	   temp2=ysq+zsq
	   if (temp2>0.0) {
	      piece[piececount] = 0.5*y*(3.0*zsq-ysq)*math.Log((x+R)*(x+R)/temp2)
	      piececount++
	   }
	   temp3=xsq+zsq
	   if (temp3>0.0) {
	      piece[piececount] = 0.5*x*(3.0*zsq-xsq)*math.Log((y+R)*(y+R)/temp3)
	      piececount++
	   }
	} else {
	  if(y>0.0) {
	  	    piece[piececount] = -y*ysq*math.Log((x+R)/y);
		    piececount++
	  }
	  if(x>0.0) {
	  	    piece[piececount] = -x*xsq*math.Log((y+R)/x);
		    piececount++
	  }
	}
	return float64(result_sign*accSum(piececount,piecePt)/6.0)
}

func CalculateSDA01(x, y, z, l, h, e float64) float64 {
     if ((x == 0.0) || (y == 0.0)) { return float64(0.0) }
     var arrPt *[]float64
     arr := *arrPt
     arr[0] = -1.0*Newell_g(x-l,y-h,z-e)
     arr[1] = -1.0*Newell_g(x-l,y-h,z+e)
     arr[2] = -1.0*Newell_g(x+l,y-h,z+e)
     arr[3] = -1.0*Newell_g(x+l,y-h,z-e)
     arr[4] = -1.0*Newell_g(x+l,y+h,z-e)
     arr[5] = -1.0*Newell_g(x+l,y+h,z+e)
     arr[6] = -1.0*Newell_g(x-l,y+h,z+e)
     arr[7] = -1.0*Newell_g(x-l,y+h,z-e)

     arr[8] =  2.0*Newell_g(x,y+h,z-e)
     arr[9] =  2.0*Newell_g(x,y+h,z+e)
     arr[10] =  2.0*Newell_g(x,y-h,z+e)
     arr[11] =  2.0*Newell_g(x,y-h,z-e)
     arr[12] =  2.0*Newell_g(x-l,y-h,z)
     arr[13] =  2.0*Newell_g(x-l,y+h,z)
     arr[14] =  2.0*Newell_g(x-l,y,z-e)
     arr[15] =  2.0*Newell_g(x-l,y,z+e)
     arr[16] =  2.0*Newell_g(x+l,y,z+e)
     arr[17] =  2.0*Newell_g(x+l,y,z-e)
     arr[18] =  2.0*Newell_g(x+l,y-h,z)
     arr[19] =  2.0*Newell_g(x+l,y+h,z)

     arr[20] = -4.0*Newell_g(x-l,y,z)
     arr[21] = -4.0*Newell_g(x+l,y,z)
     arr[22] = -4.0*Newell_g(x,y,z+e)
     arr[23] = -4.0*Newell_g(x,y,z-e)
     arr[24] = -4.0*Newell_g(x,y-h,z)
     arr[25] = -4.0*Newell_g(x,y+h,z)

     arr[26] =  8.0*Newell_g(x,y,z)

     return accSum(27,arrPt)

}

func CalculateSDA02(x, y, z, dx, dy, dz float64) float64 {
	return CalculateSDA01(x,z,y,dx,dz,dy)
}

func CalculateSDA12(x, y, z, dx, dy, dz float64) float64 {
	return CalculateSDA01(y,z,x,dy,dz,dx)
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

func RecommendSize(sz int) int {
     return GetNextPowerOfTwo(sz)
}

func (s *Newell1DFFT) SetDimensions(import_csize int) {
	s.fftsize = import_csize/2
}

func (s *Newell1DFFTS) SetDimensions(import_csize int) {
	s.fftsize = import_csize
}

func (s *Newell1DFFT) GetScaling() float64 {
	if (s.fftsize > 0) {
		return float64(1.0)/float64(s.fftsize)
	}
	return 1.0
}

func (s *Newell1DFFTS) GetScaling() float64 {
	if (s.fftsize > 0) {
		return float64(1.0)/float64(s.fftsize)
	}
	return 1.0
}

func (s *Newell1DFFT) GetLogicalDimension() int {
	if (s.fftsize > 0) {
		return 2*s.fftsize
	}
	return 1
}

func (s *Newell1DFFTS) GetLogicalDimension() int {
	if (s.fftsize > 0) {
		return s.fftsize
	}
	return 1
}

func (s *Newell3DFFT) RecommendDimensions(rdim1, rdim2, rdim3 int, cdim1, cdim2, cdim3 *int) {
	csize1 := RecommendSize(rdim1)
	csize2 := RecommendSize(rdim2)
	csize3 := RecommendSize(rdim3)
	*cdim1 = (csize1/2)+1
	*cdim2 = csize2
	*cdim3 = csize3
}

func (s *Newell3DFFT) SetDimensions(in_cdim1, in_cdim2, in_cdim3 int) {
	cdim1 := in_cdim1
	cdim2 := in_cdim2
	cdim3 := in_cdim3

	if (cdim1 == 1) {
		s.fftx.SetDimensions(1)
	} else {
		s.fftx.SetDimensions(2*(cdim1-1))
	}
	s.ffty.SetDimensions(cdim2)
	s.fftz.SetDimensions(cdim3)
}

func (s *Newell3DFFT) GetLogicalDimension(ldim1, ldim2, ldim3 *int) {
	*ldim1 = s.fftx.GetLogicalDimension()
	*ldim2 = s.ffty.GetLogicalDimension()
	*ldim3 = s.fftz.GetLogicalDimension()
}

func (s *DemagNabData) Set(import_x, import_y, import_z float64) {
	s.x, s.y, s.z = import_x, import_y, import_z
	x2 := s.x*s.x
	y2 := s.y*s.y
	s.R2 = x2 + y2
	z2 := s.z*s.z
	s.R2 += z2
	R4 := s.R2*s.R2
	s.R = math.Sqrt(s.R2)
	if (s.R2 != 0.0) {
		s.tx2 = x2 / R4
		s.ty2 = y2 / R4
		s.tz2 = z2 / R4
		s.iR2 = float64(1.0 / s.R2)
		s.iR = float64(1.0 / s.R)
	} else {
		s.tx2, s.ty2, s.tz2 = 0.0, 0.0, 0.0
		s.iR2, s.R, s.iR = 0.0, 0.0, 0.0
	}
}

func DemagNabData_SetPair(ixa, iya, iza, ixb, iyb, izb float64, pta, ptb *DemagNabData) {
	pta.Set(ixa,iya,iza)
	ptb.Set(ixb,iyb,izb)
}

func (s *DemagAsymptoticRefineData) DemagAsymptoticRefineData(dx, dy, dz, maxratio float64) {
	s.rdx, s.rdy, s.rdz = 0.0, 0.0, 0.0
	s.result_scale = 0.0
	s.xcount, s.ycount, s.zcount = 0, 0, 0
	if (dz <= dx && dz <= dy) {
		xratio := math.Ceil(dx/(maxratio*dz))
		s.xcount = int(xratio)
		s.rdx = float64(dx/xratio)
		yratio := math.Ceil(dy/(maxratio*dz))
		s.ycount = int(yratio)
		s.rdy = float64(dy/yratio)
		s.zcount = 1
		s.rdz = dz
	} else if (dy <= dx && dy <= dz) {
		xratio := math.Ceil(dx/(maxratio*dy))
		s.xcount = int(xratio)
		s.rdx = float64(dx/xratio)
		zratio := math.Ceil(dz/(maxratio*dy))
		s.zcount = int(zratio)
		s.rdz = float64(dz/zratio)
		s.ycount = 1
		s.rdy = dy
	} else {
		yratio := math.Ceil(dy/(maxratio*dx))
		s.ycount = int(yratio)
		s.rdy = float64(dy/yratio)
		zratio := math.Ceil(dz/(maxratio*dx))
		s.zcount = int(zratio)
		s.rdz = float64(dz/zratio)
		s.ycount = 1
		s.rdx = dx
	}
	s.result_scale = float64(1.0) / ( float64(s.xcount) * float64(s.ycount) * float64(s.zcount) )
}

func NxxAsymptotic(x, y, z float64) float64 {
	ptdata := new(DemagNabData)
	ptdata.Set(x,y,z)
	return DemagNxxAsymptotic_NxxAsymptotic(ptdata)
}

func DemagNxxAsymptotic_NxxAsymptotic(ptdata *DemagNabData) float64{
	return float64(0.0)
}

func DemagNxxAsymptoticBase_DemagNxxAsymptoticBase(refine_data *DemagAsymptoticRefineData) {
//	dx, dy, dz := refine_data.rdx, refine_data.rdy, refine_data.rdz
}

// Calculates the magnetostatic kernel by Newell's formulation
// 
// size: size of the kernel, usually 2 x larger than the size of the magnetization due to zero padding
// accuracy: use 2^accuracy integration points
//
// return value: A symmetric rank 5 tensor K[sourcedir][destdir][x][y][z]
// (e.g. K[X][Y][1][2][3] gives H_y at position (1, 2, 3) due to a unit dipole m_x at the origin.
// You can use the function KernIdx to convert from source-dest pairs like XX to 1D indices:
// K[KernIdx[X][X]] returns K[XX]
func Kernel_Newell(size []int, cellsize []float64, periodic []int, asymptotic_radius, zero_self_demag int, kern *host.Array) {

	Debug("Calculating demag kernel:", "size", size)

	// Sanity check
	{
		Assert((size[0] > 0 && size[1] > 1 && size[2] > 1) || (size[0] > 1 && size[1] > 0 && size[2] > 1) || (size[0] > 1 && size[1] > 1 && size[2] > 0))
		Assert(cellsize[0] > 0 && cellsize[1] > 0 && cellsize[2] > 0)
		Assert(periodic[0] >= 0 && periodic[1] >= 0 && periodic[2] >= 0)
		// Ensure only 2D periodicity
		Assert((periodic[0] + periodic[1] + periodic[2]) <= 2)
		// Ensure only 1D periodicity
		Assert((periodic[0] + periodic[1] + periodic[2]) <= 1)
		// Ensure that number of cells along a dimension is a power-of-2
		// for all dimensions having more than 1 cell
		if size[0] > 1 {
			Assert(size[0]%2 == 0)
		}
		if size[1] > 1 {
			Assert(size[1]%2 == 0)
		}
		if size[2] > 1 {
			Assert(size[2]%2 == 0)
		}
	}

//	array := kern.Array
	ffts := new(Newell3DFFT)

	var (
	    scratchPt *host.Array // Create some scratch space for doing computations
	    p1, p2, q1, q2	float64
	    R	[3]float64
	    rdimx, rdimy, rdimz, cdimx, cdimy, cdimz int
	)

	if (size[X] == 1) {
		rdimx = 1
	} else {
		rdimx = 2*size[X]
	}
	if (size[Y] == 1) {
		rdimy = 1
	} else {
		rdimy = 2*size[Y]
	}
	if (size[Z] == 1) {
		rdimz = 1
	} else {
		rdimz = 2*size[Z]
	}
	
	ffts.RecommendDimensions(rdimx, rdimy, rdimz, &cdimx, &cdimy, &cdimz)

	if (cdimx == 1) {
		ffts.fftx.SetDimensions(1)
	} else {
		ffts.fftx.SetDimensions(2*(cdimx-1))
	}
	ffts.ffty.SetDimensions(cdimy)
	ffts.fftz.SetDimensions(cdimz)

//	ldimx := ffts.fftx.GetLogicalDimension()
//	ldimy := ffts.ffty.GetLogicalDimension()
//	ldimz := ffts.fftz.GetLogicalDimension()

//	adimx := (ldimx/2) + 1
//	adimy := (ldimy/2) + 1
//	adimz := (ldimz/2) + 1

	scratch := scratchPt.Array
	dx, dy, dz := cellsize[X], cellsize[Y], cellsize[Z]

	// Determine relative sizes of dx, dy and dz, since that is all that demag
	// calculation cares about
	
	if ( FindRatApprox(dx,dy,1e-12,1000,&p1,&q1) && FindRatApprox(dz,dy,1e-12,1000,&p2,&q2) ) {
	   gcd := Gcd(q1,q2)
	   dx = p1*q2/gcd
	   dy = q1*q2/gcd
	   dz = p2*p1/gcd
	} else {
	   maxedge := dx
	   if (dy>maxedge) { maxedge = dy }
	   if (dz>maxedge) { maxedge = dz }
	   dx /= maxedge
	   dy /= maxedge
	   dz /= maxedge
	}

	// Field (destination) loop ranges
	// offset by -dx, -dy, and -dz so we can do d^2/dx^2, d^2/dy^2 and d^2/dz^2 in place
	xstop := 1
	ystop := 1
	zstop := 1
	// Handle PBC separately
	// Need to error check for only 1D PBC

	// smallest cell dimension is our typical length scale
	L := cellsize[X]
	if cellsize[Y] < L {
		L = cellsize[Y]
	}
	if cellsize[Z] < L {
		L = cellsize[Z]
	}

	scale := 1 / (4 * math.Pi * dx * dy * dz)
	scale *= ffts.fftx.GetScaling() * ffts.ffty.GetScaling() * ffts.fftz.GetScaling()

	scaled_arad := float64(asymptotic_radius) * math.Pow(dx*dy*dz, float64(1.0/3.0))

	if (rdimz > 1) { zstop = rdimz+2 }
	if (rdimy > 1) { ystop = rdimy+2 }
	if (rdimx > 1) { xstop = rdimx+2 }

	I := FullTensorIdx[0][0]

	if (scaled_arad > 0.0) {
		ztest := int(math.Ceil(scaled_arad/dz)) + 2
		if (ztest < zstop) { zstop = ztest }
		ytest := int(math.Ceil(scaled_arad/dy)) + 2
		if (ytest < ystop) { ystop = ytest }
		xtest := int(math.Ceil(scaled_arad/dx)) + 2
		if (xtest < xstop) { xstop = xtest }
	}

//	for s := 0; s < 3; s++ { // source index Ksdxyz
//		u, v, w := s, (s+1)%3, (s+2)%3 // u = direction of source (s), v & w are the orthogonal directions
// 	}
	if (periodic[0]+periodic[1]+periodic[2] == 0) {
   	   	for x := 0; x < xstop; x++ { // in each dimension, go from -(size-1)/2 to size/2 -1, wrapped.
		      xw := x
		      R[X] = float64(x-1) * cellsize[X]

		      for y := 0; y < ystop; y++ {
			    yw := y
			    R[Y] = float64(y-1) * cellsize[Y]

			    for z := 0; z < zstop; z++ {
			    	  zw := z
			    	  R[Z] = float64(z-1) * cellsize[Z]

				  // For Nxx
				  I = FullTensorIdx[0][0]
				  scratch[I][xw][yw][zw]=float32(scale*Newell_f(R[X],R[Y],R[Z]))
				  // For Nxy
				  I = FullTensorIdx[0][1]
				  scratch[I][xw][yw][zw]=float32(scale*Newell_g(R[X],R[Y],R[Z]))
				  // For Nxz
				  I = FullTensorIdx[0][2]
				  scratch[I][xw][yw][zw]=float32(scale*Newell_g(R[X],R[Z],R[Y]))

			    }
		      }
		}

		// Do d^2/dz^2
		if (zstop == 1) {
			zw := 0
			for y := 0; y < ystop; y++ {
				yw := y
				R[Y] = float64(y-1) * cellsize[Y]

				for x := 0; x < xstop; x++ {
					xw := x
					R[X] = float64(x-1) * cellsize[X]

					I = FullTensorIdx[0][0]
					scratch[I][xw][yw][zw] -= float32(scale*Newell_f(R[X],R[Y],0.0))
					scratch[I][xw][yw][zw] *= float32(2.0)

					I = FullTensorIdx[0][1]
					scratch[I][xw][yw][zw] -=float32(scale*Newell_g(R[X],R[Y],0.0))
					scratch[I][xw][yw][zw] *= float32(2.0)

					I = FullTensorIdx[0][2]
					scratch[I][xw][yw][zw] = float32(0.0)

				}
			}
		} else {
			for z := 0; z < rdimz; z++ {
				zw := z
				for y := 0; y < ystop; y++ {
					yw := y
					for x := 0; x < xstop; x++ {
						xw := x

						I = FullTensorIdx[0][0]
						scratch[I][xw][yw][zw] += -2.0*scratch[I][xw][yw][zw+1] + scratch[I][xw][yw][zw+2]
						I = FullTensorIdx[0][1]
						scratch[I][xw][yw][zw] += -2.0*scratch[I][xw][yw][zw+1] + scratch[I][xw][yw][zw+2]
						I = FullTensorIdx[0][2]
						scratch[I][xw][yw][zw] += -2.0*scratch[I][xw][yw][zw+1] + scratch[I][xw][yw][zw+2]
					}
				}
			}
		}

		// Do d^2/dy^2
		if (ystop == 1) {
			yw := 0
			for z := 0; z < zstop; z++ {
				zw := z
				R[Z] = float64(z) * cellsize[Z]

				for x := 0; x < xstop; x++ {
					xw := x
					R[X] = float64(x-1) * cellsize[X]

					I = FullTensorIdx[0][0]
					scratch[I][xw][yw][zw] -= float32(scale*(Newell_f(R[X],0.0,R[Z]-cellsize[Z])+Newell_f(R[X],0.0,R[Z]+cellsize[Z])-2.0*Newell_f(R[X],0.0,R[Z])))
					scratch[I][xw][yw][zw] *= float32(2.0)

					I = FullTensorIdx[0][1]
					scratch[I][xw][yw][zw] =float32(0.0)

					I = FullTensorIdx[0][2]
					scratch[I][xw][yw][zw] -= float32(scale*(Newell_g(R[X],R[Z]-cellsize[Z],0.0)+Newell_g(R[X],R[Z]+cellsize[Z],0.0)-2.0*Newell_g(R[X],R[Z],0.0)))
					scratch[I][xw][yw][zw] *= float32(2.0)

				}
			}
		} else {
			for z := 0; z < rdimz; z++ {
				zw := z
				for y := 0; y < ystop; y++ {
					yw := y
					for x := 0; x < xstop; x++ {
						xw := x

						I = FullTensorIdx[0][0]
						scratch[I][xw][yw][zw] += -2.0*scratch[I][xw][yw+1][zw] + scratch[I][xw][yw+2][zw]
						I = FullTensorIdx[0][1]
						scratch[I][xw][yw][zw] += -2.0*scratch[I][xw][yw+1][zw] + scratch[I][xw][yw+2][zw]
						I = FullTensorIdx[0][2]
						scratch[I][xw][yw][zw] += -2.0*scratch[I][xw][yw+1][zw] + scratch[I][xw][yw+2][zw]
					}
				}
			}
		}

		// Do d^2/dx^2
		if (xstop == 1) {
			xw := 0
			for z := 0; z < zstop; z++ {
				zw := z
				R[Z] = float64(z) * cellsize[Z]

				for y := 0; y < ystop; y++ {
					yw := y
					R[Y] = float64(y) * cellsize[Y]

					I = FullTensorIdx[0][0]
					scratch[I][xw][yw][zw] -= float32(scale*((4.0*Newell_f(0.0,R[Y],R[Z])+Newell_f(0.0,R[Y]+cellsize[Y],R[Z]+cellsize[Z])+Newell_f(0.0,R[Y]-cellsize[Y],R[Z]+cellsize[Z])+Newell_f(0.0,R[Y]+cellsize[Y],R[Z]-cellsize[Z])+Newell_f(0.0,R[Y]-cellsize[Y],R[Z]-cellsize[Z]))-2.0*(Newell_f(0,R[Y]+cellsize[Y],R[Z])+Newell_f(0,R[Y]-cellsize[Y],R[Z])+Newell_f(0,R[Y],R[Z]+cellsize[Z])+Newell_f(0,R[Y],R[Z]-cellsize[Z]))))
					scratch[I][xw][yw][zw] *= float32(2.0)

					I = FullTensorIdx[0][1]
					scratch[I][xw][yw][zw] =float32(0.0)

					I = FullTensorIdx[0][2]
					scratch[I][xw][yw][zw] = float32(0.0)

				}
			}
		} else {
			for z := 0; z < rdimz; z++ {
				zw := z
				for y := 0; y < ystop; y++ {
					yw := y
					for x := 0; x < xstop; x++ {
						xw := x

						I = FullTensorIdx[0][0]
						scratch[I][xw][yw][zw] += -2.0*scratch[I][xw+1][yw][zw] + scratch[I][xw+2][yw][zw]
						I = FullTensorIdx[0][1]
						scratch[I][xw][yw][zw] += -2.0*scratch[I][xw+1][yw][zw] + scratch[I][xw+2][yw][zw]
						I = FullTensorIdx[0][2]
						scratch[I][xw][yw][zw] += -2.0*scratch[I][xw+1][yw][zw] + scratch[I][xw+2][yw][zw]
					}
				}
			}
		}

		// Correct for self-demag
		selfscale := float32(-1.0 * ffts.fftx.GetScaling() * ffts.ffty.GetScaling() * ffts.fftz.GetScaling())
		I = FullTensorIdx[0][0]
		scratch[I][0][0][0] = float32(SelfDemagNx(cellsize[X],cellsize[Y],cellsize[Z]))
		if (zero_self_demag > 0) { scratch[I][0][0][0] -= float32(1.0/3.0) }
		scratch[I][0][0][0] *= selfscale
		I = FullTensorIdx[0][1]
		scratch[I][0][0][0] = float32(0.0)
		I = FullTensorIdx[0][2]
		scratch[I][0][0][0] = float32(0.0)

		// Use asymptotic approximation for far field
		if (scaled_arad >= 0.0) {
			scaled_arad_sq := scaled_arad*scaled_arad
			fft_scaling := float64(-1.0 * ffts.fftx.GetScaling() * ffts.ffty.GetScaling() * ffts.fftz.GetScaling())
			Assert(scaled_arad_sq > 0.0 && fft_scaling > 0.0)
		}

		xtest := float64(rdimx)*float64(cellsize[X])
		xtest *= xtest
		
	}
}
