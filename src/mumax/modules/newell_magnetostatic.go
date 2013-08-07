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

type DemagNabPairData struct {
	ubase, uoff float64
	ptp *DemagNabData // ubase + uoff
	ptm *DemagNabData // ubase - uoff
}

type DemagAsymptoticRefineData struct {
	rdx, rdy, rdz float64
	result_scale float64
	xcount, ycount, zcount int
}

type DemagNxxAsymptoticBase struct {
	cubic_cell bool
	self_demag, lead_weight float64
	a1, a2, a3, a4, a5, a6 float64
	b1, b2, b3, b4, b5, b6, b7, b8, b9, b10 float64
	c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15 float64
}

type DemagNxxAsymptotic struct {
	refine_data DemagAsymptoticRefineData
	Nxx DemagNxxAsymptoticBase
}

type DemagNxyAsymptoticBase struct {
	cubic_cell bool
	lead_weight float64
	a1, a2, a3 float64
	b1, b2, b3, b4, b5, b6 float64
	c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 float64
}

type DemagNxyAsymptotic struct {
	refine_data DemagAsymptoticRefineData
	Nxy DemagNxyAsymptoticBase
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

func (s *DemagNabData) Set(import_x, import_y, import_z float64) { // OxsDemagNabData::Set
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

func DemagNabData_SetPair(ixa, iya, iza, ixb, iyb, izb float64, pta, ptb *DemagNabData) { // OxsDemagNabData::SetPair
	pta.Set(ixa,iya,iza)
	ptb.Set(ixb,iyb,izb)
}

func (s *DemagAsymptoticRefineData) DemagAsymptoticRefineData(dx, dy, dz, maxratio float64) { // OxsDemagAsymptoticRefineData::OxsDemagAsymptoticRefineData
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

func (s *DemagNxxAsymptoticBase) NxxAsymptoticPairBase(ptA, ptB *DemagNabData) float64 {  // Oxs_DemagNxxAsymptoticBase::NxxAsymptoticPair
	return s.NxxAsymptoticBaseF(ptA) + s.NxxAsymptoticBaseF(ptB)
}

func (s *DemagNxxAsymptoticBase) NxxAsymptoticBaseF(ptdata *DemagNabData) float64{ // Oxs_DemagNxxAsymptoticBase::NxxAsymptotic
	if (ptdata.iR2 <= 0.0) { return s.self_demag }

	tx2, ty2, tz2 := ptdata.tx2, ptdata.ty2, ptdata.tz2
	tz4 := tz2*tz2
	tz6 := tz4*tz2
	term3 := (2.0*tx2 - ty2 - tz2)*s.lead_weight
	term5 := 0.0
	term7 := 0.0

	if(s.cubic_cell) {
		ty4 := ty2*ty2
		term7 = ((s.b1*tx2 + (s.b2*ty2 + s.b3*tz2))*tx2 + (s.b4*ty4 + s.b6*tz4))*tx2 + s.b7*ty4*ty2 + s.b10*tz6
	} else {
		term5 = (s.a1*tx2 + (s.a2*ty2 + s.a3*tz2))*tx2 + (s.a4*ty2 + s.a5*tz2)*ty2 + s.a6*tz4
		term7 = ((s.b1*tx2 + (s.b2*ty2 + s.b3*tz2))*tx2 + ((s.b4*ty2 + s.b5*tz2)*ty2 + s.b6*tz4))*tx2 + ((s.b7*ty2 + s.b8*tz2)*ty2 + s.b9*tz4)*ty2 + s.b10*tz6
	}
	term9 :=  (((s.c1*tx2 + (s.c2*ty2 + s.c3*tz2))*tx2 + ((s.c4*ty2 + s.c5*tz2)*ty2 + s.c6*tz4))*tx2 + ( ((s.c7*ty2 + s.c8*tz2)*ty2 + s.c9*tz4)*ty2 + s.c10*tz6 ))*tx2 + (((s.c11*ty2 + s.c12*tz2)*ty2 + s.c13*tz4)*ty2 + s.c14*tz6)*ty2 + s.c15*tz4*tz4

	Nxx := (term9 + term7 + term5 + term3)*ptdata.iR;
	// Error should be of order 1/R^11

	return Nxx
}

func (s *DemagNxxAsymptoticBase) DemagNxxAsymptoticBaseF(refine_data *DemagAsymptoticRefineData) { // Oxs_DemagNxxAsymptoticBase::Oxs_DemagNxxAsymptoticBase
	dx, dy, dz := refine_data.rdx, refine_data.rdy, refine_data.rdz
	s.self_demag = SelfDemagNx(dx,dy,dz)
	dx2, dy2, dz2 := dx*dx, dy*dy, dz*dz
	dx4, dy4, dz4 := dx2*dx2, dy2*dy2, dz2*dz2
	dx6, dy6, dz6 := dx4*dx2, dy4*dy2, dz4*dz2
	s.lead_weight = (-dx*dy*dz)/(4*math.Pi)
	// Initialize coefficients for 1/R^5 term
	if ( (dx2 != dy2) || (dx2 != dz2) || (dy2 != dz2) ) { // Non-cube case
		s.cubic_cell = false
		s.a1 = s.lead_weight / float64(4.0)
		s.a2, s.a3, s.a4, s.a5, s.a6 = s.a1, s.a1, s.a1, s.a1, s.a1
		s.a1 *=   8.0*dx2 -  4.0*dy2 -  4.0*dz2
		s.a2 *= -24.0*dx2 + 27.0*dy2 -  3.0*dz2
		s.a3 *= -24.0*dx2 -  3.0*dy2 + 27.0*dz2
		s.a4 *=   3.0*dx2 -  4.0*dy2 +      dz2
		s.a5 *=   6.0*dx2 -  3.0*dy2 -  3.0*dz2
		s.a6 *=   3.0*dx2 +      dy2 -  4.0*dz2
	} else { // Cube
		s.cubic_cell = true
		s.a1, s.a2, s.a3, s.a4, s.a5, s.a6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	}

	// Initialize coefficients for 1/R^7 term
	s.b1, s.b2, s.b3, s.b4, s.b5, s.b6, s.b7, s.b8, s.b9, s.b10 = s.lead_weight/float64(16.0), s.lead_weight/float64(16.0), s.lead_weight/float64(16.0), s.lead_weight/float64(16.0), s.lead_weight/float64(16.0), s.lead_weight/float64(16.0), s.lead_weight/float64(16.0), s.lead_weight/float64(16.0), s.lead_weight/float64(16.0), s.lead_weight/float64(16.0)
	if (s.cubic_cell) {
		s.b1  *=  -14.0*dx4
		s.b2  *=  105.0*dx4
		s.b3  *=  105.0*dx4
		s.b4  *= -105.0*dx4
		s.b6  *= -105.0*dx4
		s.b7  *=    7.0*dx4
		s.b10 *=    7.0*dx4
		s.b5, s.b8, s.b9 = 0.0, 0.0, 0.0
	} else {
		s.b1  *=   32.0*dx4 -  40.0*dx2*dy2 -  40.0*dx2*dz2 +  12.0*dy4 +  10.0*dy2*dz2 +  12.0*dz4
		s.b2  *= -240.0*dx4 + 580.0*dx2*dy2 +  20.0*dx2*dz2 - 202.0*dy4 -  75.0*dy2*dz2 +  22.0*dz4
		s.b3  *= -240.0*dx4 +  20.0*dx2*dy2 + 580.0*dx2*dz2 +  22.0*dy4 -  75.0*dy2*dz2 - 202.0*dz4
		s.b4  *=  180.0*dx4 - 505.0*dx2*dy2 +  55.0*dx2*dz2 + 232.0*dy4 -  75.0*dy2*dz2 +   8.0*dz4
		s.b5  *=  360.0*dx4 - 450.0*dx2*dy2 - 450.0*dx2*dz2 - 180.0*dy4 + 900.0*dy2*dz2 - 180.0*dz4
		s.b6  *=  180.0*dx4 +  55.0*dx2*dy2 - 505.0*dx2*dz2 +   8.0*dy4 -  75.0*dy2*dz2 + 232.0*dz4
		s.b7  *=  -10.0*dx4 +  30.0*dx2*dy2 -   5.0*dx2*dz2 -  16.0*dy4 +  10.0*dy2*dz2 -   2.0*dz4
		s.b8  *=  -30.0*dx4 +  55.0*dx2*dy2 +  20.0*dx2*dz2 +   8.0*dy4 -  75.0*dy2*dz2 +  22.0*dz4
		s.b9  *=  -30.0*dx4 +  20.0*dx2*dy2 +  55.0*dx2*dz2 +  22.0*dy4 -  75.0*dy2*dz2 +   8.0*dz4
		s.b10 *=  -10.0*dx4 -   5.0*dx2*dy2 +  30.0*dx2*dz2 -   2.0*dy4 +  10.0*dy2*dz2 -  16.0*dz4
	}

	// Initialize coefficients for 1/R^9 term
	s.c1, s.c2, s.c3, s.c4, s.c5, s.c6, s.c7, s.c8 = s.lead_weight/float64(192.0), s.lead_weight/float64(192.0), s.lead_weight/float64(192.0), s.lead_weight/float64(192.0), s.lead_weight/float64(192.0), s.lead_weight/float64(192.0), s.lead_weight/float64(192.0), s.lead_weight/float64(192.0)
	s.c9, s.c10, s.c11, s.c12, s.c13, s.c14, s.c15 = s.lead_weight/float64(192.0), s.lead_weight/float64(192.0), s.lead_weight/float64(192.0), s.lead_weight/float64(192.0), s.lead_weight/float64(192.0), s.lead_weight/float64(192.0), s.lead_weight/float64(192.0)
	if(s.cubic_cell) {
		s.c1  *=    32.0 * dx6
		s.c2  *=  -448.0 * dx6
		s.c3  *=  -448.0 * dx6
		s.c4  *=  -150.0 * dx6
		s.c5  *=  7620.0 * dx6
		s.c6  *=  -150.0 * dx6
		s.c7  *=   314.0 * dx6
		s.c8  *= -3810.0 * dx6
		s.c9  *= -3810.0 * dx6
		s.c10 *=   314.0 * dx6
		s.c11 *=   -16.0 * dx6
		s.c12 *=   134.0 * dx6
		s.c13 *=   300.0 * dx6
		s.c14 *=   134.0 * dx6
		s.c15 *=   -16.0 * dx6
	} else {
		s.c1  *=    384.0 *dx6 +   -896.0 *dx4*dy2 +   -896.0 *dx4*dz2 +    672.0 *dx2*dy4 +    560.0 *dx2*dy2*dz2 +    672.0 *dx2*dz4 +   -120.0 *dy6 +   -112.0 *dy4*dz2 +   -112.0 *dy2*dz4 +   -120.0 *dz6
		s.c2  *=  -5376.0 *dx6 +  22624.0 *dx4*dy2 +   2464.0 *dx4*dz2 + -19488.0 *dx2*dy4 +  -7840.0 *dx2*dy2*dz2 +    672.0 *dx2*dz4 +   3705.0 *dy6 +   2198.0 *dy4*dz2 +    938.0 *dy2*dz4 +   -345.0 *dz6
		s.c3  *=  -5376.0 *dx6 +   2464.0 *dx4*dy2 +  22624.0 *dx4*dz2 +    672.0 *dx2*dy4 +  -7840.0 *dx2*dy2*dz2 + -19488.0 *dx2*dz4 +   -345.0 *dy6 +    938.0 *dy4*dz2 +   2198.0 *dy2*dz4 +   3705.0 *dz6
		s.c4  *=  10080.0 *dx6 + -48720.0 *dx4*dy2 +   1680.0 *dx4*dz2 +  49770.0 *dx2*dy4 +  -2625.0 *dx2*dy2*dz2 +   -630.0 *dx2*dz4 + -10440.0 *dy6 +  -1050.0 *dy4*dz2 +   2100.0 *dy2*dz4 +   -315.0 *dz6
		s.c5  *=  20160.0 *dx6 + -47040.0 *dx4*dy2 + -47040.0 *dx4*dz2 +  -6300.0 *dx2*dy4 + 133350.0 *dx2*dy2*dz2 +  -6300.0 *dx2*dz4 +   7065.0 *dy6 + -26670.0 *dy4*dz2 + -26670.0 *dy2*dz4 +   7065.0 *dz6
		s.c6  *=  10080.0 *dx6 +   1680.0 *dx4*dy2 + -48720.0 *dx4*dz2 +   -630.0 *dx2*dy4 +  -2625.0 *dx2*dy2*dz2 +  49770.0 *dx2*dz4 +   -315.0 *dy6 +   2100.0 *dy4*dz2 +  -1050.0 *dy2*dz4 + -10440.0 *dz6
		s.c7  *=  -3360.0 *dx6 +  17290.0 *dx4*dy2 +  -1610.0 *dx4*dz2 + -19488.0 *dx2*dy4 +   5495.0 *dx2*dy2*dz2 +   -588.0 *dx2*dz4 +   4848.0 *dy6 +  -3136.0 *dy4*dz2 +    938.0 *dy2*dz4 +    -75.0 *dz6
		s.c8  *= -10080.0 *dx6 +  32970.0 *dx4*dy2 +  14070.0 *dx4*dz2 +  -6300.0 *dx2*dy4 + -66675.0 *dx2*dy2*dz2 +  12600.0 *dx2*dz4 + -10080.0 *dy6 +  53340.0 *dy4*dz2 + -26670.0 *dy2*dz4 +   3015.0 *dz6
		s.c9  *= -10080.0 *dx6 +  14070.0 *dx4*dy2 +  32970.0 *dx4*dz2 +  12600.0 *dx2*dy4 + -66675.0 *dx2*dy2*dz2 +  -6300.0 *dx2*dz4 +   3015.0 *dy6 + -26670.0 *dy4*dz2 +  53340.0 *dy2*dz4 + -10080.0 *dz6
		s.c10 *=  -3360.0 *dx6 +  -1610.0 *dx4*dy2 +  17290.0 *dx4*dz2 +   -588.0 *dx2*dy4 +   5495.0 *dx2*dy2*dz2 + -19488.0 *dx2*dz4 +    -75.0 *dy6 +    938.0 *dy4*dz2 +  -3136.0 *dy2*dz4 +   4848.0 *dz6
		s.c11 *=    105.0 *dx6 +   -560.0 *dx4*dy2 +     70.0 *dx4*dz2 +    672.0 *dx2*dy4 +   -280.0 *dx2*dy2*dz2 +     42.0 *dx2*dz4 +   -192.0 *dy6 +    224.0 *dy4*dz2 +   -112.0 *dy2*dz4 +     15.0 *dz6
		s.c12 *=    420.0 *dx6 +  -1610.0 *dx4*dy2 +   -350.0 *dx4*dz2 +    672.0 *dx2*dy4 +   2345.0 *dx2*dy2*dz2 +   -588.0 *dx2*dz4 +    528.0 *dy6 +  -3136.0 *dy4*dz2 +   2198.0 *dy2*dz4 +   -345.0 *dz6
		s.c13 *=    630.0 *dx6 +  -1470.0 *dx4*dy2 +  -1470.0 *dx4*dz2 +   -630.0 *dx2*dy4 +   5250.0 *dx2*dy2*dz2 +   -630.0 *dx2*dz4 +    360.0 *dy6 +  -1050.0 *dy4*dz2 +  -1050.0 *dy2*dz4 +    360.0 *dz6
		s.c14 *=    420.0 *dx6 +   -350.0 *dx4*dy2 +  -1610.0 *dx4*dz2 +   -588.0 *dx2*dy4 +   2345.0 *dx2*dy2*dz2 +    672.0 *dx2*dz4 +   -345.0 *dy6 +   2198.0 *dy4*dz2 +  -3136.0 *dy2*dz4 +    528.0 *dz6
		s.c15 *=    105.0 *dx6 +     70.0 *dx4*dy2 +   -560.0 *dx4*dz2 +     42.0 *dx2*dy4 +   -280.0 *dx2*dy2*dz2 +    672.0 *dx2*dz4 +     15.0 *dy6 +   -112.0 *dy4*dz2 +    224.0 *dy2*dz4 +   -192.0 *dz6
	}
}

func (s *DemagNxxAsymptotic) NxxAsymptotic(x, y, z float64) float64 { // Oxs_DemagNxxAsymptotic::Oxs_DemagNxxAsymptotic
	ptdata := new(DemagNabData)
	ptdata.Set(x,y,z)
	return s.NxxAsymptoticF(ptdata)
}

func (s *DemagNxxAsymptotic) NxxAsymptoticF(ptdata *DemagNabData) float64 { // Oxs_DemagNxxAsymptotic::NxxAsymptotic
	xcount := s.refine_data.xcount
	ycount := s.refine_data.ycount
	zcount := s.refine_data.zcount
	rdx := s.refine_data.rdx
	rdy := s.refine_data.rdy
	rdz := s.refine_data.rdz
	result_scale := s.refine_data.result_scale

	var (
		rptdata, mrptdata DemagNabData
	)
	zsum := float64(0.0)
	for k := 1-zcount; k<zcount; k++ {
		zoff := ptdata.z + float64(k)*rdz
		ysum := float64(0.0)
		for j := 1-ycount; j < ycount; j++ {
			// Compute interactions for x-strip
			yoff := ptdata.y + float64(j)*rdy
			rptdata.Set(ptdata.x,yoff,zoff)
			xsum := float64(xcount) * s.Nxx.NxxAsymptoticBaseF(&rptdata);
			for i := 1; i < xcount; i++ {
				rptdata.Set(ptdata.x+float64(i)*rdx,yoff,zoff);
				mrptdata.Set(ptdata.x-float64(i)*rdx,yoff,zoff);
				xsum += float64(xcount-i) * s.Nxx.NxxAsymptoticPairBase(&rptdata,&mrptdata);
			}
			// Weight x-strip interactions into xy-plate
			ysum += (float64(ycount) - math.Abs(float64(j)))*xsum;
		}
		// Weight xy-plate interactions into total sum
		zsum += (float64(zcount) - math.Abs(float64(k)))*ysum;
	}
	return zsum*result_scale
}

// To repeat for Nxy and Nxz
func (s *DemagNxyAsymptoticBase) NxyAsymptoticPairBase(ptA, ptB *DemagNabData) float64 { // Oxs_DemagNxyAsymptoticBase::NxyAsymptoticPair
	return s.NxyAsymptoticBaseF(ptA) + s.NxyAsymptoticBaseF(ptB)
}

func (s *DemagNxyAsymptoticBase) NxyAsymptoticBaseF(ptdata *DemagNabData) float64{ // Oxs_DemagNxyAsymptoticBase::NxyAsymptotic
	if (ptdata.iR2 <= 0.0) { return float64(0.0) }

	tx2, ty2, tz2 := ptdata.tx2, ptdata.ty2, ptdata.tz2

	term3 := 3.0*s.lead_weight

	term5 := 0.0

	if (!s.cubic_cell) {
		term5 = s.a1*tx2 + s.a2*ty2 +s.a3*tz2
	}

	tz4 := tz2*tz2

	term7 := (s.b1*tx2 + (s.b2*ty2 + s.b3*tz2))*tx2 + (s.b4*ty2 + s.b5*tz2)*ty2 + s.b6*tz4

	term9 := ((s.c1*tx2 + (s.c2*ty2 + s.c3*tz2))*tx2 + ((s.c4*ty2 + s.c5*tz2)*ty2 + s.c6*tz4))*tx2 + ((s.c7*ty2 + s.c8*tz2)*ty2 + s.c9*tz4)*ty2 + s.c10*tz4*tz2

	x := ptdata.x
	y := ptdata.y
	iR2 := ptdata.iR2
	iR := ptdata.iR
	iR5 := iR*iR2*iR2

	Nxy := (term9 + term7 + term5 + term3)*iR5*x*y
	// Error should be of order 1/R^11

	return Nxy
}

func (s *DemagNxyAsymptoticBase) DemagNxyAsymptoticBaseF(refine_data *DemagAsymptoticRefineData) { // Oxs_DemagNxyAsymptoticBase::Oxs_DemagNxyAsymptoticBase
	dx, dy, dz := refine_data.rdx, refine_data.rdy, refine_data.rdz

	dx2 := dx*dx
	dy2 := dy*dy
	dz2 := dz*dz

	dx4 := dx2*dx2
	dy4 := dy2*dy2
	dz4 := dz2*dz2

	dx6 := dx4*dx2
	dy6 := dy4*dy2
	dz6 := dz4*dz2

	s.lead_weight = (-dx*dy*dz/(4*math.Pi))

	// Initialize coefficients for 1/R^5 term
	if(dx2!=dy2 || dx2!=dz2 || dy2!=dz2) { // Non-cube case
		s.cubic_cell = false
		s.a1, s.a2, s.a3 = (s.lead_weight*5.0)/4.0, (s.lead_weight*5.0)/4.0, (s.lead_weight*5.0)/4.0
		s.a1 *=  4*dx2  -  3*dy2  -  1*dz2
		s.a2 *= -3*dx2  +  4*dy2  -  1*dz2
		s.a3 *= -3*dx2  -  3*dy2  +  6*dz2
	} else { // Cube
		s.cubic_cell = true;
		s.a1, s.a2, s.a3 = 0.0, 0.0, 0.0
	}

	// Initialize coefficients for 1/R^7 term
	s.b1, s.b2, s.b3, s.b4, s.b5, s.b6 = (s.lead_weight*7.0)/16.0, (s.lead_weight*7.0)/16.0, (s.lead_weight*7.0)/16.0, (s.lead_weight*7.0)/16.0, (s.lead_weight*7.0)/16.0, (s.lead_weight*7.0)/16.0
	if (s.cubic_cell) {
		s.b1  *=  -7*dx4
		s.b2  *=  19*dx4
		s.b3  *=  13*dx4
		s.b4  *=  -7*dx4
		s.b5  *=  13*dx4
		s.b6  *= -13*dx4
	} else {
		s.b1 *=  16*dx4 -  30*dx2*dy2 -  10*dx2*dz2 +  10*dy4 +   5*dy2*dz2 +  2*dz4
		s.b2 *= -40*dx4 + 105*dx2*dy2 -   5*dx2*dz2 -  40*dy4 -   5*dy2*dz2 +  4*dz4
		s.b3 *= -40*dx4 -  15*dx2*dy2 + 115*dx2*dz2 +  20*dy4 -  35*dy2*dz2 - 32*dz4
		s.b4 *=  10*dx4 -  30*dx2*dy2 +   5*dx2*dz2 +  16*dy4 -  10*dy2*dz2 +  2*dz4
		s.b5 *=  20*dx4 -  15*dx2*dy2 -  35*dx2*dz2 -  40*dy4 + 115*dy2*dz2 - 32*dz4
		s.b6 *=  10*dx4 +  15*dx2*dy2 -  40*dx2*dz2 +  10*dy4 -  40*dy2*dz2 + 32*dz4
	}

	// Initialize coefficients for 1/R^9 term
	s.c1, s.c2, s.c3, s.c4, s.c5, s.c6, s.c7, s.c8, s.c9, s.c10 = s.lead_weight/64.0, s.lead_weight/64.0, s.lead_weight/64.0, s.lead_weight/64.0, s.lead_weight/64.0, s.lead_weight/64.0, s.lead_weight/64.0, s.lead_weight/64.0, s.lead_weight/64.0, s.lead_weight/64.0
	if (s.cubic_cell) {
		s.c1  *=   48 *dx6
		s.c2  *= -142 *dx6
		s.c3  *= -582 *dx6
		s.c4  *= -142 *dx6
		s.c5  *= 2840 *dx6
		s.c6  *= -450 *dx6
		s.c7  *=   48 *dx6
		s.c8  *= -582 *dx6
		s.c9  *= -450 *dx6
		s.c10 *=  180 *dx6
	} else {
		s.c1  *=   576 *dx6 +  -2016 *dx4*dy2 +   -672 *dx4*dz2 +   1680 *dx2*dy4 +    840 *dx2*dy2*dz2 +   336 *dx2*dz4 +  -315 *dy6 +   -210 *dy4*dz2 +  -126 *dy2*dz4 +   -45 *dz6
		s.c2  *= -3024 *dx6 +  13664 *dx4*dy2 +    448 *dx4*dz2 + -12670 *dx2*dy4 +  -2485 *dx2*dy2*dz2 +   546 *dx2*dz4 +  2520 *dy6 +    910 *dy4*dz2 +    84 *dy2*dz4 +  -135 *dz6
		s.c3  *= -3024 *dx6 +   1344 *dx4*dy2 +  12768 *dx4*dz2 +   2730 *dx2*dy4 + -10185 *dx2*dy2*dz2 + -8694 *dx2*dz4 +  -945 *dy6 +   1680 *dy4*dz2 +  2394 *dy2*dz4 +  1350 *dz6
		s.c4  *=  2520 *dx6 + -12670 *dx4*dy2 +    910 *dx4*dz2 +  13664 *dx2*dy4 +  -2485 *dx2*dy2*dz2 +    84 *dx2*dz4 + -3024 *dy6 +    448 *dy4*dz2 +   546 *dy2*dz4 +  -135 *dz6
		s.c5  *=  5040 *dx6 +  -9940 *dx4*dy2 + -13580 *dx4*dz2 +  -9940 *dx2*dy4 +  49700 *dx2*dy2*dz2 + -6300 *dx2*dz4 +  5040 *dy6 + -13580 *dy4*dz2 + -6300 *dy2*dz4 +  2700 *dz6
		s.c6  *=  2520 *dx6 +   2730 *dx4*dy2 + -14490 *dx4*dz2 +    420 *dx2*dy4 +  -7875 *dx2*dy2*dz2 + 17640 *dx2*dz4 +  -945 *dy6 +   3990 *dy4*dz2 +  -840 *dy2*dz4 + -3600 *dz6
		s.c7  *=  -315 *dx6 +   1680 *dx4*dy2 +   -210 *dx4*dz2 +  -2016 *dx2*dy4 +    840 *dx2*dy2*dz2 +  -126 *dx2*dz4 +   576 *dy6 +   -672 *dy4*dz2 +   336 *dy2*dz4 +   -45 *dz6
		s.c8  *=  -945 *dx6 +   2730 *dx4*dy2 +   1680 *dx4*dz2 +   1344 *dx2*dy4 + -10185 *dx2*dy2*dz2 +  2394 *dx2*dz4 + -3024 *dy6 +  12768 *dy4*dz2 + -8694 *dy2*dz4 +  1350 *dz6
		s.c9  *=  -945 *dx6 +    420 *dx4*dy2 +   3990 *dx4*dz2 +   2730 *dx2*dy4 +  -7875 *dx2*dy2*dz2 +  -840 *dx2*dz4 +  2520 *dy6 + -14490 *dy4*dz2 + 17640 *dy2*dz4 + -3600 *dz6
		s.c10 *=  -315 *dx6 +   -630 *dx4*dy2 +   2100 *dx4*dz2 +   -630 *dx2*dy4 +   3150 *dx2*dy2*dz2 + -3360 *dx2*dz4 +  -315 *dy6 +   2100 *dy4*dz2 + -3360 *dy2*dz4 +  1440 *dz6
	}
}

func (s *DemagNxyAsymptoticBase) NxyAsymptoticPairXBase(ptdata *DemagNabPairData) float64{ // Oxs_DemagNxyAsymptoticBase::NxyAsymptoticPairX
	// Evaluates asymptotic approximation to
	//    Nxy(x+xoff,y,z) + Nxy(x-xoff,y,z)
	// on the assumption that |xoff| >> |x|.

	Assert(ptdata.ptp.y == ptdata.ptm.y && ptdata.ptp.z == ptdata.ptm.z)

	xp := ptdata.ptp.x
	y := ptdata.ptp.y
	z := ptdata.ptp.z
	xm := ptdata.ptm.x

	R2p := ptdata.ptp.R2
	R2m := ptdata.ptm.R2

	// Both R2p and R2m must be positive, since asymptotics
	// don't apply for R==0.
		if (R2p<=0.0) { return s.NxyAsymptoticBaseF(ptdata.ptm) }
		if (R2m<=0.0) { return s.NxyAsymptoticBaseF(ptdata.ptp) }

	// Cancellation primarily in 1/R^3 term.
	xbase := ptdata.ubase
	term3x := 3*s.lead_weight*xbase // Main non-canceling part
	term3cancel := float64(0.0) // Main canceling part
	{
		xoff  := ptdata.uoff
		A := xbase*xbase + xoff*xoff + y*y + z*z
		B := 2*xbase*xoff
		R5p := R2p*R2p*ptdata.ptp.R
		R5m := R2m*R2m*ptdata.ptm.R
		A2 := A*A
		B2 := B*B
		Rdiff := -2*B*(B2*B2 + 5*A2*(A2+2*B2)) / (R5p*R5m*(R5p+R5m))
		term3cancel = 3*s.lead_weight*xoff*Rdiff
	}

	// 1/R^5 terms; Note these are zero if cells are cubes
	tx2p := ptdata.ptp.tx2
	ty2p := ptdata.ptp.ty2
	tz2p := ptdata.ptp.tz2
	tx2m := ptdata.ptm.tx2
	ty2m := ptdata.ptm.ty2
	tz2m := ptdata.ptm.tz2
	term5p := 0.0
	term5m := 0.0
	if (!s.cubic_cell) {
		term5p = s.a1*tx2p + s.a2*ty2p + s.a3*tz2p
		term5m = s.a1*tx2m + s.a2*ty2m + s.a3*tz2m
	}

	// 1/R^7 terms
	tz4p := tz2p*tz2p
	tz4m := tz2m*tz2m
	term7p := (s.b1*tx2p + (s.b2*ty2p + s.b3*tz2p))*tx2p + (s.b4*ty2p + s.b5*tz2p)*ty2p + s.b6*tz4p
	term7m := (s.b1*tx2m + (s.b2*ty2m + s.b3*tz2m))*tx2m + (s.b4*ty2m + s.b5*tz2m)*ty2m + s.b6*tz4m

	// 1/R^9 terms
	term9p := ((s.c1*tx2p + (s.c2*ty2p + s.c3*tz2p))*tx2p + ((s.c4*ty2p + s.c5*tz2p)*ty2p + s.c6*tz4p))*tx2p + ((s.c7*ty2p + s.c8*tz2p)*ty2p + s.c9*tz4p)*ty2p + s.c10*tz4p*tz2p
	term9m := ((s.c1*tx2m + (s.c2*ty2m + s.c3*tz2m))*tx2m + ((s.c4*ty2m + s.c5*tz2m)*ty2m + s.c6*tz4m))*tx2m + ((s.c7*ty2m + s.c8*tz2m)*ty2m + s.c9*tz4m)*ty2m + s.c10*tz4m*tz2m

	// Totals
	iRp := ptdata.ptp.iR
	iR2p := ptdata.ptp.iR2
	iR5p := iR2p*iR2p*iRp

	iRm := ptdata.ptm.iR
	iR2m := ptdata.ptm.iR2
	iR5m := iR2m*iR2m*iRm

	Nxy :=  y*(term3cancel + (xp*(term9p + term7p + term5p)+term3x)*iR5p + (xm*(term9m + term7m + term5m)+term3x)*iR5m)
	// Error should be of order 1/R^11

	return Nxy
}

func (s *DemagNxyAsymptotic) NxyAsymptotic(x, y, z float64) float64 { // Oxs_DemagNxyAsymptotic::NxyAsymptotic
	ptdata := new(DemagNabData)
	ptdata.Set(x,y,z)
	return s.NxyAsymptoticF(ptdata)
}

func (s *DemagNxyAsymptotic) NxyAsymptoticF(ptdata *DemagNabData) float64{ // Oxs_DemagNxyAsymptotic::NxyAsymptotic
	xcount := s.refine_data.xcount
	ycount := s.refine_data.ycount
	zcount := s.refine_data.zcount;
	rdx := s.refine_data.rdx;
	rdy := s.refine_data.rdy;
	rdz := s.refine_data.rdz;
	result_scale := s.refine_data.result_scale;

	var (
		rptdata, mrptdata DemagNabData
	)

	zsum := float64(0.0)

	for k:=1-zcount; k<zcount; k++ {
		zoff := ptdata.z + float64(k)*rdz
		ysum := float64(0.0)
		for j:=1-ycount; j<ycount; j++ {
			// Compute interactions for x-strip
			yoff := ptdata.y+float64(j)*rdy
			rptdata.Set(ptdata.x,yoff,zoff)
			xsum := float64(xcount) * s.Nxy.NxyAsymptoticBaseF(&rptdata)
			for i:=1; i<xcount; i++ {
				rptdata.Set(ptdata.x+float64(i)*rdx,yoff,zoff)
				mrptdata.Set(ptdata.x-float64(i)*rdx,yoff,zoff)
				xsum += float64(xcount-i) * s.Nxy.NxyAsymptoticPairBase(&rptdata,&mrptdata)
			}
			// Weight x-strip interactions into xy-plate
			ysum += (float64(ycount) - math.Abs(float64(j)))*xsum;
		}
		// Weight xy-plate interactions into total sum
		zsum += (float64(zcount) - math.Abs(float64(k)))*ysum;
	}
	return zsum*result_scale;
}

func (s *DemagNxyAsymptotic) NxyAsymptoticPairX(ptdata *DemagNabPairData) float64 { // Oxs_DemagNxyAsymptotic::NxyAsymptoticPairX
	// Evaluates asymptotic approximation to
	//    Nxy(x+xoff,y,z) + Nxy(x-xoff,y,z)
	// on the assumption that |xoff| >> |x|.

	Assert(ptdata.ptp.y == ptdata.ptm.y && ptdata.ptp.z == ptdata.ptm.z)

	// Presumably at least one of xcount, ycount, or zcount is 1, but this
	// fact is not used in following code.

	// Alias data from refine_data structure.
	xcount := s.refine_data.xcount
	ycount := s.refine_data.ycount
	zcount := s.refine_data.zcount
	rdx := s.refine_data.rdx
	rdy := s.refine_data.rdy
	rdz := s.refine_data.rdz
	result_scale := s.refine_data.result_scale


	work := new(DemagNabPairData)
	work.ubase = ptdata.ubase
	zsum := float64(0.0)
	for k:=1-zcount; k<zcount; k++ {
		zoff := ptdata.ptp.z+float64(k)*rdz // .ptm.z == .ptp.z
		ysum := float64(0.0)
		for j:=1-ycount; j<ycount; j++ {
			// Compute interactions for x-strip
			yoff := ptdata.ptp.y+float64(j)*rdy // .ptm.y == .ptp.y
			work.uoff = ptdata.uoff
			DemagNabData_SetPair(work.ubase+work.uoff,yoff,zoff,work.ubase-work.uoff,yoff,zoff,work.ptp,work.ptm)
			xsum := float64(xcount) * s.Nxy.NxyAsymptoticPairXBase(work)
			for i:=1; i<xcount; i++ {
				work.uoff = ptdata.uoff + float64(i)*rdx
				DemagNabData_SetPair(work.ubase+work.uoff,yoff,zoff,work.ubase-work.uoff,yoff,zoff,work.ptp,work.ptm)
				tmpsum := s.Nxy.NxyAsymptoticPairXBase(work)
				work.uoff = ptdata.uoff - float64(i)*rdx
				DemagNabData_SetPair(work.ubase+work.uoff,yoff,zoff,work.ubase-work.uoff,yoff,zoff,work.ptp,work.ptm)
				tmpsum += s.Nxy.NxyAsymptoticPairXBase(work)
				xsum += float64(xcount-i) * tmpsum
			}
			// Weight x-strip interactions into xy-plate
			ysum += (float64(ycount) - math.Abs(float64(j)))*xsum;
		}
		// Weight xy-plate interactions into total sum
		zsum += (float64(zcount) - math.Abs(float64(k)))*ysum;
	}
	return zsum*result_scale;
}

func (s *DemagNxyAsymptotic) NxyAsymptoticPairZ(ptdata *DemagNabPairData) float64 { // Oxs_DemagNxyAsymptotic::NxyAsymptoticPairZ
	return s.NxyAsymptoticF(ptdata.ptp) + s.NxyAsymptoticF(ptdata.ptm)
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

	// Use far field approximation past the asymptotic radius
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
		// Calculate Nxx, Nxy and Nxz in the first octant, non-periodic case.
		// Step 1: Evaluate f & g at each cell site. Offset by (-dx, -dy, -dz)
		// so that we can do 2nd derivative operations "in-place".
   	   	for x := 0; x < xstop; x++ { // in each dimension, we stay in the region up to the asymptotic radius
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

		// Step: 2a: Do d^2/dz^2
		if (zstop == 1) {
			// Only 1 layer in z-direction of f/g stored in scratch array.
			zw := 0
			for y := 0; y < ystop; y++ {
				yw := y
				R[Y] = float64(y-1) * cellsize[Y]

				for x := 0; x < xstop; x++ {
					xw := x
					R[X] = float64(x-1) * cellsize[X]

					// Function f is even in each variable, so for example
					//	f(x, y, -dz) - 2f(x, y, 0) + f(x, y, dz)
					//		= 2( f(x, y, -dz) - f(f, y, 0) )
					// Function g is even in z, and odd in x and y, so for example
					//	g(x, -dz, y) - 2g(x, 0, y) + g(x, dz, y)
					//		= 2g(x, 0, y) = 0
					// Nyy(x, y, z) = Nxx(y, x, z); Nzz(x, y, z) = Nxx(z, y, x);
					// Nxz(x, y, z) = Nxy(x, z, y); Nyz(x, y, z) = Nxy(y, z, x);

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

		// Step 2b: Do d^2/dy^2
		if (ystop == 1) {
			yw := 0
			for z := 0; z < zstop; z++ {
				zw := z
				R[Z] = float64(z) * cellsize[Z]

				for x := 0; x < xstop; x++ {
					xw := x
					R[X] = float64(x-1) * cellsize[X]

					// Function f is even in each variable, so for example
					//	f(x, y, -dz) - 2f(x, y, 0) + f(x, y, dz)
					//		= 2( f(x, y, -dz) - f(f, y, 0) )
					// Function g is even in z, and odd in x and y, so for example
					//	g(x, -dz, y) - 2g(x, 0, y) + g(x, dz, y)
					//		= 2g(x, 0, y) = 0
					// Nyy(x, y, z) = Nxx(y, x, z); Nzz(x, y, z) = Nxx(z, y, x);
					// Nxz(x, y, z) = Nxy(x, z, y); Nyz(x, y, z) = Nxy(y, z, x);

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

		// Step 2c: Do d^2/dx^2
		if (xstop == 1) {
			xw := 0
			for z := 0; z < zstop; z++ {
				zw := z
				R[Z] = float64(z) * cellsize[Z]

				for y := 0; y < ystop; y++ {
					yw := y
					R[Y] = float64(y) * cellsize[Y]

					// Function f is even in each variable, so for example
					//	f(x, y, -dz) - 2f(x, y, 0) + f(x, y, dz)
					//		= 2( f(x, y, -dz) - f(f, y, 0) )
					// Function g is even in z, and odd in x and y, so for example
					//	g(x, -dz, y) - 2g(x, 0, y) + g(x, dz, y)
					//		= 2g(x, 0, y) = 0
					// Nyy(x, y, z) = Nxx(y, x, z); Nzz(x, y, z) = Nxx(z, y, x);
					// Nxz(x, y, z) = Nxy(x, z, y); Nyz(x, y, z) = Nxy(y, z, x);

					I = FullTensorIdx[0][0]
					scratch[I][xw][yw][zw] -= float32(scale*((4.0*Newell_f(0.0,R[Y],R[Z])+Newell_f(0.0,R[Y]+cellsize[Y],R[Z]+cellsize[Z])+Newell_f(0.0,R[Y]-cellsize[Y],R[Z]+cellsize[Z])+Newell_f(0.0,R[Y]+cellsize[Y],R[Z]-cellsize[Z])+Newell_f(0.0,R[Y]-cellsize[Y],R[Z]-cellsize[Z]))-2.0*(Newell_f(0,R[Y]+cellsize[Y],R[Z])+Newell_f(0,R[Y]-cellsize[Y],R[Z])+Newell_f(0,R[Y],R[Z]+cellsize[Z])+Newell_f(0,R[Y],R[Z]-cellsize[Z]))))
					scratch[I][xw][yw][zw] *= float32(2.0) // For Nxx

					I = FullTensorIdx[0][1]
					scratch[I][xw][yw][zw] = float32(0.0) // For Nxy

					I = FullTensorIdx[0][2]
					scratch[I][xw][yw][zw] = float32(0.0) // For Nxz

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

		// Special "SelfDemag" code may be more accurate at index 0,0,0.
		selfscale := float32(-1.0 * ffts.fftx.GetScaling() * ffts.ffty.GetScaling() * ffts.fftz.GetScaling())
		I = FullTensorIdx[0][0]
		scratch[I][0][0][0] = float32(SelfDemagNx(cellsize[X],cellsize[Y],cellsize[Z]))
		if (zero_self_demag > 0) { scratch[I][0][0][0] -= float32(1.0/3.0) }
		scratch[I][0][0][0] *= selfscale

		I = FullTensorIdx[0][1]
		scratch[I][0][0][0] = float32(0.0)  // Nxy[0] = 0

		I = FullTensorIdx[0][2]
		scratch[I][0][0][0] = float32(0.0) // Nxz[0] = 0

		// Step 2.5: Use asymptotic (dipolar + higher) approximation for far field
		// Dipole approximation:
		//
		//			 / 3x^2-R^2   3xy       3xz    \
		//            dx.dy.dz   |			       |
		// H_demag = ----------- |   3xy   3y^2-R^2     3yz    |
		//            4.pi.R^5   |			       |
		//			 \   3xz      3yz     3z^2-R^2 /
		//

		if (scaled_arad >= 0.0) {
			scaled_arad_sq := scaled_arad*scaled_arad
			fft_scaling := float64(-1.0 * ffts.fftx.GetScaling() * ffts.ffty.GetScaling() * ffts.fftz.GetScaling())
			Assert(scaled_arad_sq > 0.0 && fft_scaling > 0.0)
		}

		// Since H = -N.M, and by convention with the rest of this code,
		// we store "-N" instead of "N" so we don't have to multiply the
		// output from the FFT + iFFT by -1 when calculating the energy

		xtest := float64(rdimx)*float64(cellsize[X])
		xtest *= xtest

		ANxx := new(DemagNxxAsymptotic)
		Default_Refine_Data := float64(1.5)
		ANxx.refine_data.DemagAsymptoticRefineData(dx,dy,dz,Default_Refine_Data)
		ANxx.Nxx.DemagNxxAsymptoticBaseF(&ANxx.refine_data)

		//ANxy := new(DemagNxyAsymptotic)
		//ANxz := new(DemagNxzAsymptotic)
	}
}
