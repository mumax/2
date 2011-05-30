//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

// This files implements splice I/O
// Author: Arne Vansteenkiste

package gpu

import (
//. "mumax/common"
//"io"
//"os"
//"fmt"
)


const IO_BUF_LEN = 4096

//func (v *VSplice) Fprint(w io.Writer) (n int, error os.Error){
//
//}

//func (s *splice) Fprint(w io.Writer) (n int, error os.Error) {
//	buffer := make([]float32, s.Len())
//	s.CopyToHost(buffer)
//	return fmt.Fprint(w, buffer)
//}
//
//func (s *splice) Print() (n int, error os.Error) {
//	return s.Fprint(os.Stdout)
//}
//
//func (s *splice) Println() (n int, error os.Error) {
//	defer fmt.Println()
//	return s.Fprint(os.Stdout)
//}
//
////func (v *vSplice) Fprint(w io.Writer) (n int, error os.Error) {
////	for i := range v.Comp {
////		ni, erri := v.Comp[i].Fprint(w)
////		n += ni
////		error = ErrCat(error, erri)
////	}
////	return
////}
//
////func (s *vSplice) Print() (n int, error os.Error) {
////	return s.Fprint(os.Stdout)
////}
//
////func (v *vSplice) Println() (n int, error os.Error) {
////	defer fmt.Println()
////	return v.Fprint(os.Stdout)
////}
//
////func (v *VSplice) WriteTo(w io.Writer) (n int64, err os.Error){
////
////}
