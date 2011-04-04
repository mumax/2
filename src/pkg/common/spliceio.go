//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

// This files implements splice I/O
// Author: Arne Vansteenkiste

package common

import (
	"io"
	"os"
	"fmt"
)


const IO_BUF_LEN = 4096

//func (v *VSplice) Fprint(w io.Writer) (n int, error os.Error){
//
//}

func (s *Splice) Fprint(w io.Writer) (n int, error os.Error) {
	buffer := make([]float32, s.Len())
	s.CopyToHost(buffer)
	return fmt.Fprint(w, buffer)
}

func (s *Splice) Print() (n int, error os.Error) {
	return s.Fprint(os.Stdout)
}

func (s *Splice) Println() (n int, error os.Error) {
	defer fmt.Println()
	return s.Fprint(os.Stdout)
}

func (v *VSplice) Fprint(w io.Writer) (n int, error os.Error) {
	for i := range v.Comp {
		ni, erri := v.Comp[i].Fprint(w)
		n += ni
		error = ErrCat(error, erri)
	}
	return
}

func (s *VSplice) Print() (n int, error os.Error) {
	return s.Fprint(os.Stdout)
}

func (v *VSplice) Println() (n int, error os.Error) {
	defer fmt.Println()
	return v.Fprint(os.Stdout)
}

//func (v *VSplice) WriteTo(w io.Writer) (n int64, err os.Error){
//
//}


// Combines two Errors into one.
// If a and b are nil, the returned error is nil.
// If either is not nil, it is returned.
// If both are not nil, the first one is returned.
func ErrCat(a, b os.Error) os.Error {
	if a != nil {
		return a
	}
	if b != nil {
		return b
	}
	return nil
}
