//  This file is part of MuMax, a high-perfomrance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// File for playing around with the gob package.
// Author: Arne Vansteenkiste

import (
	"testing"
	"gob"
	"io"
)


func BenchmarkGob(b *testing.B) {
	b.StopTimer()
	r, w := io.Pipe()
	dec := gob.NewDecoder(r)
	enc := gob.NewEncoder(w)

	N := 32 * 1024 * 1024
	in := make([]float32, N)
	out := make([]float32, N)

	// warm-up gob
	go enc.Encode(in)
	dec.Decode(&out)

	b.SetBytes(int64(4 * N))
	go func() {
		for i := 0; i < b.N; i++ {
			enc.Encode(in)
		}
	}()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		dec.Decode(&out)
	}

}

func TestGob(t *testing.T) {
	r, w := io.Pipe()
	dec := gob.NewDecoder(r)
	enc := gob.NewEncoder(w)

	N := 1024
	in := make([]float32, N)
	out := make([]float32, N)

	go enc.Encode(in)
	err2 := dec.Decode(&out)

	if err2 != nil {
		t.Fatal(err2)
	}
}
